"""WebSocket verification + liveness stream.

This is the browser/mobile replacement for the notebook's ``cv2.imshow`` loop.
The camera lives on the client; the server sees a stream of JPEGs.

Wire protocol (JSON text frames)
--------------------------------
client -> server

* ``{"type": "hello"}``                    -> capabilities + ready state
* ``{"type": "start_liveness", "user_id": "EMP101"}``
* ``{"type": "frame", "data": "<base64 jpeg>", "seq": 12}``
* ``{"type": "punch"}``  -- the identity comes from the last frame *this
  server* matched, never from the message. A ``user_id`` or ``match_distance``
  in the message is checked against the server's own result and refused on
  disagreement.
* ``{"type": "ping"}`` / ``{"type": "stop"}``

server -> client

* ``{"type": "ready"|"capabilities"|"liveness"|"result"|"punch"|"pong"|"error"|"bye"}``

Efficiency notes
----------------
Each incoming frame triggers **one** face analysis, fanned out to both the
match result and the liveness state machine. Frames arriving faster than
``max_fps`` are counted and dropped rather than queued, because queueing a
video frame backlog only makes the round trip worse.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import datetime as dt
import hashlib
import hmac
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from ..config import settings
from ..db import session_scope
from ..deps import rate_limiter
from ..errors import AppError, RateLimitedError
from ..services import events, verifications
from ..services.liveness.active import SUPPORTED_CHALLENGES
from ..services.liveness.service import (
    advance_session,
    create_session,
    liveness_available,
)
from ..services.liveness.service import (
    registry as liveness_registry,
)
from ..services.pipeline import annotate, verify_frame
from ..services.punch import punch_attendance
from ..utils.images import ImageDecodeError, decode_image, encode_jpeg

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/stream", tags=["stream"])

#: Frames per second the server will process. A 15 fps ceiling is plenty for
#: blink/yaw detection and halves the inference load versus 30 fps.
MAX_FPS = 15.0
MIN_FRAME_GAP = 1.0 / MAX_FPS
MAX_JPEG_QUALITY = 60


def _authenticate(websocket: WebSocket, api_key: str | None) -> tuple[str, str | None]:
    """Validate the API key.

    Returns ``(identity, None)`` on success or ``("", error_message)`` on
    failure. A 2-tuple rather than a single optional string: the identity and
    the error message are both strings, so returning either one as ``None``-
    detectable would make the caller unable to tell a successful handshake from
    a rejected one.

    Browsers cannot set headers on a WebSocket handshake, so ``?api_key=`` is
    accepted as a fallback. It is visible in proxy logs -- prefer a same-origin
    deployment, or a short-lived per-session token, over putting a long-lived
    key in a query string.
    """
    client = websocket.client
    fallback = f"ip:{client.host if client else 'ws'}"

    if not settings.auth_required:
        return fallback, None

    presented = websocket.headers.get("x-api-key") or api_key
    if not presented:
        return "", "missing X-API-Key (header or ?api_key= query param)"

    got = hashlib.sha256(presented.encode("utf-8")).digest()
    # compare_digest across every configured key: same runtime either way, but
    # it keeps the intent explicit and constant-time per comparison.
    allowed = [hashlib.sha256(k.encode("utf-8")).digest() for k in settings.api_key_list]
    if not any(hmac.compare_digest(got, candidate) for candidate in allowed):
        return "", "invalid API key"

    return f"key:{hashlib.sha256(presented.encode('utf-8')).hexdigest()[:16]}", None


@router.websocket("/verify")
async def stream_verify(
    websocket: WebSocket,
    api_key: str | None = Query(default=None, max_length=256),
    user_id: str | None = Query(default=None, max_length=64),
    annotate_frames: bool = Query(default=False),
    with_liveness: bool = Query(default=True),
) -> None:
    """Bidirectional frame stream. See the module docstring for the protocol."""
    # _authenticate returns the identity on success, or a *message* describing
    # the failure. Both are strings, so the failure cases must be distinguished
    # by a sentinel rather than by `is not None` -- which would treat every
    # successful handshake as an auth error.
    identity, identity_error = _authenticate(websocket, api_key)
    if identity_error is not None:
        # 1008 = policy violation. Accept first so the close reason is delivered.
        await websocket.accept()
        await websocket.send_json({"type": "error", "code": "unauthorized", "message": identity_error})
        await websocket.close(code=1008)
        return

    try:
        rate_limiter.enforce(f"ws:{identity}")
    except RateLimitedError as exc:
        # Only RateLimitedError means "slow down". Catching Exception here
        # would relabel a real bug as a 1013 backpressure close.
        await websocket.accept()
        await websocket.send_json(
            {"type": "error", "code": "rate_limited", "message": str(exc)}
        )
        await websocket.close(code=1013)
        return

    await websocket.accept()

    liveness_state: dict[str, Any] = {"session_id": None, "status": None}
    counters = {"frames": 0, "dropped": 0, "processed": 0}
    last_processed = 0.0
    last_match: dict[str, Any] | None = None

    availability = liveness_available()

    await websocket.send_json(
        {
            "type": "ready",
            "identity": identity,
            "liveness_available": availability["available"],
            "liveness_reasons": availability["reasons"],
            "require_liveness": settings.require_liveness,
            "challenges": list(SUPPORTED_CHALLENGES),
            "max_fps": MAX_FPS,
            "match_threshold": settings.match_threshold,
        }
    )

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "code": "bad_json", "message": "frame is not valid JSON"}
                )
                continue

            kind = message.get("type")
            seq = message.get("seq")

            if kind == "ping":
                await websocket.send_json({"type": "pong", "seq": seq, "t": time.time()})
                continue

            if kind == "stop":
                await websocket.send_json({"type": "bye"})
                break

            if kind == "punch":
                response = await _handle_punch(message, last_match, liveness_state)
                await websocket.send_json({"type": "punch", **response})
                # Only an *accepted* punch spends the socket's state. Clearing
                # it unconditionally was a denial of service dressed up as
                # safety: a client that sent a wrong `user_id`, or punched
                # before starting the challenge, threw away a liveness session
                # the honest user had just spent a minute earning -- and the
                # server's own match evidence with it.
                #
                # Spending on success is sufficient. The liveness token is
                # single-use in the database, so it cannot be double-spent even
                # if this state were kept; and the match evidence is this
                # process's own measurement, which a fresh frame replaces
                # anyway.
                if response.get("ok"):
                    liveness_state.update({"session_id": None, "status": None})
                    last_match = None
                continue

            if kind == "start_liveness":
                target_user = message.get("user_id") or user_id
                result = await run_in_threadpool(
                    _start_liveness, target_user
                )
                liveness_state.update(
                    {
                        "session_id": result.get("session_id"),
                        "status": result.get("status"),
                    }
                )
                await websocket.send_json({"type": "liveness", **result})
                continue

            if kind != "frame":
                await websocket.send_json(
                    {"type": "error", "code": "unknown_type", "message": f"unsupported type {kind!r}"}
                )
                continue

            data = message.get("data") or ""
            if not data:
                await websocket.send_json(
                    {"type": "error", "code": "empty_frame", "message": "no image data", "seq": seq}
                )
                continue

            counters["frames"] += 1
            now = time.monotonic()
            if now - last_processed < MIN_FRAME_GAP:
                counters["dropped"] += 1
                continue
            last_processed = now

            try:
                frame = await run_in_threadpool(_decode_frame, data)
            except ImageDecodeError as exc:
                await websocket.send_json(
                    {"type": "error", "code": "bad_image", "message": str(exc), "seq": seq}
                )
                continue

            # One analysis, fanned out to matching and (if active) liveness.
            result, jpeg = await run_in_threadpool(
                _analyse, frame, user_id, bool(annotate_frames)
            )
            counters["processed"] += 1
            last_match = result

            response: dict[str, Any] = {
                "type": "result",
                "seq": seq,
                **result,
                "counters": dict(counters),
            }

            session_id = liveness_state.get("session_id")
            if session_id and liveness_state.get("status") in (None, "pending"):
                face_bbox = result.get("primary_bbox")
                response["liveness"] = await run_in_threadpool(
                    _advance,
                    frame,
                    session_id,
                    face_bbox,
                    bool(result.get("face_present")),
                    result.get("passive_liveness"),
                    result.get("passive_flags") or [],
                    bool(result.get("face_quality_ok", True)),
                )
                liveness_state["status"] = response["liveness"].get("status")

            if jpeg:
                response["annotated"] = base64.b64encode(jpeg).decode("ascii")
            await websocket.send_json(response)
            continue

    except WebSocketDisconnect:
        logger.debug("stream client disconnected (%s)", identity)
    # A WebSocket handler has no narrower contract to catch: any unhandled
    # error becomes a close frame. logger.exception above satisfies BLE001.
    except Exception as exc:
        logger.exception("stream failed")
        # The peer is usually already gone, so this notify is best-effort.
        # It must not raise: a second exception here would replace the real
        # one and lose the traceback above.
        with contextlib.suppress(RuntimeError, OSError, WebSocketDisconnect):
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "internal_error",
                    # Same policy as the HTTP middleware: the detail is logged,
                    # not shipped to the browser, in prod.
                    "message": str(exc) if not settings.is_prod else "internal server error",
                }
            )
        with contextlib.suppress(RuntimeError, OSError, WebSocketDisconnect):
            await websocket.close(code=1011)
    finally:
        if liveness_state.get("session_id"):
            # Leave the DB row as the record of truth; drop the memory handle.
            liveness_registry.drop(liveness_state["session_id"])
        # `or {}` on the *value*, not just the key: a frame with no face stores
        # `match: None`, and `.get("match", {})` only substitutes its default
        # when the key is absent -- so the None survives and the next `.get`
        # raises. That is an AttributeError inside a `finally`, which means it
        # fired on every disconnect after a faceless frame and replaced whatever
        # the handler was really doing with a confusing traceback.
        last_name = ((last_match or {}).get("match") or {}).get("name")
        logger.info(
            "stream closed identity=%s frames=%d processed=%d dropped=%d last_match=%s",
            identity,
            counters["frames"],
            counters["processed"],
            counters["dropped"],
            last_name,
        )


# ---------------------------------------------------------------------------
# Thread-pool workers (SQLite sessions are not thread-safe to share)
# ---------------------------------------------------------------------------
def _decode_frame(data: str) -> Any:
    if "," in data[:64] and data.lstrip().startswith("data:"):
        data = data.partition(",")[2]
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageDecodeError("frame is not valid base64") from exc
    if len(raw) > settings.max_upload_bytes:
        raise ImageDecodeError(
            f"frame is {len(raw)} bytes, limit is {settings.max_upload_bytes}"
        )
    return decode_image(raw)


def _analyse(
    frame: Any, expected_user_id: str | None, want_annotate: bool
) -> tuple[dict[str, Any], bytes | None]:
    with session_scope() as db:
        result = verify_frame(db, frame, expected_user_id=expected_user_id)
        primary = max(result.observations, key=lambda o: o.area) if result.observations else None
        payload = {
            "face_present": result.face_present,
            "faces_detected": len(result.observations),
            "primary_bbox": list(primary.bbox) if primary else None,
            "frame_quality_ok": result.frame_quality.ok,
            "face_quality_ok": bool(result.face_quality.ok) if result.face_quality else False,
            "passive_liveness": result.passive.score if result.passive else None,
            "passive_verdict": result.passive.verdict if result.passive else None,
            "passive_flags": result.passive.flags if result.passive else [],
            "match": result.match.to_dict() if result.match else None,
            "eligible_for_attendance": result.eligible_for_attendance,
            "block_reasons": result.block_reasons,
            "warnings": result.warnings,
        }
        jpeg = None
        if want_annotate:
            jpeg = encode_jpeg(annotate(frame, result), MAX_JPEG_QUALITY)
        return payload, jpeg


async def _handle_punch(
    message: dict[str, Any],
    last_match: dict[str, Any] | None,
    liveness_state: dict[str, Any],
) -> dict[str, Any]:
    """Punch over the socket, from the server's own last match result.

    Nothing in ``message`` is used as evidence. The client may still send a
    ``user_id`` or a ``match_distance`` -- older kiosk builds do -- but they
    are compared against what this server measured and a disagreement is
    refused, not resolved in the caller's favour. The identity and the distance
    that reach :func:`punch_attendance` are the ones from ``last_match``,
    which came out of :func:`verify_frame` on a frame this process decoded.

    That is the same guarantee the HTTP path gets from a verification grant,
    produced by the same shared punch rules.
    """
    match = (last_match or {}).get("match") or {}
    if match.get("decision") != "match" or not match.get("user_id"):
        return {
            "ok": False,
            "code": "no_match_result",
            "message": "no recognised face to punch; send a frame first",
        }

    target = str(match["user_id"])
    claimed = message.get("user_id")
    if claimed and str(claimed) != target:
        # Not a liveness problem: the client thinks a different person is at the
        # camera than the matcher does. Refusing is the safe reading, and the
        # disagreement is worth an alert rather than a silent preference.
        with session_scope() as db:
            events.record_event(
                db,
                kind="attendance_blocked",
                severity="warning",
                user_id=target,
                message=(
                    f"socket punch claimed {claimed!r} but the server matched {target!r}"
                ),
                context={"claimed_user_id": str(claimed), "matched_user_id": target},
            )
        return {
            "ok": False,
            "code": "identity_mismatch",
            "message": (
                f"the frame matched {target!r}, not {claimed!r}; punch the person "
                "who is actually in front of the camera"
            ),
        }

    try:
        result = await run_in_threadpool(
            _do_punch,
            target,
            float(match["distance"]),
            liveness_state.get("session_id"),
            (last_match or {}).get("passive_liveness"),
            (last_match or {}).get("passive_flags") or [],
            (last_match or {}).get("passive_verdict"),
        )
    except AppError as exc:
        payload = exc.to_payload()["error"]
        return {"ok": False, "code": payload.get("code"), "message": payload.get("message")}

    return {
        "ok": True,
        "user_id": result["user_id"],
        "work_date": result["work_date"],
        "created": result["created"],
        "duplicate_suppressed": result["duplicate_suppressed"],
        "match_distance": result["match_distance"],
        "message": result["message"],
    }


def _do_punch(
    user_id: str,
    match_distance: float,
    liveness_session_id: str | None,
    passive: float | None,
    passive_flags: list[str],
    passive_verdict: str | None,
) -> dict[str, Any]:
    with session_scope() as db:
        # The socket mints its own grant from the match it already computed, so
        # the punch goes through exactly one code path and needs no second
        # inference pass. Flags only count as evidence when the server's own
        # passive check called it a spoof; advisory cues on a live face are not.
        grant = verifications.issue(
            db,
            user_id=user_id,
            distance=match_distance,
            passive_liveness=passive,
            spoof_flags=list(passive_flags) if passive_verdict == "spoof_suspect" else [],
            source="stream",
        )
        result, message = punch_attendance(
            db,
            verification_id=grant.id,
            liveness_session_id=liveness_session_id,
            source="stream",
        )
        return {
            "user_id": result.record.user_id,
            "work_date": result.record.work_date,
            "created": result.created,
            "duplicate_suppressed": result.duplicate_suppressed,
            "match_distance": result.record.match_distance,
            "message": message,
        }


def _json_safe(value: Any) -> Any:
    """Make a service-layer payload safe for ``websocket.send_json``.

    The HTTP routes declare Pydantic response models, so they coerce types for
    free. The socket sends raw dicts, and anything ``json.dumps`` cannot handle
    -- notably ``expires_at`` on a liveness session -- would raise mid-send and
    tear down the connection. Coerce here, once, instead.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    return value


def _start_liveness(target_user: str | None) -> dict[str, Any]:
    with session_scope() as db:
        return _json_safe(create_session(db, user_id=target_user))


def _advance(
    frame: Any,
    session_id: str,
    face_bbox: list[int] | None,
    face_present: bool,
    passive: float | None,
    flags: list[str],
    face_quality_ok: bool,
) -> dict[str, Any]:
    with session_scope() as db:
        return _json_safe(
            advance_session(
                db,
                session_id,
                frame,
                face_bbox=tuple(face_bbox) if face_bbox else None,
                face_present=face_present,
                passive_score=passive,
                passive_flags=flags,
                face_quality_ok=face_quality_ok,
                source="stream",
            )
        )
