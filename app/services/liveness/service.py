"""Liveness orchestration: session lifecycle, frame processing, token issuing.

Invariant worth stating explicitly
---------------------------------
The **in-memory engine is a fast path, not the source of truth.** A punch is
authorised only by the ``liveness_sessions`` *database* row: it must read
``status='passed'`` and ``consumed_at IS NULL``. So attendance integrity never
depends on process-local state surviving, which is what lets you run multiple
API workers.

The second invariant is the ``user_id`` on that row: a session authorises a
punch for *the person it was spent against* and nobody else. Without it, the
challenge is transferable -- pass the blink once and you may punch a
colleague. See :func:`consume_token`.

The trade-off, stated plainly: the interactive challenge state (baselines,
streaks, reveal clock) lives in memory for the duration of one session, so
liveness sessions are pinned to the worker that created them. Either run a
single worker, or enable sticky sessions on the liveness routes. A ``passed``
row can always be re-read from the database afterwards, so a dropped
connection at the last frame loses the session, not the audit trail.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ...config import settings
from ...models import LivenessSession as LivenessSessionRow
from ...models import utcnow
from ...services import events
from ..quality import assess_face, to_gray
from .active import LivenessEngine
from .geometry import extract_signals, eye_detector_status

logger = logging.getLogger(__name__)

__all__ = [
    "LivenessHandle",
    "LivenessRegistry",
    "advance_session",
    "consume_token",
    "create_session",
    "get_session",
    "liveness_available",
    "registry",
    "reset_registry",
    "submit_frame",
]


@dataclass
class LivenessHandle:
    row_id: str
    engine: LivenessEngine
    user_id: str | None
    created_at: dt.datetime
    expires_at: dt.datetime
    lock: threading.Lock = field(default_factory=threading.Lock)
    prev_eye_crops: dict[int, np.ndarray] = field(default_factory=dict)
    last_passive: float | None = None
    last_passive_flags: list[str] = field(default_factory=list)
    passive_checked: bool = False


class LivenessRegistry:
    """TTL-evicted in-memory handles."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._handles: dict[str, LivenessHandle] = {}

    def put(self, handle: LivenessHandle) -> None:
        with self._lock:
            self._handles[handle.row_id] = handle

    def get(self, session_id: str) -> LivenessHandle | None:
        with self._lock:
            return self._handles.get(session_id)

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._handles.pop(session_id, None)

    def sweep(self) -> int:
        with self._lock:
            stale = [k for k, h in self._handles.items() if h.expires_at <= utcnow()]
            for key in stale:
                self._handles.pop(key, None)
        return len(stale)

    def size(self) -> int:
        with self._lock:
            return len(self._handles)

    def clear(self) -> None:
        with self._lock:
            self._handles.clear()


registry = LivenessRegistry()


def reset_registry() -> None:
    """Test hook."""
    registry.clear()


def liveness_available() -> dict[str, Any]:
    eyes = eye_detector_status()
    reasons: list[str] = []
    if not eyes["available"]:
        reasons.append(f"eye_detector_unavailable: {eyes['error']}")
    if not settings.challenge_pool:
        reasons.append("no_challenges_configured")
    return {
        "available": not reasons,
        "reasons": reasons,
        "require_liveness": settings.require_liveness,
        "eye_detector": eyes,
    }


def _plan_snapshot(engine: LivenessEngine) -> list[dict[str, Any]]:
    """Store challenge state as offsets from session start (JSON-stable)."""
    return [
        {
            "name": ch.name,
            "index": ch.index,
            "instruction": ch.instruction,
            "reveal_offset_s": round(ch.revealed_at - engine.started_at, 3),
            "completed": ch.completed_at is not None,
            "detail": ch.detector.summary(),
        }
        for ch in engine.challenges
    ]


def _locked_out(db: Session, user_id: str | None) -> dt.datetime | None:
    """When this person may not start another challenge, or ``None``.

    ``LIVENESS_MAX_ATTEMPTS`` is enforced *per session* by the engine, which on
    its own means nothing to someone grinding the challenge: a failed session
    just ends, and the next one starts free. This is the cross-session half.

    Scoped to a *named* identity, because that is the attack it exists for --
    working on one employee until their face is captured. An anonymous session
    has no identity to count against, and inventing one from the socket address
    would need a proxy-aware client identity threaded this far down; the honest
    behaviour is to skip the lockout rather than pretend to enforce it.

    The count is taken over a trailing window of the same length, but the lock
    ends ``LIVENESS_LOCKOUT_SECONDS`` after the *last* failure, not when the
    window slides past. Window-only expiry reports an unlock time that has
    already passed, so the countdown in the refusal ("try again in 1s") is a
    lie the client will act on.
    """
    if user_id is None or settings.liveness_lockout_seconds <= 0:
        return None

    window_start = utcnow() - dt.timedelta(seconds=settings.liveness_lockout_seconds)
    recent_failures, last_failure_at = db.execute(
        select(
            func.count(LivenessSessionRow.id),
            func.max(LivenessSessionRow.completed_at),
        ).where(
            LivenessSessionRow.user_id == user_id,
            LivenessSessionRow.status == "failed",
            LivenessSessionRow.completed_at.isnot(None),
            LivenessSessionRow.completed_at >= window_start,
        )
    ).one()
    if last_failure_at is None or recent_failures < settings.liveness_max_attempts:
        return None

    unlock_at = last_failure_at + dt.timedelta(seconds=settings.liveness_lockout_seconds)
    return unlock_at if unlock_at > utcnow() else None


def create_session(db: Session, *, user_id: str | None = None) -> dict[str, Any]:
    """Start an active liveness session. Returns the client-facing payload."""
    availability = liveness_available()
    if not availability["available"]:
        events.record_event(
            db,
            kind="liveness_unavailable",
            severity="critical",
            user_id=user_id,
            message="Active liveness requested but the eye detector is unavailable",
            context=availability,
        )
        return {
            "session_id": None,
            "status": "unavailable",
            "reason": "; ".join(availability["reasons"]),
            "challenges": [],
            "prompt": None,
        }

    unlock_at = _locked_out(db, user_id)
    if unlock_at is not None:
        events.record_event(
            db,
            kind="liveness_locked_out",
            severity="warning",
            user_id=user_id,
            message=(
                f"liveness refused: {settings.liveness_max_attempts} failed attempts "
                f"in the last {settings.liveness_lockout_seconds}s"
            ),
            context={"unlock_at": unlock_at.isoformat()},
        )
        return {
            "session_id": None,
            "status": "locked_out",
            "reason": (
                f"too many failed liveness attempts; try again in "
                f"{max(1, int((unlock_at - utcnow()).total_seconds()))}s"
            ),
            "challenges": [],
            "prompt": None,
        }

    engine = LivenessEngine(
        settings.challenge_pool,
        min_challenges=settings.active_liveness_min_challenges,
        ttl_seconds=settings.liveness_session_ttl_seconds,
        max_attempts=settings.liveness_max_attempts,
    )
    now = utcnow()
    row = LivenessSessionRow(
        id=uuid.uuid4().hex,
        user_id=user_id,
        status="pending",
        challenges=_plan_snapshot(engine),
        attempts=0,
        max_attempts=settings.liveness_max_attempts,
        created_at=now,
        updated_at=now,
        expires_at=now + dt.timedelta(seconds=settings.liveness_session_ttl_seconds),
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    registry.put(
        LivenessHandle(
            row_id=row.id,
            engine=engine,
            user_id=user_id,
            created_at=now,
            expires_at=row.expires_at,
        )
    )

    verdict = engine.verdict()
    return {
        "session_id": row.id,
        "status": row.status,
        # ISO string, not a datetime: this payload is also sent over the
        # WebSocket, where json.dumps has no datetime encoder. A raw datetime
        # serialises fine over REST (pydantic) and raises "Object of type
        # datetime is not JSON serializable" on the socket path.
        "expires_at": row.expires_at.isoformat(),
        "max_challenges": len(engine.challenges),
        "required": settings.active_liveness_min_challenges,
        **verdict.to_dict(),
    }


def get_session(db: Session, session_id: str) -> LivenessSessionRow | None:
    return db.get(LivenessSessionRow, session_id)


def submit_frame(
    db: Session,
    session_id: str,
    frame_bgr: np.ndarray,
    *,
    source: str = "api",
) -> dict[str, Any]:
    """Self-contained path: analyse the frame, then advance the challenge.

    Use this from HTTP routes. The WebSocket route calls
    :func:`advance_session` directly instead, because it has already run
    analysis for the match result and a second DeepFace pass would double the
    cost of every frame.
    """
    handle = registry.get(session_id)
    row = get_session(db, session_id)
    guard = _guard(row, handle)
    if guard is not None:
        return guard
    assert handle is not None and row is not None

    with handle.lock:
        from ..face import get_analyzer  # local import avoids an import cycle

        now_ts = engine_now(handle)
        try:
            observations = get_analyzer().analyze(frame_bgr, strict=False)
        except Exception as exc:  # noqa: BLE001 - a frame failure is not a session failure
            logger.warning("frame analysis failed: %s", exc)
            observations = []

        return _advance(
            db,
            row,
            handle,
            frame_bgr,
            observations,
            now_ts=now_ts,
            source=source,
        )


def advance_session(
    db: Session,
    session_id: str,
    frame_bgr: np.ndarray,
    *,
    face_bbox: tuple[int, int, int, int] | None,
    face_present: bool,
    passive_score: float | None = None,
    passive_flags: list[str] | None = None,
    face_quality_ok: bool = True,
    source: str = "stream",
) -> dict[str, Any]:
    """Advance a session from an already-analysed frame.

    The caller (the WebSocket route) has run face detection + embedding
    already; this only extracts the cheap eye/geometry signals and drives the
    state machine.
    """
    handle = registry.get(session_id)
    row = get_session(db, session_id)
    guard = _guard(row, handle)
    if guard is not None:
        return guard
    assert handle is not None and row is not None

    with handle.lock:
        if not face_present or not face_quality_ok:
            verdict = handle.engine.update(_absent_signals(engine_now(handle)))
            _persist(db, row, handle.engine, verdict, passive=passive_score, flags=passive_flags)
            return _response(row, verdict, signals=None, passive=passive_score)

        assert face_bbox is not None
        signals, prev_crops = extract_signals(
            to_gray(frame_bgr), face_bbox, engine_now(handle), handle.prev_eye_crops
        )
        handle.prev_eye_crops = prev_crops
        if passive_score is not None:
            handle.last_passive = passive_score
            handle.last_passive_flags = list(passive_flags or [])
            handle.passive_checked = True

        return _advance(
            db,
            row,
            handle,
            frame_bgr,
            observations=[],
            now_ts=signals.timestamp,
            source=source,
            signals=signals,
            passive_score=passive_score,
            passive_flags=passive_flags,
        )


def engine_now(handle: LivenessHandle) -> float:
    return handle.engine.now_fn()


def _guard(row: LivenessSessionRow | None, handle: LivenessHandle | None) -> dict[str, Any] | None:
    """Shared pre-flight checks. Returns an error payload, or None to proceed."""
    if row is None:
        return {"status": "unknown_session", "reason": "session not found"}
    if handle is None:
        return {
            "status": "stale_session",
            "reason": (
                "interactive state for this session is not held by this worker; "
                "retry against the worker that created it"
            ),
            "db_status": row.status,
        }
    if row.status in ("passed", "failed", "expired"):
        return {
            "session_id": row.id,
            "db_status": row.status,
            "status": row.status,
            "reason": row.failure_reason,
        }
    return None


def _advance(
    db: Session,
    row: LivenessSessionRow,
    handle: LivenessHandle,
    frame_bgr: np.ndarray,
    observations: list,
    *,
    now_ts: float,
    source: str,
    signals=None,
    passive_score: float | None = None,
    passive_flags: list[str] | None = None,
) -> dict[str, Any]:
    from ..quality import to_gray as _to_gray
    from .passive import passive_liveness as _passive

    engine = handle.engine
    passive = passive_score
    flags = list(passive_flags or [])

    if signals is None:
        if not observations:
            verdict = engine.update(_absent_signals(now_ts))
            _persist(db, row, engine, verdict, passive=None, flags=None)
            return _response(row, verdict, signals=None, passive=None)

        # Track the largest face: closest to camera, least occluded.
        face = max(observations, key=lambda o: o.area)
        crop = face.crop(frame_bgr)
        quality = assess_face(_to_gray(crop), face.w, face.h)

        if quality.ok and passive is None:
            passive_verdict = _passive(crop, threshold=settings.passive_liveness_min)
            passive = passive_verdict.score
            flags = passive_verdict.flags
            handle.last_passive = passive_verdict.score
            handle.last_passive_flags = passive_verdict.flags
            handle.passive_checked = True

        signals, prev_crops = extract_signals(
            _to_gray(frame_bgr), face.bbox, now_ts, handle.prev_eye_crops
        )
        handle.prev_eye_crops = prev_crops

    if not flags and handle.last_passive_flags:
        flags = handle.last_passive_flags

    verdict = engine.update(signals)
    _persist(db, row, engine, verdict, passive=passive, flags=flags)
    _maybe_alert(db, row, engine, verdict, signals, passive, flags, source)
    return _response(row, verdict, signals=signals, passive=passive)


def _absent_signals(ts: float):
    from .geometry import FrameSignals

    return FrameSignals(timestamp=ts, face_present=False)


def _persist(
    db: Session,
    row: LivenessSessionRow,
    engine: LivenessEngine,
    verdict,
    *,
    passive: float | None,
    flags: list[str] | None,
) -> None:
    row.challenges = _plan_snapshot(engine)
    row.status = verdict.status if verdict.status != "pending" else "pending"
    row.attempts = engine.attempts
    if passive is not None:
        row.passive_score = passive
    if verdict.status == "passed":
        # The engine's own score, stored verbatim. Without this the column stays
        # NULL forever, which is not cosmetic: `consume_token` falls back to half
        # the passive score, so the `liveness_score` written onto the attendance
        # row would be a number this service never measured, and
        # `GET /liveness/sessions/{id}` reports `score: 0.0` for a session that
        # passed. The live response already carried the real value via
        # `verdict.to_dict()`; this makes the persisted row agree with it.
        row.active_score = verdict.score
    if verdict.status in ("passed", "failed", "expired"):
        if row.completed_at is None:
            row.completed_at = utcnow()
        row.failure_reason = verdict.reason
    row.updated_at = utcnow()
    db.commit()

    if verdict.status in ("failed", "expired"):
        registry.drop(row.id)


def _maybe_alert(
    db: Session,
    row: LivenessSessionRow,
    engine: LivenessEngine,
    verdict,
    signals,
    passive: float | None,
    flags: list[str],
    source: str,
) -> None:
    """Raise an alert on a failed challenge test.

    ``insufficient_eye_evidence`` is a *warning*, not a critical alert: it
    usually means sunglasses or bad lighting rather than an attack, and crying
    wolf on honest users is how liveness systems get switched off.
    """
    if verdict.status != "failed":
        return

    reason = verdict.reason or "unknown"
    benign = reason in ("insufficient_eye_evidence", "insufficient_eye_evidence_budget")
    severity = "warning" if benign else "critical"
    context: dict[str, Any] = {
        "session_id": row.id,
        "reason": reason,
        "source": source,
        "passive_score": passive,
        "passive_flags": flags,
        "challenges": _plan_snapshot(engine),
    }
    if signals is not None:
        context["signals"] = signals.to_dict()

    events.record_event(
        db,
        kind="spoof_detected" if not benign else "liveness_failed",
        severity=severity,
        user_id=row.user_id,
        message=f"Active liveness {verdict.status}: {reason}",
        context=context,
    )


def _response(row: LivenessSessionRow, verdict, *, signals, passive: float | None) -> dict[str, Any]:
    payload = {
        "session_id": row.id,
        "status": row.status,
        **verdict.to_dict(),
        "passive_score": round(passive, 4) if passive is not None else None,
    }
    if signals is not None:
        payload["signals"] = signals.to_dict()
        payload["frame_ok"] = signals.reliable
    return payload


def consume_token(db: Session, session_id: str, *, user_id: str | None = None) -> tuple[bool, str, float]:
    """Authorise one attendance punch against a passed liveness session.

    Returns ``(ok, reason, score)``. This is the single gate that makes a
    replayed photo useless: without a fresh, unconsumed, passed session there
    is no way to write attendance when ``REQUIRE_LIVENESS=true``.

    ``user_id`` is the person the *server* verified the punch for, and passing
    it is what makes the session non-transferable. Without that check, passing
    the challenge authorises a punch for whoever the caller names, and the
    challenge stops being evidence of anything.

    A session that was started without a claimed identity (the common 1:N
    case, where nobody knows who is in front of the camera yet) adopts the
    first person that spends it, and is consumed in the same transaction. The
    adoption is therefore not a race: there is exactly one winner, and the
    loser sees ``liveness_token_already_used`` rather than a second punch.
    """
    row = get_session(db, session_id)
    if row is None:
        return False, "unknown_liveness_session", 0.0
    if row.status != "passed":
        return False, f"liveness_{row.status}", 0.0
    if row.consumed_at is not None:
        return False, "liveness_token_already_used", 0.0
    if row.expires_at <= utcnow():
        return False, "liveness_token_expired", 0.0

    if user_id is not None:
        if row.user_id is not None and row.user_id != user_id:
            # Deliberately *not* consumed: the rightful owner of this session
            # may still be standing there.
            return False, "liveness_session_user_mismatch", 0.0
        if row.user_id is None:
            row.user_id = user_id

    score = float(row.active_score or 0.0)
    if row.passive_score is not None:
        # Corroborate, but do not average the weak signal into the strong one.
        score = max(score, float(row.passive_score) * 0.5)

    row.consumed_at = utcnow()
    db.commit()
    return True, "ok", score
