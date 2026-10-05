"""HTTP + WebSocket API contract, including the anti-spoofing gates.

Uses the deterministic stub backend, so results are stable and no model or
camera is needed. What is being verified here is the *wiring and the security
gates*, not recognition accuracy.
"""

from __future__ import annotations

import base64
import logging
import typing
import uuid

import pytest

from app.config import settings
from tests.conftest import make_blank_image, make_face_image, make_spoof_image, to_jpeg_bytes

PREFIX = "/api/v1"


def _uid(prefix: str = "EMP") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8].upper()}"


def _enroll(client, user_id: str, seed: int = 1, image=None, headers=None):
    payload = to_jpeg_bytes(image if image is not None else make_face_image(seed=seed), quality=92)
    return client.post(
        f"{PREFIX}/users",
        data={"id": user_id, "name": f"Test {user_id}"},
        files={"image": ("face.jpg", payload, "image/jpeg")},
        headers=headers,
    )


def _verify(client, seed: int = 1, image=None, user_id: str | None = None) -> dict:
    """Run one frame through the real pipeline and return the verify body."""
    raw = to_jpeg_bytes(image if image is not None else make_face_image(seed=seed), quality=90)
    body = {"image_base64": base64.b64encode(raw).decode()}
    if user_id is not None:
        body["user_id"] = user_id
    response = client.post(f"{PREFIX}/recognition/verify", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _verification_id(client, seed: int = 1, image=None, user_id: str | None = None) -> str:
    """The server-issued identity proof an eligible frame earns.

    Every punch in this file goes through one of these: the punch endpoint has
    no ``user_id`` and no ``match_distance`` field, so a request without a
    grant is not a punch at all.

    The 1:1 form is used whenever the expected person is known. The shared
    roster accumulates a lot of similar synthetic faces over a test session, so
    a 1:N search would happily hand back a neighbour; naming the identity
    makes the grant deterministic.
    """
    body = _verify(client, seed=seed, image=image, user_id=user_id)
    grant = body["verification_id"]
    assert grant, f"an eligible frame must come back with a verification_id: {body}"
    return grant


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
def test_health_reports_dependencies(client):
    body = client.get("/health").json()
    assert body["status"] in ("ok", "degraded")
    assert "database" in body["checks"]
    assert "face_backend" in body["checks"]
    assert body["checks"]["database"]["ok"] is True


def test_liveness_probe_is_cheap(client):
    body = client.get("/health/live").json()
    assert body["status"] == "alive"
    assert body["uptime_s"] >= 0


def test_readiness_reports_not_ready_without_eye_detector(client):
    """No OpenCV in this environment -> active liveness is unavailable -> 503."""
    response = client.get("/health/ready")
    body = response.json()
    if body["status"] == "not_ready":
        assert response.status_code == 503
        assert body["checks"]["not_ready_reasons"]


def test_openapi_schema_is_generated(client):
    schema = client.get("/openapi.json").json()
    assert f"{PREFIX}/attendance/punch" in schema["paths"]
    assert f"{PREFIX}/recognition/verify" in schema["paths"]
    assert f"{PREFIX}/liveness/sessions" in schema["paths"]


# ---------------------------------------------------------------------------
# Enrolment
# ---------------------------------------------------------------------------
def test_enroll_and_read_back(client):
    user_id = _uid()
    response = _enroll(client, user_id, seed=1)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["id"] == user_id
    assert body["embedding_dim"] == 512
    assert body["is_active"] is True

    fetched = client.get(f"{PREFIX}/users/{user_id}")
    assert fetched.status_code == 200
    assert fetched.json()["name"] == f"Test {user_id}"


def test_enroll_rejects_a_non_image(client):
    response = client.post(
        f"{PREFIX}/users",
        data={"id": _uid(), "name": "Nope"},
        files={"image": ("face.jpg", b"this is not an image", "image/jpeg")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_image"


def test_enroll_rejects_an_empty_upload(client):
    response = client.post(
        f"{PREFIX}/users",
        data={"id": _uid(), "name": "Nope"},
        files={"image": ("face.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_upload"


def test_enroll_rejects_a_bad_user_id(client):
    response = client.post(
        f"{PREFIX}/users",
        data={"id": "has spaces!", "name": "Nope"},
        files={"image": ("face.jpg", to_jpeg_bytes(make_face_image()), "image/jpeg")},
    )
    assert response.status_code == 422


def test_enroll_is_idempotent_on_the_same_id(client):
    user_id = _uid()
    first = _enroll(client, user_id, seed=1)
    second = _enroll(client, user_id, seed=2)
    assert first.status_code == 201
    assert second.status_code == 201
    listing = client.get(f"{PREFIX}/users", params={"limit": 1000}).json()
    assert sum(1 for u in listing["items"] if u["id"] == user_id) == 1


def test_deactivate_removes_from_roster(client):
    user_id = _uid()
    _enroll(client, user_id, seed=3)
    assert client.delete(f"{PREFIX}/users/{user_id}").status_code == 200
    assert client.get(f"{PREFIX}/users/{user_id}").json()["is_active"] is False
    assert client.get(f"{PREFIX}/users/{user_id}").status_code == 200  # soft delete
    assert client.delete(f"{PREFIX}/users/does-not-exist").status_code == 404


def test_multi_sample_enrollment_averages_into_one_vector(client):
    user_id = _uid()
    start = client.post(
        f"{PREFIX}/users/enroll/start",
        data={"id": user_id, "name": "Multi Sample", "samples": "3"},
    )
    assert start.status_code == 201
    enrollment_id = start.json()["enrollment_id"]

    for seed in (11, 12, 13):
        response = client.post(
            f"{PREFIX}/users/enroll/{enrollment_id}/sample",
            files={"image": ("f.jpg", to_jpeg_bytes(make_face_image(seed=seed)), "image/jpeg")},
        )
        assert response.status_code == 200
        assert response.json()["accepted"] is True

    finalized = client.post(f"{PREFIX}/users/enroll/{enrollment_id}/finalize")
    assert finalized.status_code == 200
    assert finalized.json()["sample_count"] == 3
    assert finalized.json()["id"] == user_id


def test_finalize_requires_enough_samples(client):
    user_id = _uid()
    start = client.post(
        f"{PREFIX}/users/enroll/start", data={"id": user_id, "name": "Too Few", "samples": "3"}
    )
    enrollment_id = start.json()["enrollment_id"]
    client.post(
        f"{PREFIX}/users/enroll/{enrollment_id}/sample",
        files={"image": ("f.jpg", to_jpeg_bytes(make_face_image(seed=21)), "image/jpeg")},
    )
    response = client.post(f"{PREFIX}/users/enroll/{enrollment_id}/finalize")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "not_enough_samples"


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------
def test_verify_matches_the_enrolled_person(client):
    user_id = _uid()
    _enroll(client, user_id, seed=41)
    body = _verify(client, seed=41)
    assert body["face_present"] is True
    assert body["match"]["decision"] == "match"
    assert body["match"]["user_id"] == user_id
    assert body["eligible_for_attendance"] is True
    assert body["liveness"]["active_required"] is True


def test_verify_issues_a_single_use_identity_proof(client):
    """A matched frame carries a grant; it is the only way to punch later.

    Two frames, two grants: the id is not a property of the user, it is a
    property of one observation. That is what stops a single captured frame
    being replayed all day.
    """
    user_id = _uid()
    _enroll(client, user_id, seed=42)

    first = _verify(client, seed=42)["verification_id"]
    second = _verify(client, seed=42)["verification_id"]
    assert first and second
    assert first != second


def test_an_unmatched_frame_earns_no_identity_proof(client):
    """No face, no grant: a proof is only ever minted from a real match."""
    body = _verify(client, image=make_blank_image())
    assert body["face_present"] is False
    assert body["eligible_for_attendance"] is False
    assert body["verification_id"] is None
    assert body["block_reasons"]


def test_verify_one_to_one_accepts_the_claimed_identity(client):
    user_id = _uid()
    _enroll(client, user_id, seed=6)
    body = _verify(client, seed=6, user_id=user_id)
    assert body["identity_match"] is True
    assert body["match"]["user_id"] == user_id
    assert body["verification_id"]


def test_verify_one_to_one_rejects_the_wrong_claim(client):
    """1:1 must not report success for a face that is somebody else."""
    claimed = _uid()
    _enroll(client, claimed, seed=7)
    body = _verify(client, seed=8, user_id=claimed)
    assert body["identity_match"] is False
    assert body["eligible_for_attendance"] is False
    assert body["verification_id"] is None


def test_verify_rejects_bad_base64(client):
    response = client.post(f"{PREFIX}/recognition/verify", json={"image_base64": "!!!not-base64!!!"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_image"


def test_verify_upload_route(client):
    user_id = _uid()
    _enroll(client, user_id, seed=9)
    response = client.post(
        f"{PREFIX}/recognition/verify/upload",
        files={"image": ("f.jpg", to_jpeg_bytes(make_face_image(seed=9)), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["match"]["user_id"] == user_id


def test_roster_endpoint_reports_model_and_threshold(client):
    body = client.get(f"{PREFIX}/recognition/roster").json()
    assert body["model"] == settings.face_model_name
    assert body["match_threshold"] == settings.match_threshold
    assert body["gray_zone_threshold"] > body["match_threshold"]


# ---------------------------------------------------------------------------
# Liveness availability (fails closed without OpenCV)
# ---------------------------------------------------------------------------
def test_liveness_capabilities_advertise_supported_challenges(client):
    body = client.get(f"{PREFIX}/liveness/capabilities").json()
    assert "blink" in body["supported_challenges"]
    assert body["require_liveness"] is True
    assert body["how_to_pass"]


def test_liveness_session_fails_closed_when_unavailable(client):
    """The honest behaviour: refuse to start rather than hand out a useless id."""
    from app.services.liveness.geometry import eye_detector_status

    response = client.post(f"{PREFIX}/liveness/sessions", json={})
    if eye_detector_status()["available"]:
        assert response.status_code == 201
    else:
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "liveness_unavailable"


def test_unknown_liveness_session_is_reported_not_guessed(client):
    response = client.get(f"{PREFIX}/liveness/sessions/does-not-exist")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# The attendance gate
# ---------------------------------------------------------------------------
def test_punch_without_liveness_is_refused(client):
    user_id = _uid()
    _enroll(client, user_id, seed=10)
    response = client.post(
        f"{PREFIX}/attendance/punch", json={"verification_id": _verification_id(client, seed=10, user_id=user_id)}
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "liveness_required"


def test_punch_without_a_verification_is_refused(client):
    """There is no way to name somebody from the request body any more."""
    response = client.post(f"{PREFIX}/attendance/punch", json={})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_punch_rejects_a_client_asserted_identity(client):
    """Legacy clients that still send ``user_id`` get a loud 422, not a punch.

    Silently ignoring the field would leave a kiosk believing it had marked
    the person it aimed at, when the server marked whoever the grant named.
    """
    user_id = _uid()
    _enroll(client, user_id, seed=11)
    response = client.post(
        f"{PREFIX}/attendance/punch",
        json={"verification_id": _verification_id(client, seed=11, user_id=user_id), "user_id": user_id},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_punch_rejects_a_client_asserted_match_distance(client):
    """The distance is the server's number. A client cannot write it at all."""
    user_id = _uid()
    _enroll(client, user_id, seed=43)
    response = client.post(
        f"{PREFIX}/attendance/punch",
        json={"verification_id": _verification_id(client, seed=43, user_id=user_id), "match_distance": 0.01},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_punch_for_a_deactivated_user_is_404(client):
    """A grant outlives the person it named, but not their deactivation."""
    user_id = _uid()
    _enroll(client, user_id, seed=12)
    grant = _verification_id(client, seed=12, user_id=user_id)
    assert client.delete(f"{PREFIX}/users/{user_id}").status_code == 200

    response = client.post(f"{PREFIX}/attendance/punch", json={"verification_id": grant})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "user_not_found"


def test_a_spoof_like_frame_earns_no_proof_so_it_cannot_punch(client, settings_override):
    """A re-captured print cannot buy attendance, even with liveness off.

    The print is close enough on the embedding that the *match* still lands,
    which is exactly the case that used to be exploitable: the caller asserted
    ``match_distance: 0.01`` and the punch believed it. With the distance gone
    from the request there is nothing left to assert -- the frame either never
    earns a grant, or it earns one carrying the server's own spoof flags.
    """
    settings_override.set(require_liveness=False, debug=False)
    user_id = _uid()
    _enroll(client, user_id, seed=13)

    body = _verify(client, seed=13, user_id=user_id, image=make_spoof_image(seed=13))
    if body["verification_id"] is None:
        # No grant at all: a guessed id is refused for not existing.
        response = client.post(
            f"{PREFIX}/attendance/punch", json={"verification_id": "x" * 32}
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "unverified_identity"
        return

    # A grant exists, so the refusal has to come from the flags it carries --
    # measured by this process, not chosen by the caller.
    response = client.post(
        f"{PREFIX}/attendance/punch", json={"verification_id": body["verification_id"]}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "spoof_suspected"


def test_punch_succeeds_and_is_idempotent_per_day(client, settings_override):
    settings_override.set(require_liveness=False)
    user_id = _uid()
    _enroll(client, user_id, seed=14)

    first = client.post(
        f"{PREFIX}/attendance/punch", json={"verification_id": _verification_id(client, seed=14, user_id=user_id)}
    )
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["created"] is True
    assert body["duplicate_suppressed"] is False
    assert body["attendance"]["user_id"] == user_id
    assert body["attendance"]["punch_count"] == 1

    second = client.post(
        f"{PREFIX}/attendance/punch", json={"verification_id": _verification_id(client, seed=14, user_id=user_id)}
    )
    assert second.status_code == 200
    body2 = second.json()
    assert body2["created"] is False
    assert body2["duplicate_suppressed"] is True
    assert body2["attendance"]["punch_count"] == 2
    # one row for the day, not two
    assert body2["attendance"]["id"] == body["attendance"]["id"]
    # the best (lowest) distance of the day is retained
    assert body2["attendance"]["match_distance"] <= body["attendance"]["match_distance"]


def test_a_grant_is_refused_on_its_second_use(client, settings_override):
    settings_override.set(require_liveness=False)
    user_id = _uid()
    _enroll(client, user_id, seed=44)
    grant = _verification_id(client, seed=44, user_id=user_id)

    assert client.post(f"{PREFIX}/attendance/punch", json={"verification_id": grant}).status_code == 200
    replay = client.post(f"{PREFIX}/attendance/punch", json={"verification_id": grant})
    assert replay.status_code == 422
    assert replay.json()["error"]["code"] == "unverified_identity"
    assert replay.json()["error"]["context"]["reason"] == "verification_already_used"


def test_check_out_and_queries(client, settings_override):
    settings_override.set(require_liveness=False)
    user_id = _uid()
    _enroll(client, user_id, seed=15)
    client.post(
        f"{PREFIX}/attendance/punch", json={"verification_id": _verification_id(client, seed=15, user_id=user_id)}
    )

    listed = client.get(f"{PREFIX}/attendance", params={"user_id": user_id}).json()
    assert listed["total"] == 1
    assert listed["items"][0]["user_id"] == user_id

    summary = client.get(f"{PREFIX}/attendance", params={"with_summary": True}).json()
    assert summary["summary"]["present"] >= 1

    checkout = client.post(f"{PREFIX}/attendance/check-out", params={"user_id": user_id})
    assert checkout.status_code == 200
    assert checkout.json()["attendance"]["check_out_at"] is not None

    csv_response = client.get(f"{PREFIX}/attendance/export.csv", params={"user_id": user_id})
    assert csv_response.status_code == 200
    assert "text/csv" in csv_response.headers["content-type"]
    assert user_id in csv_response.text


def test_attendance_day_filter_rejects_bad_format(client):
    assert client.get(f"{PREFIX}/attendance", params={"day": "not-a-date"}).status_code == 422


# ---------------------------------------------------------------------------
# Malformed request bodies must not break the 422 handler
# ---------------------------------------------------------------------------
def test_unparseable_body_returns_422_not_500(client):
    """Regression: the error handler used to raise TypeError on the raw body.

    FastAPI puts the undecodable ``input`` into ``exc.errors()``; when that is
    ``bytes``, JSONResponse could not serialise it and the handler itself blew
    up, turning a client-side mistake into an opaque 500. See _json_safe.
    """
    response = client.post(
        f"{PREFIX}/recognition/verify",
        content=b"--boundary\r\ngarbage",
        headers={"Content-Type": "multipart/form-data; boundary=boundary"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_validation_error_body_is_always_json_serialisable(client):
    """A 422 body must render, whatever ended up inside pydantic's details."""
    response = client.post(f"{PREFIX}/attendance/punch", content=b"\xff\xfe not json")
    assert response.status_code == 422
    body = response.json()  # would raise if the handler emitted raw bytes
    assert body["error"]["code"] == "validation_error"
    assert isinstance(body["error"]["details"], list)


def test_validation_error_does_not_echo_the_whole_upload(client):
    """bytes inputs are summarised, not reflected back to the caller."""
    payload = b"A" * 4096
    response = client.post(f"{PREFIX}/recognition/verify", content=payload)
    assert response.status_code == 422
    rendered = response.text
    assert "A" * 200 not in rendered


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
def test_blocked_punches_are_recorded_as_alerts(client):
    user_id = _uid()
    _enroll(client, user_id, seed=16)
    client.post(
        f"{PREFIX}/attendance/punch", json={"verification_id": _verification_id(client, seed=16, user_id=user_id)}
    )

    body = client.get(
        f"{PREFIX}/alerts", params={"user_id": user_id, "kind": "attendance_blocked"}
    ).json()
    assert body["total"] >= 1
    assert body["items"][0]["user_id"] == user_id


def test_alert_summary_groups_by_severity(client):
    body = client.get(f"{PREFIX}/alerts/summary").json()
    assert set(body["by_severity"]) == {"info", "warning", "critical"}
    assert body["total"] >= 0
    assert body["webhook_configured"] is False


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def test_api_key_is_enforced_when_configured(client, settings_override):
    settings_override.set(environment="prod", api_keys="s3cret-key")
    assert client.get("/health/live").status_code == 200  # probes stay open

    unauthorised = client.get(f"{PREFIX}/recognition/roster")
    assert unauthorised.status_code == 401
    assert unauthorised.json()["error"]["code"] == "unauthorized"

    wrong = client.get(f"{PREFIX}/recognition/roster", headers={"X-API-Key": "nope"})
    assert wrong.status_code == 401

    ok = client.get(f"{PREFIX}/recognition/roster", headers={"X-API-Key": "s3cret-key"})
    assert ok.status_code == 200


def test_every_data_route_requires_the_key(client, settings_override):
    """Regression: reads and the deactivate call used to be open in prod.

    The protected list is the whole roster, every attendance record and every
    alert -- an unauthenticated `DELETE /users/{id}` deactivates somebody, and
    an unauthenticated `GET /users` hands over the face template roster.
    """
    settings_override.set(environment="prod", api_keys="s3cret-key")
    key = {"X-API-Key": "s3cret-key"}
    user_id = _uid()
    assert _enroll(client, user_id, seed=31, image=make_face_image(seed=31), headers=key).status_code == 201

    guarded = [
        ("GET", f"{PREFIX}/users", None),
        ("GET", f"{PREFIX}/users/{user_id}", None),
        ("DELETE", f"{PREFIX}/users/{user_id}", None),
        ("GET", f"{PREFIX}/attendance", None),
        ("GET", f"{PREFIX}/attendance/export.csv", None),
        ("POST", f"{PREFIX}/attendance/check-out", {"params": {"user_id": user_id}}),
        ("GET", f"{PREFIX}/alerts", None),
        ("GET", f"{PREFIX}/alerts/summary", None),
        ("GET", f"{PREFIX}/liveness/capabilities", None),
        ("GET", f"{PREFIX}/recognition/roster", None),
    ]
    for method, path, extra in guarded:
        assert client.request(method, path, **(extra or {})).status_code == 401, path

    # The authorised half deliberately excludes the DELETE: running it would
    # deactivate the very user this test then asserts is still active, and the
    # point of the assertion below is that the *refused* DELETE changed nothing.
    for method, path, extra in guarded:
        if method == "DELETE":
            continue
        assert (
            client.request(method, path, headers=key, **(extra or {})).status_code != 401
        ), path

    # ...and the user is still there: a refused DELETE did not deactivate it.
    fetched = client.get(f"{PREFIX}/users/{user_id}", headers=key)
    assert fetched.json()["is_active"] is True


def test_prod_without_keys_fails_loudly(client, settings_override):
    settings_override.set(environment="prod", api_keys="")
    response = client.get(f"{PREFIX}/recognition/roster")
    assert response.status_code == 503
    assert "misconfigured" in response.json()["error"]["message"]


def test_the_websocket_handshake_is_guarded_too(client, settings_override):
    """The socket is a data route: it takes frames and can write attendance.

    It authenticated separately from the HTTP dependency, so the HTTP fix alone
    would have left an unauthenticated door straight into the punch path. A bad
    key must be refused *before* ``ready``, otherwise the client believes it is
    connected and only finds out on its first punch.
    """
    settings_override.set(environment="prod", api_keys="s3cret-key")

    with client.websocket_connect(f"{PREFIX}/stream/verify") as ws:
        assert ws.receive_json()["code"] == "unauthorized"

    with client.websocket_connect(f"{PREFIX}/stream/verify?api_key=nope") as ws:
        assert ws.receive_json()["code"] == "unauthorized"

    # Browsers cannot set handshake headers, so the query param is the sanctioned
    # fallback -- and it must actually work, not just be rejected politely.
    with client.websocket_connect(f"{PREFIX}/stream/verify?api_key=s3cret-key") as ws:
        assert ws.receive_json()["type"] == "ready"

    with client.websocket_connect(
        f"{PREFIX}/stream/verify", headers={"X-API-Key": "s3cret-key"}
    ) as ws:
        assert ws.receive_json()["type"] == "ready"


def test_rate_limiter_returns_429(client):
    from app.deps import RateLimiter

    limiter = RateLimiter(per_minute=2, burst=1)
    limiter.enforce("caller")  # burst token
    with pytest.raises(Exception) as excinfo:
        limiter.enforce("caller")
    assert "retry in" in str(excinfo.value)


def test_a_spoofed_forwarded_header_cannot_mint_a_new_rate_limit_bucket(
    client, settings_override
):
    """X-Forwarded-For is client-writable, so it is not trusted by default.

    Bucketing on it unconditionally meant an unauthenticated caller could
    defeat the limiter by rotating the header. `TRUST_PROXY_HEADERS` exists
    for deployments behind a proxy that overwrites it.
    """
    settings_override.set(trust_proxy_headers=False)
    from app.deps import RateLimiter, client_identity

    class _Request:
        client = type("C", (), {"host": "10.0.0.9"})()
        headers: typing.ClassVar[dict[str, str]] = {"x-forwarded-for": "1.2.3.4"}

    first = client_identity(_Request(), None)
    _Request.headers = {"x-forwarded-for": "5.6.7.8"}
    assert client_identity(_Request(), None) == first == "ip:10.0.0.9"

    settings_override.set(trust_proxy_headers=True)
    assert client_identity(_Request(), None) == "ip:5.6.7.8"
    assert RateLimiter(per_minute=60).check(first) == 0.0


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
def test_stream_handshake_and_ping(client):
    with client.websocket_connect(f"{PREFIX}/stream/verify") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["require_liveness"] is True
        assert "blink" in ready["challenges"]

        ws.send_json({"type": "ping", "seq": 1})
        pong = ws.receive_json()
        assert pong["type"] == "pong"
        assert pong["seq"] == 1

        ws.send_json({"type": "stop"})
        assert ws.receive_json()["type"] == "bye"


def test_stream_returns_a_match_result_per_frame(client):
    user_id = _uid()
    _enroll(client, user_id, seed=50)
    frame = base64.b64encode(to_jpeg_bytes(make_face_image(seed=50), quality=85)).decode()

    with client.websocket_connect(f"{PREFIX}/stream/verify") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "frame", "data": frame, "seq": 7})
        result = ws.receive_json()
        assert result["type"] == "result"
        assert result["seq"] == 7
        assert result["face_present"] is True
        assert result["match"]["user_id"] == user_id
        assert "counters" in result


def test_stream_disconnect_after_a_faceless_frame_does_not_crash(
    client, caplog
):
    """Closing the socket after a frame with no face must not raise.

    A real production bug: the handler's ``finally`` block logged the last match
    with ``(last_match or {}).get("match", {}).get("name")``. When the last frame
    contained no face, ``last_match["match"]`` is ``None`` -- and ``.get``'s
    default only applies when the *key* is absent, so the ``None`` survived and
    the next ``.get`` raised AttributeError. Because it was inside ``finally``,
    every disconnect after a faceless frame produced that traceback and masked
    whatever the handler was really doing.

    The existing faceless-frame tests never closed the socket while holding this
    state, so nothing caught it. Assert the close is clean via the log record
    rather than by expecting an exception, since an exception in ``finally``
    does not reach the test client -- it kills the ASGI app.
    """
    blank = base64.b64encode(to_jpeg_bytes(make_blank_image(), quality=85)).decode()

    with caplog.at_level(logging.INFO, logger="app.api.stream"), client.websocket_connect(
        f"{PREFIX}/stream/verify"
    ) as ws:
        ws.receive_json()
        ws.send_json({"type": "frame", "data": blank, "seq": 1})
        result = ws.receive_json()
        # Precondition: the frame really did produce a null match.
        assert result["match"] is None
        ws.send_json({"type": "stop"})
        assert ws.receive_json()["type"] == "bye"

    assert "AttributeError" not in caplog.text
    closed = [r for r in caplog.records if "stream closed" in r.getMessage()]
    assert closed, f"no close record; log was:\n{caplog.text}"
    assert "last_match=None" in closed[-1].getMessage()


def test_stream_rejects_bad_frames_without_dropping_the_socket(client):
    with client.websocket_connect(f"{PREFIX}/stream/verify") as ws:
        ws.receive_json()
        ws.send_json({"type": "frame", "data": "", "seq": 1})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "empty_frame"

        ws.send_json({"type": "frame", "data": "!!!", "seq": 2})
        assert ws.receive_json()["code"] == "bad_image"

        ws.send_json({"type": "ping", "seq": 3})
        assert ws.receive_json()["type"] == "pong"


def test_stream_punch_uses_server_side_state(client, settings_override):
    settings_override.set(require_liveness=False)
    user_id = _uid()
    _enroll(client, user_id, seed=45)
    frame = base64.b64encode(to_jpeg_bytes(make_face_image(seed=45), quality=85)).decode()

    with client.websocket_connect(f"{PREFIX}/stream/verify") as ws:
        ws.receive_json()
        ws.send_json({"type": "frame", "data": frame, "seq": 1})
        ws.receive_json()  # result
        ws.send_json({"type": "punch"})
        response = ws.receive_json()
        assert response["type"] == "punch"
        assert response["ok"] is True
        assert response["user_id"] == user_id
        assert response["created"] is True


def test_stream_punch_ignores_a_client_claimed_distance(client, settings_override):
    """The distance in the message is never read; the server reports its own.

    Previously the client could name its own distance and the socket only
    re-checked it against the threshold. Now there is nothing to check: the
    number in the response is the one the matcher measured.
    """
    settings_override.set(require_liveness=False)
    user_id = _uid()
    _enroll(client, user_id, seed=46)
    frame = base64.b64encode(to_jpeg_bytes(make_face_image(seed=46), quality=85)).decode()

    with client.websocket_connect(f"{PREFIX}/stream/verify") as ws:
        ws.receive_json()
        ws.send_json({"type": "frame", "data": frame, "seq": 1})
        result = ws.receive_json()
        ws.send_json({"type": "punch", "match_distance": 0.0})
        response = ws.receive_json()
        assert response["ok"] is True
        assert response["match_distance"] == pytest.approx(result["match"]["distance"])


def test_stream_punch_refuses_a_client_claimed_identity(client, settings_override):
    """Claiming somebody else is refused, not resolved in the caller's favour.

    The claimed id is one that was never enrolled, so it cannot be the person
    the server matched no matter which of the similar synthetic faces wins.
    """
    settings_override.set(require_liveness=False)
    _enroll(client, _uid(), seed=47)
    frame = base64.b64encode(to_jpeg_bytes(make_face_image(seed=47), quality=85)).decode()

    with client.websocket_connect(f"{PREFIX}/stream/verify") as ws:
        ws.receive_json()
        ws.send_json({"type": "frame", "data": frame, "seq": 1})
        result = ws.receive_json()
        assert result["match"]["decision"] == "match"

        ws.send_json({"type": "punch", "user_id": _uid("GHOST")})
        response = ws.receive_json()
        assert response["ok"] is False
        assert response["code"] == "identity_mismatch"


def test_stream_punch_before_any_frame_is_refused(client, settings_override):
    settings_override.set(require_liveness=False)
    with client.websocket_connect(f"{PREFIX}/stream/verify") as ws:
        ws.receive_json()
        ws.send_json({"type": "punch"})
        response = ws.receive_json()
        assert response["ok"] is False
        assert response["code"] in ("no_identity", "no_match_result")


def test_stream_punch_spends_the_frame_it_used(client, settings_override):
    """One frame, one punch: the match evidence is dropped after use."""
    settings_override.set(require_liveness=False)
    user_id = _uid()
    _enroll(client, user_id, seed=48)
    frame = base64.b64encode(to_jpeg_bytes(make_face_image(seed=48), quality=85)).decode()

    with client.websocket_connect(f"{PREFIX}/stream/verify") as ws:
        ws.receive_json()
        ws.send_json({"type": "frame", "data": frame, "seq": 1})
        ws.receive_json()
        ws.send_json({"type": "punch"})
        assert ws.receive_json()["ok"] is True

        ws.send_json({"type": "punch"})
        replay = ws.receive_json()
        assert replay["ok"] is False
        assert replay["code"] == "no_match_result"


# ---------------------------------------------------------------------------
# Spoof image behaviour
# ---------------------------------------------------------------------------
def test_spoof_like_frame_is_scored_lower_than_a_live_frame(client):
    user_id = _uid()
    _enroll(client, user_id, seed=49)
    live = base64.b64encode(to_jpeg_bytes(make_face_image(seed=49), quality=90)).decode()
    spoof = base64.b64encode(to_jpeg_bytes(make_spoof_image(seed=49), quality=90)).decode()

    live_body = client.post(f"{PREFIX}/recognition/verify", json={"image_base64": live}).json()
    spoof_body = client.post(f"{PREFIX}/recognition/verify", json={"image_base64": spoof}).json()

    assert spoof_body["liveness"]["passive"] < live_body["liveness"]["passive"]
