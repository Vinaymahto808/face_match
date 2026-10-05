"""``POST /kiosk/punch``: the liveness-free in/out path.

What is proved here is the bookkeeping, not recognition: first punch of the
day is the *in*, every later one moves the *out*, a forced mode is honoured,
a dry run writes nothing, and a spoof-looking frame is refused.
"""

from __future__ import annotations

import itertools
import uuid

import numpy as np

from tests.conftest import make_face_image, make_spoof_image, to_jpeg_bytes
from tests.test_api import _enroll

PREFIX = "/api/v1"

# The stub analyzer derives its embedding from an 8x8 thumbnail of the face
# box, so every plain synthetic face lands within ~0.04 of every other and the
# shared test roster is one big tie. Stamping a Hadamard pattern over the box
# gives each test a face that is orthogonal to the others: after a JPEG round
# trip, pairwise distance >= 0.87 and >= 0.62 from a plain face (threshold
# 0.40, gray zone 0.50), while the added noise keeps the quality gate and the
# passive check happy. So a claim resolves to the right person every time and
# a never-enrolled pattern is a genuine stranger.
_H = np.array([[1]])
while _H.shape[0] < 16:
    _H = np.block([[_H, _H], [_H, -_H]])
_pattern_ids = itertools.count(1)


def make_patterned_face(k: int, seed: int = 1) -> np.ndarray:
    img = make_face_image(seed=seed).astype(np.float32)
    h, w = img.shape[:2]
    x0, y0, bw, bh = int(w * 0.3), int(h / 6), int(w * 0.4), int(h * 0.5)  # stub face box
    pattern = _H[k % 16].reshape(4, 4)
    for r in range(4):
        for c in range(4):
            ys = slice(y0 + r * bh // 4, y0 + (r + 1) * bh // 4)
            xs = slice(x0 + c * bw // 4, x0 + (c + 1) * bw // 4)
            img[ys, xs] = img[ys, xs] * 0.15 + (165 if pattern[r, c] > 0 else 75) * 0.85
    box = img[y0 : y0 + bh, x0 : x0 + bw]
    box += np.random.default_rng(seed).normal(0.0, 9.0, size=box.shape)  # sensor noise
    return np.clip(img, 0, 255).astype(np.uint8)


def _person(client) -> tuple[str, np.ndarray]:
    """Enrol a fresh, uniquely patterned person; returns ``(id, face)``."""
    uid = f"KIOSK{uuid.uuid4().hex[:8].upper()}"
    face = make_patterned_face(next(_pattern_ids))
    assert _enroll(client, uid, image=face).status_code == 201
    return uid, face


def _punch(client, user_id: str | None, image: np.ndarray, **form) -> dict:
    data = {k: str(v) for k, v in form.items()}
    if user_id is not None:
        data["user_id"] = user_id
    response = client.post(
        f"{PREFIX}/kiosk/punch",
        data=data,
        files={"image": ("frame.jpg", to_jpeg_bytes(image, quality=92), "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_first_punch_is_in_then_out(client):
    uid, face = _person(client)

    first = _punch(client, uid, face)
    assert first["action"] == "in", first
    assert first["user_id"] == uid
    # Box + frame size let the browser draw the detection over its preview.
    assert first["frame_size"] == [640, 480]
    x, y, w, h = first["face_box"]
    assert 0 <= x < x + w <= 640 and 0 <= y < y + h <= 480
    assert first["attendance"]["check_out_at"] is None
    assert first["attendance"]["source"] == "kiosk"
    assert first["attendance"]["liveness_session_id"] is None

    second = _punch(client, uid, face)
    assert second["action"] == "out", second
    assert second["attendance"]["check_out_at"] is not None
    assert second["attendance"]["id"] == first["attendance"]["id"]

    third = _punch(client, uid, face)
    assert third["action"] == "out"
    assert third["attendance"]["check_out_at"] >= second["attendance"]["check_out_at"]


def test_roster_search_without_claimed_id(client):
    uid, face = _person(client)
    body = _punch(client, None, face, dry_run="true")
    assert body["action"] == "identified", body
    assert body["user_id"] == uid


def test_forced_in_after_in_only_updates_last_seen(client):
    uid, face = _person(client)
    _punch(client, uid, face)
    again = _punch(client, uid, face, mode="in")
    assert again["action"] == "in", again
    assert again["attendance"]["punch_count"] == 2
    assert again["attendance"]["check_out_at"] is None


def test_forced_out_with_no_in_records_both(client):
    uid, face = _person(client)
    out = _punch(client, uid, face, mode="out")
    assert out["action"] == "out", out
    assert out["attendance"]["first_in_at"] is not None
    assert out["attendance"]["check_out_at"] is not None
    assert "recorded together" in out["message"]


def test_dry_run_writes_nothing(client):
    uid, face = _person(client)
    body = _punch(client, uid, face, dry_run="true")
    assert body["action"] == "identified", body
    assert body["attendance"] is None
    rows = client.get(f"{PREFIX}/attendance", params={"user_id": uid}).json()
    assert rows["total"] == 0


def test_spoof_frame_is_refused(client):
    uid, _ = _person(client)
    body = _punch(client, uid, make_spoof_image(seed=1))
    # Either the quality gate stops it (blurry print) or passive liveness does;
    # what matters is that nothing is written.
    assert body["action"] in ("refused", "no_match"), body
    assert body["attendance"] is None
    rows = client.get(f"{PREFIX}/attendance", params={"user_id": uid}).json()
    assert rows["total"] == 0


def test_unknown_face_is_no_match(client):
    _person(client)
    stranger = make_patterned_face(next(_pattern_ids))  # never enrolled
    body = _punch(client, None, stranger)
    assert body["action"] == "no_match", body
    assert body["attendance"] is None
    assert body["face_box"] is not None  # a face was seen, it just isn't anyone
