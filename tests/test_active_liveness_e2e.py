"""End-to-end active liveness, through the real HTTP and WebSocket surfaces.

What is being tested
--------------------
The integrity chain, not the computer vision:

``POST /liveness/sessions`` -> N frames -> ``status == "passed"`` -> a database
row that ``consume_token`` accepts -> exactly one attendance punch.

Every step in that chain is real. The only thing stubbed is signal extraction:
:class:`FrameSignals` are injected at the ``extract_signals`` seam and the
engine's clock is driven by a fake, so the test is deterministic and finishes in
seconds instead of waiting out randomised reveal delays.

That is a deliberate trade. Driving the *real* Haar cascade needs a real face,
which a synthetic ellipse is not -- asserting on that would test OpenCV, not
this application. The signal-to-challenge logic is covered separately in
``test_active_liveness.py`` against the engine directly.
"""

from __future__ import annotations

import base64
import datetime as dt
import functools
import re
import uuid

import pytest

from app.config import settings
from app.models import LivenessSession as LivenessSessionRow
from app.models import utcnow
from app.services.liveness.geometry import EyeSignal, FrameSignals, eye_detector_status
from tests.conftest import make_face_image, to_jpeg_bytes

PREFIX = "/api/v1"

pytestmark = pytest.mark.skipif(
    not eye_detector_status()["available"],
    reason="active liveness needs the OpenCV Haar eye detector",
)

# Two challenges instead of the default pool: the engine always needs
# min_challenges, so a two-item pool means two items and no wasted frames.
TWO_CHALLENGES = "blink,head_turn_left"


# ---------------------------------------------------------------------------
# fake clock
# ---------------------------------------------------------------------------
class FakeClock:
    """Monotonic clock the test advances by hand.

    Reveal times are randomised over a 0.6-2.2s window plus up to 1.4s between
    challenges. Waiting that out in real time would make this file slow and
    flaky; advancing a fake clock keeps it exact.
    """

    #: Comfortably above the engine's 1/20s "frame_rate_implausible" floor, so a
    #: legitimate frame sequence is never mistaken for a video replay.
    FRAME_DT = 0.08

    def __init__(self, start: float = 10_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def tick(self) -> float:
        self.t += self.FRAME_DT
        return self.t


# ---------------------------------------------------------------------------
# virtual user: emits the signals that satisfy whichever challenge is active
# ---------------------------------------------------------------------------
_FACE = (192, 80, 256, 240)
_FACE_NEAR = (96, 40, 448, 360)  # 3x the area, clearing MoveCloserDetector.growth


def _signals(
    clock: FakeClock,
    *,
    box: tuple[int, int, int, int] = _FACE,
    eyes: int = 2,
    openness: float = 0.9,
    motion: float = 0.0,
    span: float = 0.45,
    asym: float = 0.0,
) -> FrameSignals:
    out = FrameSignals(timestamp=clock.tick(), face_present=True, face_box=box)
    out.face_area = box[2] * box[3]
    out.eyes_found = eyes
    out.openness = openness
    out.eye_motion = motion
    out.eye_span_ratio = span
    out.yaw_asym = asym
    out.eyes = [
        EyeSignal(box=(0, 0, 10, 6), openness=openness, motion=motion, usable=True)
        for _ in range(eyes)
    ]
    return out


class VirtualUser:
    """Replays the frame sequence that completes each challenge type.

    Values clear the detector thresholds with margin rather than sitting on
    them: a test that only passes at the exact boundary is a test that fails on
    the next legitimate retune.
    """

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def next(self, clock: FakeClock, challenge: str | None) -> FrameSignals:
        if challenge is None:
            # Not revealed yet. Hold neutral; the engine ignores these.
            return _signals(clock)

        n = self._seen.get(challenge, 0)
        self._seen[challenge] = n + 1

        if challenge == "blink":
            return self._blink(clock, n)
        if challenge in ("head_turn_left", "head_turn_right"):
            return self._head_turn(clock, n)
        if challenge == "move_closer":
            return self._move_closer(clock, n)
        raise AssertionError(f"no virtual-user script for challenge {challenge!r}")

    @staticmethod
    def _blink(clock: FakeClock, n: int) -> FrameSignals:
        # 8 calibration frames open, then close/open/open, twice.
        if n < 8:
            return _signals(clock)
        return _signals(clock, openness=0.18, motion=0.45) if (n - 8) % 3 == 0 else _signals(clock)

    @staticmethod
    def _head_turn(clock: FakeClock, n: int) -> FrameSignals:
        # 10 calibration frames neutral, 2 turned away, then hold neutral.
        # Turned: span 62% of baseline (needs <0.86) and asym 0.16 (needs >0.10).
        if n < 10:
            return _signals(clock)
        return _signals(clock, span=0.28, asym=0.16) if n < 12 else _signals(clock)

    @staticmethod
    def _move_closer(clock: FakeClock, n: int) -> FrameSignals:
        return _signals(clock) if n < 8 else _signals(clock, box=_FACE_NEAR)


@pytest.fixture
def clock(monkeypatch):
    """Give every liveness engine created in this test a controllable clock."""
    from app.services.liveness.active import LivenessEngine

    fake = FakeClock()
    original_init = LivenessEngine.__init__

    def patched_init(self, challenge_names, **kwargs):
        kwargs.setdefault("now_fn", fake)
        original_init(self, challenge_names, **kwargs)

    monkeypatch.setattr(LivenessEngine, "__init__", patched_init)
    return fake


@pytest.fixture
def driver(monkeypatch, clock, settings_override):
    """Install the scripted signal extractor, scoped to one test.

    ``driver["challenge"]`` is written by the driving loop before each frame and
    ``driver["user"]`` is reset per session -- sharing one script across two
    sessions would feed the second one mid-routine.
    """
    from app.services.liveness import service as service_module

    settings_override.set(liveness_challenges=TWO_CHALLENGES)
    state = {"challenge": None, "user": VirtualUser()}

    def patched(gray, face_box, timestamp, prev_eye_crops=None):
        return state["user"].next(clock, state["challenge"]), {}

    monkeypatch.setattr(service_module, "extract_signals", patched)
    return state


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _uid(prefix: str = "LIVE") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8].upper()}"


@functools.lru_cache(maxsize=8)
def _jpeg(seed: int) -> bytes:
    """Encode once per seed.

    PIL re-encoding a JPEG for every one of ~40 frames was the slowest part of
    this file by a wide margin.
    """
    return to_jpeg_bytes(make_face_image(seed=seed), quality=80)


@functools.lru_cache(maxsize=8)
def _b64(seed: int) -> str:
    return base64.b64encode(_jpeg(seed)).decode("ascii")


@functools.lru_cache(maxsize=8)
def _jpeg_hq(seed: int) -> bytes:
    """The enrolment-grade encode, used for identity frames.

    A 1:1 verification only reports ``match`` when the claimed person is also
    the *globally* nearest in the roster -- if some other enrolled employee is
    marginally closer, it is a ``review`` and earns no grant. Querying with a
    different JPEG quality than the one the template was built from adds
    encoding noise to the distance, which is enough to lose that race against a
    neighbour in a session-wide roster. Matching the enrolment's own settings
    removes the artefact without weakening the assertion.
    """
    return to_jpeg_bytes(make_face_image(seed=seed), quality=92)


def _enroll(client, user_id: str, seed: int = 1):
    return client.post(
        f"{PREFIX}/users",
        data={"id": user_id, "name": f"Test {user_id}"},
        files={"image": ("face.jpg", to_jpeg_bytes(make_face_image(seed=seed), quality=92), "image/jpeg")},
    )


def _verification_id(client, user_id: str, seed: int) -> str:
    """The server-issued identity proof a recognised frame earns.

    The punch body has no ``user_id`` and no ``match_distance`` field, so this
    is the only way to say *who* is being marked and *how well* they matched.
    Every punch in this file goes through one.

    The 1:1 form (``user_id`` on the verify request) is used deliberately: the
    shared session-wide roster accumulates hundreds of similar synthetic faces,
    and a 1:N search would hand back whichever neighbour happened to be
    nearest. Naming the identity keeps the grant -- and the test -- exact.
    """
    raw = _jpeg_hq(seed)
    response = client.post(
        f"{PREFIX}/recognition/verify",
        json={
            "image_base64": base64.b64encode(raw).decode("ascii"),
            "user_id": user_id,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    grant = body["verification_id"]
    assert grant, (
        "an eligible frame must earn a grant: "
        f"decision={body['match'] and body['match']['decision']} "
        f"blocks={body['block_reasons']}"
    )
    return grant


def _punch(client, verification_id: str, session_id: str | None = None):
    body: dict[str, object] = {"verification_id": verification_id}
    if session_id is not None:
        body["liveness_session_id"] = session_id
    return client.post(f"{PREFIX}/attendance/punch", json=body)


def _run_http_session(client, driver, *, seed: int = 41, max_frames: int = 200,
                      user_id: str | None = None) -> dict:
    """Submit frames over HTTP until the session reaches a terminal state.

    ``user_id`` binds the session to one person. Sessions started that way can
    only authorise a punch for that person, which is the binding the
    ``identity_mismatch`` tests below rely on.
    """
    driver["user"] = VirtualUser()
    started = client.post(f"{PREFIX}/liveness/sessions", json={"user_id": user_id} if user_id else {})
    assert started.status_code == 201, started.text
    body = started.json()

    for _ in range(max_frames):
        active = body.get("active_challenge")
        driver["challenge"] = active["name"] if active else None
        response = client.post(
            f"{PREFIX}/liveness/sessions/{body['session_id']}/frames",
            files={"image": ("frame.jpg", _jpeg(seed), "image/jpeg")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] in ("passed", "failed", "expired"):
            break
    return body


def _run_ws_session(ws, driver, *, seed: int = 42, max_sends: int = 200) -> dict:
    """Drive the socket's own liveness session to a terminal state.

    The server drops frames that arrive faster than ``MIN_FRAME_GAP`` and sends
    no reply for them, so the reply to send N may be the reply to send N-1.
    Sending and receiving alternately handles that without tracking which.
    """
    driver["user"] = VirtualUser()
    ws.send_json({"type": "start_liveness", "seq": 0})
    started = ws.receive_json()
    assert started["type"] == "liveness", started
    status = started["status"]

    for seq in range(1, max_sends + 1):
        if status in ("passed", "failed", "expired"):
            break
        ws.send_json({"type": "frame", "data": _b64(seed), "seq": seq})
        message = ws.receive_json()
        assert message["type"] == "result", message
        live = message.get("liveness")
        if not live:
            continue
        active = live.get("active_challenge")
        driver["challenge"] = active["name"] if active else None
        status = live["status"]

    return {"status": status, "session_id": started.get("session_id")}


# ---------------------------------------------------------------------------
# the challenge plan
# ---------------------------------------------------------------------------
def test_session_reveals_a_randomised_challenge_plan(client, driver, settings_override):
    settings_override.set(liveness_challenges="blink,head_turn_left,head_turn_right,move_closer")

    plans = []
    for _ in range(6):
        started = client.post(f"{PREFIX}/liveness/sessions", json={})
        assert started.status_code == 201
        body = started.json()
        assert body["status"] == "pending"
        assert body["required"] == settings.active_liveness_min_challenges
        assert len(body["challenges"]) >= settings.active_liveness_min_challenges
        assert set(body["challenges"][0]) >= {"name", "index", "instruction", "revealed"}
        # Nothing is revealed at t=0: the delay is what a pre-cut tape cannot hit.
        assert body["active_challenge"] is None

        stored = client.get(f"{PREFIX}/liveness/sessions/{body['session_id']}").json()
        plans.append([c["reveal_offset_s"] for c in stored["challenges"]])

    # Randomised per session: two sessions must not share a reveal schedule.
    assert len({tuple(p) for p in plans}) > 1
    for offsets in plans:
        assert offsets[0] > 0.0, "a challenge must not be revealed instantly"


def test_challenge_names_come_from_the_configured_pool(client, driver, settings_override):
    settings_override.set(liveness_challenges="blink,move_closer")
    for _ in range(8):
        body = client.post(f"{PREFIX}/liveness/sessions", json={}).json()
        names = [c["name"] for c in body["challenges"]]
        assert set(names) <= {"blink", "move_closer"}
        assert len(names) == 2


# ---------------------------------------------------------------------------
# the integrity chain
# ---------------------------------------------------------------------------
def test_a_completed_session_reaches_passed_and_yields_a_usable_token(client, driver):
    user_id = _uid()
    assert _enroll(client, user_id, seed=51).status_code < 400

    final = _run_http_session(client, driver, user_id=user_id)
    assert final["status"] == "passed", final
    assert len(final["passed_challenges"]) >= settings.active_liveness_min_challenges
    assert 0.0 < final["score"] <= 1.0

    session_id = final["session_id"]
    read = client.get(f"{PREFIX}/liveness/sessions/{session_id}").json()
    assert read["status"] == "passed"
    assert read["consumed"] is False

    punch = _punch(client, _verification_id(client, user_id, 51), session_id)
    assert punch.status_code == 200, punch.text
    assert punch.json()["created"] is True
    assert punch.json()["attendance"]["liveness_session_id"] == session_id


def test_a_passed_token_cannot_be_replayed_for_a_second_punch(client, driver):
    """The headline guarantee: one liveness session authorises one punch.

    The second attempt also has to present a *fresh* grant, because the first
    one was spent on the first punch. Two independent single-use links, so
    neither the challenge nor the face frame can be doubled.
    """
    user_id = _uid()
    _enroll(client, user_id, seed=52)
    session_id = _run_http_session(client, driver, user_id=user_id)["session_id"]

    first = _punch(client, _verification_id(client, user_id, 52), session_id)
    assert first.status_code == 200

    replay = _punch(client, _verification_id(client, user_id, 52), session_id)
    assert replay.status_code == 403
    assert replay.json()["error"]["code"] == "liveness_required"
    assert replay.json()["error"]["context"]["reason"] == "liveness_token_already_used"


def test_a_grant_cannot_be_replayed_for_a_second_punch(client, driver):
    """The other half of the pair: one recognised frame authorises one punch."""
    user_id = _uid()
    _enroll(client, user_id, seed=67)
    session_id = _run_http_session(client, driver, user_id=user_id)["session_id"]
    grant = _verification_id(client, user_id, 67)

    assert _punch(client, grant, session_id).status_code == 200
    # A second, *valid* liveness session: the only stale thing is the grant.
    other = _run_http_session(client, driver, user_id=user_id)["session_id"]
    replay = _punch(client, grant, other)
    assert replay.status_code == 422
    assert replay.json()["error"]["code"] == "unverified_identity"
    assert replay.json()["error"]["context"]["reason"] == "verification_already_used"


def test_the_session_row_shows_consumed_after_a_punch(client, driver):
    user_id = _uid()
    _enroll(client, user_id, seed=53)
    session_id = _run_http_session(client, driver, user_id=user_id)["session_id"]

    _punch(client, _verification_id(client, user_id, 53), session_id)
    assert client.get(f"{PREFIX}/liveness/sessions/{session_id}").json()["consumed"] is True


def test_a_second_punch_of_the_day_needs_a_second_session(client, driver):
    """Re-punching is legitimate (re-entry, late correction) but not free."""
    user_id = _uid()
    _enroll(client, user_id, seed=54)

    first = _punch(
        client,
        _verification_id(client, user_id, 54),
        _run_http_session(client, driver, user_id=user_id)["session_id"],
    )
    assert first.status_code == 200, first.text

    no_token = _punch(client, _verification_id(client, user_id, 54))
    assert no_token.status_code == 403
    assert no_token.json()["error"]["code"] == "liveness_required"

    with_token = _punch(
        client,
        _verification_id(client, user_id, 54),
        _run_http_session(client, driver, user_id=user_id)["session_id"],
    )
    assert with_token.status_code == 200, with_token.text
    assert with_token.json()["created"] is False
    assert with_token.json()["attendance"]["punch_count"] == 2


def test_a_liveness_session_cannot_punch_for_somebody_else(client, driver):
    """The hole this whole change exists to close.

    Before: pass the challenge yourself, then POST a punch naming a colleague
    and inventing the distance. Nothing in the request ever had to agree with
    the face. Now the session is bound to whoever it was started for, and the
    person to mark comes from the server's own verification -- so a colleague's
    grant plus your honest liveness gets you nothing.
    """
    honest = _uid()
    victim = _uid()
    _enroll(client, honest, seed=55)
    _enroll(client, victim, seed=56)

    # An attacker passes the challenge in their own right...
    session_id = _run_http_session(client, driver, user_id=honest)["session_id"]
    # ...then tries to spend it on a different, enrolled person.
    stolen = _punch(client, _verification_id(client, victim, 56), session_id)
    assert stolen.status_code == 403
    assert stolen.json()["error"]["code"] == "identity_mismatch"

    # Crucially, the refused punch did not spend the attacker's own session.
    assert client.get(f"{PREFIX}/liveness/sessions/{session_id}").json()["consumed"] is False
    assert _punch(
        client, _verification_id(client, honest, 55), session_id
    ).status_code == 200


def test_an_anonymous_session_is_claimed_by_its_first_punch(client, driver):
    """A kiosk that does not know who is at the camera still gets one punch.

    Starting a session without a ``user_id`` leaves it unclaimed, and the first
    person to complete the challenge adopts it -- so the person named by the
    verification grant gets marked, and the token is spent. The convenience
    costs exactly what it should: the session cannot then be reused, by them or
    anyone else.
    """
    user_id = _uid()
    _enroll(client, user_id, seed=57)
    session_id = _run_http_session(client, driver)["session_id"]

    first = _punch(client, _verification_id(client, user_id, 57), session_id)
    assert first.status_code == 200, first.text
    assert first.json()["attendance"]["user_id"] == user_id
    assert client.get(f"{PREFIX}/liveness/sessions/{session_id}").json()["consumed"] is True

    other = _uid()
    _enroll(client, other, seed=58)
    stolen = _punch(client, _verification_id(client, other, 58), session_id)
    assert stolen.status_code == 403
    assert stolen.json()["error"]["context"]["reason"] == "liveness_token_already_used"


def test_a_refused_punch_leaves_the_token_usable(client, driver):
    """A request that fails an early gate must not cost the user the challenge.

    The gate order is what buys this: the grant is looked up and the distance
    checked *before* anything is consumed, so a stale id or a stopped roster
    entry cannot burn a session the user just spent a minute earning.
    """
    user_id = _uid()
    _enroll(client, user_id, seed=59)
    session_id = _run_http_session(client, driver, user_id=user_id)["session_id"]

    bad = _punch(client, "f" * 32, session_id)
    assert bad.status_code == 422
    assert bad.json()["error"]["code"] == "unverified_identity"
    assert client.get(f"{PREFIX}/liveness/sessions/{session_id}").json()["consumed"] is False

    good = _punch(client, _verification_id(client, user_id, 59), session_id)
    assert good.status_code == 200, good.text


def test_a_passed_session_alone_still_cannot_punch(client, driver):
    """Liveness is not a substitute for identity verification.

    Without a verification grant there is nothing to punch with, and without a
    user id in the body there is no way to supply one on the fly either.
    """
    user_id = _uid()
    _enroll(client, user_id, seed=60)
    session_id = _run_http_session(client, driver, user_id=user_id)["session_id"]

    response = client.post(
        f"{PREFIX}/attendance/punch", json={"liveness_session_id": session_id}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"

    # Even sending the forbidden fields loudly does not work.
    legacy = client.post(
        f"{PREFIX}/attendance/punch",
        json={"user_id": user_id, "match_distance": 0.01, "liveness_session_id": session_id},
    )
    assert legacy.status_code == 422
    assert client.get(f"{PREFIX}/liveness/sessions/{session_id}").json()["consumed"] is False


def test_frames_for_an_unknown_session_are_refused(client, driver):
    response = client.post(
        f"{PREFIX}/liveness/sessions/not-a-real-session/frames",
        files={"image": ("frame.jpg", _jpeg(1), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "unknown_session"


def test_a_passed_session_persists_its_active_score(client, driver):
    """The engine's own score is stored, not left NULL.

    Regression: ``active_score`` was never written, so ``GET`` on a session that
    had just passed reported ``score: 0.0`` and ``consume_token`` fell back to
    half the passive score. The number on the attendance row was then a score
    this service never measured.
    """
    user_id = _uid()
    _enroll(client, user_id, seed=68)
    final = _run_http_session(client, driver, user_id=user_id)
    assert final["status"] == "passed", final
    reported = float(final["score"])
    assert 0.0 < reported <= 1.0

    stored = client.get(f"{PREFIX}/liveness/sessions/{final['session_id']}").json()
    assert stored["score"] == pytest.approx(reported, abs=1e-3)
    assert stored["score"] > 0.0

    # And it is that number, not a re-derivation, that lands on the punch.
    punch = _punch(client, _verification_id(client, user_id, 68), final["session_id"])
    assert punch.status_code == 200, punch.text
    assert punch.json()["attendance"]["liveness_score"] == pytest.approx(reported, abs=1e-3)


def _fail_sessions(db, user_id: str, count: int, *, age_seconds: float = 0.0) -> None:
    """Insert terminal ``failed`` rows, the way the FSM would have left them."""
    from app.models import LivenessSession, utcnow

    now = utcnow()
    for _ in range(count):
        db.add(
            LivenessSession(
                id=uuid.uuid4().hex,
                user_id=user_id,
                status="failed",
                challenges=[],
                attempts=3,
                max_attempts=3,
                failure_reason="insufficient_eye_evidence",
                created_at=now,
                updated_at=now,
                completed_at=now - dt.timedelta(seconds=age_seconds),
                expires_at=now,
            )
        )
    db.commit()


def test_repeated_failures_lock_the_identity_out(client, db, settings_override):
    """A per-session attempt cap means nothing across sessions.

    ``LIVENESS_MAX_ATTEMPTS`` is enforced inside one engine, so a failed session
    simply ends and the next one starts free -- a target could be worked on
    indefinitely. ``LIVENESS_LOCKOUT_SECONDS`` is the cross-session half, and
    until it was wired up the setting existed and did nothing.
    """
    settings_override.set(liveness_max_attempts=3, liveness_lockout_seconds=300)
    victim = _uid()
    _fail_sessions(db, victim, 3)

    refused = client.post(f"{PREFIX}/liveness/sessions", json={"user_id": victim})
    assert refused.status_code == 429
    assert refused.json()["error"]["code"] == "liveness_locked_out"
    assert "s" in refused.json()["error"]["message"]  # carries a countdown

    # A 201 with a null session_id would be worse than useless: the kiosk would
    # prompt for a challenge that can never start.
    assert refused.json()["error"]["code"] != "liveness_unavailable"


def test_the_lockout_countdown_is_truthful(client, db, settings_override):
    """A refusal that says "try again in 1s" and then keeps refusing is a bug.

    The failure count is taken over a trailing window, so a lockout that expired
    *with the window* would hand back an unlock time that had already passed: the
    kiosk would show a one-second countdown, retry, and walk into the same 429.
    The lock has to end ``LIVENESS_LOCKOUT_SECONDS`` after the last failure, and
    the number in the message has to be the real one.
    """
    settings_override.set(liveness_max_attempts=3, liveness_lockout_seconds=300)
    victim = _uid()
    _fail_sessions(db, victim, 3, age_seconds=60)

    first = client.post(f"{PREFIX}/liveness/sessions", json={"user_id": victim})
    assert first.status_code == 429
    # 300s of lockout, started 60s ago, so ~240s left (truncated, and the test
    # has by then spent a few ms).
    seconds = int(re.search(r"in (\d+)s", first.json()["error"]["message"]).group(1))
    assert 230 <= seconds <= 240, seconds

    # And retrying immediately must still be refused: the countdown is the
    # earliest a retry can work, not a promise that one will.
    assert client.post(f"{PREFIX}/liveness/sessions", json={"user_id": victim}).status_code == 429

    # The audited unlock time agrees with the message.
    from app.models import Event

    unlock_at = [
        row.context["unlock_at"]
        for row in db.query(Event).filter(Event.kind == "liveness_locked_out").all()
        if row.user_id == victim
    ][-1]
    assert dt.datetime.fromisoformat(unlock_at) > utcnow()


def test_a_lockout_is_scoped_to_one_identity(client, db, settings_override):
    """Locking one employee out must not lock out the whole gate."""
    settings_override.set(liveness_max_attempts=3, liveness_lockout_seconds=300)
    blocked, other = _uid(), _uid()
    _fail_sessions(db, blocked, 3)

    assert client.post(f"{PREFIX}/liveness/sessions", json={"user_id": blocked}).status_code == 429
    assert client.post(f"{PREFIX}/liveness/sessions", json={"user_id": other}).status_code == 201
    # Anonymous is not locked either: there is no identity to count failures
    # against, and inventing one from the socket address would be a guess.
    assert client.post(f"{PREFIX}/liveness/sessions", json={}).status_code == 201


def test_a_lockout_expires(client, db, settings_override):
    """Otherwise a bad camera or a nervous first-timer is locked out forever."""
    settings_override.set(liveness_max_attempts=3, liveness_lockout_seconds=300)
    victim = _uid()
    _fail_sessions(db, victim, 3, age_seconds=299)
    assert client.post(f"{PREFIX}/liveness/sessions", json={"user_id": victim}).status_code == 429

    # A failure from before the window does not count towards it.
    _fail_sessions(db, victim, 1, age_seconds=1000)
    db.query(LivenessSessionRow).filter(
        LivenessSessionRow.user_id == victim,
        LivenessSessionRow.completed_at >= utcnow() - dt.timedelta(seconds=400),
    ).update({"completed_at": utcnow() - dt.timedelta(seconds=1000)})
    db.commit()
    assert client.post(f"{PREFIX}/liveness/sessions", json={"user_id": victim}).status_code == 201


def test_a_zero_lockout_disables_the_control(client, db, settings_override):
    """An operator can turn it off, and 0 must mean "off", not "always locked"."""
    settings_override.set(liveness_max_attempts=3, liveness_lockout_seconds=0)
    victim = _uid()
    _fail_sessions(db, victim, 99)
    assert client.post(f"{PREFIX}/liveness/sessions", json={"user_id": victim}).status_code == 201


def test_a_spoofed_frame_cannot_reach_a_passed_session(client, driver):
    """Passive evidence is recorded on the session even while it is pending."""
    user_id = _uid()
    _enroll(client, user_id, seed=61)
    driver["user"] = VirtualUser()
    started = client.post(f"{PREFIX}/liveness/sessions", json={"user_id": user_id}).json()
    session_id = started["session_id"]

    from tests.conftest import make_spoof_image

    spoof = to_jpeg_bytes(make_spoof_image(seed=61), quality=80)
    for _ in range(3):
        body = client.post(
            f"{PREFIX}/liveness/sessions/{session_id}/frames",
            files={"image": ("frame.jpg", spoof, "image/jpeg")},
        ).json()
    assert body["status"] == "pending", "a spoof cannot pass challenges it never performs"

    stored = client.get(f"{PREFIX}/liveness/sessions/{session_id}").json()
    assert stored["status"] != "passed"
    assert stored["consumed"] is False

    refused = _punch(client, _verification_id(client, user_id, 61), session_id)
    assert refused.status_code == 403



# ---------------------------------------------------------------------------
# websocket parity
# ---------------------------------------------------------------------------
def _ws_url(seed_user: str | None = None) -> str:
    """Socket URL, pinned to one identity where the test needs it.

    The ``user_id`` query parameter makes the socket run 1:1 verification and
    also binds the liveness session it starts, which is what a kiosk aimed at a
    specific employee would send.
    """
    if seed_user is None:
        return f"{PREFIX}/stream/verify"
    return f"{PREFIX}/stream/verify?user_id={seed_user}"


def test_websocket_punches_once_then_refuses_the_replay(client, driver):
    """The socket path must enforce the same single-use rule as HTTP."""
    user_id = _uid()
    _enroll(client, user_id, seed=62)

    with client.websocket_connect(_ws_url(user_id)) as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["liveness_available"] is True
        assert ready["require_liveness"] is True

        outcome = _run_ws_session(ws, driver, seed=62)
        assert outcome["status"] == "passed", outcome

        ws.send_json({"type": "punch", "seq": 1})
        punched = ws.receive_json()
        assert punched["type"] == "punch"
        assert punched["ok"] is True, punched
        assert punched["user_id"] == user_id
        assert punched["created"] is True

        # The socket cleared its own liveness state *and* its match evidence
        # after the punch, so the replay is refused for the stricter reason:
        # there is nothing left to punch with. Re-sending the session id cannot
        # resurrect the token either.
        ws.send_json(
            {"type": "punch", "seq": 2, "liveness_session_id": outcome["session_id"]}
        )
        replayed = ws.receive_json()
        assert replayed["type"] == "punch"
        assert replayed["ok"] is False, replayed
        assert replayed["code"] == "no_match_result"
        assert client.get(
            f"{PREFIX}/liveness/sessions/{outcome['session_id']}"
        ).json()["consumed"] is True


def test_websocket_punch_is_refused_before_liveness(client, driver):
    user_id = _uid()
    _enroll(client, user_id, seed=63)

    with client.websocket_connect(_ws_url(user_id)) as ws:
        ws.receive_json()
        ws.send_json({"type": "frame", "data": _b64(63), "seq": 1})
        assert ws.receive_json()["match"]["user_id"] == user_id

        ws.send_json({"type": "punch", "seq": 2})
        response = ws.receive_json()
        assert response["type"] == "punch"
        assert response["ok"] is False
        assert response["code"] == "liveness_required"


def test_websocket_punch_ignores_a_client_claimed_distance(client, driver):
    """The distance in the message is never read; the server reports its own.

    Previously the client named its own distance and the socket only re-checked
    it against the threshold -- so "punch" was a way to skip verification. Now
    the number in the response is the one the matcher measured, and the claimed
    one could not change it even if the message schema were loose.
    """
    user_id = _uid()
    _enroll(client, user_id, seed=64)

    with client.websocket_connect(_ws_url(user_id)) as ws:
        ws.receive_json()
        outcome = _run_ws_session(ws, driver, seed=64)
        assert outcome["status"] == "passed", outcome

        # One more frame, so the punch has match evidence the server just took.
        ws.send_json({"type": "frame", "data": _b64(64), "seq": 900})
        seen = ws.receive_json()
        assert seen["type"] == "result"
        assert seen["match"]["decision"] == "match"

        ws.send_json({"type": "punch", "seq": 901, "match_distance": 0.0})
        response = ws.receive_json()
        assert response["ok"] is True, response
        assert response["match_distance"] == pytest.approx(seen["match"]["distance"])
        assert response["match_distance"] > 0.0


def test_websocket_punch_refuses_a_claimed_identity(client, driver):
    """A socket client cannot name somebody other than who was matched.

    This is the socket-side twin of the HTTP identity-mismatch rule, and it
    lands on the same shared ``punch_attendance`` that minted its own grant from
    the server's match, so there is no second code path to get it wrong.
    """
    user_id = _uid()
    other = _uid()
    _enroll(client, user_id, seed=65)
    _enroll(client, other, seed=66)

    with client.websocket_connect(_ws_url(user_id)) as ws:
        ws.receive_json()
        ws.send_json({"type": "frame", "data": _b64(65), "seq": 1})
        result = ws.receive_json()
        assert result["match"]["user_id"] == user_id

        ws.send_json({"type": "punch", "seq": 2, "user_id": other})
        response = ws.receive_json()
        assert response["type"] == "punch"
        assert response["ok"] is False, response
        assert response["code"] == "identity_mismatch"

        # The refused punch kept the server's match evidence, so the person who
        # is actually there can still be punched without re-sending a frame.
        ws.send_json({"type": "punch", "seq": 3})
        retry = ws.receive_json()
        assert retry["ok"] is False
        assert retry["code"] == "liveness_required"

        # ...and once liveness is passed, it succeeds -- for the matched user,
        # whoever the client claimed.
        assert _run_ws_session(ws, driver, seed=65)["status"] == "passed"
        ws.send_json({"type": "punch", "seq": 4})
        final = ws.receive_json()
        assert final["ok"] is True, final
        assert final["user_id"] == user_id
