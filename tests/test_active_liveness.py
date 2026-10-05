"""Active liveness: the randomised challenge-response state machine.

Driven with synthetic :class:`FrameSignals` so the logic is testable with no
OpenCV, no camera and no model. The assertions are about *behaviour under
attack*, not just the happy path.
"""

from __future__ import annotations

import pytest

from app.services.liveness.active import (
    INSTRUCTIONS,
    SUPPORTED_CHALLENGES,
    BlinkDetector,
    HeadTurnDetector,
    LivenessEngine,
    MoveCloserDetector,
)
from app.services.liveness.geometry import EyeSignal, FrameSignals

FRAME_DT = 1.0 / 15.0  # 15 fps: above the 1/20 s plausibility floor


class Clock:
    """Injectable monotonic clock so reveal times are deterministic."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def tick(self, dt: float = FRAME_DT) -> float:
        self.now += dt
        return self.now


def _eyes(n: int = 2, openness: float = 0.55) -> list[EyeSignal]:
    return [EyeSignal(box=(10 + 40 * i, 10, 30, 14), openness=openness, motion=0.0) for i in range(n)]


def signals(
    clock: Clock,
    *,
    dt: float = FRAME_DT,
    face_present: bool = True,
    eyes: int = 2,
    openness: float = 0.55,
    eye_motion: float = 0.0,
    face_area: int = 20_000,
    eye_span_ratio: float = 0.42,
    yaw_asym: float = 0.0,
) -> FrameSignals:
    return FrameSignals(
        timestamp=clock.tick(dt),
        face_present=face_present,
        face_box=(100, 80, 300, 300) if face_present else None,
        face_area=face_area if face_present else 0,
        eyes=_eyes(eyes, openness) if eyes else [],
        openness=openness,
        eye_motion=eye_motion,
        eyes_found=eyes,
        yaw_asym=yaw_asym,
        eye_span_ratio=eye_span_ratio,
    )


def engine(clock: Clock, challenges: list[str], **kwargs) -> LivenessEngine:
    eng = LivenessEngine(challenges, now_fn=clock, **kwargs)
    clock.now = eng.started_at + 30.0  # clear every randomised reveal time
    return eng


def feed(eng: LivenessEngine, clock: Clock, n: int, **kwargs):
    """Feed n frames, stopping early if the session ends."""
    verdict = eng.verdict()
    for _ in range(n):
        if verdict.status != "pending":
            break
        verdict = eng.update(signals(clock, **kwargs))
    return verdict


def perform(clock: Clock, eng: LivenessEngine, name: str) -> None:
    """Drive the currently-active challenge `name` to completion."""
    if name == "blink":
        _blink(clock, eng)
    elif name.startswith("head_turn"):
        feed(eng, clock, 12)  # calibrate
        feed(eng, clock, 5, eye_span_ratio=0.26, yaw_asym=0.22)  # turn away
        feed(eng, clock, 5)  # and return to neutral
    elif name == "move_closer":
        feed(eng, clock, 10)  # calibrate
        feed(eng, clock, 6, face_area=45_000)
    else:  # pragma: no cover
        raise AssertionError(f"no driver for challenge {name!r}")


# ---------------------------------------------------------------------------
# Blink
# ---------------------------------------------------------------------------
def _blink(clock: Clock, eng: LivenessEngine) -> None:
    feed(eng, clock, 10)  # calibrate with eyes open
    for _ in range(2):
        feed(eng, clock, 2, openness=0.12, eye_motion=0.35)  # shut + motion spike
        feed(eng, clock, 4, openness=0.55)  # reopen and hold


def test_blink_completes_after_two_blinks():
    clock = Clock()
    eng = engine(clock, ["blink"], min_challenges=1)
    assert eng.verdict().status == "pending"
    _blink(clock, eng)
    verdict = eng.verdict()
    assert verdict.status == "passed"
    assert verdict.passed_challenges == ["blink"]
    assert verdict.score >= 0.6


def test_single_blink_is_not_enough():
    clock = Clock()
    eng = engine(clock, ["blink"], min_challenges=1)
    feed(eng, clock, 10)
    feed(eng, clock, 2, openness=0.12, eye_motion=0.35)
    feed(eng, clock, 4, openness=0.55)
    assert eng.verdict().status == "pending"


def test_a_held_shut_eye_does_not_count_as_a_blink():
    """Without the recovery requirement, one long blink scores twice."""
    clock = Clock()
    eng = engine(clock, ["blink"], min_challenges=1)
    feed(eng, clock, 10)
    feed(eng, clock, 20, openness=0.10, eye_motion=0.4)  # never reopens
    assert eng.verdict().status == "pending"


def test_openness_drop_without_motion_is_ignored():
    """A static dark band (sunglasses, a shadow) must not register as a blink."""
    clock = Clock()
    eng = engine(clock, ["blink"], min_challenges=1)
    feed(eng, clock, 10)
    feed(eng, clock, 6, openness=0.10, eye_motion=0.0)
    feed(eng, clock, 6, openness=0.55)
    assert eng.verdict().status == "pending"


def test_eyes_closed_during_calibration_is_not_viable():
    """Guards against a session calibrated on a shut-eye baseline."""
    detector = BlinkDetector()
    sigs = [FrameSignals(timestamp=float(i), face_present=True, eyes_found=2, eyes=_eyes(2, 0.05))
            for i in range(detector.calibration_frames + 2)]
    for sig in sigs:
        detector.update(sig)
    assert detector.calibrated
    assert detector.viable is False


# ---------------------------------------------------------------------------
# Head turn
# ---------------------------------------------------------------------------
def test_head_turn_requires_a_return_to_neutral():
    """The anti-splice property: a turn-and-hold clip cannot complete."""
    clock = Clock()
    eng = engine(clock, ["head_turn_left"], min_challenges=1)
    feed(eng, clock, 12)  # calibrate
    feed(eng, clock, 8, eye_span_ratio=0.28, yaw_asym=0.20)  # turned and held
    verdict = eng.verdict()
    assert verdict.status == "pending"
    assert not eng.challenges[0].completed_at


def test_head_turn_completes_on_turn_then_return():
    clock = Clock()
    eng = engine(clock, ["head_turn_left"], min_challenges=1)
    feed(eng, clock, 12)
    feed(eng, clock, 5, eye_span_ratio=0.28, yaw_asym=0.20)  # turn
    feed(eng, clock, 5)  # return to neutral
    verdict = eng.verdict()
    assert verdict.status == "passed"
    assert verdict.passed_challenges == ["head_turn_left"]


def test_head_turn_needs_two_eyes():
    """Inter-eye distance is undefined with one eye; must not pass on guesswork."""
    clock = Clock()
    eng = engine(clock, ["head_turn_right"], min_challenges=1)
    feed(eng, clock, 12, eyes=1)
    feed(eng, clock, 10, eyes=1, eye_span_ratio=0.1, yaw_asym=0.3)
    assert eng.verdict().status == "pending"


def test_head_turn_detector_is_sign_agnostic():
    """Foreshortening must work on a mirrored front camera too."""
    detector = HeadTurnDetector("left")
    for i in range(detector.calibration_frames + 2):
        detector.update(FrameSignals(timestamp=float(i), face_present=True, eyes_found=2,
                                     eyes=_eyes(), eye_span_ratio=0.42, yaw_asym=-0.02))
    # Head goes one way, asymmetry moves the *opposite* way: either sign must
    # still register a turn, because the mapping is not knowable server-side.
    for i in range(3):
        detector.update(FrameSignals(timestamp=100.0 + i, face_present=True, eyes_found=2,
                                     eyes=_eyes(), eye_span_ratio=0.24, yaw_asym=+0.22))
    assert detector.peak > 0.0
    assert detector._turned


# ---------------------------------------------------------------------------
# Move closer
# ---------------------------------------------------------------------------
def test_move_closer_completes_on_sustained_growth():
    clock = Clock()
    eng = engine(clock, ["move_closer"], min_challenges=1)
    feed(eng, clock, 10)
    verdict = feed(eng, clock, 6, face_area=40_000)  # 2x the baseline
    assert verdict.status == "passed"


def test_move_closer_needs_sustained_frames_not_one_spike():
    clock = Clock()
    eng = engine(clock, ["move_closer"], min_challenges=1)
    feed(eng, clock, 10)
    eng.update(signals(clock, face_area=40_000))
    assert eng.verdict().status == "pending"


# ---------------------------------------------------------------------------
# Session-level guarantees
# ---------------------------------------------------------------------------
def test_randomised_order_and_reveal_times_differ_between_sessions():
    pool = list(SUPPORTED_CHALLENGES)
    plans = []
    for seed in range(12):
        eng = LivenessEngine(pool, min_challenges=2, seed=seed)
        plans.append(
            (
                tuple(c.name for c in eng.challenges),
                tuple(round(c.revealed_at - eng.started_at, 4) for c in eng.challenges),
            )
        )
    assert len({p[0] for p in plans}) > 1, "challenge order is not being shuffled"
    assert len({p[1] for p in plans}) > 1, "reveal times are not being randomised"


def test_unsupported_challenges_are_dropped_not_fatal():
    eng = LivenessEngine(["blink", "head_nod", "smile"], min_challenges=1, seed=1)
    assert [c.name for c in eng.challenges] == ["blink"]


def test_empty_pool_raises():
    with pytest.raises(ValueError, match="no supported liveness challenges"):
        LivenessEngine(["teleport"], min_challenges=1)


def test_missing_face_eventually_fails_the_session():
    clock = Clock()
    eng = engine(clock, ["blink"], min_challenges=1)
    for _ in range(45):
        eng.update(signals(clock, face_present=False))
    verdict = eng.verdict()
    assert verdict.status == "failed"
    assert verdict.reason == "no_face_detected"


def test_unreliable_eyes_fail_closed():
    """Sunglasses or a hat brim must not produce a pass on missing evidence."""
    clock = Clock()
    eng = engine(clock, ["blink"], min_challenges=1)
    for _ in range(20):
        eng.update(signals(clock, eyes=0, face_present=True))
    verdict = eng.verdict()
    assert verdict.status == "failed"
    assert verdict.reason == "insufficient_eye_evidence"


def test_implausible_frame_rate_fails_the_session():
    """A 2-4x video replay shows up as sub-50 ms inter-frame gaps."""
    clock = Clock()
    eng = engine(clock, ["blink"], min_challenges=1)
    for _ in range(10):
        eng.update(signals(clock, dt=0.01))  # 100 fps
    verdict = eng.verdict()
    assert verdict.status == "failed"
    assert verdict.reason == "frame_rate_implausible"


def test_session_expires():
    clock = Clock()
    eng = LivenessEngine(["blink"], min_challenges=1, ttl_seconds=5.0, now_fn=clock)
    clock.now = eng.started_at + 10.0
    verdict = eng.update(signals(clock))
    assert verdict.status == "expired"
    assert verdict.reason == "session_expired"


def test_a_completely_static_subject_never_passes():
    """The core guarantee: no movement, no attendance."""
    clock = Clock()
    eng = engine(clock, list(SUPPORTED_CHALLENGES), min_challenges=2)
    verdict = feed(eng, clock, 200)
    assert verdict.status == "pending"
    assert not verdict.passed_challenges


def test_min_challenges_gates_the_pass():
    """One completed challenge out of two required must not authorise a punch."""
    clock = Clock()
    eng = engine(clock, ["blink", "move_closer"], min_challenges=2, seed=17)
    assert len(eng.challenges) == 2

    perform(clock, eng, eng.challenges[0].name)
    verdict = eng.verdict()
    assert len(verdict.passed_challenges) == 1
    assert verdict.status == "pending"

    perform(clock, eng, eng.challenges[1].name)
    verdict = eng.verdict()
    assert verdict.status == "passed"
    assert len(verdict.passed_challenges) == 2


def test_verdict_exposes_prompts_to_the_client():
    clock = Clock()
    eng = engine(clock, ["blink", "move_closer"], min_challenges=2)
    verdict = eng.verdict()
    assert verdict.prompt in INSTRUCTIONS.values()
    assert len(verdict.challenges) >= 2
    for ch in verdict.challenges:
        assert ch["instruction"] in INSTRUCTIONS.values()


def test_terminal_states_do_not_resurrect():
    clock = Clock()
    eng = engine(clock, ["blink"], min_challenges=1)
    eng._fail("insufficient_eye_evidence")
    for _ in range(10):
        verdict = eng.update(signals(clock))
    assert verdict.status == "failed"
    assert verdict.reason == "insufficient_eye_evidence"


def test_move_closer_detector_handles_tiny_baseline():
    detector = MoveCloserDetector()
    for i in range(detector.calibration_frames + 2):
        detector.update(FrameSignals(timestamp=float(i), face_present=True, eyes_found=2,
                                     eyes=_eyes(), face_area=1))
    assert detector._area0 >= 1.0
    for i in range(6):
        detector.update(FrameSignals(timestamp=50.0 + i, face_present=True, eyes_found=2,
                                     eyes=_eyes(), face_area=100))
    assert detector.done
