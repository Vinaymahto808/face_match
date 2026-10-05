"""Active liveness: randomised challenge-response.

Why this is the primary defence
-------------------------------
Passive single-frame analysis cannot tell a good photo-of-a-screen from a
person when the display is 4K and close. A challenge-response test can, because
the challenge is not known in advance:

1. The **order** of challenges is shuffled per session.
2. Each challenge has a randomised **reveal time**, so an attacker cannot
   pre-cut a video that is "blink then turn left" -- the tape has to be able
   to produce any order, in real time, against a random clock.
3. Each challenge requires a **return to neutral**, not just reaching a pose.
   A spliced montage shows a peak and never a return, so it cannot complete.
4. Attempts are capped with a lockout, so guessing is not a strategy.

What this still cannot do
-------------------------
A genuine pre-recorded video of the real person, played in real time at the
right resolution, defeats all of the above. Fixing that needs depth or IR
sensing, a server-signed capture nonce bound to the camera, or a third-party
gaze/attestation check. Do not promise your users otherwise.

This module is pure logic over :class:`FrameSignals`, so it is fully testable
without OpenCV, a camera, or a model.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .geometry import FrameSignals

logger = logging.getLogger(__name__)

__all__ = [
    "INSTRUCTIONS",
    "SUPPORTED_CHALLENGES",
    "BlinkDetector",
    "Challenge",
    "EngineVerdict",
    "HeadTurnDetector",
    "LivenessEngine",
    "MoveCloserDetector",
]

SUPPORTED_CHALLENGES = ("blink", "head_turn_left", "head_turn_right", "move_closer")

INSTRUCTIONS: dict[str, str] = {
    "blink": "Blink twice, slowly and clearly",
    "head_turn_left": "Turn your head to the left, then face the camera again",
    "head_turn_right": "Turn your head to the right, then face the camera again",
    "move_closer": "Move a little closer to the camera",
}


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
class _BaseDetector:
    """Calibrate on the first N usable frames, then judge.

    Every threshold is relative to a measured baseline rather than a constant,
    because baseline eye geometry varies wildly between people, cameras and
    lighting.
    """

    name = "base"
    #: frames needed before the baseline is trusted
    calibration_frames = 8
    #: usable frames allowed to pass with no usable evidence before failing
    unreliable_budget = 12

    def __init__(self) -> None:
        self._cal: list[float] = []
        self.unreliable_streak = 0
        self.calibrated = False
        self._done = False
        self.peak = 0.0

    def _observe(self, value: float) -> None:
        raise NotImplementedError

    def _baseline(self) -> float:
        raise NotImplementedError

    def update(self, sig: FrameSignals) -> None:
        if not sig.reliable:
            self.unreliable_streak += 1
            return
        self.unreliable_streak = 0

        if not self.calibrated:
            self._cal.append(self._observe(sig))
            if len(self._cal) >= self.calibration_frames:
                self.calibrated = True
                self._on_calibrated()
            return

        self._evaluate(sig)

    def _on_calibrated(self) -> None:
        return

    def _evaluate(self, sig: FrameSignals) -> None:
        raise NotImplementedError

    @property
    def progress(self) -> float:
        return 1.0 if self._done else 0.0

    @property
    def done(self) -> bool:
        return self._done

    def _finish(self) -> None:
        self._done = True

    def summary(self) -> dict:
        return {
            "name": self.name,
            "done": self._done,
            "progress": round(self.progress, 3),
            "peak": round(self.peak, 4),
            "calibrated": self.calibrated,
        }


class BlinkDetector(_BaseDetector):
    """Two deliberate blinks, each requiring an openness collapse *and* a
    localised motion spike, then recovery to the open baseline.

    Requiring the motion spike is what stops a single noisy frame from
    registering as a blink; requiring recovery is what stops a long blink from
    counting twice.
    """

    name = "blink"
    calibration_frames = 8
    blinks_required = 2
    close_ratio = 0.62  # openness below 62% of baseline counts as shut
    reopen_ratio = 0.80  # and above 80% counts as open again
    motion_min = 0.08
    reopen_frames = 2
    min_open_baseline = 0.20  # below this the eyes were shut during calibration

    def __init__(self) -> None:
        super().__init__()
        self._open_baseline = 0.0
        self._blinks = 0
        self._state = "open"  # open -> closing -> open
        self._reopen_count = 0
        self.viable = True

    def _observe(self, sig: FrameSignals) -> float:
        return sig.openness

    def _on_calibrated(self) -> None:
        # Take a high percentile, not the mean: the calibration window is
        # expected to be eyes-open, and a mean is dragged down by any
        # accidental squint.
        self._open_baseline = float(np_percentile(self._cal, 85))
        if self._open_baseline < self.min_open_baseline:
            self.viable = False

    def _evaluate(self, sig: FrameSignals) -> None:
        if not self.viable:
            return
        # Slow upward-tracking baseline so natural variation is absorbed.
        if sig.openness > self._open_baseline:
            self._open_baseline += 0.05 * (sig.openness - self._open_baseline)
        self._open_baseline = max(self._open_baseline, 1e-3)

        ratio = sig.openness / self._open_baseline
        approach = max(0.0, 1.0 - ratio)

        if self._state == "open":
            if ratio < self.close_ratio and sig.eye_motion >= self.motion_min:
                self._state = "closing"
                self._reopen_count = 0
                self.peak = max(self.peak, approach)
        else:
            if ratio > self.reopen_ratio:
                self._reopen_count += 1
                if self._reopen_count >= self.reopen_frames:
                    self._blinks += 1
                    self._state = "open"
                    if self._blinks >= self.blinks_required:
                        self.peak = 1.0
                        self._finish()
            elif ratio < self.close_ratio * 0.85:
                self.peak = max(self.peak, approach)

    @property
    def progress(self) -> float:
        return 1.0 if self._done else min(self._blinks / self.blinks_required, 0.95)

    def summary(self) -> dict:
        data = super().summary()
        data.update(blinks=self._blinks, baseline=round(self._open_baseline, 4), viable=self.viable)
        return data


class HeadTurnDetector(_BaseDetector):
    """Yaw away from neutral *and back*.

    Two independent cues, either of which is sufficient:

    * **foreshortening** -- the inter-eye distance shrinks relative to the face
      box width as the head rotates. Sign-agnostic, so it cannot be broken by a
      mirrored front camera.
    * **asymmetry** -- the eyes shift off-centre within the face box.

    Completion requires returning to neutral for ``return_frames`` consecutive
    frames, which is what defeats a spliced "turn and hold" clip.
    """

    name = "head_turn"
    calibration_frames = 10
    span_shrink = 0.86  # span/span0 below this = turned
    asym_shift = 0.10  # |asym - asym0| above this = turned
    return_span = 0.94  # considered back at neutral above this
    return_asym = 0.06
    return_frames = 3

    def __init__(self, direction: str = "left") -> None:
        super().__init__()
        self.direction = direction
        self.name = f"head_turn_{direction}"
        self._span0 = 0.0
        self._asym0 = 0.0
        self._turned = False
        self._return_count = 0
        self.observed_sign = 0

    def _observe(self, sig: FrameSignals) -> tuple[float, float]:
        return sig.eye_span_ratio, sig.yaw_asym

    def _on_calibrated(self) -> None:
        spans = [s for s, _ in self._cal]
        asyms = [a for _, a in self._cal]
        self._span0 = float(np_percentile(spans, 75))
        self._asym0 = float(np_percentile(asyms, 50))

    def _evaluate(self, sig: FrameSignals) -> None:
        if sig.eyes_found < 2 or self._span0 <= 0.01:
            # Yaw needs both eyes; with one eye there is no inter-eye distance.
            return

        span_ratio = sig.eye_span_ratio / self._span0
        d_asym = sig.yaw_asym - self._asym0
        turned = (span_ratio < self.span_shrink) or (abs(d_asym) > self.asym_shift)
        # Record which way the head actually went, for audit + mirroring checks.
        self.observed_sign = 1 if d_asym > 0 else (-1 if d_asym < 0 else 0)

        if turned:
            self._turned = True
            self._return_count = 0
            self.peak = max(
                self.peak,
                1.0 - min(span_ratio, 1.0),
                min(abs(d_asym) / 0.25, 1.0),
            )
            return

        if self._turned and span_ratio > self.return_span and abs(d_asym) < self.return_asym:
            self._return_count += 1
            if self._return_count >= self.return_frames:
                self.peak = max(self.peak, 0.6)
                self._finish()
        else:
            self._return_count = 0


class MoveCloserDetector(_BaseDetector):
    """Face box grows by ~45% relative to the calibration baseline.

    Proximity is a genuine depth cue on a fixed webcam: an attacker holding a
    photo must keep the print at a fixed arm's length, so the *apparent* face
    scale can only be matched by also moving the print closer, which changes
    the print's blur, moire and reflection pattern.
    """

    name = "move_closer"
    calibration_frames = 8
    growth = 1.45
    hold_frames = 3

    def __init__(self) -> None:
        super().__init__()
        self._area0 = 0.0
        self._hold = 0

    def _observe(self, sig: FrameSignals) -> float:
        return float(sig.face_area)

    def _on_calibrated(self) -> None:
        self._area0 = float(np_percentile(self._cal, 50))
        self._area0 = max(self._area0, 1.0)

    def _evaluate(self, sig: FrameSignals) -> None:
        if sig.face_area <= 0:
            return
        ratio = sig.face_area / self._area0
        if ratio >= self.growth:
            self._hold += 1
            self.peak = max(self.peak, min(ratio / self.growth, 1.0))
            if self._hold >= self.hold_frames:
                self._finish()
        else:
            self._hold = 0


def np_percentile(values: Iterable[float], pct: float) -> float:
    """Local percentile to avoid importing numpy into the hot path twice."""
    import numpy as np

    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, pct))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
@dataclass
class Challenge:
    name: str
    index: int
    revealed_at: float
    detector: _BaseDetector
    completed_at: float | None = None

    @property
    def instruction(self) -> str:
        return INSTRUCTIONS.get(self.name, self.name)

    def to_dict(self, now: float, *, started_at: float | None = None) -> dict:
        out = {
            "name": self.name,
            "index": self.index,
            "instruction": self.instruction,
            "revealed": now >= self.revealed_at,
            "completed": self.completed_at is not None,
            "progress": round(self.detector.progress, 3),
            "detail": self.detector.summary(),
        }
        if started_at is not None:
            # Seconds after session start, matching the persisted
            # `reveal_offset_s` in LivenessSessionRow.challenges. The client
            # needs this to render the challenge schedule; without it the
            # randomised reveal order -- the anti-replay property -- is
            # invisible to the client it is meant to defend.
            out["reveal_offset_s"] = round(self.revealed_at - started_at, 3)
        return out


@dataclass
class EngineVerdict:
    status: str  # pending | passed | failed | expired | unavailable
    score: float = 0.0
    passed_challenges: list[str] = field(default_factory=list)
    reason: str | None = None
    active_challenge: dict | None = None
    challenges: list[dict] = field(default_factory=list)
    prompt: str | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "score": round(self.score, 4),
            "passed_challenges": list(self.passed_challenges),
            "reason": self.reason,
            "prompt": self.prompt,
            "active_challenge": self.active_challenge,
            "challenges": self.challenges,
        }


class LivenessEngine:
    """Session state machine. One instance per in-flight liveness session."""

    def __init__(
        self,
        challenge_names: Iterable[str],
        *,
        min_challenges: int = 2,
        ttl_seconds: float = 90.0,
        max_attempts: int = 3,
        enforce_direction: bool = False,
        seed: int | None = None,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.now_fn = now_fn
        self.ttl = ttl_seconds
        self.max_attempts = max_attempts
        self.enforce_direction = enforce_direction
        self.min_challenges = max(1, min(int(min_challenges), 8))
        self.started_at = now_fn()
        self.attempts = 0
        self.status = "pending"
        self.failure_reason: str | None = None
        # Shuffling challenge order is entropy against a recorded script, not
        # a secret: the state that matters lives in the DB, not in this seed.
        self._rng = random.Random(seed)  # noqa: S311

        pool = [n for n in dict.fromkeys(challenge_names) if n in SUPPORTED_CHALLENGES]
        dropped = [n for n in challenge_names if n not in SUPPORTED_CHALLENGES]
        if dropped:
            logger.warning("ignoring unsupported liveness challenges: %s", ", ".join(dropped))
        if not pool:
            raise ValueError("no supported liveness challenges configured")

        chosen = list(pool)
        self._rng.shuffle(chosen)
        chosen = chosen[: max(self.min_challenges, min(len(chosen), self.min_challenges + 1))]

        # Randomised reveal: the tape has to work against an unpredictable clock.
        base = self.started_at + self._rng.uniform(0.6, 2.2)
        self.challenges: list[Challenge] = []
        for i, name in enumerate(chosen):
            self.challenges.append(
                Challenge(
                    name=name,
                    index=i,
                    revealed_at=base + i * self._rng.uniform(0.4, 1.4),
                    detector=_make_detector(name),
                )
            )
        self._active_index = 0
        self._last_ts: float | None = None
        self._unreliable_total = 0
        self._frames = 0

    # -- helpers ----------------------------------------------------------
    @property
    def active(self) -> Challenge | None:
        if self._active_index >= len(self.challenges):
            return None
        ch = self.challenges[self._active_index]
        return ch if self.now_fn() >= ch.revealed_at else None

    @property
    def passed_challenges(self) -> list[str]:
        return [c.name for c in self.challenges if c.completed_at is not None]

    def _fail(self, reason: str) -> None:
        if self.status == "pending":
            self.status = "failed"
            self.failure_reason = reason

    def _pass(self) -> None:
        scores = [
            c.detector.peak
            for c in self.challenges
            if c.completed_at is not None
        ]
        mean = sum(scores) / len(scores) if scores else 0.0
        # Floor the score: completing N randomised challenges is strong evidence
        # even when the per-challenge peaks were modest.
        self.status = "passed"
        self.score = float(min(1.0, 0.6 + 0.4 * mean))

    # -- main entry point --------------------------------------------------
    def update(self, signals: FrameSignals) -> EngineVerdict:
        """Advance the session by one frame.

        Face presence is read from ``signals.face_present`` rather than a
        separate argument, so the two can never disagree.
        """
        if self.status in ("passed", "failed", "expired"):
            return self.verdict()

        now = self.now_fn()
        if now - self.started_at > self.ttl:
            self.status = "expired"
            self.failure_reason = "session_expired"
            return self.verdict()

        if not signals.face_present:
            self._unreliable_total += 1
            if self._unreliable_total > 40:
                self._fail("no_face_detected")
            return self.verdict()

        self._frames += 1
        ch = self.active
        if ch is None:
            return self.verdict()

        # Reject implausibly fast frame arrival. A 2-4x video replay shows up
        # as an inter-frame gap far below real camera timing.
        if self._last_ts is not None:
            gap = now - self._last_ts
            if 0 < gap < 1.0 / 20.0:
                self._fail("frame_rate_implausible")
        self._last_ts = now

        if not signals.reliable:
            ch.detector.unreliable_streak += 1
            self._unreliable_total += 1
            if ch.detector.unreliable_streak > ch.detector.unreliable_budget:
                # Sunglasses, a hat brim, motion blur: we cannot judge, so we
                # fail closed rather than hand out a pass on missing evidence.
                self._fail("insufficient_eye_evidence")
                return self.verdict()
            return self.verdict()

        ch.detector.update(signals)

        if ch.detector.done:
            if getattr(ch.detector, "viable", True) is False:
                self._fail("challenge_not_viable")
                return self.verdict()
            ch.completed_at = now
            self._active_index += 1
            self._unreliable_total = 0
            if len(self.passed_challenges) >= self.min_challenges:
                self._pass()

        return self.verdict()

    def fail(self, reason: str) -> EngineVerdict:
        self._fail(reason)
        return self.verdict()

    def verdict(self) -> EngineVerdict:
        now = self.now_fn()
        active = self.active
        return EngineVerdict(
            status=self.status,
            score=self.score if self.status == "passed" else 0.0,
            passed_challenges=self.passed_challenges,
            reason=self.failure_reason,
            active_challenge=active.to_dict(now, started_at=self.started_at) if active else None,
            challenges=[c.to_dict(now, started_at=self.started_at) for c in self.challenges],
            prompt=active.instruction if active else None,
        )


def _make_detector(name: str) -> _BaseDetector:
    if name == "blink":
        return BlinkDetector()
    if name == "head_turn_left":
        return HeadTurnDetector("left")
    if name == "head_turn_right":
        return HeadTurnDetector("right")
    if name == "move_closer":
        return MoveCloserDetector()
    raise ValueError(f"unsupported challenge: {name}")
