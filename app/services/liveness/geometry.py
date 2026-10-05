"""Per-frame signal extraction for active liveness.

Turns a frame into a :class:`FrameSignals` record: eye openness, eye-region
motion, head-yaw proxies and face scale. The challenge state machine in
:mod:`app.services.liveness.active` consumes only this record, so it can be
unit-tested with synthetic signals and no OpenCV, no camera, no model.

How eye openness is measured without a landmark model
-----------------------------------------------------
A 6-point EAR (Soukupova & Turlikova) needs eye *landmarks*. Getting those
means dlib, MediaPipe or a 68/478-point model -- heavy dependencies that, like
DeepFace, have no CPython 3.14 wheels. Instead, openness is measured from the
appearance of the eye bounding box itself:

    An OPEN eye has a vertically thick mass of dark pixels (iris, pupil,
    lashes) sitting in the middle of the crop, with brighter sclera above and
    below. A CLOSED eye collapses that dark mass into a thin horizontal lid
    line, so the dark pixels' vertical extent collapses.

So ``openness`` = the fraction of crop rows that contain "enough" dark pixels.
It is computed against a *local* median + MAD, which makes it invariant to
overall exposure -- the user's face brightness does not matter, only whether
the dark mass is vertically thick.

This is a heuristic, not a landmark-grade EAR, and the blind spot is real: a
hat brim or sunglasses over the eyes will suppress blinks. Callers get a
``reliable`` flag and are expected to fail the session rather than pass it
when eye evidence is missing.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["EyeDetector", "EyeSignal", "FrameSignals", "extract_signals", "eye_detector_status"]


@dataclass(slots=True)
class EyeSignal:
    box: tuple[int, int, int, int]  # x, y, w, h
    openness: float  # 0..1, vertical extent of the dark mass
    motion: float  # 0..1, localised frame-to-frame change
    usable: bool = True


@dataclass(slots=True)
class FrameSignals:
    timestamp: float
    face_present: bool = False
    face_box: tuple[int, int, int, int] | None = None
    face_area: int = 0
    eyes: list[EyeSignal] = field(default_factory=list)

    # Derived, filled in by extract_signals
    openness: float = 0.0
    eye_motion: float = 0.0
    eyes_found: int = 0
    yaw_asym: float = 0.0
    eye_span_ratio: float = 0.0

    @property
    def reliable(self) -> bool:
        """Enough evidence to judge a challenge this frame."""
        return self.face_present and self.eyes_found >= 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_present": self.face_present,
            "face_box": list(self.face_box) if self.face_box else None,
            "openness": round(self.openness, 4),
            "eye_motion": round(self.eye_motion, 4),
            "eyes_found": self.eyes_found,
            "yaw_asym": round(self.yaw_asym, 4),
            "eye_span_ratio": round(self.eye_span_ratio, 4),
            "reliable": self.reliable,
        }


class EyeDetector:
    """Haar cascade eye detector, loaded lazily from the OpenCV data dir."""

    def __init__(self) -> None:
        self._cascade: Any = None
        self._lock = threading.Lock()
        self.error: str | None = None

    @property
    def available(self) -> bool:
        return self._load() is not None

    def _load(self):
        if self._cascade is not None:
            return self._cascade
        with self._lock:
            if self._cascade is not None:
                return self._cascade
            try:
                import cv2

                path = f"{cv2.data.haarcascades}haarcascade_eye.xml"
                cascade = cv2.CascadeClassifier(path)
                if cascade.empty():
                    raise RuntimeError(f"cascade failed to load: {path}")
                self._cascade = cascade
                self.error = None
            except Exception as exc:  # noqa: BLE001 - degrade to "unavailable", never raise
                self.error = f"{type(exc).__name__}: {exc}"
                logger.warning("eye detector unavailable: %s", self.error)
        return self._cascade

    def detect(self, gray: np.ndarray, face_box: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
        """Eyes live in the upper ~60% of the face box; searching the whole box
        picks up eyebrow and mouth edges and produces phantom signals."""
        cascade = self._load()
        if cascade is None:
            return []

        fx, fy, fw, fh = face_box
        y0 = max(0, fy)
        y1 = max(y0 + 1, fy + int(fh * 0.6))
        x0 = max(0, fx)
        x1 = max(x0 + 1, fx + fw)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0 or min(roi.shape[:2]) < 12:
            return []

        min_edge = max(8, int(fw * 0.08))
        found = cascade.detectMultiScale(
            roi,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(min_edge, min_edge),
            maxSize=(int(fw * 0.6), int(fh * 0.45)),
        )
        out: list[tuple[int, int, int, int]] = []
        for (ex, ey, ew, eh) in found:
            if ew < 6 or eh < 6:
                continue
            # Translate back to full-frame coordinates and drop implausible
            # aspect ratios (Haar sometimes returns a sliver next to the brow).
            if not 0.25 <= ew / max(eh, 1) <= 3.0:
                continue
            out.append((int(x0 + ex), int(y0 + ey), int(ew), int(eh)))
        return out

    def reset(self) -> None:
        self._cascade = None


_detector = EyeDetector()


def eye_detector_status() -> dict[str, Any]:
    return {"available": _detector.available, "error": _detector.error}


def _openness(eye_crop: np.ndarray) -> float:
    """Vertical extent of the eye's dark mass, 0..1. See module docstring."""
    patch = np.asarray(eye_crop, dtype=np.float64)
    h, w = patch.shape[:2]
    if h < 5 or w < 5:
        return 0.0
    med = float(np.median(patch))
    mad = float(np.median(np.abs(patch - med))) * 1.4826
    threshold = med - max(6.0, 2.0 * mad)
    dark = patch < threshold
    row_dark = dark.mean(axis=1)
    # A row counts as "part of the eye" if >=7% of it is dark. Below that the
    # row is skin, sclera or noise.
    rows = row_dark > 0.07
    return float(rows.sum() / h)


def _motion(curr: np.ndarray, prev: np.ndarray | None) -> float:
    if prev is None or prev.shape != curr.shape:
        return 0.0
    diff = np.abs(np.asarray(curr, dtype=np.float64) - np.asarray(prev, dtype=np.float64))
    return float(np.clip(np.mean(diff) / 12.0, 0.0, 1.0))


def extract_signals(
    gray: np.ndarray,
    face_box: tuple[int, int, int, int] | None,
    timestamp: float,
    prev_eye_crops: dict[int, np.ndarray] | None = None,
) -> tuple[FrameSignals, dict[int, np.ndarray]]:
    """Build :class:`FrameSignals` for one frame.

    Returns the signals plus the eye crops needed as ``prev_eye_crops`` on the
    next call, so motion can be computed without re-cropping.
    """
    signals = FrameSignals(timestamp=timestamp)

    if face_box is None:
        return signals, {}

    fx, fy, fw, fh = face_box
    signals.face_present = True
    signals.face_box = (int(fx), int(fy), int(fw), int(fh))
    signals.face_area = int(fw * fh)
    if fw < 8 or fh < 8:
        signals.face_present = False
        return signals, {}

    eye_boxes = _detector.detect(gray, face_box)
    signals.eyes_found = len(eye_boxes)

    prev = prev_eye_crops or {}
    new_crops: dict[int, np.ndarray] = {}
    eyes: list[EyeSignal] = []

    h_img, w_img = gray.shape[:2]
    for i, (ex, ey, ew, eh) in enumerate(sorted(eye_boxes, key=lambda b: b[0])):
        x0, y0 = max(0, ex), max(0, ey)
        x1, y1 = min(w_img, ex + ew), min(h_img, ey + eh)
        if x1 - x0 < 5 or y1 - y0 < 5:
            continue
        crop = gray[y0:y1, x0:x1]
        new_crops[i] = crop
        eyes.append(
            EyeSignal(
                box=(x0, y0, x1 - x0, y1 - y0),
                openness=_openness(crop),
                motion=_motion(crop, prev.get(i)),
                usable=True,
            )
        )

    signals.eyes = eyes
    if eyes:
        signals.openness = float(np.mean([e.openness for e in eyes]))
        signals.eye_motion = float(max(e.motion for e in eyes))

    if len(eyes) == 2:
        (lbox, rbox) = (eyes[0].box, eyes[1].box)
        lcx = lbox[0] + lbox[2] / 2.0
        rcx = rbox[0] + rbox[2] / 2.0
        signals.eye_span_ratio = float(abs(rcx - lcx) / max(fw, 1))
        gap_left = (lcx - fx) / max(fw, 1)
        gap_right = ((fx + fw) - rcx) / max(fw, 1)
        # Positive => eyes sit toward the frame-right of the face box.
        signals.yaw_asym = float(gap_right - gap_left)
    elif len(eyes) == 1:
        cx = eyes[0].box[0] + eyes[0].box[2] / 2.0
        signals.yaw_asym = float(((fx + fw) - cx) - (cx - fx)) / max(fw, 1)
        signals.eye_span_ratio = 0.0
    else:
        signals.yaw_asym = 0.0
        signals.eye_span_ratio = 0.0

    return signals, new_crops
