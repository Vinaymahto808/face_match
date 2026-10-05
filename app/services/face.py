"""Face detection + embedding, behind a swappable backend interface.

Why an interface: ``deepface`` drags in TensorFlow, which has no wheels for
CPython 3.14. Rather than making the whole service unbootable on a machine
without it, the analyzer is a small protocol with two implementations:

* :class:`DeepFaceAnalyzer` -- production. Model loads lazily on first use.
* :class:`StubAnalyzer` -- deterministic, dependency-free. Used by the test
  suite and local development, and hard-blocked from writing attendance
  unless ``ALLOW_STUB_BACKEND=true``.

A note on frames: the notebook wrote every webcam frame to ``temp_live.jpg``,
ran DeepFace on the path, then deleted the file -- disk I/O plus a race between
concurrent requests sharing one filename. DeepFace accepts a NumPy BGR array
directly, so frames are passed in memory and the temp file disappears entirely.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)

__all__ = [
    "DeepFaceAnalyzer",
    "FaceAnalyzer",
    "FaceObservation",
    "StubAnalyzer",
    "analyzer_status",
    "get_analyzer",
    "reset_analyzer",
]


class _Unraisable(Exception):
    """Stand-in for deepface's FaceNotDetected when the import path is unknown."""


@dataclass(slots=True)
class FaceObservation:
    """One detected face in a frame."""

    x: int
    y: int
    w: int
    h: int
    confidence: float = 0.0
    embedding: np.ndarray | None = None

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Safe ROI slice: negative indices clamp, so out-of-bounds boxes
        degrade to a smaller crop instead of an exception."""
        h_img, w_img = image.shape[:2]
        x0 = max(0, min(int(self.x), w_img - 1))
        y0 = max(0, min(int(self.y), h_img - 1))
        x1 = max(x0 + 1, min(int(self.x + self.w), w_img))
        y1 = max(y0 + 1, min(int(self.y + self.h), h_img))
        return image[y0:y1, x0:x1]


@runtime_checkable
class FaceAnalyzer(Protocol):
    """Contract every backend implements."""

    name: str
    is_stub: bool
    ready: bool

    def warmup(self) -> None: ...

    def analyze(self, image_bgr: np.ndarray, *, strict: bool = True) -> list[FaceObservation]: ...


class _BaseAnalyzer:
    is_stub = False

    def __init__(self) -> None:
        self._ready = False
        self._lock = threading.Lock()
        self.last_error: str | None = None
        self.load_seconds: float | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    def analyze(self, image_bgr: np.ndarray, *, strict: bool = True) -> list[FaceObservation]:
        raise NotImplementedError

    def _mark_ready(self, started: float) -> None:
        self._ready = True
        self.load_seconds = round(time.perf_counter() - started, 2)
        logger.info("face backend %r ready in %ss", self.name, self.load_seconds)

    def _mark_failed(self, exc: BaseException) -> None:
        self.last_error = f"{type(exc).__name__}: {exc}"
        logger.exception("face backend %r failed to initialise", self.name)


class DeepFaceAnalyzer(_BaseAnalyzer):
    """Production backend. Weights load once, on first use, under a lock."""

    name = "deepface"

    def __init__(self, model_name: str | None = None, detector: str | None = None) -> None:
        super().__init__()
        self.model_name = model_name or settings.face_model_name
        self.detector = detector or settings.face_detector
        self._deepface: Any = None

    # -- lifecycle ---------------------------------------------------------
    def warmup(self) -> None:
        with self._lock:
            if self._ready:
                return
            started = time.perf_counter()
            try:
                from deepface import DeepFace  # heavy import, kept lazy

                self._deepface = DeepFace
                self._build_model()
                self._mark_ready(started)
            except Exception as exc:  # noqa: BLE001 - record and serve 503 instead of crashing
                self._mark_failed(exc)

    @staticmethod
    def _face_not_detected_error() -> type[BaseException]:
        """The exception class deepface raises instead of returning no faces.

        Resolved by name at call time rather than imported at module scope: the
        import is deepface-internal and could move between releases, and this
        module has to stay importable without TensorFlow installed. Falls back to
        a private subclass nobody raises, which simply means the translate-back
        never fires and the original error still propagates.
        """
        try:
            from deepface.modules.exceptions import FaceNotDetected

            return FaceNotDetected
        except Exception:  # noqa: BLE001 - deepface layout changed; keep old behaviour
            return _Unraisable

    def _build_model(self) -> None:
        """Build the network without needing a real face in a test image.

        DeepFace builds the model lazily inside its inference path, so a warmup
        on a synthetic image can return early with zero faces and never load
        anything. Call its model factory directly, and only fall back to a
        throwaway inference if that private path has moved.
        """
        assert self._deepface is not None
        last_error: Exception | None = None
        for module_path, attr in (
            ("deepface.modules.facialRepresentation", "build_model"),
            ("deepface.modules", "facialRepresentation"),
        ):
            try:
                import importlib

                mod = importlib.import_module(module_path)
                factory = getattr(mod, attr, None) or getattr(
                    getattr(mod, "facialRepresentation", None), "build_model", None
                )
                if callable(factory):
                    factory(self.model_name)
                    return
            except Exception as exc:  # noqa: BLE001 - trying a second construction strategy
                last_error = exc
        logger.debug("direct model build unavailable (%s); warming via inference", last_error)
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            self._deepface.represent(
                img_path=frame, model_name=self.model_name, enforce_detection=False
            )
        except Exception:  # noqa: BLE001 - warmup is best-effort by design
            # A failure here is not fatal: the next real request will build the
            # model anyway. We just don't get eager warmup.
            logger.warning("inference warmup did not succeed; relying on lazy load")

    # -- inference ---------------------------------------------------------
    def analyze(self, image_bgr: np.ndarray, *, strict: bool = True) -> list[FaceObservation]:
        if not self._ready:
            self.warmup()
        if not self._ready or self._deepface is None:
            raise RuntimeError(
                f"face backend unavailable ({self.last_error or 'not loaded'}). "
                "Check that deepface/tensorflow are installed."
            )
        if image_bgr is None or image_bgr.size == 0:
            return []

        # ndarray in, ndarray out -- no temp files.
        # The kwarg is `detector_backend`, not `detector`. Passing the wrong
        # name raises TypeError on every single request, and because the test
        # suite runs on StubAnalyzer it would never be caught there -- so this
        # call is asserted against the installed signature in
        # tests/test_embeddings.py.
        try:
            results = self._deepface.represent(
                img_path=image_bgr,
                model_name=self.model_name,
                detector_backend=self.detector,
                enforce_detection=strict,
            )
        except self._face_not_detected_error() as exc:
            # With `enforce_detection=True`, deepface raises instead of
            # returning an empty list. Callers are written against the
            # empty-list contract -- "no observations" is the same fact as
            # "no face in this frame" -- so translate the raise back into
            # that shape. Without this, uploading a photo with no face to
            # /users/enroll/{id}/sample escapes as a 500 instead of the
            # intended 400 no_face_detected.
            logger.debug("no face detected in frame: %s", exc)
            return []
        if not results:
            return []

        out: list[FaceObservation] = []
        for item in results:
            area = item.get("facial_area") or {}
            if not area or int(area.get("w", 0)) <= 0 or int(area.get("h", 0)) <= 0:
                continue
            embedding = item.get("embedding")
            out.append(
                FaceObservation(
                    x=int(area.get("x", 0)),
                    y=int(area.get("y", 0)),
                    w=int(area["w"]),
                    h=int(area["h"]),
                    confidence=float(item.get("face_confidence") or 0.0),
                    embedding=None if embedding is None else np.asarray(embedding, dtype=np.float64),
                )
            )
        return out


class StubAnalyzer(_BaseAnalyzer):
    """Dependency-free deterministic backend for tests and local dev.

    Detects one centred "face" box and derives a 512-d embedding from a
    16x16 grayscale signature of the crop. Identical crops give identical
    vectors, different crops give different vectors, so matching, throttling
    and the liveness state machine are all exercisable without a webcam.
    """

    is_stub = True
    name = "stub"

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.dim = dim
        self._ready = True

    def warmup(self) -> None:
        self._ready = True

    def _signature(self, crop: np.ndarray) -> np.ndarray:
        gray = crop.mean(axis=2) if crop.ndim == 3 else crop.astype(np.float64)
        if gray.ndim != 2 or gray.size == 0:
            return np.zeros(self.dim)
        h, w = gray.shape
        ys = np.linspace(0, h - 1, 8)
        xs = np.linspace(0, w - 1, 8)
        thumb = gray[np.ix_(ys.astype(int), xs.astype(int))].astype(np.float64)
        flat = (thumb - thumb.mean()) / (thumb.std() + 1e-6)
        reps = int(np.ceil(self.dim / flat.size))
        return np.tile(flat.ravel(), reps)[: self.dim]

    def analyze(self, image_bgr: np.ndarray, *, strict: bool = True) -> list[FaceObservation]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        h, w = image_bgr.shape[:2]
        bw, bh = int(w * 0.4), int(h * 0.5)
        x, y = (w - bw) // 2, max(0, (h - bh) // 3)
        obs = FaceObservation(x=x, y=y, w=bw, h=bh, confidence=0.99)
        obs.embedding = self._signature(obs.crop(image_bgr))
        return [obs]


_analyzer: FaceAnalyzer | None = None
_analyzer_lock = threading.Lock()


def get_analyzer() -> FaceAnalyzer:
    """Process-wide singleton."""
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                if settings.embedding_backend == "stub":
                    logger.warning(
                        "EMBEDDING_BACKEND=stub -- deterministic fake embeddings, "
                        "NOT usable for real recognition"
                    )
                    _analyzer = StubAnalyzer()
                else:
                    _analyzer = DeepFaceAnalyzer()
    return _analyzer


def reset_analyzer() -> None:
    """Test hook: force the next ``get_analyzer()`` to rebuild."""
    global _analyzer
    with _analyzer_lock:
        _analyzer = None


def analyzer_status() -> dict[str, Any]:
    a = get_analyzer()
    return {
        "backend": a.name,
        "model": settings.face_model_name if a.name == "deepface" else "deterministic-signature",
        "detector": settings.face_detector if a.name == "deepface" else None,
        "ready": bool(a.ready),
        "is_stub": bool(a.is_stub),
        "attendance_blocked": bool(a.is_stub and not settings.allow_stub_backend),
        "load_seconds": getattr(a, "load_seconds", None),
        "last_error": getattr(a, "last_error", None),
    }
