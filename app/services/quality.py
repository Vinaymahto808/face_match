"""Frame and face quality gating.

Pure NumPy on purpose: no OpenCV import, so it is testable anywhere and adds
nothing to the dependency surface. Everything here is a *rejection* decision
made before an embedding is ever computed.

Why it matters for anti-spoofing too: a photo held up to the camera is often
motion-blurred, off-axis and low-contrast. Failing those frames early also
denies an attacker a stream of garbage frames to brute-force a match with.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import settings

__all__ = ["QualityReport", "assess_face", "assess_frame", "to_gray"]


@dataclass(slots=True)
class QualityReport:
    ok: bool
    score: float  # 0..1, higher is better
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
        }


def to_gray(image: np.ndarray) -> np.ndarray:
    """Rec.601 luma. Accepts HxWx3 (BGR/RGB) or HxW."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"unsupported image shape {arr.shape}")
    b, g, r = arr[..., 0], arr[..., 1], arr[..., 2]
    return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.float64)


def _laplacian(gray: np.ndarray) -> np.ndarray:
    """4-neighbour discrete Laplacian, edge-preserving via slicing."""
    p = np.pad(gray, 1, mode="reflect")
    return (
        p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * gray
    )


def _standardize(gray: np.ndarray) -> np.ndarray:
    mu, sd = gray.mean(), gray.std()
    return (gray - mu) / (sd + 1e-6)


def _basic_stats(gray: np.ndarray) -> dict[str, float]:
    lap = _laplacian(gray)
    # Saturated pixels carry no gradient information regardless of local detail.
    saturated = (gray <= 2) | (gray >= 253)
    grad = np.abs(lap)[~saturated] if saturated.any() else np.abs(lap)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(grad.var()) if grad.size else 0.0,
        "flat_fraction": float((grad < 4.0).mean()) if grad.size else 1.0,
    }


def assess_frame(image: np.ndarray) -> QualityReport:
    """Whole-frame sanity: dimensions, exposure, contrast, sharpness."""
    reasons: list[str] = []
    arr = np.asarray(image)
    if arr.ndim < 2 or arr.size == 0:
        return QualityReport(ok=False, score=0.0, reasons=["empty_frame"])

    h, w = arr.shape[:2]
    if min(h, w) < settings.min_frame_edge_px:
        reasons.append(f"frame_too_small:{w}x{h}")

    gray = to_gray(arr)
    stats = _basic_stats(gray)
    score = 1.0

    if stats["brightness"] < settings.min_brightness:
        reasons.append("underexposed")
        score -= 0.35
    if stats["brightness"] > settings.max_brightness:
        reasons.append("overexposed")
        score -= 0.35
    if stats["contrast"] < settings.min_contrast:
        reasons.append("low_contrast")
        score -= 0.25
    if stats["sharpness"] < settings.min_sharpness:
        reasons.append("blurry")
        score -= 0.3
    if stats["flat_fraction"] > settings.max_flat_region_fraction:
        # Large textureless areas: blank wall, a flat synthetic patch, or a
        # heavily smoothed/downscaled image.
        reasons.append("mostly_flat")
        score -= 0.2

    return QualityReport(
        ok=not reasons,
        score=max(0.0, min(1.0, score)),
        reasons=reasons,
        metrics=stats,
    )


def assess_face(gray_face: np.ndarray, face_w: int, face_h: int) -> QualityReport:
    """Per-face gate. Input is the *luma* crop of the detected face box."""
    reasons: list[str] = []
    if min(face_w, face_h) < settings.min_face_edge_px:
        reasons.append(f"face_too_small:{face_w}x{face_h}")

    gray = np.asarray(gray_face, dtype=np.float64)
    if gray.size == 0:
        return QualityReport(ok=False, score=0.0, reasons=["empty_face_crop"])

    if gray.shape[0] < 8 or gray.shape[1] < 8:
        # Too few pixels for a meaningful Laplacian; size check already fired.
        stats = {"brightness": float(gray.mean()), "contrast": float(gray.std())}
        return QualityReport(
            ok=not reasons, score=0.2 if not reasons else 0.0, reasons=reasons, metrics=stats
        )

    stats = _basic_stats(gray)
    stats["fill_ratio"] = float(gray.shape[1]) / float(max(1, face_w))
    score = 1.0

    if stats["brightness"] < settings.min_brightness:
        reasons.append("underexposed")
        score -= 0.3
    if stats["brightness"] > settings.max_brightness:
        reasons.append("overexposed")
        score -= 0.3
    if stats["sharpness"] < settings.min_sharpness:
        reasons.append("blurry")
        score -= 0.3
    if stats["contrast"] < settings.min_contrast:
        reasons.append("low_contrast")
        score -= 0.2
    if stats["flat_fraction"] > settings.max_flat_region_fraction:
        reasons.append("low_texture")
        score -= 0.2

    return QualityReport(
        ok=not reasons,
        score=max(0.0, min(1.0, score)),
        reasons=reasons,
        metrics=stats,
    )


def texture_signature(gray_face: np.ndarray, grid: int = 8) -> np.ndarray:
    """Normalised grid signature of a crop.

    Used by both the liveness heuristic and the stub backend: two crops of the
    same person under similar lighting produce near-identical signatures.
    """
    gray = np.asarray(gray_face, dtype=np.float64)
    if gray.size == 0:
        return np.zeros(grid * grid)
    ys = np.linspace(0, gray.shape[0] - 1, grid).astype(int)
    xs = np.linspace(0, gray.shape[1] - 1, grid).astype(int)
    thumb = gray[np.ix_(ys, xs)]
    return ((thumb - thumb.mean()) / (thumb.std() + 1e-6)).ravel()
