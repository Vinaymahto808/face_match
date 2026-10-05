"""Passive (single-frame) anti-spoofing signals.

What this can and cannot do
---------------------------
Presentation attacks here mean one of:

* a **printed photo** held up to the lens
* a **photo/video on a phone or laptop screen** re-photographed by the camera
* a **pre-recorded video** fed into a virtual camera

Passive analysis looks for artefacts of those re-captures. It is a *weak*
signal, on purpose-weighted terms, and this module never claims otherwise:

* Skin micro-texture survives a camera-to-disk round trip but is smoothed by
  print+dust and by screenshot-then-upscale, so texture energy drops.
* A camera aimed at a screen re-samples the panel's RGB subpixel grid, which
  shows up as periodic energy in the high-frequency spectrum (moire).
* Screenshot-and-recompress pipelines leave large flat plateaus and very low
  sensor-noise residuals, because the original sensor noise is already gone.
* Print/ICC/screenshot colour handling reduces chroma spread.

None of these survive a well-made 4K display at the right distance, which is
exactly why :mod:`app.services.liveness.active` (randomised challenge-response)
is the primary defence and this module is the secondary one. Both are
required by the attendance endpoint.

All maths here is pure NumPy so it is unit-testable without OpenCV, TensorFlow
or a camera.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["PassiveVerdict", "extract_features", "passive_liveness", "score_features"]

Verdict = Literal["alive", "inconclusive", "spoof_suspect", "unavailable"]

# Each feature is mapped to a 0..1 "human-ness" sub-score, then combined with
# weights. Weights favour the cues that are most reliable across capture
# pipelines and deliberately keep the shakier spectral ones small.
#
# Two kinds of mapping:
#   _RAMPS      monotone: (suspicious, healthy). Invert if the healthy end is high.
#   _BAND_RAMPS band-pass: (very_low, healthy_low, healthy_high, very_high).
#               Some cues are *not* monotone -- a spectrum with almost no
#               high-frequency energy is a smoothed print, and a spectrum with
#               far too much is pure sensor noise or JPEG mush. Both are bad, so
#               a linear ramp would mark every real photograph as suspicious.
_RAMPS: dict[str, tuple[float, float]] = {
    "flat_fraction": (0.30, 0.75),          # inverted below
    "laplacian_var": (8.0, 60.0),
    "noise_sigma": (0.5, 3.0),
    "saturation_std": (0.02, 0.10),
    "colorfulness": (8.0, 30.0),
    "skin_fraction": (0.25, 0.70),
    "radial_peakiness": (1.6, 6.0),         # inverted below
}

_BAND_RAMPS: dict[str, tuple[float, float, float, float]] = {
    # (too_low, healthy_low, healthy_high, too_high)
    "spectral_flatness": (0.0010, 0.008, 0.050, 0.250),
    "hf_energy_ratio": (0.0012, 0.004, 0.030, 0.120),
}

_WEIGHTS: dict[str, float] = {
    "flat_fraction": 0.20,
    "laplacian_var": 0.16,
    "noise_sigma": 0.14,
    "saturation_std": 0.11,
    "colorfulness": 0.11,
    "hf_energy_ratio": 0.09,
    "skin_fraction": 0.08,
    "spectral_flatness": 0.06,
    "radial_peakiness": 0.05,
}

_INVERTED = {"flat_fraction", "radial_peakiness"}

_FLAG_NAMES: dict[str, str] = {
    "flat_fraction": "smoothed_texture_print_or_screenshot",
    "laplacian_var": "low_micro_detail",
    "noise_sigma": "missing_sensor_noise_recompressed",
    "saturation_std": "flat_chroma_distribution",
    "colorfulness": "washed_out_colour",
    "hf_energy_ratio": "low_high_frequency_detail",
    "spectral_flatness": "unnatural_spectral_profile",
    "radial_peakiness": "possible_screen_replay_moire",
    "skin_fraction": "weak_skin_signature",
}

_GRAY_W = 128


@dataclass(slots=True)
class PassiveVerdict:
    score: float
    verdict: Verdict
    flags: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    sub_scores: dict[str, float] = field(default_factory=dict)
    engine: str = "heuristics"

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "verdict": self.verdict,
            "flags": list(self.flags),
            "engine": self.engine,
            "features": {k: round(v, 5) for k, v in self.features.items()},
            "sub_scores": {k: round(v, 3) for k, v in self.sub_scores.items()},
        }


def _gray(face_bgr: np.ndarray) -> np.ndarray:
    arr = np.asarray(face_bgr, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    b, g, r = arr[..., 0], arr[..., 1], arr[..., 2]
    return 0.114 * b + 0.587 * g + 0.299 * r


def _laplacian(gray: np.ndarray) -> np.ndarray:
    p = np.pad(gray, 1, mode="reflect")
    return p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * gray


def _resize_gray(gray: np.ndarray, size: int) -> np.ndarray:
    """Area-average downscale to ``size``x``size``. Averaging, not sampling:
    sampling would alias away exactly the high-frequency evidence we want."""
    h, w = gray.shape[:2]
    if h == size and w == size:
        return gray.copy()
    h_edges = np.linspace(0, h, size + 1).astype(int)
    w_edges = np.linspace(0, w, size + 1).astype(int)
    out = np.empty((size, size), dtype=np.float64)
    for i in range(size):
        y0, y1 = h_edges[i], max(h_edges[i] + 1, h_edges[i + 1])
        for j in range(size):
            x0, x1 = w_edges[j], max(w_edges[j] + 1, w_edges[j + 1])
            out[i, j] = gray[y0:y1, x0:x1].mean()
    return out


def _ramp(value: float, key: str) -> float:
    """Map a raw feature onto a 0..1 "looks like a live capture" sub-score."""
    if key in _BAND_RAMPS:
        too_low, good_low, good_high, too_high = _BAND_RAMPS[key]
        if value <= too_low or value >= too_high:
            return 0.0
        if value < good_low:
            return (value - too_low) / (good_low - too_low)
        if value > good_high:
            return (too_high - value) / (too_high - good_high)
        return 1.0

    lo, hi = _RAMPS[key]
    if hi <= lo:
        return 0.0
    t = (value - lo) / (hi - lo)
    t = float(np.clip(t, 0.0, 1.0))
    return 1.0 - t if key in _INVERTED else t


def extract_features(face_bgr: np.ndarray) -> dict[str, float]:
    """Compute the nine re-capture cues from a face crop."""
    arr = np.asarray(face_bgr, dtype=np.float64)
    gray = _gray(arr)
    if gray.shape[0] < 16 or gray.shape[1] < 16:
        raise ValueError("face crop too small for liveness analysis")

    feats: dict[str, float] = {}

    # -- micro-texture ----------------------------------------------------
    lap = np.abs(_laplacian(gray))
    feats["laplacian_var"] = float(lap.var())
    feats["flat_fraction"] = float((lap < 4.0).mean())

    # -- sensor-noise residual -------------------------------------------
    # High-pass residual, measured by median absolute deviation so that the
    # real texture content does not dominate the estimate.
    high = _laplacian(gray)
    feats["noise_sigma"] = float(1.4826 * np.median(np.abs(high - np.median(high))))

    # -- colour ----------------------------------------------------------
    if arr.ndim == 3 and arr.shape[2] >= 3:
        b, g, r = arr[..., 0], arr[..., 1], arr[..., 2]
        mx, mn = arr.max(axis=2), arr.min(axis=2)
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
        feats["saturation_std"] = float(sat.std())

        rg = r - g
        yb = 0.5 * (r + g) - b
        rg_mean, rg_std = float(rg.mean()), float(rg.std())
        yb_mean, yb_std = float(yb.mean()), float(yb.std())
        # Hasler & Suesstrunk colourfulness.
        feats["colorfulness"] = float(
            np.hypot(rg_std, yb_std) + 0.3 * np.hypot(rg_mean, yb_mean)
        )

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cr = (r - y) * 0.713 + 128.0
        cb = (b - y) * 0.564 + 128.0
        skin = (cr >= 133) & (cr <= 178) & (cb >= 77) & (cb <= 132)
        feats["skin_fraction"] = float(skin.mean())
    else:  # grayscale input: colour cues are undefined, neutral value
        feats["saturation_std"] = 0.06
        feats["colorfulness"] = 19.0
        feats["skin_fraction"] = 0.5

    # -- spectrum --------------------------------------------------------
    small = _resize_gray(gray, _GRAY_W)
    centered = small - small.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    total = power.sum()
    if total <= 0:
        feats.update(hf_energy_ratio=0.0, spectral_flatness=0.0, radial_peakiness=0.0)
        return feats

    # Radially-binned power spectrum, ignoring the DC centre.
    yy, xx = np.indices(power.shape)
    cy, cx = _GRAY_W // 2, _GRAY_W // 2
    radius = np.hypot(yy - cy, xx - cx)
    rmax = float(radius.max())
    norm_r = radius / rmax

    high_mask = norm_r >= 0.6
    feats["hf_energy_ratio"] = float(power[high_mask].sum() / total)

    p = power[power > 0]
    feats["spectral_flatness"] = float(
        np.exp(np.log(p).mean()) / (p.mean() + 1e-12)
    )

    # Peakiness of the outer half of the spectrum: periodic screen-grid
    # interference shows up here as an isolated strong radial bin.
    bands = 24
    outer = power[(norm_r >= 0.45) & (norm_r <= 0.95)]
    if outer.size >= bands:
        edges = np.linspace(0, outer.size, bands + 1).astype(int)
        radial = np.array(
            [outer[edges[i] : max(edges[i] + 1, edges[i + 1])].mean() for i in range(bands)]
        )
        med = float(np.median(radial)) + 1e-9
        feats["radial_peakiness"] = float(radial.max() / med)
    else:
        feats["radial_peakiness"] = 1.0

    return feats


def score_features(features: dict[str, float]) -> tuple[float, dict[str, float], list[str]]:
    """Weighted fusion of feature sub-scores into one 0..1 score + flags."""
    sub = {key: _ramp(features.get(key, 0.0), key) for key in _WEIGHTS}
    total_w = sum(_WEIGHTS.values())
    score = sum(sub[k] * w for k, w in _WEIGHTS.items()) / total_w

    flags = [
        _FLAG_NAMES[k]
        for k, v in sub.items()
        if v < 0.35 and _WEIGHTS[k] >= 0.08
    ]
    return float(np.clip(score, 0.0, 1.0)), sub, flags


# ---------------------------------------------------------------------------
# Optional CNN scorer (mini-FASNet). Enable with LIVENESS_FASNET_MODEL=/path.onnx
# ---------------------------------------------------------------------------
class _FasNetScorer:
    """Wraps a mini-FASNet ONNX model: 80x80x3 -> P(live)."""

    def __init__(self, path: str) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.path = path

    @staticmethod
    def _preprocess(face_bgr: np.ndarray) -> np.ndarray:
        from ..utils.images import resize_bilinear

        rgb = np.asarray(face_bgr, dtype=np.float32)[:, :, ::-1] / 255.0
        small = resize_bilinear(rgb, 80, 80)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return ((small - mean) / std).transpose(2, 0, 1)[None, ...].astype(np.float32)

    def score(self, face_bgr: np.ndarray) -> float:
        out = self.session.run(None, {self.input_name: self._preprocess(face_bgr)})
        logits = np.asarray(out[0]).ravel()
        if logits.size == 1:
            return float(logits[0])
        # Two-class head: column 0 is the spoof/label-0 class in the reference
        # mini-FASNet export, so P(live) is the softmax probability of column 1.
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return float(probs[-1])


_fasnet: _FasNetScorer | None = None
_fasnet_lock = threading.Lock()
_fasnet_error: str | None = None


def _get_fasnet(path: str | None) -> _FasNetScorer | None:
    global _fasnet, _fasnet_error
    if not path:
        return None
    if _fasnet is not None:
        return _fasnet
    with _fasnet_lock:
        if _fasnet is None:
            try:
                _fasnet = _FasNetScorer(path)
                _fasnet_error = None
                logger.info("loaded mini-FASNet liveness model from %s", path)
            except Exception as exc:  # noqa: BLE001 - heuristics are the fallback
                _fasnet_error = f"{type(exc).__name__}: {exc}"
                logger.warning("mini-FASNet unavailable (%s); heuristics only", _fasnet_error)
    return _fasnet


def reset_fasnet() -> None:
    """Test hook."""
    global _fasnet, _fasnet_error
    with _fasnet_lock:
        _fasnet = None
        _fasnet_error = None


def fasnet_status() -> dict[str, Any]:
    scorer = _fasnet
    return {
        "loaded": scorer is not None,
        "path": getattr(scorer, "path", None),
        "error": _fasnet_error,
    }


def passive_liveness(face_bgr: np.ndarray, *, threshold: float) -> PassiveVerdict:
    """Score one face crop. ``threshold`` is the minimum score to call it alive."""
    try:
        features = extract_features(face_bgr)
    except ValueError as exc:
        return PassiveVerdict(
            score=0.0, verdict="inconclusive", flags=[str(exc)], engine="heuristics"
        )

    score, sub, flags = score_features(features)
    engine = "heuristics"

    # Fusion with the CNN scorer when one is configured. The CNN is the more
    # trustworthy signal, so it gets the majority weight.
    scorer = _get_fasnet(_settings_path())
    if scorer is not None:
        try:
            cnn = float(np.clip(scorer.score(face_bgr), 0.0, 1.0))
            score = 0.65 * cnn + 0.35 * score
            engine = "fused"
            features["cnn_live_prob"] = cnn
            if cnn < 0.5:
                flags = sorted(set(flags) | {"cnn_spoof_probability"})
        except Exception as exc:  # noqa: BLE001 - heuristics are the fallback
            logger.warning("mini-FASNet inference failed, using heuristics: %s", exc)

    # Hysteresis: just under the bar is "inconclusive", not a spoof accusation.
    # Accusing a real user of holding a photo is worse than a missed alert.
    if score < threshold:
        verdict: Verdict = "spoof_suspect"
    elif score < threshold + 0.12:
        verdict = "inconclusive"
    else:
        verdict = "alive"

    return PassiveVerdict(
        score=score, verdict=verdict, flags=flags, features=features, sub_scores=sub, engine=engine
    )


def _settings_path() -> str | None:
    # app.services.liveness.passive -> app.config is three levels up.
    from ...config import settings

    return settings.fasnet_path
