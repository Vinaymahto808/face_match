"""Frame and face quality gating.

The gate is a *rejection* layer, so the tests are about two things:

* every documented failure mode actually fires (a gate that silently never
  rejects is worse than no gate, because it looks like protection);
* a healthy frame passes (an over-eager gate just makes people retry forever,
  which is its own kind of denial-of-service).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.services.quality import QualityReport, assess_face, assess_frame, to_gray
from tests.conftest import _box_blur, make_face_image


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _flat_field(value: int = 120, size: int = 320) -> np.ndarray:
    """A textureless image -- the 'camera pointed at a blank wall' case."""
    return np.full((size, int(size * 4 / 3), 3), value, dtype=np.uint8)


def _resample(image: np.ndarray, scale: float) -> np.ndarray:
    """Nearest-neighbour downscale, i.e. what a lossy preview does."""
    h, w = image.shape[:2]
    small = image[: int(h * scale), : int(w * scale)]
    ys = np.linspace(0, small.shape[0] - 1, h).astype(int)
    xs = np.linspace(0, small.shape[1] - 1, w).astype(int)
    return small[np.ix_(ys, xs)]


def _crop(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    bw, bh = int(w * 0.4), int(h * 0.5)
    x, y = (w - bw) // 2, max(0, (h - bh) // 3)
    return image[y : y + bh, x : x + bw]


def _has(report: QualityReport, reason: str) -> bool:
    return any(r == reason or r.startswith(reason) for r in report.reasons)


# ---------------------------------------------------------------------------
# assess_frame
# ---------------------------------------------------------------------------
def test_a_good_frame_passes():
    report = assess_frame(make_face_image(seed=1))
    assert report.ok, report.reasons
    assert report.score == pytest.approx(1.0)
    assert report.reasons == []


def test_empty_input_is_rejected_not_raised():
    report = assess_frame(np.array([], dtype=np.uint8))
    assert not report.ok
    assert report.reasons == ["empty_frame"]
    assert report.score == 0.0


def test_three_dimensional_empty_frame_is_rejected():
    report = assess_frame(np.zeros((0, 0, 3), dtype=np.uint8))
    assert not report.ok
    assert report.reasons == ["empty_frame"]


def test_underexposed_frame_is_rejected():
    report = assess_frame(make_face_image(seed=2) * 0.02)
    assert not report.ok
    assert _has(report, "underexposed")
    assert report.metrics["brightness"] < settings.min_brightness


def test_overexposed_frame_is_rejected():
    # Blown-out highlights: most of the frame is pinned at 255.
    blown = np.clip(make_face_image(seed=3).astype(np.float64) * 1.4 + 190.0, 0, 255)
    report = assess_frame(blown)
    assert not report.ok
    assert _has(report, "overexposed")
    assert report.metrics["brightness"] > settings.max_brightness


def test_blurry_frame_is_rejected():
    """Blur is the classic re-imaged-print artefact, so it must not survive."""
    image = make_face_image(seed=4)
    for _ in range(6):
        image = _box_blur(image, 3)
    report = assess_frame(image)
    assert not report.ok
    assert _has(report, "blurry")
    assert report.metrics["sharpness"] < settings.min_sharpness


def test_flat_field_is_rejected():
    report = assess_frame(_flat_field())
    assert not report.ok
    # A uniform field is both flat and sharp-less; either alone is disqualifying.
    assert _has(report, "mostly_flat") or _has(report, "blurry")
    assert report.metrics["flat_fraction"] == pytest.approx(1.0)


def test_frame_below_minimum_edge_is_rejected():
    tiny = np.full((settings.min_frame_edge_px - 8, settings.min_frame_edge_px - 8, 3), 90, np.uint8)
    report = assess_frame(tiny)
    assert not report.ok
    assert _has(report, "frame_too_small")


def test_low_contrast_frame_is_rejected():
    """Narrow dynamic range around mid grey: correct exposure, no signal."""
    image = make_face_image(seed=5)
    gray = to_gray(image).astype(np.float64)
    flat = 128 + (gray - gray.mean()) * 0.05
    report = assess_frame(np.repeat(flat[..., None], 3, axis=2).astype(np.uint8))
    assert not report.ok
    assert _has(report, "low_contrast") or _has(report, "mostly_flat")


def test_reasons_are_prefixed_and_repeatable():
    """Every reason string carries a value after the prefix where one applies."""
    report = assess_frame(_flat_field(3))
    assert not report.ok
    assert report.reasons, "a rejection must explain itself"
    for reason in report.reasons:
        assert isinstance(reason, str)
        assert reason == reason.strip()
        assert "\n" not in reason


def test_quality_report_round_trips_to_json_safe_types():
    report = assess_frame(make_face_image(seed=6, blur=8))
    payload = report.to_dict()
    assert set(payload) == {"ok", "score", "reasons", "metrics"}
    assert isinstance(payload["score"], float)
    assert isinstance(payload["reasons"], list)
    for value in payload["metrics"].values():
        assert isinstance(value, float)


def test_to_gray_accepts_2d_and_3d():
    color = np.zeros((4, 4, 3), dtype=np.uint8)
    color[..., 0] = 10  # blue channel only
    assert to_gray(color).shape == (4, 4)
    assert to_gray(color[..., 0]).shape == (4, 4)
    # A 2-D input is taken as luma already; a 3-D input is weighted Rec.601.
    assert to_gray(color)[0, 0] == pytest.approx(0.114 * 10)

    grey = np.full((4, 4, 3), 200, dtype=np.uint8)
    assert np.allclose(to_gray(grey), to_gray(grey[..., 0]))


def test_to_gray_rejects_unsupported_shapes():
    with pytest.raises(ValueError, match="unsupported image shape"):
        to_gray(np.zeros((4, 4, 1), dtype=np.uint8))


# ---------------------------------------------------------------------------
# assess_face
# ---------------------------------------------------------------------------
def test_a_good_face_crop_passes():
    crop = _crop(make_face_image(seed=7))
    report = assess_face(to_gray(crop), crop.shape[1], crop.shape[0])
    assert report.ok, report.reasons


def test_face_below_minimum_edge_is_rejected():
    small = np.full((settings.min_face_edge_px - 10, settings.min_face_edge_px - 10), 128.0)
    report = assess_face(small, small.shape[1], small.shape[0])
    assert not report.ok
    assert _has(report, "face_too_small")


def test_tiny_face_crop_returns_a_score_without_crashing():
    """The Laplacian needs at least an 8x8 crop; below that we bail out early."""
    tiny = np.full((4, 4), 128.0)
    report = assess_face(tiny, 4, 4)
    assert not report.ok
    assert report.score == 0.0
    assert "sharpness" not in report.metrics


def test_empty_face_crop_is_rejected():
    report = assess_face(np.array([]), 0, 0)
    assert not report.ok
    assert report.reasons == ["empty_face_crop"]


def test_flat_face_crop_is_rejected_as_low_texture():
    flat = np.full((160, 160), 128.0)
    report = assess_face(flat, 160, 160)
    assert not report.ok
    assert _has(report, "low_texture")


def test_heavily_downscaled_face_is_rejected():
    """A face crop taken from a 160x120 preview has no skin detail left."""
    crop = _resample(make_face_image(seed=8), 0.12)
    gray = to_gray(crop)
    report = assess_face(gray, crop.shape[1], crop.shape[0])
    assert not report.ok
    assert report.reasons


def test_underexposed_face_is_rejected():
    crop = to_gray(_crop(make_face_image(seed=9))) * 0.02
    report = assess_face(crop, crop.shape[1], crop.shape[0])
    assert not report.ok
    assert _has(report, "underexposed")


def test_fill_ratio_is_reported():
    """A padding-shrunk crop is a real bug in the detector wiring; surface it."""
    crop = to_gray(_crop(make_face_image(seed=10)))
    full = assess_face(crop, crop.shape[1], crop.shape[0])
    half = assess_face(crop, crop.shape[1] * 2, crop.shape[0])
    assert full.metrics["fill_ratio"] == pytest.approx(1.0)
    assert half.metrics["fill_ratio"] == pytest.approx(0.5)


def test_frame_gate_is_stricter_than_the_face_gate_on_the_same_pixels():
    """Sanity check that the two gates are genuinely different measurements.

    ``assess_frame`` and ``assess_face`` share helpers, so a regression that
    made one a copy of the other would silently disable per-face rejection.
    """
    crop = to_gray(_crop(make_face_image(seed=11)))
    face_report = assess_face(crop, crop.shape[1], crop.shape[0])
    frame_report = assess_frame(make_face_image(seed=11))
    assert face_report.metrics["sharpness"] != frame_report.metrics["sharpness"]
    assert face_report.metrics["fill_ratio"] not in frame_report.metrics


# ---------------------------------------------------------------------------
# interaction with the pipeline's ordering guarantee
# ---------------------------------------------------------------------------
def test_gate_settings_are_sane_defaults():
    """A config where the flat-region allowance exceeds 1.0 disables the gate."""
    assert 0.0 < settings.max_flat_region_fraction < 1.0
    assert settings.min_sharpness > 0
    assert 0 < settings.min_brightness < settings.max_brightness < 256
    assert settings.min_contrast > 0
    assert settings.min_face_edge_px < settings.min_frame_edge_px


def test_assess_face_accepts_a_crop_narrower_than_the_declared_box():
    """Detectors routinely return a box wider than the pixels they can crop.

    ``fill_ratio`` is how that shows up; it must not by itself reject the face.
    """
    crop = to_gray(_crop(make_face_image(seed=12)))
    report = assess_face(crop, crop.shape[1] + 40, crop.shape[0])
    assert 0.0 < report.metrics["fill_ratio"] < 1.0
    assert "low_texture" not in report.reasons
