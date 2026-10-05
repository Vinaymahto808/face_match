"""Passive (single-frame) anti-spoofing heuristics.

The assertion that matters is *relative*: a live-capture-like frame must score
higher than its own re-captured analogue. Pinning absolute numbers would only
brittle the test suite against a legitimate retune.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.liveness.passive import (
    _BAND_RAMPS,
    _RAMPS,
    _WEIGHTS,
    extract_features,
    passive_liveness,
    score_features,
)
from tests.conftest import make_face_image, make_spoof_image

CROP = (150, 110, 340, 300)  # x, y, w, h over the synthetic 640x480 frame


def _crop(image: np.ndarray) -> np.ndarray:
    x, y, w, h = CROP
    return image[y : y + h, x : x + w]


def test_all_features_are_computed():
    features = extract_features(_crop(make_face_image(seed=1)))
    assert set(features) == set(_WEIGHTS)
    for name, value in features.items():
        assert np.isfinite(value), f"{name} is not finite"
        assert value >= 0.0


def test_weights_sum_to_one():
    """Otherwise the score silently rescales and the threshold drifts."""
    assert sum(_WEIGHTS.values()) == pytest.approx(1.0)


def test_every_weighted_feature_has_a_mapping():
    """A weight with no mapping is a silently dead cue."""
    mapped = set(_RAMPS) | set(_BAND_RAMPS)
    assert set(_WEIGHTS) == mapped
    assert mapped <= set(extract_features(_crop(make_face_image(seed=11))))


def test_ramps_are_well_formed():
    for key, (lo, hi) in _RAMPS.items():
        assert lo < hi, f"ramp for {key} is not increasing"
        assert key in _WEIGHTS
    for key, bounds in _BAND_RAMPS.items():
        too_low, good_low, good_high, too_high = bounds
        assert too_low < good_low < good_high < too_high, f"band for {key} is not ordered"
        assert key in _WEIGHTS


def test_band_ramp_peaks_in_the_healthy_range():
    """Too little high-frequency energy and too much are both suspicious."""
    from app.services.liveness.passive import _ramp

    too_low, good_low, good_high, too_high = _BAND_RAMPS["hf_energy_ratio"]
    assert _ramp(good_low, "hf_energy_ratio") == pytest.approx(1.0)
    assert _ramp(good_high, "hf_energy_ratio") == pytest.approx(1.0)
    assert _ramp(too_low, "hf_energy_ratio") == 0.0
    assert _ramp(too_high, "hf_energy_ratio") == 0.0
    assert _ramp((good_low + good_high) / 2, "hf_energy_ratio") == 1.0


def test_score_is_bounded():
    score, sub, flags = score_features(extract_features(_crop(make_face_image(seed=2))))
    assert 0.0 <= score <= 1.0
    assert set(sub) == set(_WEIGHTS)
    assert isinstance(flags, list)


def test_live_scores_higher_than_recaptured_print():
    """The core anti-spoof claim, on a controlled pair."""
    live = extract_features(_crop(make_face_image(seed=3)))
    spoof = extract_features(_crop(make_spoof_image(seed=3)))

    live_score, _, _ = score_features(live)
    spoof_score, _, _ = score_features(spoof)

    assert live_score > spoof_score, (
        f"re-captured print scored {spoof_score:.3f} vs live {live_score:.3f}; "
        f"live={ {k: round(v, 4) for k, v in live.items()} } "
        f"spoof={ {k: round(v, 4) for k, v in spoof.items()} }"
    )


def test_recompression_lowers_texture_and_noise_cues():
    """A print loses the two cues the weights lean on hardest."""
    live = extract_features(_crop(make_face_image(seed=4)))
    spoof = extract_features(_crop(make_spoof_image(seed=4)))
    assert spoof["flat_fraction"] > live["flat_fraction"]
    assert spoof["laplacian_var"] < live["laplacian_var"]
    assert spoof["noise_sigma"] < live["noise_sigma"]


def test_desaturation_lowers_chroma_cues():
    live = extract_features(_crop(make_face_image(seed=5)))
    spoof = extract_features(_crop(make_spoof_image(seed=5)))
    assert spoof["saturation_std"] < live["saturation_std"]
    assert spoof["colorfulness"] < live["colorfulness"]


def test_synthetic_moire_scores_worse_than_clean():
    """A periodic grid models a screen's subpixel structure re-photographed."""
    clean = make_face_image(seed=6)
    x, y, w, h = CROP
    moire = clean.copy()
    patch = moire[y : y + h, x : x + w].astype(np.float64)
    _, gx = np.mgrid[0:h, 0:w]
    grid = 40.0 * np.sin(2 * np.pi * gx / 3.0)[:, :, None]
    moire[y : y + h, x : x + w] = np.clip(patch + grid, 0, 255).astype(np.uint8)

    clean_score, _, _ = score_features(extract_features(_crop(clean)))
    moire_score, _, _ = score_features(extract_features(_crop(moire)))
    assert moire_score < clean_score


def test_flat_synthetic_image_is_flagged():
    dead = np.full((300, 300, 3), 128, dtype=np.uint8)
    verdict = passive_liveness(dead, threshold=0.55)
    assert verdict.score < 0.55
    assert verdict.verdict in ("spoof_suspect", "inconclusive")
    assert verdict.flags, "a dead-flat crop must raise at least one flag"


def test_verdict_thresholds_and_hysteresis():
    live = passive_liveness(_crop(make_face_image(seed=8)), threshold=0.55)
    assert live.verdict == "alive"
    assert live.score > 0.55

    # An impossible threshold must not be reported as confidently alive.
    strict = passive_liveness(_crop(make_spoof_image(seed=8)), threshold=0.97)
    assert strict.verdict == "spoof_suspect"
    assert strict.score < 0.97


def test_tiny_crop_degrades_to_inconclusive_not_a_crash():
    verdict = passive_liveness(np.zeros((8, 8, 3), dtype=np.uint8), threshold=0.55)
    assert verdict.verdict == "inconclusive"
    assert verdict.score == 0.0
    assert verdict.flags


def test_grayscale_input_is_accepted():
    gray = np.linspace(0, 255, 300 * 300, dtype=np.uint8).reshape(300, 300)
    features = extract_features(gray)
    assert set(features) == set(_WEIGHTS)


def test_verdict_to_dict_is_json_safe():
    import json

    payload = passive_liveness(_crop(make_face_image(seed=9)), threshold=0.55).to_dict()
    json.dumps(payload)  # must not raise
    assert set(payload) >= {"score", "verdict", "flags", "features"}
