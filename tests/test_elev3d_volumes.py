"""Tests for cut/fill volumetrics and uncertainty."""

import math

import numpy as np
import pytest

from elev3d.diff import difference
from elev3d.synthetic import CUT_M3, FILL_M3, TRANSFORM, synthetic_dems
from elev3d.volumes import _confidence, zone_volumes
from elev3d.zones import Zone

SITE = Zone(
    id="site",
    name="site",
    geometry={
        "type": "Polygon",
        "coordinates": [
            [[500000, 4599000], [501000, 4599000], [501000, 4600000],
             [500000, 4600000], [500000, 4599000]]
        ],
    },
)


def _diff(floor=0.08):
    dem1, dem2, t = synthetic_dems()
    return difference(dem1, dem2, noise_floor=floor).diff_thresholded, t


def test_planted_cut_recovered():
    diff, t = _diff()
    v = zone_volumes(diff, SITE, t, noise_floor=0.08)
    assert v.cut_m3 == pytest.approx(CUT_M3, rel=0.03)


def test_planted_fill_recovered():
    diff, t = _diff()
    v = zone_volumes(diff, SITE, t, noise_floor=0.08)
    assert v.fill_m3 == pytest.approx(FILL_M3, rel=0.03)


def test_net_volume_sign():
    diff, t = _diff()
    v = zone_volumes(diff, SITE, t, noise_floor=0.08)
    assert v.net_m3 == pytest.approx(v.fill_m3 - v.cut_m3)
    assert v.net_m3 < 0  # more cut than fill in the synthetic site


def test_hand_computed_volume():
    # 2×2 px at 10 m, uniform -0.5 m → 4 × 100 m² × 0.5 m = 200 m³ cut
    diff = np.full((2, 2), -0.5)
    v = zone_volumes(diff, SITE, TRANSFORM, noise_floor=0.0)
    assert v.cut_m3 == pytest.approx(200.0)
    assert v.fill_m3 == pytest.approx(0.0)
    assert v.max_cut_depth == pytest.approx(-0.5)
    assert v.max_fill_height == pytest.approx(0.0)


def test_noise_floor_excludes_noise():
    rng = np.random.default_rng(0)
    diff = rng.normal(0, 0.02, (20, 20))
    v = zone_volumes(diff, SITE, TRANSFORM, noise_floor=0.08)
    assert v.cut_m3 == pytest.approx(0.0, abs=500.0)  # near zero
    assert v.changed_pixels < 20 * 20 * 0.05


def test_uncertainty_math():
    # σ_V = px_area × √(σ1²+σ2²) × √N
    diff = np.full((10, 10), -1.0)  # N = 100 changed px
    v = zone_volumes(diff, SITE, TRANSFORM, noise_floor=0.0,
                     rmse1=0.03, rmse2=0.04)
    expected = 100.0 * math.sqrt(0.03**2 + 0.04**2) * math.sqrt(100)
    assert v.volume_sigma_m3 == pytest.approx(expected)


def test_uncertainty_none_without_rmse():
    diff = np.full((4, 4), -1.0)
    v = zone_volumes(diff, SITE, TRANSFORM, noise_floor=0.0)
    assert v.volume_sigma_m3 is None
    assert v.confidence == "low"  # honest: unquantified uncertainty


def test_confidence_bands():
    assert _confidence(1000.0, 100.0) == "high"  # SNR 10
    assert _confidence(300.0, 100.0) == "medium"  # SNR 3
    assert _confidence(100.0, 100.0) == "low"  # SNR 1
    assert _confidence(0.0, 100.0) == "low"


def test_negative_rmse_raises():
    diff = np.full((4, 4), -1.0)
    with pytest.raises(ValueError, match="RMSE"):
        zone_volumes(diff, SITE, TRANSFORM, rmse1=-0.1, rmse2=0.1)


def test_nan_pixels_excluded():
    diff = np.full((4, 4), -0.5)
    diff[0, 0] = np.nan
    v = zone_volumes(diff, SITE, TRANSFORM, noise_floor=0.0)
    assert v.valid_pixels == 15
    assert v.cut_m3 == pytest.approx(15 * 100.0 * 0.5)


def test_to_dict_roundtrip():
    diff, t = _diff()
    v = zone_volumes(diff, SITE, t, noise_floor=0.08)
    d = v.to_dict()
    assert d["zone_id"] == "site"
    assert d["cut_m3"] > 0 and d["fill_m3"] > 0
