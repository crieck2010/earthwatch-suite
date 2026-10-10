"""Tests for the coregistration bias diagnostic."""

import numpy as np
import pytest

from elev3d.coreg import coregistration_check
from elev3d.synthetic import TRANSFORM, synthetic_dems, synthetic_zones_geojson
from elev3d.diff import difference
from elev3d.zones import Zone


def _stable_zone():
    gj = synthetic_zones_geojson()
    feat = next(f for f in gj["features"]
                if f["properties"]["kind"] == "stable")
    props = feat["properties"]
    return Zone(id=props["id"], name=props["name"],
                geometry=feat["geometry"], kind=props["kind"])


def test_synthetic_pair_passes():
    dem1, dem2, t = synthetic_dems()
    diff = difference(dem1, dem2, noise_floor=0.0).diff
    rep = coregistration_check(diff, _stable_zone(), t, threshold_m=0.10)
    assert rep.passed
    assert abs(rep.bias_m) < 0.02  # noise is zero-mean


def test_shifted_dem_fails():
    dem1, dem2, t = synthetic_dems()
    diff = difference(dem1, dem2 + 0.5, noise_floor=0.0).diff
    rep = coregistration_check(diff, _stable_zone(), t, threshold_m=0.10)
    assert not rep.passed
    assert rep.bias_m == pytest.approx(0.5, abs=0.05)


def test_median_robust_to_outliers():
    # a few wild pixels inside the stable zone don't fail the check
    dem1, dem2, t = synthetic_dems()
    diff = difference(dem1, dem2, noise_floor=0.0).diff
    diff[20, 40] = 5.0
    diff[21, 41] = -5.0
    rep = coregistration_check(diff, _stable_zone(), t, threshold_m=0.10)
    assert rep.passed


def test_empty_stable_zone_raises():
    zone = Zone(
        id="x", name="x",
        geometry={"type": "Polygon",
                  "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
    )
    with pytest.raises(ValueError, match="no valid pixels"):
        coregistration_check(np.full((4, 4), np.nan), zone, TRANSFORM)


def test_report_dict():
    dem1, dem2, t = synthetic_dems()
    diff = difference(dem1, dem2, noise_floor=0.0).diff
    d = coregistration_check(diff, _stable_zone(), t).to_dict()
    assert set(d) == {"zone_id", "bias_m", "mad_m", "mean_dh", "std_dh",
                      "stable_pixels", "threshold_m", "passed"}
