"""Tests for water indices (reuse-first, NaN-honest fallback)."""

import numpy as np
import pytest

from coast.indices_water import (
    SUPPORTED_INDICES,
    compute_index,
    describe_indices,
)


def _bands():
    green = np.array([[0.20, 0.30], [0.10, 0.40]])
    nir = np.array([[0.40, 0.10], [0.30, 0.10]])
    swir1 = np.array([[0.10, 0.05], [0.20, 0.05]])
    return {"green": green, "nir": nir, "swir1": swir1}


def test_ndwi_hand_computed():
    bands = _bands()
    out = compute_index("ndwi", bands)
    expected = (bands["green"] - bands["nir"]) / (bands["green"] + bands["nir"])
    np.testing.assert_allclose(out, expected)


def test_mndwi_hand_computed():
    bands = _bands()
    out = compute_index("mndwi", bands)
    expected = (bands["green"] - bands["swir1"]) / (bands["green"] + bands["swir1"])
    np.testing.assert_allclose(out, expected)


def test_nan_propagates():
    bands = _bands()
    bands["green"][0, 0] = np.nan
    out = compute_index("ndwi", bands)
    assert np.isnan(out[0, 0])
    assert np.isfinite(out[0, 1])


def test_zero_denominator_is_nan_not_inf():
    bands = {"green": np.zeros((2, 2)), "nir": np.zeros((2, 2))}
    out = compute_index("ndwi", bands)
    assert np.isnan(out).all()


def test_unknown_index_raises():
    with pytest.raises(ValueError, match="unknown index"):
        compute_index("evi", _bands())


def test_supported_indices_declared():
    assert set(SUPPORTED_INDICES) == {"ndwi", "mndwi"}


def test_describe_indices_has_references():
    info = describe_indices()
    assert "McFeeters" in info["ndwi"]["reference"]
    assert "Xu" in info["mndwi"]["reference"]
