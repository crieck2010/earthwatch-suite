import numpy as np
import pytest

from flood.indices_water import (
    LOCAL_INDICES,
    WATER_INDICES,
    compute,
    mndwi,
    required_bands,
)


def test_water_indices_registry():
    assert WATER_INDICES == ("ndwi", "mndwi")
    assert LOCAL_INDICES == ("mndwi",)


def test_mndwi_hand_computed():
    green = np.array([[0.4, 0.2]], dtype=np.float32)
    swir1 = np.array([[0.1, 0.3]], dtype=np.float32)
    out = mndwi(green, swir1)
    assert out[0, 0] == pytest.approx((0.4 - 0.1) / (0.4 + 0.1))
    assert out[0, 1] == pytest.approx((0.2 - 0.3) / (0.2 + 0.3))


def test_mndwi_zero_denominator_is_nan():
    out = mndwi(np.array([0.0]), np.array([0.0]))
    assert np.isnan(out[0])


def test_mndwi_nan_honest():
    out = mndwi(np.array([np.nan]), np.array([0.2]))
    assert np.isnan(out[0])
    out = mndwi(np.array([0.4]), np.array([np.nan]))
    assert np.isnan(out[0])


def test_mndwi_water_positive_land_negative():
    # Open water: green >> swir1 -> positive; dry land: swir1 >= green.
    out = mndwi(np.array([0.35]), np.array([0.05]))
    assert out[0] > 0.5
    out = mndwi(np.array([0.15]), np.array([0.30]))
    assert out[0] < 0


def test_required_bands_mndwi():
    assert required_bands("mndwi") == ("green", "swir1")


def test_compute_mndwi_dispatch():
    bands = {"green": np.array([0.4]), "swir1": np.array([0.1])}
    out = compute("mndwi", bands)
    assert out[0] == pytest.approx(0.6)


def test_ndwi_delegates_to_imagery_or_raises_helpfully():
    imagery = pytest.importorskip("imagery")
    bands = {"green": np.array([0.4]), "nir": np.array([0.1])}
    out = compute("ndwi", bands)
    expected = imagery.indices.compute("ndwi", bands)
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_required_bands_ndwi_needs_imagery():
    imagery = pytest.importorskip("imagery")
    assert "green" in required_bands("ndwi") and "nir" in required_bands("ndwi")
