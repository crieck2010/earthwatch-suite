"""Tests for vegetation.indices_ext: local indices + registry reuse."""

import math

import numpy as np
import pytest

from vegetation import indices_ext as ix


def test_gndvi_hand_computed():
    nir = np.array([[0.6]], dtype=np.float32)
    green = np.array([[0.2]], dtype=np.float32)
    out = ix.gndvi(nir, green)
    assert out[0, 0] == pytest.approx(0.5)


def test_msavi_hand_computed():
    nir = np.array([[0.5]], dtype=np.float32)
    red = np.array([[0.2]], dtype=np.float32)
    out = ix.msavi(nir, red)
    inner = (2 * 0.5 + 1) ** 2 - 8 * (0.5 - 0.2)
    expected = (2 * 0.5 + 1 - math.sqrt(inner)) / 2
    assert out[0, 0] == pytest.approx(expected)


def test_nan_propagation():
    nir = np.array([[0.6, np.nan]], dtype=np.float32)
    green = np.array([[0.2, 0.2]], dtype=np.float32)
    out = ix.gndvi(nir, green)
    assert out[0, 0] == pytest.approx(0.5)
    assert math.isnan(out[0, 1])
    # zero denominator -> NaN, not inf
    out2 = ix.gndvi(np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32))
    assert math.isnan(out2[0, 0])


def test_compute_local_indices():
    bands = {
        "nir": np.full((2, 2), 0.6, dtype=np.float32),
        "green": np.full((2, 2), 0.2, dtype=np.float32),
        "red": np.full((2, 2), 0.2, dtype=np.float32),
    }
    assert ix.compute("gndvi", bands)[0, 0] == pytest.approx(0.5)
    assert np.isfinite(ix.compute("msavi", bands)).all()


def test_compute_unknown_index():
    with pytest.raises(ValueError, match="unknown vigor index"):
        ix.compute("ndvi-fake", {})


def test_compute_missing_band():
    with pytest.raises(ValueError, match="needs bands"):
        ix.compute("gndvi", {"nir": np.zeros((1, 1))})


def test_compute_shape_mismatch():
    bands = {
        "nir": np.zeros((2, 2), dtype=np.float32),
        "green": np.zeros((3, 3), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="mismatched shapes"):
        ix.compute("gndvi", bands)


def test_required_bands_local():
    assert ix.required_bands("gndvi") == ("nir", "green")
    assert ix.required_bands("msavi") == ("nir", "red")


def test_vigor_index_list():
    assert set(ix.list_vigor_indices()) == {"ndvi", "evi", "savi", "gndvi", "msavi"}


def test_registry_indices_delegate():
    """ndvi/evi/savi come from survey-imagery, not reimplemented here."""
    imagery = pytest.importorskip("imagery")
    bands = {
        "nir": np.full((2, 2), 0.6, dtype=np.float32),
        "red": np.full((2, 2), 0.2, dtype=np.float32),
        "blue": np.full((2, 2), 0.1, dtype=np.float32),
    }
    ours = ix.compute("ndvi", bands)
    theirs = imagery.indices.compute("ndvi", bands)
    assert np.allclose(ours, theirs, equal_nan=True)
    assert ix.required_bands("ndvi") == ("nir", "red")
