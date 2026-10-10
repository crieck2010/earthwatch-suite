"""Tests for burn.severity: USGS band classification."""

import numpy as np
import pytest

from burn.severity import (
    BURNED_CLASS_IDS,
    BURNED_THRESHOLD,
    CLASS_BY_ID,
    SEVERITY_CLASSES,
    burned_mask,
    class_areas,
    classify,
)


def test_band_edges_belong_to_higher_class():
    # Exact USGS break points classify into the *higher* band.
    d = np.array([[-0.10, 0.10, 0.27, 0.44, 0.66]], dtype=np.float32)
    out = classify(d)
    assert list(out[0]) == [0, 1, 2, 3, 4]


def test_regrowth_band():
    d = np.array([[-0.5, -0.11]], dtype=np.float32)
    out = classify(d)
    assert list(out[0]) == [5, 5]


def test_nan_is_nodata_255():
    d = np.array([[np.nan]], dtype=np.float32)
    assert classify(d)[0, 0] == 255


def test_class_ids_stable():
    ids = [cid for cid, *_ in SEVERITY_CLASSES]
    assert ids == [0, 1, 2, 3, 4, 5]
    assert set(BURNED_CLASS_IDS) == {1, 2, 3, 4}
    assert BURNED_THRESHOLD == pytest.approx(0.10)


def test_burned_mask_threshold_and_nan():
    d = np.array([[0.09, 0.10, 0.5, np.nan]], dtype=np.float32)
    m = burned_mask(d)
    assert list(m[0]) == [False, True, True, False]


def test_burned_mask_custom_threshold():
    d = np.array([[0.3]], dtype=np.float32)
    assert not burned_mask(d, threshold=0.5)[0, 0]
    assert burned_mask(d, threshold=0.2)[0, 0]


def test_class_areas_known_fractions():
    # 10x10 grid: 25 high, 25 low, 50 unburned. 10 m pixels -> 0.01 ha/px.
    sev = np.full((10, 10), 0, dtype=np.uint8)
    sev[:5, :5] = 4
    sev[:5, 5:] = 1
    areas = class_areas(sev, None, 10.0)
    assert areas["high_severity_ha"] == pytest.approx(0.25)
    assert areas["low_severity_ha"] == pytest.approx(0.25)
    assert areas["unburned_ha"] == pytest.approx(0.50)
    assert areas["burned_ha"] == pytest.approx(0.50)
    assert areas["burned_fraction"] == pytest.approx(0.5)
    assert areas["valid_pixels"] == 100.0


def test_class_areas_respects_valid_mask():
    sev = np.full((4, 4), 4, dtype=np.uint8)
    valid = np.zeros((4, 4), dtype=bool)
    valid[:2, :] = True
    areas = class_areas(sev, valid, 10.0)
    assert areas["high_severity_ha"] == pytest.approx(0.08)
    assert areas["burned_fraction"] == pytest.approx(1.0)


def test_class_areas_bad_pixel_size():
    with pytest.raises(ValueError):
        class_areas(np.zeros((2, 2), dtype=np.uint8), None, 0.0)


def test_class_areas_shape_mismatch():
    with pytest.raises(ValueError):
        class_areas(np.zeros((2, 2), dtype=np.uint8), np.zeros((3, 3), dtype=bool), 10.0)


def test_class_by_id_names():
    assert CLASS_BY_ID[4][0] == "high severity"
    assert CLASS_BY_ID[5][0] == "enhanced regrowth"
