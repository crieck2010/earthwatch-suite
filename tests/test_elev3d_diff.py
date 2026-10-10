"""Tests for DEM differencing."""

import numpy as np
import pytest

from elev3d.diff import check_grids, difference, pixel_area


def test_difference_basic():
    dem1 = np.array([[100.0, 101.0], [102.0, 103.0]])
    dem2 = np.array([[99.0, 101.5], [102.0, 104.0]])
    res = difference(dem1, dem2)
    np.testing.assert_allclose(res.diff, [[-1.0, 0.5], [0.0, 1.0]])
    assert res.valid_pixels == 4
    assert res.changed_pixels == 4  # noise_floor=0: everything counts


def test_difference_nan_honest():
    dem1 = np.array([[100.0, np.nan], [102.0, 103.0]])
    dem2 = np.array([[99.0, 101.5], [np.nan, 104.0]])
    res = difference(dem1, dem2)
    assert np.isnan(res.diff[0, 1])
    assert np.isnan(res.diff[1, 0])
    assert res.diff[0, 0] == pytest.approx(-1.0)
    assert res.valid_pixels == 2


def test_noise_floor_zeroes_small_changes():
    dem1 = np.zeros((4, 4))
    dem2 = np.full((4, 4), 0.05)
    dem2[0, 0] = 0.5
    res = difference(dem1, dem2, noise_floor=0.1)
    assert res.diff[1, 1] == pytest.approx(0.05)  # raw untouched
    assert res.diff_thresholded[1, 1] == pytest.approx(0.0)
    assert res.diff_thresholded[0, 0] == pytest.approx(0.5)
    assert res.changed_pixels == 1


def test_noise_floor_nan_stays_nan():
    dem = np.full((3, 3), np.nan)
    res = difference(np.zeros((3, 3)), dem, noise_floor=0.1)
    assert np.isnan(res.diff_thresholded).all()


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        difference(np.zeros((2, 2)), np.zeros((3, 3)))


def test_negative_noise_floor_raises():
    with pytest.raises(ValueError, match="noise_floor"):
        difference(np.zeros((2, 2)), np.zeros((2, 2)), noise_floor=-0.1)


def test_pixel_area():
    assert pixel_area((10.0, 0.0, 500000.0, 0.0, -10.0, 4600000.0)) == pytest.approx(100.0)


def test_pixel_area_rejects_rotation():
    with pytest.raises(ValueError, match="rotated"):
        pixel_area((10.0, 1.0, 0.0, 1.0, -10.0, 0.0))


def test_check_grids_ok():
    check_grids((10, 10), (10, 10), (1, 0, 0, 0, -1, 0), (1, 0, 0, 0, -1, 0))


def test_check_grids_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        check_grids((10, 10), (10, 11))


def test_check_grids_transform_mismatch():
    with pytest.raises(ValueError, match="resample"):
        check_grids((10, 10), (10, 10), (1, 0, 0, 0, -1, 0), (2, 0, 0, 0, -2, 0))


def test_summary_stats():
    dem1 = np.zeros((2, 2))
    dem2 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    s = difference(dem1, dem2).summary()
    assert s["mean_dh"] == pytest.approx(0.0)
    assert s["min_dh"] == pytest.approx(-4.0)
    assert s["max_dh"] == pytest.approx(3.0)
    assert s["valid_pixels"] == 4.0
