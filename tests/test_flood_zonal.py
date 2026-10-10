import numpy as np
import pytest

from flood.zonal import rasterize_zone_rect, water_stats


def test_water_stats_known_mask():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0:2, 0:2] = True  # 4 water pixels
    stats = water_stats(mask, None, pixel_size_m=10.0)
    assert stats["water_pixels"] == 4
    assert stats["valid_pixels"] == 16
    assert stats["total_pixels"] == 16
    assert stats["water_fraction"] == pytest.approx(0.25)
    assert stats["water_ha"] == pytest.approx(4 * 100 / 10_000)  # 0.04 ha


def test_water_stats_respects_valid():
    mask = np.ones((4, 4), dtype=bool)
    valid = np.ones((4, 4), dtype=bool)
    valid[0, :] = False  # top row invalid (cloud)
    stats = water_stats(mask, valid, pixel_size_m=10.0)
    assert stats["water_pixels"] == 12
    assert stats["valid_pixels"] == 12
    assert stats["water_fraction"] == pytest.approx(1.0)


def test_water_stats_all_invalid_no_crash():
    mask = np.ones((2, 2), dtype=bool)
    stats = water_stats(mask, np.zeros((2, 2), dtype=bool), pixel_size_m=10.0)
    assert stats["water_fraction"] == 0.0
    assert stats["water_pixels"] == 0


def test_water_stats_bad_pixel_size():
    with pytest.raises(ValueError):
        water_stats(np.ones((2, 2), bool), None, pixel_size_m=0)


def test_water_stats_shape_mismatch():
    with pytest.raises(ValueError):
        water_stats(np.ones((2, 2), bool), np.ones((3, 3), bool), 10.0)


def test_rasterize_zone_rect_half_grid():
    # 4x4 grid over (0,0)-(4,4); zone covers the left half.
    rast = rasterize_zone_rect((4, 4), (0, 0, 4, 4), (0, 0, 2, 4))
    assert rast.shape == (4, 4)
    assert rast[:, 0:2].all()
    assert not rast[:, 2:4].any()


def test_rasterize_zone_rect_pixel_centres():
    # 2x2 grid over (0,0)-(2,2); zone exactly the top-left pixel's extent.
    rast = rasterize_zone_rect((2, 2), (0, 0, 2, 2), (0, 1, 1, 2))
    assert rast.tolist() == [[True, False], [False, False]]
