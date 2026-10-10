"""Tests for ``render_surface``: the density-surface compositing layer."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from aesthetics import render_surface

BG = np.array([3, 11, 14, 255], dtype=np.uint8)     # default background * 255
GREY = np.array([41, 41, 41, 255], dtype=np.uint8)    # default nodata * 255


def _blob(ny=30, nx=40):
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    grid = (np.exp(-(((x - 14) / 7.0) ** 2 + ((y - 15) / 6.0) ** 2)) * 100.0
            + np.exp(-(((x - 30) / 5.0) ** 2 + ((y - 10) / 4.0) ** 2)) * 60.0)
    return grid


def test_output_shape_dtype_and_opaque():
    out = render_surface(_blob(), width_px=80, height_px=60)
    assert out.shape == (60, 80, 4)
    assert out.dtype == np.uint8
    assert np.all(out[..., 3] == 255)  # default background is opaque


def test_determinism_byte_identical():
    kw = dict(width_px=80, height_px=60, smoothing=1.5)
    a = render_surface(_blob(), **kw)
    b = render_surface(_blob(), **kw)
    assert np.array_equal(a, b)


def test_nodata_on_land_is_flat_grey():
    grid = _blob(21, 21)
    grid[8:13, 8:13] = np.nan
    land = np.ones_like(grid, dtype=bool)
    out = render_surface(grid, width_px=21, height_px=21, land_mask=land,
                         contours=False, shadow=False)
    assert np.array_equal(out[10, 10], GREY)


def test_nodata_off_land_shows_background():
    grid = _blob(21, 21)
    grid[8:13, 8:13] = np.nan
    land = np.zeros_like(grid, dtype=bool)
    out = render_surface(grid, width_px=21, height_px=21, land_mask=land,
                         contours=False, shadow=False)
    assert np.array_equal(out[10, 10], BG)


def test_no_data_bleed_under_smoothing():
    # One finite cell in a NaN field on land: under NaN-aware smoothing
    # the neighbours stay no-data (grey), they do not pick up color.
    grid = np.full((21, 21), np.nan)
    grid[10, 10] = 5.0
    land = np.ones_like(grid, dtype=bool)
    out = render_surface(grid, width_px=21, height_px=21, land_mask=land,
                         smoothing=3.0, contours=False, shadow=False)
    assert not np.array_equal(out[10, 10], GREY)  # the finite cell is colored
    for j, i in ((10, 11), (10, 9), (11, 10), (9, 10)):
        assert np.array_equal(out[j, i], GREY)


def test_log_scale_maps_positives_and_drops_nonpositives():
    grid = np.array([[1.0, 10.0, 100.0],
                     [0.0, -5.0, 10.0],
                     [1.0, 10.0, 1000.0]])
    land = np.ones_like(grid, dtype=bool)
    out = render_surface(grid, width_px=3, height_px=3, scale="log",
                         land_mask=land, contours=False, shadow=False)
    # Non-positive cells are no-data -> grey on land.
    assert np.array_equal(out[1, 0], GREY)
    assert np.array_equal(out[1, 1], GREY)
    # Positive cells are colormapped: dark teal low, hot pink high.
    assert not np.array_equal(out[0, 0], GREY)
    assert out[2, 2, 0] > out[0, 0, 0]  # pink core is redder than teal base
    # Without a land mask the dropped cells show the background instead.
    out2 = render_surface(grid, width_px=3, height_px=3, scale="log",
                          contours=False, shadow=False)
    assert np.array_equal(out2[1, 1], BG)


def test_shadow_offsets_below_right_of_land():
    grid = np.full((40, 40), np.nan)
    land = np.zeros_like(grid, dtype=bool)
    land[5:15, 5:15] = True
    light_bg = (0.5, 0.5, 0.5, 1.0)
    kw = dict(width_px=40, height_px=40, land_mask=land,
              background=light_bg, contour_color=(1, 1, 1, 0.35))
    with_shadow = render_surface(grid, shadow=True,
                                 shadow_offset_px=(6, 10), **kw)
    without = render_surface(grid, shadow=False, **kw)
    # Below-right of the land (inside the shifted silhouette) is darker.
    assert with_shadow[22, 17, 0] < without[22, 17, 0]
    # Far above-left is untouched by the shadow.
    assert np.array_equal(with_shadow[1, 1], without[1, 1])


def test_shadow_uses_data_footprint_without_land_mask():
    grid = np.full((40, 40), np.nan)
    grid[5:15, 5:15] = 10.0
    light_bg = (0.5, 0.5, 0.5, 1.0)
    kw = dict(width_px=40, height_px=40, background=light_bg,
              contours=False)
    with_shadow = render_surface(grid, shadow=True,
                                 shadow_offset_px=(6, 10), **kw)
    without = render_surface(grid, shadow=False, **kw)
    assert with_shadow[22, 17, 0] < without[22, 17, 0]


def test_contours_change_pixels_but_not_over_nodata():
    grid = _blob(21, 21)
    grid[8:13, 8:13] = np.nan
    land = np.ones_like(grid, dtype=bool)
    kw = dict(width_px=42, height_px=42, land_mask=land, shadow=False)
    with_c = render_surface(grid, contours=True, **kw)
    without_c = render_surface(grid, contours=False, **kw)
    assert np.any(with_c != without_c)
    # The heart of the hole stays flat grey either way (suppressed ink).
    assert np.array_equal(with_c[21, 21], GREY)
    assert np.array_equal(without_c[21, 21], GREY)


def test_all_nan_is_graceful():
    grid = np.full((12, 12), np.nan)
    land = np.ones_like(grid, dtype=bool)
    out_land = render_surface(grid, width_px=12, height_px=12,
                              land_mask=land, contours=False)
    assert np.array_equal(out_land[6, 6], GREY)
    out_plain = render_surface(grid, width_px=12, height_px=12,
                               contours=False, shadow=False)
    assert np.all(out_plain == BG)


def test_accepts_matplotlib_builtin_cmap():
    out = render_surface(_blob(10, 10), width_px=20, height_px=20,
                         cmap="viridis", contours=False, shadow=False)
    assert out.shape == (20, 20, 4)


def test_bad_inputs_raise():
    grid = _blob(10, 10)
    with pytest.raises(ValueError, match="same shape as grid"):
        render_surface(grid, width_px=10, height_px=10,
                       land_mask=np.zeros((4, 4), dtype=bool))
    with pytest.raises(ValueError, match='scale must be'):
        render_surface(grid, width_px=10, height_px=10, scale="sqrt")
    with pytest.raises(ValueError, match="non-empty"):
        render_surface(np.zeros((0, 4)), width_px=10, height_px=10)
    with pytest.raises(ValueError, match="must be 2D"):
        render_surface(np.zeros(4), width_px=10, height_px=10)
    with pytest.raises(ValueError, match="vmax > vmin"):
        render_surface(grid, width_px=10, height_px=10, vmin=5, vmax=5)
    with pytest.raises(ValueError, match="positive vmin"):
        render_surface(grid, width_px=10, height_px=10,
                       scale="log", vmin=0.0, vmax=10.0)
    with pytest.raises(ValueError, match="unknown colormap"):
        render_surface(grid, width_px=10, height_px=10, cmap="not_a_cmap")
    with pytest.raises(ValueError, match="positive integers"):
        render_surface(grid, width_px=0, height_px=10)
