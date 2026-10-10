import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from aesthetics.glow import accumulate_glow, glow_from_grid, light_beam


def test_shape_and_range(glow_points):
    xs, ys, vals = glow_points
    rgba = accumulate_glow(xs, ys, vals, (180, 120), vmax=50.0)
    assert rgba.shape == (180, 120, 4)
    assert rgba.min() >= 0.0 and rgba.max() <= 1.0


def test_deterministic(glow_points):
    xs, ys, vals = glow_points
    a = accumulate_glow(xs, ys, vals, (180, 120), vmax=50.0)
    b = accumulate_glow(xs, ys, vals, (180, 120), vmax=50.0)
    assert np.array_equal(a, b)


def test_empty_points_transparent():
    rgba = accumulate_glow([], [], [], (60, 80), vmax=10.0)
    assert np.all(rgba[..., 3] == 0.0)  # alpha zero everywhere: invisible
    assert np.all(np.isfinite(rgba))


def test_glow_from_grid_matches_binning():
    # A grid fed directly must equal binning one point per pixel.
    rng = np.random.default_rng(4)
    grid = rng.uniform(0, 50, (40, 30))
    mask = grid > 20.0
    ys, xs = np.nonzero(mask)
    a = glow_from_grid(grid * mask, vmax=50.0, sigma=3.0, bloom_gain=0.4)
    b = accumulate_glow(xs, ys, grid[ys, xs], (40, 30), vmax=50.0,
                        sigma=3.0, bloom_gain=0.4)
    assert a.shape == b.shape
    assert np.array_equal(a, b)  # identical code path after binning


def test_glow_from_grid_dense_field_visible():
    grid = np.zeros((100, 100))
    grid[40:60, 40:60] = 800.0  # dense hot block
    rgba = glow_from_grid(grid, vmax=1000.0, sigma=3.0, bloom_gain=0.0)
    assert rgba[50, 50, 3] > 0.5  # core burns bright
    assert rgba[50, 50, :3].max() > 0.5
    assert rgba[0, 0, 3] == 0.0  # empty stays transparent


def test_glow_from_grid_bad_inputs_raise():
    with pytest.raises(ValueError):
        glow_from_grid(np.zeros(10), vmax=1.0)
    with pytest.raises(ValueError):
        glow_from_grid(np.zeros((4, 4)), vmin=2.0, vmax=2.0)


def test_additive_overlap():
    # Energy accumulates linearly: coincident points double the alpha.
    kw = dict(shape=(80, 80), vmin=0.0, vmax=100.0, sigma=2.0,
              bloom_sigma=6.0, bloom_gain=0.0)
    one = accumulate_glow([40.0], [40.0], [5.0], **kw)
    two = accumulate_glow([40.0, 40.0], [40.0, 40.0], [5.0, 5.0], **kw)
    assert np.allclose(two[..., 3], 2.0 * one[..., 3], rtol=1e-6)


def test_bloom_widens_halo():
    kw = dict(shape=(120, 120), vmin=0.0, vmax=10.0, sigma=2.0)
    nobloom = accumulate_glow([60.0], [60.0], [8.0], bloom_gain=0.0, **kw)
    withbloom = accumulate_glow([60.0], [60.0], [8.0], bloom_gain=0.8,
                                bloom_sigma=18.0, **kw)
    # Alpha channel carries the linear energy: bloom reaches far pixels.
    ring = withbloom[60, 85, 3] / max(nobloom[60, 85, 3], 1e-12)
    assert ring > 3.0  # bloom carries energy far from the core


def test_denser_region_burns_brighter():
    rng = np.random.default_rng(0)
    kw = dict(shape=(100, 100), vmin=0.0, vmax=200.0, sigma=3.0,
              bloom_gain=0.0)
    sparse = accumulate_glow(rng.uniform(0, 100, 5), rng.uniform(0, 100, 5),
                             np.ones(5), **kw)
    dense = accumulate_glow(rng.uniform(40, 60, 40), rng.uniform(40, 60, 40),
                            np.ones(40), **kw)
    # Composited brightness (rgb * alpha): dense regions burn brighter.
    bright = lambda r: (r[..., :3] * r[..., 3:4]).sum()
    assert bright(dense) > bright(sparse)


def test_light_beam_direction():
    shape = (120, 120)
    beam = light_beam(shape, 60.0, 60.0, angle_deg=0.0, length_px=40.0,
                      width_px=6.0, intensity=1.0)
    assert beam.shape == (120, 120, 4)
    ahead = beam[60, 90, :3].sum()    # along +x
    side = beam[90, 60, :3].sum()     # perpendicular
    behind = beam[60, 30, :3].sum()   # behind the source
    assert ahead > 10 * side
    assert behind == 0.0


def test_light_beam_fades_with_distance():
    shape = (120, 120)
    beam = light_beam(shape, 10.0, 60.0, angle_deg=0.0, length_px=80.0,
                      width_px=6.0, intensity=1.0)
    near = beam[60, 30, :3].sum()
    far = beam[60, 80, :3].sum()
    assert near > far > 0.0


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        accumulate_glow([1.0], [1.0], [1.0, 2.0], (10, 10), vmax=1.0)
    with pytest.raises(ValueError):
        accumulate_glow([1.0], [1.0], [1.0], (10, 10), vmin=5.0, vmax=5.0)
    with pytest.raises(ValueError):
        light_beam((10, 10), 5, 5, 0.0, -4.0, 3.0)
