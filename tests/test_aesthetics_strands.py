"""Tests for the strand render layer (warming.watch wind-strand look)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from aesthetics.strands import _resample_mask, _split_nan, render_strands


def _trails(n=50, m=12, seed=3):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x0, y0 = rng.random(), rng.random()
        t = np.linspace(0, 1, m)
        out.append(np.column_stack([
            np.clip(x0 + 0.06 * t + 0.008 * rng.standard_normal(m), 0, 1),
            np.clip(y0 + 0.04 * t + 0.008 * rng.standard_normal(m), 0, 1),
        ]))
    return out


def test_deterministic():
    trails = _trails()
    vals = np.linspace(-4, 30, len(trails))
    a = render_strands(trails, vals, vmin=-4, vmax=30)
    b = render_strands(trails, vals, vmin=-4, vmax=30)
    assert np.array_equal(a, b)


def test_output_shape_and_range():
    rgba = render_strands(_trails(10, 6), np.zeros(10), vmin=-4, vmax=30,
                          width_px=540, height_px=960)
    assert rgba.shape == (960, 540, 4)
    assert rgba.dtype == np.float64
    assert rgba.min() >= 0.0 and rgba.max() <= 1.0
    assert np.all(rgba[..., 3] == 1.0)  # opaque composite


def test_color_mapping_cold_vs_hot():
    import matplotlib as mpl
    cmap = mpl.colormaps["turbo"]
    trail = [np.array([[0.4, 0.5], [0.6, 0.5]])]
    cold = render_strands(trail, [-4.0], vmin=-4, vmax=30, head_alpha=0.8)
    hot = render_strands(trail, [30.0], vmin=-4, vmax=30, head_alpha=0.8)
    # head region around x=0.6 -> col 648, y=0.5 -> row 960
    cold_px = cold[950:970, 630:670, :3].max(axis=(0, 1))
    hot_px = hot[950:970, 630:670, :3].max(axis=(0, 1))
    assert np.allclose(cold_px, 0.8 * np.array(cmap(0.0)[:3]), atol=0.08)
    assert np.allclose(hot_px, 0.8 * np.array(cmap(1.0)[:3]), atol=0.08)


def test_head_brighter_than_tail():
    t = np.linspace(0.2, 0.8, 12)
    trail = [np.column_stack([t, np.full(12, 0.5)])]
    rgba = render_strands(trail, [10.0], vmin=-4, vmax=30)
    head = rgba[950:970, 820:860, :3].max()
    tail = rgba[950:970, 200:240, :3].max()
    assert head > 1.5 * tail  # alpha ramp tail -> head


def test_per_point_values():
    t = np.linspace(0.3, 0.7, 8)
    trail = [np.column_stack([t, np.full(8, 0.5)])]
    vals = [np.linspace(-4, 30, 8)]  # cold tail -> hot head
    rgba = render_strands(trail, vals, vmin=-4, vmax=30)
    assert rgba[..., :3].max() > 0.1
    # Head (x~0.68, hot) is redder than the tail (x~0.32, cold).
    head_r = rgba[950:970, 710:750, 0].max()
    tail_r = rgba[950:970, 330:370, 0].max()
    assert head_r > tail_r + 0.1


def test_per_point_length_mismatch_raises():
    trail = [np.array([[0.4, 0.5], [0.6, 0.5], [0.7, 0.5]])]
    with pytest.raises(ValueError, match="per-point"):
        render_strands(trail, [np.array([1.0, 2.0])], vmin=0, vmax=1)


def test_mask_all_false_is_background():
    trails = _trails(40, 8)
    vals = np.linspace(0, 1, 40)
    mask = np.zeros((24, 12), dtype=bool)
    rgba = render_strands(trails, vals, vmin=0, vmax=1, mask=mask)
    assert np.allclose(rgba[..., :3], 0.0)


def test_mask_all_true_keeps_strands():
    trails = _trails(40, 8)
    vals = np.linspace(0, 1, 40)
    mask = np.ones((24, 12), dtype=bool)
    rgba = render_strands(trails, vals, vmin=0, vmax=1, mask=mask)
    assert rgba[..., :3].max() > 0.2


def test_mask_clips_half():
    # Left half masked out: strands only on the right.
    trails = [np.array([[0.1, 0.5], [0.9, 0.5]])]
    mask = np.zeros((10, 10), dtype=bool)
    mask[:, 5:] = True
    rgba = render_strands(trails, [0.5], vmin=0, vmax=1, mask=mask,
                          head_alpha=0.9, tail_alpha=0.9)
    left = rgba[950:970, 100:200, :3].max()
    right = rgba[950:970, 800:900, :3].max()
    assert left < 0.01 and right > 0.2


def test_mask_feather_softens_edge():
    trails = [np.array([[0.1, 0.5], [0.9, 0.5]])]
    mask = np.zeros((10, 10), dtype=bool)
    mask[:, 5:] = True
    kw = dict(vmin=0, vmax=1, mask=mask, head_alpha=0.9, tail_alpha=0.9)
    hard = render_strands(trails, [0.5], **kw)
    soft = render_strands(trails, [0.5], mask_feather=3.0, **kw)
    row = 960
    # Hard edge: dark immediately left of the boundary, bright right of it.
    assert hard[row, 535, 0] < 0.01 and hard[row, 545, 0] > 0.2
    # Feathered: a gradual ramp across the boundary.
    assert 0.01 < soft[row, 535, 0] < soft[row, 540, 0] < soft[row, 545, 0]
    # Deterministic.
    again = render_strands(trails, [0.5], mask_feather=3.0, **kw)
    assert np.array_equal(soft, again)


def test_nan_vertex_splits_trail():
    trail = [np.array([[0.2, 0.5], [0.3, np.nan], [0.5, 0.5], [0.6, 0.5]])]
    rgba = render_strands(trail, [0.5], vmin=0, vmax=1)
    assert rgba[..., :3].max() > 0.1  # the valid sub-trail still draws


def test_nan_value_is_transparent():
    trail = [np.array([[0.4, 0.5], [0.6, 0.5]])]
    rgba = render_strands(trail, [np.nan], vmin=0, vmax=1)
    assert np.allclose(rgba[..., :3], 0.0)


def test_degenerate_and_empty_trails():
    # Single-point and empty inputs draw nothing but don't crash.
    rgba = render_strands(
        [np.array([[0.5, 0.5]]), np.zeros((0, 2))],
        [0.5, 0.5], vmin=0, vmax=1)
    assert np.allclose(rgba[..., :3], 0.0)
    rgba = render_strands([], [], vmin=0, vmax=1)
    assert rgba.shape == (1920, 1080, 4)
    assert np.allclose(rgba[..., :3], 0.0)


def test_bad_inputs_raise():
    with pytest.raises(ValueError, match="vmax > vmin"):
        render_strands(_trails(2, 4), [0.0, 1.0], vmin=1, vmax=1)
    with pytest.raises(ValueError, match="unknown colormap"):
        render_strands(_trails(2, 4), [0.0, 1.0], vmin=0, vmax=1,
                       cmap="not_a_cmap")
    with pytest.raises(ValueError, match="\\(M, 2\\)"):
        render_strands([np.array([0.5, 0.5])], [0.5], vmin=0, vmax=1)
    with pytest.raises(ValueError, match="values must be"):
        render_strands(_trails(3, 4), [0.0, 1.0], vmin=0, vmax=1)


def test_split_nan_unit():
    poly = np.array([[0, 0], [1, np.nan], [2, 2], [3, 3], [4, np.nan]])
    parts = _split_nan(poly)
    assert len(parts) == 1  # only the (2,2)-(3,3) run has >= 2 points
    sub, a0 = parts[0]
    assert a0 == 2 and len(sub) == 2


def test_resample_mask_orientation():
    # Row 0 of the mask is the top (y-fraction 1).
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, :] = True
    px = _resample_mask(mask, 8, 8)
    assert px.shape == (8, 8)
    assert px[:2].all() and not px[2:].any()  # 2 px rows per mask row
