"""Tests for the bivariate brightness channel in ``render_strands`` (v0.3.0).

HUE = temperature (values/vmin/vmax), BRIGHTNESS = speed — the mapped.earth
bivariate encoding. No live network; all inputs are synthetic.
"""

import io

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from PIL import Image

from aesthetics.strands import render_strands


def _trails(n=40, m=10, seed=7):
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


def _lum(rgba):
    rgb = rgba[..., :3]
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _png_bytes(rgba):
    buf = io.BytesIO()
    Image.fromarray(np.round(rgba * 255.0).astype(np.uint8)).save(
        buf, format="PNG")
    return buf.getvalue()


KW = dict(vmin=-4, vmax=30, width_px=540, height_px=960)


def test_brightness_zeros_near_background():
    # Brightness 0 must render near-invisible even where the alpha ramp
    # is high (luminance AND alpha scale, not just alpha).
    trails = _trails()
    vals = np.linspace(-4, 30, len(trails))
    rgba = render_strands(trails, vals, brightness=np.zeros(len(trails)),
                          **KW)
    assert rgba[..., :3].max() < 1e-3
    assert np.all(rgba[..., :3] >= 0.0)


def test_brightness_scalar_zero_is_background():
    trails = _trails()
    vals = np.linspace(-4, 30, len(trails))
    rgba = render_strands(trails, vals, brightness=0.0, **KW)
    assert rgba[..., :3].max() < 1e-3


def test_brightness_ones_identical_to_none():
    # The all-ones run must be byte-identical to the legacy path.
    trails = _trails()
    vals = np.linspace(-4, 30, len(trails))
    plain = render_strands(trails, vals, **KW)
    ones = render_strands(trails, vals,
                          brightness=np.ones(len(trails)), **KW)
    assert np.array_equal(plain, ones)
    # Scalar 1.0 takes the same code path.
    scalar = render_strands(trails, vals, brightness=1.0, **KW)
    assert np.array_equal(plain, scalar)


def test_brightness_nan_trails_match_removal():
    # NaN brightness -> those trails contribute nothing: identical to
    # rendering without them at all.
    trails = _trails()
    n = len(trails)
    vals = np.linspace(-4, 30, n)
    drop = (3, 17)
    keep = [i for i in range(n) if i not in drop]
    b = np.linspace(0.2, 1.0, n)
    b[list(drop)] = np.nan
    full = render_strands(trails, vals, brightness=b, **KW)
    sub = render_strands([trails[i] for i in keep], vals[keep],
                         brightness=b[keep], **KW)
    assert np.array_equal(full, sub)


def test_brightness_nan_per_vertex_drops_adjacent_segments():
    # Single horizontal trail, one NaN vertex mid-trail: the two adjacent
    # segments vanish, the rest of the trail still glows.
    t = np.linspace(0.2, 0.8, 12)
    trail = [np.column_stack([t, np.full(12, 0.5)])]
    b = np.ones((1, 12))
    b[0, 6] = np.nan
    rgba = render_strands(trail, [10.0], brightness=b, vmin=-4, vmax=30)
    lum = _lum(rgba)
    gap = lum[900:1020, 540:600].max()   # x ~ 0.50-0.56, around vertex 6
    head = lum[900:1020, 820:900].max()  # x ~ 0.76-0.83, the head
    assert gap < 0.05
    assert head > 0.1


def test_brightness_deterministic_png_bytes():
    # Same inputs -> byte-identical PNG bytes, twice.
    trails = _trails()
    n, m = len(trails), len(trails[0])
    vals = np.linspace(-4, 30, n)
    grad = np.tile(np.linspace(0.1, 1.0, m), (n, 1))  # (n, max_M)
    a = render_strands(trails, vals, brightness=grad, **KW)
    b = render_strands(trails, vals, brightness=grad, **KW)
    assert _png_bytes(a) == _png_bytes(b)


def _half_maxes(rgba):
    # Tight bands on the strand line (y = 0.5 -> row 960 at 1920 px):
    # head at x = 0.8, tail at x = 0.2.
    head = rgba[950:970, 840:890, :3].max()
    tail = rgba[950:970, 190:240, :3].max()
    return head, tail


def test_per_vertex_gradient_head_brighter_than_tail():
    # (1, max_M) per-vertex brightness ramping 0 -> 1 along the trail:
    # the head glows, the tail vanishes beyond the existing alpha fade.
    t = np.linspace(0.2, 0.8, 12)
    trail = [np.column_stack([t, np.full(12, 0.5)])]
    grad = np.linspace(0.0, 1.0, 12)[None, :]
    rgba = render_strands(trail, [10.0], brightness=grad,
                          vmin=-4, vmax=30)
    head, tail = _half_maxes(rgba)
    assert head > 0.3
    assert tail < 0.01
    assert head > 10 * tail  # glow ramp, not just the alpha fade


def test_gradient_adds_contrast_beyond_alpha_fade():
    # The brightness gradient must steepen the head/tail falloff beyond
    # what the alpha ramp alone produces.
    t = np.linspace(0.2, 0.8, 12)
    trail = [np.column_stack([t, np.full(12, 0.5)])]
    kw = dict(vmin=-4, vmax=30)
    flat = render_strands(trail, [10.0], **kw)
    grad = render_strands(trail, [10.0],
                          brightness=np.linspace(0.0, 1.0, 12)[None, :],
                          **kw)
    head_f, tail_f = _half_maxes(flat)
    head_g, tail_g = _half_maxes(grad)
    # The gradient crushes the tail far past the alpha ramp's own fade...
    assert tail_g < tail_f - 0.05
    # ...while the bright head is essentially untouched.
    assert head_g > 0.85 * head_f


def test_brightness_per_vertex_list_of_arrays():
    # Sequence of (M,) arrays mirrors the per-point values convention.
    trails = _trails()
    n = len(trails)
    vals = np.linspace(-4, 30, n)
    per_vertex = [np.linspace(0.2, 1.0, len(tr)) for tr in trails]
    rgba = render_strands(trails, vals, brightness=per_vertex, **KW)
    assert rgba[..., :3].max() > 0.1
    # Same content as the equivalent (n, max_M) array.
    stacked = np.stack([np.linspace(0.2, 1.0, len(trails[0]))
                        for _ in range(n)])
    twin = render_strands(trails, vals, brightness=stacked, **KW)
    assert np.array_equal(rgba, twin)


def test_brightness_clips_out_of_range():
    # Values outside [0, 1] clip, they do not raise.
    trails = _trails()
    n = len(trails)
    vals = np.linspace(-4, 30, n)
    hi = render_strands(trails, vals, brightness=np.full(n, 2.0), **KW)
    ones = render_strands(trails, vals, brightness=np.ones(n), **KW)
    assert np.array_equal(hi, ones)
    lo = render_strands(trails, vals, brightness=np.full(n, -1.0), **KW)
    assert lo[..., :3].max() < 1e-3


def test_brightness_invalid_shapes_raise():
    trails = _trails()  # (40, 10)
    vals = np.linspace(-4, 30, 40)
    with pytest.raises(ValueError, match="ndim"):
        render_strands(trails, vals,
                       brightness=np.zeros((40, 10, 2)), **KW)
    with pytest.raises(ValueError, match=r"\(n,\)"):
        render_strands(trails, vals, brightness=np.zeros(39), **KW)
    with pytest.raises(ValueError, match=r"\(n, max_M\)"):
        render_strands(trails, vals, brightness=np.zeros((40, 9)), **KW)
    # Genuinely ragged input falls back to the per-trail-array path and
    # reports the offending trail.
    with pytest.raises(ValueError, match=r"brightness\[0\]"):
        render_strands(trails, vals,
                       brightness=[np.zeros(5)] + [np.zeros(10)] * 39,
                       **KW)
    # A rectangular list of the wrong length takes the (n, max_M) path.
    with pytest.raises(ValueError, match=r"\(n, max_M\)"):
        render_strands(trails, vals,
                       brightness=[np.zeros(10)] * 39, **KW)
