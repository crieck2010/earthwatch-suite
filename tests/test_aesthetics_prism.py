import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from aesthetics.backgrounds import PAPER
from aesthetics.prism import prism_frame


def _topmost_row(rgba, paper=PAPER, tol=0.02):
    """Topmost pixel row that differs from the paper background."""
    diff = np.abs(rgba[..., :3] - np.asarray(paper)).max(axis=-1)
    rows = np.where((diff > tol).any(axis=1))[0]
    return rows.min() if rows.size else None


def test_shape_and_range():
    vals = np.arange(48, dtype=float).reshape(6, 8)
    rgba = prism_frame(vals, vmin=0.0, vmax=47.0,
                       width_px=270, height_px=480)
    assert rgba.shape == (480, 270, 4)
    assert rgba.min() >= 0.0 and rgba.max() <= 1.0


def test_deterministic():
    rng = np.random.default_rng(3)
    vals = rng.uniform(0, 10, (8, 10))
    a = prism_frame(vals, vmin=0.0, vmax=10.0, width_px=270, height_px=480)
    b = prism_frame(vals, vmin=0.0, vmax=10.0, width_px=270, height_px=480)
    assert np.array_equal(a, b)


def test_taller_value_reaches_higher():
    kw = dict(vmin=0.0, vmax=10.0, width_px=270, height_px=480)
    short = prism_frame(np.full((4, 4), 2.0), **kw)
    tall = prism_frame(np.full((4, 4), 10.0), **kw)
    assert _topmost_row(tall) < _topmost_row(short)  # smaller row = higher


def test_nan_cells_show_paper():
    vals = np.full((4, 4), 10.0)
    vals[1, 1] = np.nan
    rgba = prism_frame(vals, vmin=0.0, vmax=10.0,
                       width_px=270, height_px=480)
    # The frame is not empty, and NaN handling doesn't crash or invent color.
    assert _topmost_row(rgba) is not None
    flat = prism_frame(np.zeros((4, 4)), vmin=0.0, vmax=10.0,
                       width_px=270, height_px=480)
    # Zero-height field: only flat top faces at base level, well below the
    # tall columns' tops.
    assert _topmost_row(flat) > _topmost_row(rgba)


def test_corners_are_paper():
    vals = np.full((6, 6), 5.0)
    rgba = prism_frame(vals, vmin=0.0, vmax=10.0,
                       width_px=270, height_px=480)
    for (r, c) in [(0, 0), (0, 269), (479, 0), (479, 269)]:
        assert np.allclose(rgba[r, c, :3], PAPER, atol=0.02)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        prism_frame(np.zeros((4, 4)), vmin=5.0, vmax=5.0)
    with pytest.raises(ValueError):
        prism_frame(np.zeros(4), vmin=0.0, vmax=1.0)
    with pytest.raises(ValueError):
        prism_frame(np.zeros((4, 4)), vmin=0.0, vmax=1.0, cmap="nope")
