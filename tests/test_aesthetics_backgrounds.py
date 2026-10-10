import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from aesthetics.backgrounds import (
    BLACK,
    PAPER,
    close,
    draw_coastlines,
    fig_to_rgba,
    new_canvas,
)


def test_canvas_exact_pixels():
    fig, ax = new_canvas(1080, 1920, background="black")
    rgba = fig_to_rgba(fig)
    assert rgba.shape == (1920, 1080, 4)
    assert rgba.dtype == np.float64
    assert rgba.min() >= 0.0 and rgba.max() <= 1.0
    close(fig)


def test_canvas_background_colors():
    fig, ax = new_canvas(200, 100, background="black")
    rgba = fig_to_rgba(fig)
    assert np.allclose(rgba[50, 100, :3], BLACK, atol=0.005)
    close(fig)
    fig, ax = new_canvas(200, 100, background="paper")
    rgba = fig_to_rgba(fig)
    assert np.allclose(rgba[50, 100, :3], PAPER, atol=0.005)
    close(fig)


def test_canvas_has_no_chrome():
    fig, ax = new_canvas(200, 100)
    assert ax.axison is False
    assert ax.get_xlim() == (0.0, 1.0)
    assert ax.get_ylim() == (0.0, 1.0)
    close(fig)


def test_canvas_bad_background_raises():
    with pytest.raises(ValueError):
        new_canvas(100, 100, background="neon")


def test_draw_coastlines_dict_format():
    fig, ax = new_canvas(200, 100, background="black")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 5)
    before = fig_to_rgba(fig).copy()
    segs = [
        {"lons": [-8, -4, 0, 4], "lats": [0, 2, -1, 1]},
        {"lons": [5, 8], "lats": [-3, 3]},
    ]
    n = draw_coastlines(ax, segs, color="white")
    assert n == 2
    after = fig_to_rgba(fig)
    assert not np.array_equal(before, after)  # something was drawn
    close(fig)


def test_draw_coastlines_tuple_format_and_empty():
    fig, ax = new_canvas(200, 100, background="black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n = draw_coastlines(ax, [((0.1, 0.9), (0.5, 0.5))], color="white")
    assert n == 1
    assert draw_coastlines(ax, []) == 0
    close(fig)


def test_fig_to_rgba_deterministic():
    fig, ax = new_canvas(160, 120, background="paper")
    a = fig_to_rgba(fig)
    b = fig_to_rgba(fig)
    assert np.array_equal(a, b)
    close(fig)
