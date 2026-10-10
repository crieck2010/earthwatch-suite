from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import numpy as np

from aesthetics.backgrounds import close, fig_to_rgba, new_canvas
from aesthetics.legends import (
    counter,
    date_dial,
    encoding_statement,
    gradient_bar,
    timeline,
    vertical_scale_bar,
)


def _changed(draw_fn, *args, **kwargs):
    fig, ax = new_canvas(400, 600, background="black")
    before = fig_to_rgba(fig).copy()
    draw_fn(ax, *args, **kwargs)
    after = fig_to_rgba(fig)
    close(fig)
    return not np.array_equal(before, after)


def test_gradient_bar_draws_and_ramps():
    assert _changed(gradient_bar, (0.1, 0.3, 0.5, 0.02), "turbo", 51.0, 72.0,
                    label="water temperature", unit="°F")
    # The bar itself ramps: probe inside the bar (rows ~408-420 of 600).
    fig, ax = new_canvas(400, 600, background="black")
    gradient_bar(ax, (0.1, 0.3, 0.5, 0.02), "turbo", 0.0, 1.0)
    rgba = fig_to_rgba(fig)
    left = rgba[414, 60, :3]    # inside the bar, left end
    right = rgba[414, 220, :3]  # inside the bar, right end
    assert not np.allclose(left, right, atol=0.05)
    close(fig)


def test_gradient_bar_deterministic():
    kw = dict(rect=(0.1, 0.3, 0.5, 0.02), cmap="turbo", vmin=0.0, vmax=1.0)
    fig, ax = new_canvas(200, 300, background="black")
    gradient_bar(ax, **kw)
    a = fig_to_rgba(fig)
    close(fig)
    fig, ax = new_canvas(200, 300, background="black")
    gradient_bar(ax, **kw)
    b = fig_to_rgba(fig)
    close(fig)
    assert np.array_equal(a, b)


def test_vertical_scale_bar_draws():
    assert _changed(vertical_scale_bar, 0.1, 0.3, 0.4, 0.0, 50.0,
                    label="rain", unit="mm/day", color="white")


def test_date_dial_marks_month():
    fig, ax = new_canvas(400, 400, background="black")
    date_dial(ax, 0.5, 0.5, 0.2, datetime(2026, 3, 27))
    texts = [t.get_text() for t in ax.texts]
    assert "27 MAR 2026" in texts  # centre readout
    assert sum(1 for t in texts if len(t) == 1) == 12  # month ring
    close(fig)
    assert _changed(date_dial, 0.5, 0.5, 0.2, datetime(2026, 3, 27))


def test_timeline_half_filled():
    fig, ax = new_canvas(400, 400, background="black")
    t0 = datetime(2026, 9, 16)
    t1 = datetime(2026, 9, 23)
    timeline(ax, (0.1, 0.2, 0.6, 0.015), t0, t1, datetime(2026, 9, 19, 12))
    texts = [t.get_text() for t in ax.texts]
    assert "19 SEP 12:00" in texts
    close(fig)


def test_counter_thousands_separators():
    fig, ax = new_canvas(400, 400, background="black")
    counter(ax, 0.1, 0.2, 33705751, label="flashes on land")
    texts = [t.get_text() for t in ax.texts]
    assert "33\u2009705\u2009751" in texts
    assert "FLASHES ON LAND" in texts
    close(fig)


def test_encoding_statement_draws():
    assert _changed(encoding_statement, 0.5, 0.1, "BRIGHTNESS = SPEED")
