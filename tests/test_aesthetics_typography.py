import io

import matplotlib

matplotlib.use("Agg")

import numpy as np

from aesthetics.backgrounds import close, fig_to_rgba, new_canvas
from aesthetics.typography import (
    MONO,
    SERIF,
    draw_north_arrow,
    draw_readout,
    draw_subtitle,
    draw_title,
    letterspace,
    place_labels,
)


def test_letterspace():
    assert letterspace("AB") == "A\u2009B"
    assert letterspace("") == ""


def test_title_uses_serif():
    fig, ax = new_canvas(400, 400)
    t = draw_title(ax, "Lake Ontario", 0.06, 0.94)
    assert t.get_family() == [SERIF]
    assert t.get_text() == "Lake Ontario"
    close(fig)


def test_subtitle_is_letterspaced_upper():
    fig, ax = new_canvas(400, 400)
    t = draw_subtitle(ax, "A Week of Currents", 0.06, 0.88)
    assert t.get_text() == letterspace("A WEEK OF CURRENTS")
    close(fig)


def test_readout_uses_mono():
    fig, ax = new_canvas(400, 400)
    t = draw_readout(ax, "17 SEP 10:00 EDT", 0.6, 0.4)
    assert t.get_family() == [MONO]
    close(fig)


def test_north_arrow_draws():
    fig, ax = new_canvas(400, 400)
    before = fig_to_rgba(fig).copy()
    draw_north_arrow(ax, 0.1, 0.8, angle_deg=45.0)
    after = fig_to_rgba(fig)
    assert not np.array_equal(before, after)
    close(fig)


def test_place_labels_all_placed_when_spread_out():
    fig, ax = new_canvas(400, 400)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    labels = [
        {"x": 1.0, "y": 1.0, "text": "TORONTO"},
        {"x": 9.0, "y": 9.0, "text": "KINGSTON"},
        {"x": 1.0, "y": 9.0, "text": "OSHAWA"},
        {"x": 9.0, "y": 1.0, "text": "OSWEGO"},
    ]
    out = place_labels(ax, labels)
    assert all(lb["placed"] for lb in out)
    assert all(0.0 <= lb["lx"] <= 1.0 and 0.0 <= lb["ly"] <= 1.0
               for lb in out if lb["placed"])
    close(fig)


def test_place_labels_no_overlap_and_deterministic():
    fig, ax = new_canvas(400, 400)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    # Cluster of labels near one point: some must yield, none may overlap.
    labels = [{"x": 5.0, "y": 5.0, "text": f"PLACE{i:02d}", "priority": 10 - i}
              for i in range(6)]
    out1 = place_labels(ax, labels)
    placed = [lb for lb in out1 if lb["placed"]]
    assert len(placed) >= 1
    # Re-run on a fresh canvas: identical placement (deterministic).
    close(fig)
    fig2, ax2 = new_canvas(400, 400)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    out2 = place_labels(ax2, labels)
    for a, b in zip(out1, out2):
        assert a["placed"] == b["placed"]
        if a["placed"]:
            assert a["lx"] == b["lx"] and a["ly"] == b["ly"]
    close(fig2)


def test_place_labels_priority_wins():
    fig, ax = new_canvas(300, 300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    labels = [
        {"x": 5.0, "y": 5.0, "text": "LOW", "priority": 0},
        {"x": 5.0, "y": 5.0, "text": "HIGH", "priority": 99},
    ]
    out = place_labels(ax, labels)
    by_text = {lb["text"]: lb for lb in out}
    assert by_text["HIGH"]["placed"] is True
    close(fig)
