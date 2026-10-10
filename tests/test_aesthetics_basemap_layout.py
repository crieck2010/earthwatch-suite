"""Tests for basemap styles and the layout engine."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from aesthetics import (
    NO_BASEMAP,
    SUBTLE_LAND,
    VOID_BLACK,
    BasemapStyle,
    draw_basemap,
    get_basemap,
    get_preset,
    list_basemaps,
    new_canvas,
)
from aesthetics.backgrounds import close
from aesthetics.layout import (
    FurnitureSpec,
    box_spec,
    fit_font_size,
    place_furniture,
    text_extent_frac,
    text_spec,
)
from aesthetics.typography import SERIF, place_labels


@pytest.fixture()
def segs():
    return [{"lons": [0.1, 0.2, 0.3], "lats": [0.4, 0.45, 0.4]}]


def test_list_and_get_basemap():
    assert list_basemaps() == ["no_basemap", "subtle_land", "void_black"]
    assert get_basemap("void_black") is VOID_BLACK
    with pytest.raises(KeyError):
        get_basemap("topo_world")


def test_void_black_draws_coastlines(segs):
    fig, ax = new_canvas(540, 960)
    n = draw_basemap(ax, segs, VOID_BLACK)
    assert n == 1
    assert len(ax.lines) == 1
    assert ax.lines[0].get_color() == "#3a3f4a"  # v0.1.0 look preserved
    close(fig)


def test_no_basemap_draws_nothing(segs):
    fig, ax = new_canvas(540, 960)
    n = draw_basemap(ax, segs, NO_BASEMAP)
    assert n == 0
    assert len(ax.lines) == 0 and len(ax.images) == 0
    close(fig)


def test_subtle_land_fill_uses_mask(segs):
    fig, ax = new_canvas(200, 200)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True  # land block in the middle
    draw_basemap(ax, segs, SUBTLE_LAND, mask=mask)
    assert len(ax.images) == 1  # the land-fill image
    # Without a mask the fill is skipped, not an error.
    fig2, ax2 = new_canvas(200, 200)
    draw_basemap(ax2, segs, SUBTLE_LAND, mask=None)
    assert len(ax2.images) == 0
    close(fig)
    close(fig2)


def test_presets_carry_basemap_styles():
    assert get_preset("dark_flow").basemap is VOID_BLACK
    assert get_preset("dark_glow").basemap is VOID_BLACK
    assert get_preset("dark_strands").basemap is NO_BASEMAP
    # The v0.1.0 coastline look is unchanged on the old presets.
    assert get_preset("dark_flow").basemap.coastline_color == "#3a3f4a"


def test_basemap_style_is_frozen():
    with pytest.raises(Exception):
        VOID_BLACK.coastline_color = "#ffffff"


# --- layout engine ------------------------------------------------------------


def _rects(items):
    return {it.kind: it.rect for it in items if it.placed}


def test_no_overlap_property():
    specs = [
        FurnitureSpec("a", (0.05, 0.80, 0.30, 0.10), priority=3),
        FurnitureSpec("b", (0.05, 0.80, 0.30, 0.10), priority=2,
                      alternatives=((0.60, 0.80, 0.30, 0.10),)),
        FurnitureSpec("c", (0.05, 0.80, 0.30, 0.10), priority=1,
                      alternatives=((0.60, 0.80, 0.30, 0.10),
                                    (0.05, 0.60, 0.30, 0.10))),
    ]
    placed = place_furniture(specs, pad=0.01)
    rects = [it.rect for it in placed if it.placed]
    assert len(rects) == 3
    for i in range(3):
        for j in range(i + 1, 3):
            ax, ay, aw, ah = rects[i]
            bx, by, bw, bh = rects[j]
            assert not (ax - 0.01 < bx + bw + 0.01 and ax + aw + 0.01 > bx - 0.01
                        and ay - 0.01 < by + bh + 0.01 and ay + ah + 0.01 > by - 0.01)


def test_priority_wins_preferred_slot():
    specs = [
        FurnitureSpec("low", (0.05, 0.80, 0.30, 0.10), priority=1,
                      alternatives=((0.60, 0.60, 0.30, 0.10),)),
        FurnitureSpec("high", (0.05, 0.80, 0.30, 0.10), priority=9),
    ]
    placed = {it.kind: it for it in place_furniture(specs)}
    assert placed["high"].rect == (0.05, 0.80, 0.30, 0.10)
    assert placed["high"].scale == 1.0
    assert placed["low"].rect == (0.60, 0.60, 0.30, 0.10)


def test_optional_dropped_when_no_space():
    specs = [
        FurnitureSpec("req", (0.0, 0.0, 1.0, 1.0), priority=5),
        FurnitureSpec("opt", (0.0, 0.0, 1.0, 1.0), priority=1, optional=True),
    ]
    placed = {it.kind: it for it in place_furniture(specs)}
    assert placed["req"].placed and not placed["opt"].placed


def test_required_falls_back_to_preferred():
    specs = [FurnitureSpec("req", (0.0, 0.0, 1.0, 1.0), priority=5)]
    placed = place_furniture(specs)
    assert placed[0].placed and placed[0].rect == (0.0, 0.0, 1.0, 1.0)


def test_shrink_steps_used_before_dropping():
    # "wall" occupies the left half; the small rect's preferred width
    # overflows the canvas, but shrunk 0.5 it fits right of the wall.
    specs = [
        FurnitureSpec("wall", (0.0, 0.0, 0.50, 1.0), priority=5),
        FurnitureSpec("small", (0.55, 0.05, 0.90, 0.10), priority=1,
                      shrink_steps=(0.5,)),
    ]
    placed = {it.kind: it for it in place_furniture(specs, pad=0.0)}
    assert placed["small"].placed
    assert placed["small"].scale == 0.5
    x, y, w, h = placed["small"].rect
    assert x + w <= 1.0 and x >= 0.50


def test_input_order_preserved_in_output():
    specs = [FurnitureSpec("b", (0.5, 0.5, 0.1, 0.1), priority=1),
             FurnitureSpec("a", (0.1, 0.1, 0.1, 0.1), priority=9)]
    placed = place_furniture(specs)
    assert [it.kind for it in placed] == ["b", "a"]


def test_text_extent_and_shrink_to_fit():
    fig, ax = new_canvas(1080, 1920)
    w, h = text_extent_frac(ax, "Hello", family="DejaVu Sans", size=20)
    assert 0.0 < w < 0.2 and 0.0 < h < 0.05
    w2, _ = text_extent_frac(ax, "Hello", family="DejaVu Sans", size=40)
    assert w2 > 1.8 * w  # roughly linear in size
    long_title = "A Very Long Title That Would Overflow The Canvas Width"
    size = fit_font_size(ax, long_title, family=SERIF, weight="bold",
                         start_size=64, max_width_frac=0.88)
    assert size < 64
    w3, _ = text_extent_frac(ax, long_title, family=SERIF, size=size,
                             weight="bold")
    assert w3 <= 0.88 + 1e-6
    # Short titles keep full size.
    assert fit_font_size(ax, "Ok", family=SERIF, weight="bold",
                         start_size=64, max_width_frac=0.88) == 64
    close(fig)


def test_text_spec_measures_and_shrinks():
    fig, ax = new_canvas(1080, 1920)
    spec = text_spec(ax, "title", "Lake Ontario", 0.06, 0.94,
                     family=SERIF, size=64, weight="bold", priority=5)
    assert spec.kind == "title" and spec.priority == 5
    x, y, w, h = spec.rect
    assert x == 0.06 and w > 0 and h > 0
    assert spec.shrink_steps == (0.92, 0.85, 0.78, 0.70)
    close(fig)


def test_box_spec():
    s = box_spec("date_dial", (0.80, 0.70, 0.16, 0.09), priority=2,
                 optional=True)
    assert s.kind == "date_dial" and s.optional and s.priority == 2


def test_place_labels_avoids_obstacles():
    fig, ax = new_canvas(1080, 1920)
    labels = [{"x": 0.70, "y": 0.75, "text": "Lagos", "priority": 5}]
    # Obstacle covering the label's preferred right-hand side.
    out = place_labels(ax, labels, obstacles=[(0.71, 0.70, 0.20, 0.10)])
    assert out[0]["placed"]
    lx, ly = out[0]["lx"], out[0]["ly"]
    # Label text must be outside the obstacle rect.
    assert not (0.71 < lx < 0.91 and 0.70 < ly < 0.80)
    # Without obstacles the old call pattern still works.
    out2 = place_labels(ax, labels)
    assert out2[0]["placed"]
    close(fig)
