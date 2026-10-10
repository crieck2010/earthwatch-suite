"""Tests for flow.styles: presets and validation."""

import pytest

from flow.styles import STYLES, Style, get_style, list_styles


def test_get_style_known():
    s = get_style("reel-dark")
    assert isinstance(s, Style)
    assert s.scalar_cmap == "cividis"
    assert 0.0 < s.trail_alpha_tail < s.trail_alpha_head <= 1.0


def test_get_style_unknown():
    with pytest.raises(KeyError):
        get_style("nope")


def test_list_styles_sorted_and_complete():
    styles = list_styles()
    names = [s.name for s in styles]
    assert names == sorted(names)
    assert {"reel-dark", "reel-light", "print"} <= set(names)


def test_styles_are_frozen():
    s = get_style("reel-dark")
    with pytest.raises(Exception):
        s.name = "mutated"


def test_all_styles_have_sane_fields():
    for s in list_styles():
        assert s.particle_linewidth > 0
        assert s.name and s.description
