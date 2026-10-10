"""Tests for the engine-registered colormaps (v0.4.0)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from aesthetics.colormaps import get_colormap, list_colormaps


def _hex_rgb(hex_color):
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _luminance(rgba):
    return float(0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2])


def test_teal_pink_is_registered():
    assert "teal_pink" in matplotlib.colormaps
    cmap = get_colormap("teal_pink")
    assert cmap.name == "teal_pink"
    assert list_colormaps() == ["teal_pink"]


def test_teal_pink_endpoints_match_anchors():
    cmap = get_colormap("teal_pink")
    assert np.allclose(cmap(0.0)[:3], _hex_rgb("#07304a"), atol=1e-6)
    assert np.allclose(cmap(1.0)[:3], _hex_rgb("#ef5da8"), atol=1e-6)
    assert np.allclose(cmap(0.65)[:3], _hex_rgb("#f5e7c1"), atol=0.02)


def test_teal_pink_luminance_is_monotonicish_to_the_cream_shoulder():
    cmap = get_colormap("teal_pink")
    lums = [_luminance(cmap(t)) for t in (0.0, 0.25, 0.45, 0.65)]
    # Sequential build: luminance climbs teal -> turquoise -> cream.
    assert lums == sorted(lums)
    # The hot pink top end is far brighter than the deep teal low end,
    # even though the cream shoulder is the luminance peak.
    assert _luminance(cmap(1.0)) > _luminance(cmap(0.0)) + 0.2
    assert _luminance(cmap(0.65)) >= _luminance(cmap(1.0))


def test_get_colormap_passthrough_and_builtins():
    cmap = get_colormap("viridis")
    assert cmap.name == "viridis"
    assert get_colormap(cmap) is cmap


def test_get_colormap_unknown_raises():
    with pytest.raises(ValueError, match="unknown colormap"):
        get_colormap("not_a_cmap")
