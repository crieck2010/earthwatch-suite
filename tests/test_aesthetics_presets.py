import io

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from aesthetics import (
    DARK_FLOW,
    DARK_GLOW,
    PAPER_PRISM,
    draw_readout,
    draw_subtitle,
    draw_title,
    fig_to_rgba,
    frame_furniture,
    get_preset,
    gradient_bar,
    lic_texture,
    list_presets,
    new_canvas,
    preset_figure,
    timeline,
)
from aesthetics.backgrounds import BLACK, PAPER, close
from aesthetics.presets import Preset


def test_list_presets():
    assert list_presets() == ["dark_flow", "dark_glow", "dark_strands",
                              "paper_prism"]


def test_get_preset_known_and_unknown():
    p = get_preset("dark_flow")
    assert isinstance(p, Preset)
    assert p is DARK_FLOW
    with pytest.raises(KeyError):
        get_preset("neon_nights")


def test_presets_are_frozen():
    with pytest.raises(Exception):
        DARK_FLOW.cmap = "viridis"


def test_dark_flow_encoding_statement():
    assert DARK_FLOW.encoding == "BRIGHTNESS = SPEED"
    assert DARK_GLOW.background == "black"
    assert PAPER_PRISM.background == "paper"
    assert PAPER_PRISM.bg_rgb == PAPER
    assert DARK_FLOW.bg_rgb == BLACK


def test_preset_figure_canvas_matches_preset():
    for name in list_presets():
        fig, ax, preset = preset_figure(name, 270, 480)
        assert preset.name == name
        rgba = fig_to_rgba(fig)
        assert rgba.shape == (480, 270, 4)
        assert np.allclose(rgba[240, 135, :3], preset.bg_rgb, atol=0.005)
        close(fig)


def _build_dark_flow_frame(seed: int) -> bytes:
    """End-to-end composed frame: the exact call pattern survey-viz will use."""
    from datetime import datetime

    rng = np.random.default_rng(0)
    ny, nx = 96, 54
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    u = np.sin(x / 6.0) * 2.0
    v = np.cos(y / 8.0) * 1.5
    temp = 55.0 + 0.15 * x

    fig, ax, preset = preset_figure("dark_flow", 270, 480)
    tex = lic_texture(u, v, temp, vmin=51.0, vmax=72.0, cmap=preset.cmap,
                      seed=seed, speed_max=3.0)
    ax.imshow(tex, extent=[0, 1, 0, 1], origin="upper", transform=ax.transAxes,
              aspect="auto", zorder=2)
    lay = preset.legend_layout
    draw_title(ax, "Lake Ontario", *lay["title"], size=40,
               color=preset.text_color)
    draw_subtitle(ax, "A Week of Currents", *lay["subtitle"], size=15,
                  color=preset.text_color)
    draw_readout(ax, "NOAA GLOFS MODEL", 0.06, 0.83, size=11,
                 color=preset.text_color)
    gradient_bar(ax, lay["gradient_bar"], preset.cmap, 51.0, 72.0,
                 label="water temperature", unit="°F",
                 color=preset.text_color)
    timeline(ax, lay["timeline"], datetime(2026, 9, 16), datetime(2026, 9, 23),
             datetime(2026, 9, 17, 10), color=preset.text_color)
    frame_furniture(ax, brand="my-ocean-page", holder="Charles Rieck",
                    year=2026,
                    data_source="NOAA GLOFS MODEL (LOOFS)\n16-23 SEPTEMBER 2026",
                    encoding=preset.encoding, color=preset.text_color)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    close(fig)
    return buf.getvalue()


def test_composed_frame_byte_deterministic():
    a = _build_dark_flow_frame(seed=7)
    b = _build_dark_flow_frame(seed=7)
    assert a == b
    assert len(a) > 10_000  # a real frame, not an empty canvas


def test_composed_frame_seed_changes_pixels():
    a = _build_dark_flow_frame(seed=7)
    b = _build_dark_flow_frame(seed=8)
    assert a != b
