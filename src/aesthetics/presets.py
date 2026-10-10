"""Preset bundles: the mapped.earth look as one choice.

Four curated presets, each a frozen dataclass capturing background,
colormap, typography colours, legend placement, basemap style, and
furniture layout:

- ``dark_flow`` — LIC current streaks on black (hue = scalar,
  brightness = speed).
- ``dark_glow`` — additive event glow with bloom on black.
- ``paper_prism`` — 3D prism extrusion on warm paper.
- ``dark_strands`` — advected particle strands on black, color = scalar
  (the warming.watch wind-strand look); no visible basemap.

:func:`preset_figure` builds the canvas for a preset; the caller then
draws the data layer and legends with the other modules. Reel-studio will
expose these as one-click style choices; survey-viz will apply them as the
default look for new renders.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")

from aesthetics.backgrounds import BLACK, PAPER, new_canvas
from aesthetics.basemap import NO_BASEMAP, VOID_BLACK, BasemapStyle

_DARK = "dark"
_PAPER = "paper"


@dataclass(frozen=True)
class Preset:
    """A complete aesthetic bundle."""

    name: str
    background: str  # "black" | "paper"
    bg_rgb: Tuple[float, float, float]
    cmap: str
    text_color: str
    muted_color: str
    title_size: float = 64
    subtitle_size: float = 26
    readout_size: float = 20
    encoding: str = ""  # honesty line, e.g. "BRIGHTNESS = SPEED"
    legend_layout: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    basemap: BasemapStyle = VOID_BLACK


DARK_FLOW = Preset(
    name="dark_flow",
    background="black",
    bg_rgb=BLACK,
    cmap="turbo",
    text_color="white",
    muted_color="#9aa0aa",
    encoding="BRIGHTNESS = SPEED",
    legend_layout={
        "title": (0.06, 0.94),
        "subtitle": (0.06, 0.875),
        "gradient_bar": (0.60, 0.30, 0.30, 0.012),
        "timeline": (0.60, 0.36, 0.30, 0.010),
        "north_arrow": (0.06, 0.80),
    },
)
"""LIC streaks on black: hue = scalar (e.g. water temperature), brightness = speed."""

DARK_GLOW = Preset(
    name="dark_glow",
    background="black",
    bg_rgb=BLACK,
    cmap="inferno",
    text_color="white",
    muted_color="#9aa0aa",
    encoding="",
    legend_layout={
        "title": (0.06, 0.94),
        "subtitle": (0.06, 0.862),
        "counter": (0.06, 0.10),
        "date_dial": (0.80, 0.78, 0.075),
    },
)
"""Additive event glow with bloom on black (flashes, epicenters, beams)."""

PAPER_PRISM = Preset(
    name="paper_prism",
    background="paper",
    bg_rgb=PAPER,
    cmap="Blues",
    text_color="black",
    muted_color="#5a5142",
    encoding="Height is millimetres per day. Nothing else is encoded.",
    legend_layout={
        "title": (0.06, 0.96),
        "subtitle": (0.06, 0.895),
        "vertical_scale": (0.06, 0.30, 0.30),
        "date_dial": (0.82, 0.80, 0.07),
    },
)
"""3D prism extrusion on warm paper: height = value."""

DARK_STRANDS = Preset(
    name="dark_strands",
    background="black",
    bg_rgb=BLACK,
    cmap="turbo",
    text_color="white",
    muted_color="#9aa0aa",
    encoding="COLOR = VALUE",
    legend_layout={
        "title": (0.06, 0.94),
        "subtitle": (0.06, 0.875),
        "gradient_bar": (0.06, 0.22, 0.30, 0.012),
        "timeline": (0.06, 0.14, 0.60, 0.010),
    },
    basemap=NO_BASEMAP,
)
"""Advected particle strands on black, color = scalar (warming.watch look).

No visible basemap: the geography emerges from the strand mask, so callers
pass a landmask/oceanmask to ``render_strands`` instead of coastlines."""

_PRESETS: Dict[str, Preset] = {
    p.name: p for p in (DARK_FLOW, DARK_GLOW, PAPER_PRISM, DARK_STRANDS)
}


def list_presets() -> list:
    """Names of all bundled presets."""
    return sorted(_PRESETS)


def get_preset(name: str) -> Preset:
    """Return the preset dataclass for ``name`` (raises KeyError if unknown)."""
    try:
        return _PRESETS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown preset {name!r}; choose from {list_presets()}"
        ) from exc


def preset_figure(
    name: str,
    width_px: int = 1080,
    height_px: int = 1920,
    *,
    dpi: int = 100,
):
    """Build the chrome-free canvas for a preset; returns ``(fig, ax, preset)``."""
    preset = get_preset(name)
    fig, ax = new_canvas(width_px, height_px, background=preset.bg_rgb, dpi=dpi)
    return fig, ax, preset
