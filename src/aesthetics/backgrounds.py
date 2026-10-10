"""Curated backgrounds and chrome-free canvases.

Two house backgrounds — pure black void and warm paper — plus a helper that
builds exact-pixel figures with zero matplotlib chrome (no axes, ticks,
spines, or gridlines; the map fills the frame edge to edge).

Coastline drawing consumes the same segment format survey-viz's underlay
module produces: a list of ``{"lons": [...], "lats": [...]}`` dicts (plain
``(lons, lats)`` tuples are also accepted).
"""

from typing import Dict, List, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# House backgrounds -----------------------------------------------------------
BLACK: Tuple[float, float, float] = (0.0, 0.0, 0.0)
"""Pure black void: land disappears, only the phenomenon glows."""

PAPER: Tuple[float, float, float] = (0.925, 0.882, 0.780)
"""Warm paper background for the light/prism style."""

_BACKGROUNDS = {"black": BLACK, "paper": PAPER}

Segment = Union[Dict[str, Sequence[float]], Tuple[Sequence[float], Sequence[float]]]


def new_canvas(
    width_px: int = 1080,
    height_px: int = 1920,
    *,
    background: Union[str, Tuple[float, float, float]] = "black",
    dpi: int = 100,
):
    """Create an exact-pixel, chrome-free figure.

    Returns ``(fig, ax)`` where ``ax`` covers the whole figure, has its
    axis turned off, and uses axes-fraction coordinates in [0, 1].
    Deterministic for a fixed matplotlib version: bundled DejaVu fonts,
    Agg backend, no interactive state.
    """
    if isinstance(background, str):
        try:
            bg = _BACKGROUNDS[background]
        except KeyError as exc:
            raise ValueError(
                f"unknown background {background!r}; choose from "
                f"{sorted(_BACKGROUNDS)} or pass an RGB tuple"
            ) from exc
    else:
        bg = tuple(float(c) for c in background)
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.set_facecolor(bg)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    return fig, ax


def draw_coastlines(
    ax,
    segments: Sequence[Segment],
    *,
    color: str = "#3a3f4a",
    linewidth: float = 0.8,
    alpha: float = 0.9,
    zorder: int = 3,
) -> int:
    """Draw hairline coastlines; returns the number of segments drawn.

    ``segments`` is the survey-viz underlay format: each item is either a
    ``{"lons": [...], "lats": [...]}`` dict or a ``(lons, lats)`` tuple, in
    data coordinates. Coordinates are plotted in the axes' current data
    transform, so set the map extent before calling.
    """
    n = 0
    for seg in segments:
        if isinstance(seg, dict):
            lons, lats = seg["lons"], seg["lats"]
        else:
            lons, lats = seg
        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)
        if lons.size == 0:
            continue
        ax.plot(
            lons,
            lats,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        n += 1
    return n


def fig_to_rgba(fig) -> np.ndarray:
    """Render a figure to an (H, W, 4) float64 RGBA array in [0, 1]."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf.astype(np.float64) / 255.0


def close(fig) -> None:
    """Close a figure and release its memory."""
    plt.close(fig)
