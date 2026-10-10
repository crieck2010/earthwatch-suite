"""Custom legends — never a default colorbar.

Each legend states exactly what is encoded, in the mapped.earth manner:

- :func:`gradient_bar` — thin horizontal gradient with min/max mono labels.
- :func:`vertical_scale_bar` — vertical scale for prism height.
- :func:`date_dial` — circular month ring with a pointer at the current date.
- :func:`timeline` — scrubber bar with a moving timestamp readout.
- :func:`counter` — big running cumulative total ("33 705 751").
- :func:`encoding_statement` — the honesty line ("BRIGHTNESS = SPEED",
  "Height is millimetres of rain per day. Nothing else is encoded.").

All geometry in axes fractions [0, 1]; all text in DejaVu Sans Mono.
Deterministic.
"""

from datetime import date, datetime
from typing import Tuple, Union

import matplotlib
import numpy as np
from matplotlib.patches import Rectangle

matplotlib.use("Agg")

from aesthetics.typography import MONO

ColormapLike = Union[str, matplotlib.colors.Colormap]


def _as_cmap(cmap: ColormapLike) -> matplotlib.colors.Colormap:
    if isinstance(cmap, str):
        try:
            return matplotlib.colormaps[cmap]
        except KeyError as exc:
            raise ValueError(f"unknown colormap {cmap!r}") from exc
    return cmap


def gradient_bar(
    ax,
    rect: Tuple[float, float, float, float],
    cmap: ColormapLike,
    vmin: float,
    vmax: float,
    *,
    label: str = "",
    unit: str = "",
    label_size: float = 18,
    tick_size: float = 16,
    color: str = "white",
    alpha: float = 0.9,
    zorder: int = 10,
) -> None:
    """Thin horizontal gradient bar with min/max labels.

    ``rect`` = (x, y, width, height) in axes fraction.
    """
    x, y, w, h = rect
    grad = np.linspace(0.0, 1.0, 256).reshape(1, -1)
    rgb = _as_cmap(cmap)(grad)[0, :, :3]
    rgba = np.ones((8, 256, 4))
    rgba[:, :, :3] = np.repeat(rgb[np.newaxis, :, :], 8, axis=0)
    ax.imshow(rgba, extent=(x, x + w, y, y + h), transform=ax.transAxes,
              origin="lower", aspect="auto", zorder=zorder)
    if label:
        ax.text(x + w / 2, y + h + 0.008, label.upper(), transform=ax.transAxes,
                fontsize=label_size, color=color, alpha=alpha, ha="center",
                va="bottom", family=MONO, zorder=zorder)
    ax.text(x, y - 0.006, f"{vmin:g}{unit}", transform=ax.transAxes,
            fontsize=tick_size, color=color, alpha=alpha, ha="left", va="top",
            family=MONO, zorder=zorder)
    ax.text(x + w, y - 0.006, f"{vmax:g}{unit}", transform=ax.transAxes,
            fontsize=tick_size, color=color, alpha=alpha, ha="right", va="top",
            family=MONO, zorder=zorder)


def vertical_scale_bar(
    ax,
    x: float,
    y: float,
    height: float,
    vmin: float,
    vmax: float,
    *,
    label: str = "",
    unit: str = "",
    n_ticks: int = 5,
    bar_width: float = 0.012,
    size: float = 16,
    color: str = "black",
    alpha: float = 0.8,
    zorder: int = 10,
) -> None:
    """Vertical scale bar for prism height (the paper style)."""
    ax.add_patch(Rectangle(
        (x, y), bar_width, height, transform=ax.transAxes,
        facecolor=color, alpha=alpha, edgecolor="none", zorder=zorder,
    ))
    for i in range(n_ticks):
        frac = i / max(n_ticks - 1, 1)
        val = vmin + frac * (vmax - vmin)
        yy = y + frac * height
        ax.plot([x, x + bar_width * 1.6], [yy, yy], transform=ax.transAxes,
                color=color, alpha=alpha, linewidth=1.2, zorder=zorder)
        ax.text(x - 0.008, yy, f"{val:g}", transform=ax.transAxes,
                fontsize=size, color=color, alpha=alpha, ha="right",
                va="center", family=MONO, zorder=zorder)
    if label:
        ax.text(x, y + height + 0.012,
                f"{label}{(' ' + unit) if unit else ''}",
                transform=ax.transAxes, fontsize=size, color=color,
                alpha=alpha, ha="left", va="bottom", family=MONO, zorder=zorder)


def date_dial(
    ax,
    cx: float,
    cy: float,
    radius: float,
    when: Union[date, datetime],
    *,
    color: str = "white",
    alpha: float = 0.9,
    label_size: float = 15,
    date_size: float = 30,
    zorder: int = 10,
) -> None:
    """Circular month ring with a pointer at the current month and a centre
    date readout — the lightning/rain reels' way of showing time.

    ``(cx, cy)`` and ``radius`` are in axes fraction (radius measured in
    y-fraction units); the ring is aspect-corrected so it renders as a
    true circle on any canvas shape.
    """
    from matplotlib.patches import Ellipse

    fig = ax.figure
    fig_w, fig_h = fig.get_size_inches() * fig.dpi
    y_to_x = fig_h / fig_w  # x-fraction per y-fraction for a true circle

    def _xy(ang, r):
        return (cx + np.cos(ang) * r * y_to_x, cy + np.sin(ang) * r)

    months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    ax.add_patch(Ellipse(
        (cx, cy), 2 * radius * y_to_x, 2 * radius, transform=ax.transAxes,
        facecolor="none", edgecolor=color, alpha=alpha * 0.55, linewidth=2.0,
        zorder=zorder,
    ))
    for i, m in enumerate(months):
        ang = np.deg2rad(90.0 - i * 30.0)
        lx, ly = _xy(ang, radius + 0.022)
        ax.text(lx, ly, m, transform=ax.transAxes, fontsize=label_size,
                color=color, alpha=alpha * 0.7, ha="center", va="center",
                family=MONO, zorder=zorder)
    month_idx = when.month - 1
    day_frac = (when.day - 1) / 30.0
    ang = np.deg2rad(90.0 - (month_idx + day_frac) * 30.0)
    px, py = _xy(ang, radius)
    ax.plot([cx, px], [cy, py], transform=ax.transAxes, color=color,
            alpha=alpha, linewidth=3.0, solid_capstyle="round",
            zorder=zorder + 1)
    ax.add_patch(Ellipse(
        (px, py), 0.016 * y_to_x, 0.016, transform=ax.transAxes,
        facecolor=color, alpha=alpha, edgecolor="none", zorder=zorder + 1,
    ))
    label = when.strftime("%d %b %Y").upper() if isinstance(when, (date, datetime)) else str(when)
    ax.text(cx, cy, label, transform=ax.transAxes, fontsize=date_size,
            color=color, alpha=alpha, ha="center", va="center",
            family=MONO, weight="bold", zorder=zorder)


def timeline(
    ax,
    rect: Tuple[float, float, float, float],
    t_start: datetime,
    t_end: datetime,
    t_now: datetime,
    *,
    fmt: str = "%d %b %H:%M",
    color: str = "white",
    track_alpha: float = 0.25,
    size: float = 20,
    zorder: int = 10,
) -> None:
    """Scrubber bar showing where ``t_now`` sits in ``[t_start, t_end]``,
    with a mono timestamp readout above it."""
    x, y, w, h = rect
    total = (t_end - t_start).total_seconds()
    frac = 0.0 if total <= 0 else float(
        np.clip((t_now - t_start).total_seconds() / total, 0.0, 1.0))
    ax.add_patch(Rectangle(
        (x, y), w, h, transform=ax.transAxes, facecolor=color,
        alpha=track_alpha, edgecolor="none", zorder=zorder,
    ))
    ax.add_patch(Rectangle(
        (x, y), w * frac, h, transform=ax.transAxes, facecolor=color,
        alpha=0.95, edgecolor="none", zorder=zorder + 1,
    ))
    ax.text(x, y + h + 0.008, t_now.strftime(fmt).upper(),
            transform=ax.transAxes, fontsize=size, color=color, alpha=0.9,
            ha="left", va="bottom", family=MONO, zorder=zorder)


def counter(
    ax,
    x: float,
    y: float,
    value: float,
    *,
    label: str = "",
    size: float = 56,
    label_size: float = 18,
    color: str = "white",
    alpha: float = 0.95,
    ha: str = "left",
    zorder: int = 10,
) -> None:
    """Big running cumulative total with thin-space thousands separators."""
    text = f"{int(round(value)):,}".replace(",", "\u2009")
    if label:
        ax.text(x, y + 0.045, label.upper(), transform=ax.transAxes,
                fontsize=label_size, color=color, alpha=alpha * 0.7,
                ha=ha, va="bottom", family=MONO, zorder=zorder)
    ax.text(x, y, text, transform=ax.transAxes, fontsize=size, color=color,
            alpha=alpha, ha=ha, va="top", family=MONO, weight="bold",
            zorder=zorder)


def encoding_statement(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 18,
    color: str = "white",
    alpha: float = 0.7,
    ha: str = "center",
    zorder: int = 10,
) -> None:
    """The honesty line: exactly what is encoded, in words.

    E.g. ``"BRIGHTNESS = SPEED"`` or
    ``"Height is millimetres of rain per day. Nothing else is encoded."``
    """
    ax.text(x, y, text, transform=ax.transAxes, fontsize=size, color=color,
            alpha=alpha, ha=ha, va="top", family=MONO, zorder=zorder)
