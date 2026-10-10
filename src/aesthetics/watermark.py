"""Watermark, copyright, and data-source furniture.

Consistent bottom-of-frame furniture in the mapped.earth manner: the
brand mark bottom-right, the © holder line beneath it, and the data
source bottom-left — all in quiet monospace so they never compete with
the map.

The brand is a parameter, never hardcoded: survey-viz / reel-studio pass
the user's own handle. There is deliberately no default that imitates
any existing creator.
"""

import matplotlib

matplotlib.use("Agg")

from aesthetics.typography import MONO


def draw_watermark(
    ax,
    x: float,
    y: float,
    brand: str,
    *,
    size: float = 24,
    color: str = "white",
    alpha: float = 0.85,
    ha: str = "right",
    zorder: int = 10,
):
    """Brand mark, e.g. ``draw_watermark(ax, 0.94, 0.055, "my-earth-page")``."""
    return ax.text(
        x, y, brand, transform=ax.transAxes, fontsize=size, color=color,
        alpha=alpha, ha=ha, va="top", family=MONO, zorder=zorder,
    )


def draw_copyright(
    ax,
    x: float,
    y: float,
    holder: str,
    year: int,
    *,
    size: float = 15,
    color: str = "white",
    alpha: float = 0.55,
    ha: str = "right",
    zorder: int = 10,
):
    """© line, e.g. ``draw_copyright(ax, 0.94, 0.032, "Charles Rieck", 2026)``."""
    return ax.text(
        x, y, f"\u00a9 {year} {holder}", transform=ax.transAxes,
        fontsize=size, color=color, alpha=alpha, ha=ha, va="top",
        family=MONO, zorder=zorder,
    )


def draw_data_source(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 15,
    color: str = "white",
    alpha: float = 0.55,
    ha: str = "left",
    zorder: int = 10,
):
    """Data provenance, e.g. ``"NOAA GLOFS MODEL (LOOFS)\\n16-23 SEPTEMBER 2026"``.

    Multi-line strings are supported; lines stack upward from ``(x, y)``.
    """
    lines = text.split("\n")
    artists = []
    for i, line in enumerate(reversed(lines)):
        artists.append(ax.text(
            x, y + i * 0.016, line, transform=ax.transAxes, fontsize=size,
            color=color, alpha=alpha, ha=ha, va="top", family=MONO,
            zorder=zorder,
        ))
    return artists


def frame_furniture(
    ax,
    *,
    brand: str,
    holder: str,
    year: int,
    data_source: str,
    encoding: str = None,
    color: str = "white",
    zorder: int = 10,
) -> None:
    """One-call standard furniture: brand + © bottom-right, data source
    bottom-left, optional encoding honesty-line centred above them."""
    draw_data_source(ax, 0.06, 0.018, data_source, color=color, zorder=zorder)
    draw_watermark(ax, 0.94, 0.062, brand, color=color, zorder=zorder)
    draw_copyright(ax, 0.94, 0.030, holder, year, color=color, zorder=zorder)
    if encoding:
        from aesthetics.legends import encoding_statement

        encoding_statement(ax, 0.5, 0.075, encoding, color=color, zorder=zorder)
