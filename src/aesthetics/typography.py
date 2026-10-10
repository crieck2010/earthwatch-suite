"""Editorial typography: serif titles, letterspaced subtitles, mono readouts.

The mapped.earth type system in three voices:

- **Serif display** (DejaVu Serif): the phenomenon title — "Lake Ontario".
- **Letterspaced sans**: the subtitle naming the time window — "A WEEK OF CURRENTS".
- **Monospace** (DejaVu Sans Mono): every data readout — timestamps,
  scales, attributions.

Fonts are the DejaVu families bundled with matplotlib, so rendering is
identical on any machine with the same matplotlib version (no system-font
lottery — this is what keeps survey-cache fingerprints stable).

All coordinates are axes fractions in [0, 1] unless noted.
"""

from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")

SERIF = "DejaVu Serif"
SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

_THIN_SPACE = "\u2009"


def letterspace(text: str) -> str:
    """Insert thin spaces between characters for the subtitle voice."""
    return _THIN_SPACE.join(list(text))


def draw_title(
    ax,
    text: str,
    x: float,
    y: float,
    *,
    size: float = 64,
    color: str = "white",
    alpha: float = 1.0,
    ha: str = "left",
    va: str = "top",
    weight: str = "bold",
    zorder: int = 10,
):
    """Serif display title, e.g. ``draw_title(ax, "Lake Ontario", 0.06, 0.94)``."""
    return ax.text(
        x, y, text, transform=ax.transAxes, fontsize=size, color=color,
        alpha=alpha, ha=ha, va=va, weight=weight, family=SERIF, zorder=zorder,
    )


def draw_subtitle(
    ax,
    text: str,
    x: float,
    y: float,
    *,
    size: float = 26,
    color: str = "white",
    alpha: float = 0.85,
    ha: str = "left",
    va: str = "top",
    zorder: int = 10,
):
    """Letterspaced sans subtitle naming the window, e.g. "A WEEK OF CURRENTS"."""
    return ax.text(
        x, y, letterspace(text.upper()), transform=ax.transAxes, fontsize=size,
        color=color, alpha=alpha, ha=ha, va=va, family=SANS, zorder=zorder,
    )


def draw_readout(
    ax,
    text: str,
    x: float,
    y: float,
    *,
    size: float = 20,
    color: str = "white",
    alpha: float = 0.75,
    ha: str = "left",
    va: str = "top",
    zorder: int = 10,
):
    """Monospace data readout — timestamps, scales, attributions."""
    return ax.text(
        x, y, text, transform=ax.transAxes, fontsize=size, color=color,
        alpha=alpha, ha=ha, va=va, family=MONO, zorder=zorder,
    )


def draw_north_arrow(
    ax,
    x: float,
    y: float,
    angle_deg: float = 0.0,
    *,
    size: float = 0.045,
    color: str = "white",
    alpha: float = 0.8,
    zorder: int = 10,
):
    """Minimal north arrow, rotated honestly when the frame is rotated.

    ``angle_deg`` is the counter-clockwise frame rotation (from
    :func:`aesthetics.framing.optimal_rotation`); the arrow rotates with it.
    """
    import matplotlib.patches as mpatches

    theta = np.deg2rad(90.0 + angle_deg)  # axes-fraction space, y up
    dx, dy = np.cos(theta) * size, np.sin(theta) * size
    ax.annotate(
        "", xy=(x + dx, y + dy), xytext=(x, y), xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=color, alpha=alpha,
                        linewidth=1.6, shrinkA=0, shrinkB=0),
        zorder=zorder,
    )
    ax.text(x, y + dy + 0.012, "N", transform=ax.transAxes, fontsize=size * 320,
            color=color, alpha=alpha, ha="center", va="bottom",
            family=SANS, zorder=zorder)


def place_labels(
    ax,
    labels: Sequence[Dict],
    *,
    fontsize: float = 20,
    color: str = "white",
    dot_color: str = "white",
    dot_size: float = 26,
    alpha: float = 0.85,
    zorder: int = 9,
    obstacles: Sequence[Tuple[float, float, float, float]] = (),
) -> List[Dict]:
    """Place minimal dot-marker place labels with greedy collision avoidance.

    ``labels``: sequence of ``{"x": float, "y": float, "text": str}`` in
    **data** coordinates, optionally ``"priority": int`` (higher wins ties).
    Each label is tried at 8 compass offsets around its point; the first
    offset whose approximated text box overlaps neither a placed label,
    another dot, nor any ``obstacles`` rect wins. Labels that fit nowhere
    are dropped (returned with ``"placed": False``).

    ``obstacles``: sequence of ``(x, y, w, h)`` rects in axes fraction —
    e.g. furniture rects from :func:`aesthetics.layout.place_furniture`
    — that labels must avoid (the date dial used to sit on "Lagos").

    Returns the label dicts with ``"placed"``, ``"lx"``, ``"ly"``
    (axes-fraction label position) and ``"anchor"`` added.
    """
    fig = ax.figure
    dpi = fig.dpi
    fig_w, fig_h = fig.get_size_inches() * dpi

    # Approximate text box in axes fraction (full-bleed axes: axes px = fig px).
    def _rect(lx, ly, text, ha, va):
        w = len(text) * fontsize * dpi / 72.0 * 0.52 / fig_w
        h = fontsize * dpi / 72.0 * 1.25 / fig_h
        x0 = lx - w if ha == "right" else (lx - w / 2 if ha == "center" else lx)
        y0 = ly - h if va == "top" else (ly - h / 2 if va == "center" else ly)
        return (x0, y0, x0 + w, y0 + h)

    def _overlaps(r, rects):
        return any(
            r[0] < q[2] and r[2] > q[0] and r[1] < q[3] and r[3] > q[1]
            for q in rects
        )

    def _hits_obstacle(r):
        # obstacles are (x, y, w, h); convert to (x0, y0, x1, y1).
        return any(
            r[0] < ox + ow and r[2] > ox and r[1] < oy + oh and r[3] > oy
            for ox, oy, ow, oh in obstacles
        )

    ordered = sorted(
        [dict(lb) for lb in labels],
        key=lambda d: d.get("priority", 0),
        reverse=True,
    )
    placed_rects: List[Tuple[float, float, float, float]] = []
    dot_rects: List[Tuple[float, float, float, float]] = []
    out: List[Dict] = []

    # 8 compass offsets in axes fraction, with text anchor per offset.
    r = 0.018
    offsets = [
        (r, 0.0, "left", "center"), (-r, 0.0, "right", "center"),
        (0.0, r, "center", "bottom"), (0.0, -r, "center", "top"),
        (r, r, "left", "bottom"), (-r, r, "right", "bottom"),
        (r, -r, "left", "top"), (-r, -r, "right", "top"),
    ]

    for lb in ordered:
        disp = ax.transData.transform([(lb["x"], lb["y"])])[0]
        px = disp / np.array([fig_w, fig_h])  # axes fraction
        dot_rects.append((px[0] - 0.004, px[1] - 0.004,
                          px[0] + 0.004, px[1] + 0.004))
        chosen = None
        for dx, dy, ha, va in offsets:
            rect = _rect(px[0] + dx, px[1] + dy, lb["text"], ha, va)
            if not (0.0 <= rect[0] and rect[2] <= 1.0
                    and 0.0 <= rect[1] and rect[3] <= 1.0):
                continue
            if _overlaps(rect, placed_rects) or _overlaps(rect, dot_rects):
                continue
            if _hits_obstacle(rect):
                continue
            chosen = (px[0] + dx, px[1] + dy, ha, va, rect)
            break
        if chosen is None:
            out.append({**lb, "placed": False})
            continue
        lx, ly, ha, va, rect = chosen
        placed_rects.append(rect)
        ax.scatter([lb["x"]], [lb["y"]], s=dot_size, c=dot_color,
                   alpha=alpha, zorder=zorder, linewidths=0)
        ax.text(lx, ly, lb["text"], transform=ax.transAxes, fontsize=fontsize,
                color=color, alpha=alpha, ha=ha, va=va, family=SANS,
                zorder=zorder)
        out.append({**lb, "placed": True, "lx": lx, "ly": ly, "anchor": (ha, va)})
    return out
