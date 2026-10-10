"""3D prism extrusion on the warm-paper style.

Renders a 2D value grid as extruded columns — **height encodes the value**,
nothing else — in an isometric projection over the paper background: the
mapped.earth "three years of rain" look.

The projection is done manually (no mplot3d): each cell becomes three
shaded polygons (top + two visible sides) drawn back-to-front with the
painter's algorithm. This keeps rendering fully deterministic, fast, and
free of mplot3d's quirks. NaN cells are skipped so honest data gaps show
paper, never invented columns.

Deterministic: no RNG; polygon order is fixed by grid position.
"""

from typing import Tuple, Union

import matplotlib
import numpy as np
from matplotlib.patches import Polygon

from aesthetics.backgrounds import PAPER, close, fig_to_rgba, new_canvas

matplotlib.use("Agg")

ColormapLike = Union[str, matplotlib.colors.Colormap]

_C30 = float(np.cos(np.deg2rad(30.0)))
_S30 = float(np.sin(np.deg2rad(30.0)))


def _as_cmap(cmap: ColormapLike) -> matplotlib.colors.Colormap:
    if isinstance(cmap, str):
        try:
            return matplotlib.colormaps[cmap]
        except KeyError as exc:
            raise ValueError(f"unknown colormap {cmap!r}") from exc
    return cmap


def _shade(rgb: np.ndarray, factor: float) -> Tuple[float, float, float]:
    return tuple(float(np.clip(c * factor, 0.0, 1.0)) for c in rgb[:3])


def prism_frame(
    values,
    *,
    vmin: float,
    vmax: float,
    cmap: ColormapLike = "Blues",
    paper: Tuple[float, float, float] = PAPER,
    width_px: int = 1080,
    height_px: int = 1920,
    max_height_frac: float = 0.32,
    top_alpha: float = 0.92,
    footprint_frac: float = 0.94,
) -> np.ndarray:
    """Render ``values`` as an isometric prism field on paper.

    Parameters
    ----------
    values : 2D array (Ny, Nx). NaN cells are skipped (paper shows through).
    vmin, vmax : float
        Fixed value range. Height and color both normalize to this range —
        use reel-wide fixed values to avoid flicker.
    cmap : colormap for the top faces (default ``"Blues"``).
    paper : RGB background tuple.
    width_px, height_px : output size.
    max_height_frac : fraction of ``height_px`` used for the tallest column.
    top_alpha : opacity of the top faces.
    footprint_frac : fraction of ``width_px`` used by the base footprint.

    Returns
    -------
    (height_px, width_px, 4) float64 RGBA in [0, 1].
    """
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim != 2:
        raise ValueError("values must be a 2D array")
    if not vmax > vmin:
        raise ValueError("vmax must be greater than vmin")
    ny, nx = vals.shape
    if ny < 1 or nx < 1:
        raise ValueError("values must be non-empty")

    cmap_obj = _as_cmap(cmap)
    hnorm = np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0)

    # Isometric footprint: projected half-width of the (nx+ny) diagonal.
    cell = (width_px * footprint_frac) / ((nx + ny) * _C30)
    max_h = height_px * max_height_frac

    def proj(ix: float, iy: float, z: float) -> Tuple[float, float]:
        return ((ix - iy) * _C30 * cell, (ix + iy) * _S30 * cell - z * max_h)

    # Project the four base corners to size the view.
    corners = [proj(0, 0, 0), proj(nx, 0, 0), proj(0, ny, 0), proj(nx, ny, 0),
               proj(0, 0, 1), proj(nx, ny, 1)]
    px = [c[0] for c in corners]
    py = [c[1] for c in corners]
    x0, x1 = min(px), max(px)
    y0, y1 = min(py), max(py)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    # Map projected coords into axes fraction, centered with margin.
    margin = 0.04
    span_x = (x1 - x0) or 1.0
    span_y = (y1 - y0) or 1.0
    scale = min((1.0 - 2 * margin) / span_x, (1.0 - 2 * margin) / span_y)

    def to_axes(px_: float, py_: float) -> Tuple[float, float]:
        # Projected +py points down-screen; flip so columns rise upward.
        return (0.5 + (px_ - cx) * scale, 0.5 - (py_ - cy) * scale)

    fig, ax = new_canvas(width_px, height_px, background=paper)

    # Painter's algorithm: cells with smaller (ix+iy) sit higher on screen
    # (behind); draw them first. Per-cell zorder keeps each cell's sides
    # behind its own top while preserving back-to-front cell order.
    order = sorted(
        ((ix, iy) for iy in range(ny) for ix in range(nx)),
        key=lambda c: c[0] + c[1],
    )
    for depth, (ix, iy) in enumerate(order):
        z = hnorm[iy, ix]
        if not np.isfinite(vals[iy, ix]):
            continue  # honest gap: paper shows through
        top_rgb = np.asarray(cmap_obj(z)[:3])
        # Corners at base (z=0) and top (z).
        b00, b10, b11, b01 = (
            to_axes(*proj(ix, iy, 0.0)),
            to_axes(*proj(ix + 1, iy, 0.0)),
            to_axes(*proj(ix + 1, iy + 1, 0.0)),
            to_axes(*proj(ix, iy + 1, 0.0)),
        )
        t00, t10, t11, t01 = (
            to_axes(*proj(ix, iy, z)),
            to_axes(*proj(ix + 1, iy, z)),
            to_axes(*proj(ix + 1, iy + 1, z)),
            to_axes(*proj(ix, iy + 1, z)),
        )
        if z > 1e-9:
            # Two visible sides: +x face (right) and +y face (left).
            ax.add_patch(Polygon(
                [b10, b11, t11, t10], closed=True,
                facecolor=_shade(top_rgb, 0.62), edgecolor="none",
                zorder=depth,
            ))
            ax.add_patch(Polygon(
                [b01, b11, t11, t01], closed=True,
                facecolor=_shade(top_rgb, 0.80), edgecolor="none",
                zorder=depth,
            ))
        ax.add_patch(Polygon(
            [t00, t10, t11, t01], closed=True,
            facecolor=(*top_rgb, top_alpha), edgecolor="none",
            zorder=depth,
        ))

    rgba = fig_to_rgba(fig)
    close(fig)
    return rgba
