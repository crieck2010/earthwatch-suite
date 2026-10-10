"""Frame rendering: scalar colormap + advected particle streaks + decorations.

All matplotlib imports are lazy (Agg backend) so the rest of the engine —
fields, advection, manifests — imports and tests without it. Raises an
informative ImportError when matplotlib is missing.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import numpy as np

from flow.advect import ParticleSet
from flow.fields import VectorField
from flow.styles import Style, get_style

# Named output sizes: (width_px, height_px).
SIZES = {
    "reel": (1080, 1920),    # vertical 9:16 social reel
    "wide": (1920, 1080),    # 16:9 landscape
    "square": (1080, 1080),  # 1:1 feed post
}


def _require_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize, to_rgba
    except ImportError as exc:
        raise ImportError(
            "rendering needs matplotlib (pip install 'survey-flow[render]')"
        ) from exc
    return plt, LineCollection, Normalize, to_rgba


def fit_aspect(bounds: Tuple[float, float, float, float],
               size_px: Tuple[int, int]) -> Tuple[float, float, float, float]:
    """Crop ``bounds`` (minx, miny, maxx, maxy) to the target aspect ratio.

    Distances are measured in kilometres (longitude scaled by cos(lat))
    so the map is not stretched. Returns the cropped view bounds.
    """
    minx, miny, maxx, maxy = bounds
    w_px, h_px = size_px
    mean_lat = math.radians((miny + maxy) / 2.0)
    kx = 111.32 * math.cos(mean_lat)   # km per degree longitude
    ky = 110.57                        # km per degree latitude
    data_aspect = ((maxx - minx) * kx) / max((maxy - miny) * ky, 1e-9)
    target_aspect = w_px / h_px
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    if data_aspect > target_aspect:
        # Too wide: crop longitude.
        half_w = (maxy - miny) * (ky / kx) * target_aspect / 2.0
        return (cx - half_w, miny, cx + half_w, maxy)
    # Too tall: crop latitude.
    half_h = (maxx - minx) * (kx / ky) / target_aspect / 2.0
    return (minx, cy - half_h, maxx, cy + half_h)


def _pretty_timestamp(ts: str) -> str:
    """'2026-09-16T01:20:00+00:00' -> '16 Sep 2026 01:20 UTC' (best effort)."""
    import datetime as _dt
    try:
        d = _dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return str(ts)
    suffix = " UTC" if (d.tzinfo is None or d.utcoffset() == _dt.timedelta(0)) else ""
    return d.strftime("%d %b %Y %H:%M") + suffix


def _nice_scalebar_km(width_km: float) -> float:
    """Largest 1/2/5-multiple of a power of ten within ~1/6 of the width."""
    """Largest 1/2/5-multiple of a power of ten within ~1/6 of the width."""
    target = width_km / 6.0
    if target <= 0:
        return 1.0
    mag = 10.0 ** math.floor(math.log10(target))
    for mult in (10, 5, 2, 1):
        if mult * mag <= target:
            return mult * mag
    return mag


def render_frame(
    field: VectorField,
    particles: Optional[ParticleSet],
    style: Style | str,
    out_path: str,
    *,
    timestamp: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    size: str | Tuple[int, int] = "reel",
    dpi: int = 100,
    title: Optional[str] = None,
) -> str:
    """Render one map frame to ``out_path`` (PNG); return the path.

    ``particles`` may be None for a static scalar snapshot (the form a
    survey-qgis Processing algorithm would produce).
    """
    plt, LineCollection, Normalize, to_rgba = _require_mpl()
    style = get_style(style) if isinstance(style, str) else style
    if isinstance(size, str) and size not in SIZES:
        raise ValueError(f"unknown size {size!r}; want one of {sorted(SIZES)}")
    size_px = SIZES[size] if isinstance(size, str) else tuple(size)
    w_px, h_px = (int(size_px[0]), int(size_px[1]))

    if field.scalar is None:
        raise ValueError("field has no scalar grid to render")
    lo, hi = field.scalar_range()
    if style.scalar_vmin != "auto":
        lo = float(style.scalar_vmin)
    if style.scalar_vmax != "auto":
        hi = float(style.scalar_vmax)
    if vmin is not None:
        lo = float(vmin)
    if vmax is not None:
        vmax_ = float(vmax)
        hi = vmax_
    if not hi > lo:
        hi = lo + 1.0

    view = fit_aspect(field.bounds, (w_px, h_px))
    minx, miny, maxx, maxy = view

    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    fig.patch.set_facecolor(style.background)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_axis_off()

    cmap = plt.get_cmap(style.scalar_cmap).copy()
    cmap.set_bad(style.background)
    norm = Normalize(vmin=lo, vmax=hi)
    # imshow (bilinear) gives a smooth field with no cell-edge hairlines.
    lons, lats = field.lons, field.lats
    dlon = float(lons[1] - lons[0]) if lons.size > 1 else 1.0
    dlat = float(lats[1] - lats[0]) if lats.size > 1 else 1.0
    extent = (lons[0] - dlon / 2.0, lons[-1] + dlon / 2.0,
              lats[0] - dlat / 2.0, lats[-1] + dlat / 2.0)
    ax.imshow(field.scalar, extent=extent, origin="lower",
              cmap=cmap, norm=norm, interpolation="bilinear",
              aspect="auto", zorder=1)
    land_rgba = np.zeros(field.shape + (4,))
    land_rgba[field.landmask] = to_rgba(style.land_color)
    ax.imshow(land_rgba, extent=extent, origin="lower",
              interpolation="nearest", aspect="auto", zorder=2)

    if particles is not None and particles.n > 0:
        segs, valid = particles.trail_segments()          # (n, T-1, 2, 2)
        n_seg = segs.shape[1]
        base = to_rgba(style.particle_color)
        alphas = np.linspace(style.trail_alpha_tail, style.trail_alpha_head,
                             n_seg)
        flat_segs = segs.reshape(-1, 2, 2)
        flat_valid = valid.reshape(-1)
        flat_segs = flat_segs[flat_valid]
        colors = np.tile(base, (flat_valid.sum(), 1))
        colors[:, 3] = np.tile(alphas, valid.shape[0])[flat_valid]
        if len(flat_segs):
            lc = LineCollection(flat_segs, colors=colors,
                                linewidths=style.particle_linewidth,
                                zorder=3, capstyle="round")
            ax.add_collection(lc)
        if style.head_dot:
            pos = particles.positions
            ok = np.isfinite(pos).all(axis=1)
            ax.scatter(pos[ok, 0], pos[ok, 1], s=style.head_size,
                       c=[style.particle_color], zorder=4, linewidths=0)

    # Decorations in axes fraction coordinates (overlay axes).
    deco = fig.add_axes([0, 0, 1, 1])
    deco.set_axis_off()
    deco.set_xlim(0, 1)
    deco.set_ylim(0, 1)
    tc = style.text_color
    if timestamp is None:
        timestamp = field.timestamp
    if style.timestamp and timestamp:
        label = style.timestamp_format.format(
            timestamp=_pretty_timestamp(timestamp))
        deco.text(0.04, 0.965, label, transform=deco.transAxes,
                  color=tc, fontsize=13, va="top", ha="left",
                  bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none",
                            alpha=0.35))
    if title or style.title:
        deco.text(0.5, 0.965, title or style.title, transform=deco.transAxes,
                  color=tc, fontsize=15, va="top", ha="center", weight="bold",
                  bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none",
                            alpha=0.35))
    if style.scalebar:
        mean_lat = math.radians((miny + maxy) / 2.0)
        width_km = (maxx - minx) * 111.32 * math.cos(mean_lat)
        bar_km = _nice_scalebar_km(width_km)
        bar_deg = bar_km / (111.32 * math.cos(mean_lat))
        x0 = minx + (maxx - minx) * 0.04
        y0 = miny + (maxy - miny) * 0.045
        ax.plot([x0, x0 + bar_deg], [y0, y0], color=tc, lw=3,
                solid_capstyle="butt", zorder=5)
        ax.text(x0 + bar_deg / 2.0, y0 + (maxy - miny) * 0.012,
                f"{bar_km:g} km", color=tc, fontsize=11, ha="center",
                va="bottom", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none",
                          alpha=0.35))
    if style.colorbar:
        cax = fig.add_axes([0.30, 0.035, 0.40, 0.022])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(style.scalar_label, color=tc, fontsize=10)
        cb.ax.xaxis.set_tick_params(colors=tc, labelsize=9)
        for spine in cb.ax.spines.values():
            spine.set_edgecolor(tc)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path
