"""Basemap styles: curated background + coastline + land treatments.

Replaces the old hardcoded ``#3a3f4a`` hairlines-on-black with a frozen
:class:`BasemapStyle` dataclass carried by each preset and overridable by
the caller. Three house styles:

- ``VOID_BLACK`` — hairline coastlines on the black void (the v0.1.0
  look; ``dark_flow`` / ``dark_glow`` keep it, so their pixels are
  unchanged).
- ``NO_BASEMAP`` — nothing drawn at all; the geography emerges from a
  data mask (the warming.watch strand look; ``dark_strands`` default).
- ``SUBTLE_LAND`` — faint land fill (via an optional landmask) under
  hairline coastlines, for maps that need land readable but quiet.

Land fill needs a 2D boolean landmask (axes-fraction aligned, row 0 =
top) because coastline *polylines* cannot be filled. The mask contract
is shared with :func:`aesthetics.strands.render_strands`; survey-viz
builds it from its underlay/landmask source. Without a mask,
``land_fill`` is silently ignored (never an error — the map still
draws).
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import numpy as np

from aesthetics.backgrounds import draw_coastlines

Segment = Sequence  # survey-viz underlay format (dicts or (lons, lats) tuples)


@dataclass(frozen=True)
class BasemapStyle:
    """Complete basemap treatment for one preset."""

    name: str = "void_black"
    coastlines: bool = True
    coastline_color: str = "#3a3f4a"
    coastline_width: float = 0.8
    coastline_alpha: float = 0.9
    land_fill: Optional[Tuple[float, float, float]] = None
    land_fill_alpha: float = 0.55
    ocean_fill: Optional[Tuple[float, float, float]] = None
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0)


VOID_BLACK = BasemapStyle(
    name="void_black",
    coastlines=True,
    coastline_color="#3a3f4a",
    coastline_width=0.8,
    coastline_alpha=0.9,
    background=(0.0, 0.0, 0.0),
)
"""The v0.1.0 look: hairline coastlines on the black void."""

NO_BASEMAP = BasemapStyle(
    name="no_basemap",
    coastlines=False,
    background=(0.0, 0.0, 0.0),
)
"""Nothing drawn — geography emerges from a data mask (warming.watch)."""

SUBTLE_LAND = BasemapStyle(
    name="subtle_land",
    coastlines=True,
    coastline_color="#4a505c",
    coastline_width=0.8,
    coastline_alpha=0.9,
    land_fill=(0.10, 0.11, 0.13),
    land_fill_alpha=0.85,
    background=(0.0, 0.0, 0.0),
)
"""Faint landmass under hairline coastlines; needs a landmask."""

_BASEMAPS = {s.name: s for s in (VOID_BLACK, NO_BASEMAP, SUBTLE_LAND)}


def list_basemaps() -> list:
    """Names of the bundled basemap styles."""
    return sorted(_BASEMAPS)


def get_basemap(name: str) -> BasemapStyle:
    """Return the basemap style for ``name`` (raises KeyError if unknown)."""
    try:
        return _BASEMAPS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown basemap {name!r}; choose from {list_basemaps()}"
        ) from exc


def draw_basemap(
    ax,
    segments: Sequence[Segment],
    style: BasemapStyle,
    *,
    mask: Optional[np.ndarray] = None,
    zorder: int = 1,
) -> int:
    """Draw the full basemap treatment; returns coastline segments drawn.

    - Sets the axes facecolor to ``style.background``.
    - ``style.ocean_fill``: flat fill over the whole axes.
    - ``style.land_fill`` + ``mask``: fills masked (land) pixels. The
      mask is resampled to the axes' pixel size with nearest neighbor
      and drawn as an RGBA image under everything else. Without a mask
      the fill is skipped.
    - ``style.coastlines``: hairline coastlines via
      :func:`aesthetics.backgrounds.draw_coastlines` with the style's
      color/width/alpha.

    ``mask``: 2D bool array, axes-fraction aligned, row 0 = top,
    ``True`` = land.
    """
    ax.set_facecolor(style.background)
    if style.ocean_fill is not None:
        ax.add_patch(matplotlib.patches.Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes,
            facecolor=(*style.ocean_fill, 1.0), edgecolor="none",
            zorder=zorder))
    if style.land_fill is not None and mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"mask must be 2D, got shape {mask.shape}")
        my, mx = mask.shape
        land = np.zeros((my, mx, 4))
        land[..., :3] = style.land_fill
        land[..., 3] = np.where(mask, style.land_fill_alpha, 0.0)
        ax.imshow(land, extent=[0, 1, 0, 1], origin="upper",
                  transform=ax.transAxes, aspect="auto",
                  interpolation="nearest", zorder=zorder)
    n = 0
    if style.coastlines:
        n = draw_coastlines(
            ax, segments, color=style.coastline_color,
            linewidth=style.coastline_width, alpha=style.coastline_alpha,
            zorder=zorder + 1,
        )
    return n
