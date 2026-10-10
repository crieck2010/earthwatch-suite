"""Pure-numpy zonal statistics over LST rasters.

Rectangle zones take a fast path; arbitrary polygons fall back to a
point-in-polygon mask. NaN pixels (clouds, nodata, outside-AOI) never
count. Mirrors the zonal pattern of the sibling engines.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .temperature import celsius_stats
from .zones import Zone


def _is_rectangle(geometry: dict) -> bool:
    """True when the polygon ring is exactly its bbox corners."""
    try:
        ring = geometry["coordinates"][0]
    except (KeyError, IndexError, TypeError):
        return False
    if len(ring) != 5 or ring[0] != ring[-1]:
        return False
    xs = sorted({round(p[0], 9) for p in ring})
    ys = sorted({round(p[1], 9) for p in ring})
    return len(xs) == 2 and len(ys) == 2


def _point_in_ring(x: float, y: float, ring) -> bool:
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xinters:
                inside = not inside
    return inside


def zone_mask(
    zone: Zone,
    shape: Tuple[int, int],
    transform: Tuple[float, float, float, float, float, float],
):
    """Boolean mask of zone pixels for a raster.

    ``transform`` is an affine tuple (a, b, c, d, e, f) mapping
    pixel (col, row) → map (x, y): x = c + a*col, y = f + e*row.
    """
    import numpy as np

    rows, cols = shape
    a, _b, c, _d, e, f = transform
    if _is_rectangle(zone.geometry):
        x0, y0, x1, y1 = zone.bbox
        # map → pixel (assumes north-up, no rotation: b == d == 0)
        col0 = int((x0 - c) / a)
        col1 = int((x1 - c) / a)
        # e is negative for north-up rasters
        row0 = int((y1 - f) / e)
        row1 = int((y0 - f) / e)
        col0, col1 = max(0, min(col0, col1)), min(cols, max(col0, col1))
        row0, row1 = max(0, min(row0, row1)), min(rows, max(row0, row1))
        mask = np.zeros(shape, dtype=bool)
        mask[row0:row1, col0:col1] = True
        return mask

    ring = zone.geometry["coordinates"][0]
    col_idx, row_idx = np.meshgrid(np.arange(cols), np.arange(rows))
    xs = c + a * col_idx
    ys = f + e * row_idx
    inside = np.zeros(shape, dtype=bool)
    # vectorised ray casting over the ring edges
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        cond = (y1 > ys) != (y2 > ys)
        with np.errstate(divide="ignore", invalid="ignore"):
            xinters = (x2 - x1) * (ys - y1) / (y2 - y1) + x1
        inside ^= cond & (xs < xinters)
    return inside


def zonal_lst_stats(
    lst_c,
    zone: Zone,
    transform: Tuple[float, float, float, float, float, float],
) -> Dict[str, float]:
    """Zonal LST statistics (°C) for one zone on one pass.

    Returns mean/median/std/minimum/maximum/p10/p90 plus valid_pixels and
    total_pixels (total = pixels inside the zone, valid = finite LST).
    """
    import numpy as np

    arr = np.asarray(lst_c, dtype=float)
    mask = zone_mask(zone, arr.shape, transform)
    total = int(mask.sum())
    values = arr[mask]
    mean, median, std, vmin, vmax, p10, p90, valid = celsius_stats(values)
    return {
        "mean": mean,
        "median": median,
        "std": std,
        "minimum": vmin,
        "maximum": vmax,
        "p10": p10,
        "p90": p90,
        "valid_pixels": valid,
        "total_pixels": float(total),
    }
