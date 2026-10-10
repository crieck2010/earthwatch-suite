"""Pure-numpy zone masks over difference rasters.

Rectangle zones take a fast path; arbitrary polygons fall back to a
point-in-polygon mask. NaN pixels (nodata in either DEM) never count.
Mirrors the zonal pattern of the sibling engines.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

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
) -> np.ndarray:
    """Boolean mask of zone pixels for a raster.

    ``transform`` is an affine tuple (a, b, c, d, e, f) mapping
    pixel (col, row) → map (x, y): x = c + a*col, y = f + e*row.
    """
    rows, cols = shape
    a, _b, c, _d, e, f = transform
    if _is_rectangle(zone.geometry):
        x0, y0, x1, y1 = zone.bbox
        # map → pixel (assumes north-up, no rotation: b == d == 0)
        col0 = int((x0 - c) / a)
        col1 = int((x1 - c) / a)
        row0 = int((y1 - f) / e)  # e < 0 north-up: y1 (north) is the top row
        row1 = int((y0 - f) / e)
        col0, col1 = max(0, min(col0, col1)), min(cols, max(col0, col1))
        row0, row1 = max(0, min(row0, row1)), min(rows, max(row0, row1))
        mask = np.zeros(shape, dtype=bool)
        mask[row0:row1, col0:col1] = True
        return mask

    # Arbitrary polygon: point-in-polygon on pixel centres.
    mask = np.zeros(shape, dtype=bool)
    ring = zone.geometry["coordinates"][0]
    col_idx = np.arange(cols)
    row_idx = np.arange(rows)
    xs = c + a * (col_idx + 0.5)
    ys = f + e * (row_idx + 0.5)
    for r in range(rows):
        y = ys[r]
        # cheap bbox reject per row
        if y < min(p[1] for p in ring) or y > max(p[1] for p in ring):
            continue
        for col in range(cols):
            if _point_in_ring(xs[col], y, ring):
                mask[r, col] = True
    return mask
