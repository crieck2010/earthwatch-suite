"""Zonal helpers over burn rasters (pure numpy).

For rectangular zones the raster maps 1:1 onto the zone grid without any
extra library. For true polygon zones, rasterio rasterisation is used when
available; otherwise a clear error tells the caller what to install.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def rasterize_zone_rect(shape: Tuple[int, int], bbox, zone_bbox) -> np.ndarray:
    """Bool raster of a rectangular zone over a grid, no rasterio needed.

    ``shape`` is (rows, cols); ``bbox`` is the grid's (minx, miny, maxx,
    maxy) in the same units as ``zone_bbox``. Pixels whose centre falls
    inside the zone rectangle are True.
    """
    rows, cols = shape
    minx, miny, maxx, maxy = bbox
    zx0, zy0, zx1, zy1 = zone_bbox
    xs = minx + (np.arange(cols) + 0.5) * (maxx - minx) / cols
    ys = maxy - (np.arange(rows) + 0.5) * (maxy - miny) / rows
    in_x = (xs >= zx0) & (xs <= zx1)
    in_y = (ys >= zy0) & (ys <= zy1)
    return np.outer(in_y, in_x)


def rasterize_zone_polygon(shape: Tuple[int, int], transform, geometry) -> np.ndarray:
    """Bool raster of a polygon zone; needs rasterio."""
    try:
        from rasterio import features as _features
    except ImportError as exc:
        raise ImportError(
            "true polygon zones need rasterio "
            "(pip install 'survey-burn[raster]')"
        ) from exc
    rows, cols = shape
    return _features.rasterize(
        [(geometry, 1)],
        out_shape=(rows, cols),
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)


def apply_zone_mask(raster: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
    """Copy of ``raster`` with pixels outside the zone set to NaN."""
    out = np.asarray(raster, dtype=np.float32).copy()
    out[~np.asarray(zone_mask, dtype=bool)] = np.nan
    return out
