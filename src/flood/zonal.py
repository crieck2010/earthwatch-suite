"""Water-area statistics over zones (pure numpy).

For rectangular zones the raster maps 1:1 onto the zone grid without any
extra library. For true polygon zones, rasterio rasterisation is used when
available; otherwise a clear error tells the caller what to install.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

#: Square metres per hectare.
_M2_PER_HA = 10_000.0


def water_stats(
    water: np.ndarray,
    valid: np.ndarray | None,
    pixel_size_m: float,
) -> Dict[str, float]:
    """Area statistics for one water mask.

    ``water`` and ``valid`` are bool arrays; ``valid`` may be None (every
    pixel valid). Returns pixel counts, the water fraction of *valid*
    pixels, and the water area in hectares.
    """
    water = np.asarray(water, dtype=bool)
    if valid is None:
        valid_arr = np.ones(water.shape, dtype=bool)
    else:
        valid_arr = np.asarray(valid, dtype=bool)
        if valid_arr.shape != water.shape:
            raise ValueError("valid mask shape != water mask shape")
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")
    water_pixels = int(np.count_nonzero(water & valid_arr))
    valid_pixels = int(np.count_nonzero(valid_arr))
    total_pixels = int(water.size)
    fraction = water_pixels / valid_pixels if valid_pixels else 0.0
    pixel_ha = (pixel_size_m ** 2) / _M2_PER_HA
    return {
        "water_pixels": water_pixels,
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
        "water_fraction": fraction,
        "water_ha": water_pixels * pixel_ha,
        "pixel_size_m": float(pixel_size_m),
    }


def rasterize_zone_rect(shape, bbox, zone_bbox) -> np.ndarray:
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
    xx, yy = np.meshgrid(xs, ys)
    return (xx >= zx0) & (xx < zx1) & (yy >= zy0) & (yy < zy1)


def rasterize_zone_polygon(geometry: dict, shape, transform) -> np.ndarray:
    """Bool raster of a GeoJSON polygon (needs rasterio)."""
    try:
        from rasterio import features as _features
    except ImportError as exc:
        raise ImportError(
            "true polygon rasterisation needs 'rasterio' "
            "(pip install 'survey-flood[raster]')"
        ) from exc
    from affine import Affine  # rasterio depends on affine

    aff = Affine(*transform)
    burned = _features.rasterize(
        [(geometry, 1)], out_shape=shape, transform=aff, fill=0, dtype="uint8"
    )
    return burned.astype(bool)
