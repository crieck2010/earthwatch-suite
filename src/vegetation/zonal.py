"""Zonal statistics on index rasters: pure-numpy core.

The core (:func:`zonal_stats`) works on an index array plus a boolean mask —
no rasterio needed, so the synthetic pipeline and all tests run on stdlib +
numpy only. :func:`masks_for_zones` builds those masks from zone geometries:
with rasterio it rasterises true polygons; without it, zones must be
rectangles (bbox-aligned), which is exactly what the synthetic demo uses.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .zones import Zone


def zonal_stats(
    index_array: np.ndarray, mask: np.ndarray
) -> Dict[str, float]:
    """Statistics of an index raster inside a boolean mask.

    Returns mean, median, std, minimum, maximum, p10, p90, valid_pixels,
    total_pixels. NaN pixels are excluded from every statistic. An empty or
    all-NaN mask yields NaN statistics and valid_pixels = 0.
    """
    arr = np.asarray(index_array, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if m.shape != arr.shape:
        raise ValueError(
            f"mask shape {m.shape} != index array shape {arr.shape}"
        )
    vals = arr[m]
    vals = vals[np.isfinite(vals)]
    total = int(m.sum())
    out: Dict[str, float] = {
        "mean": float("nan"),
        "median": float("nan"),
        "std": float("nan"),
        "minimum": float("nan"),
        "maximum": float("nan"),
        "p10": float("nan"),
        "p90": float("nan"),
        "valid_pixels": 0,
        "total_pixels": total,
    }
    if vals.size == 0:
        return out
    out.update(
        mean=float(np.mean(vals)),
        median=float(np.median(vals)),
        std=float(np.std(vals)),
        minimum=float(np.min(vals)),
        maximum=float(np.max(vals)),
        p10=float(np.percentile(vals, 10)),
        p90=float(np.percentile(vals, 90)),
        valid_pixels=int(vals.size),
    )
    return out


def _rasterio_masks(
    zones: Sequence[Zone],
    shape: Tuple[int, int],
    transform: Sequence[float],
) -> List[np.ndarray]:
    from rasterio.features import geometry_mask

    # One mask per zone (geometry_mask inverts: True where geometry absent).
    return [
        ~geometry_mask(
            [z.geometry], out_shape=shape, transform=transform, invert=True
        )
        for z in zones
    ]


def masks_for_zones(
    zones: Sequence[Zone],
    shape: Tuple[int, int],
    transform: Sequence[float] | None = None,
    grid_bbox: Tuple[float, float, float, float] | None = None,
) -> List[np.ndarray]:
    """Boolean mask per zone over a (rows, cols) grid.

    With rasterio installed (and ``transform`` given), polygons are
    rasterised properly. Without rasterio, zones must be axis-aligned
    rectangles and ``grid_bbox`` (minx, miny, maxx, maxy) maps grid cells to
    coordinates — sufficient for the synthetic pipeline and tests.
    """
    if transform is not None:
        try:
            return _rasterio_masks(zones, shape, transform)
        except ImportError as exc:
            raise ImportError(
                "masks_for_zones with a transform needs 'rasterio' "
                "(pip install 'survey-vegetation[raster]')"
            ) from exc
    if grid_bbox is None:
        raise ValueError(
            "without rasterio/transform, grid_bbox is required and zones "
            "must be rectangles"
        )
    rows, cols = shape
    minx, miny, maxx, maxy = grid_bbox
    # Cell centres in coordinate space (y decreases with row, north-up).
    xs = minx + (np.arange(cols) + 0.5) * (maxx - minx) / cols
    ys = maxy - (np.arange(rows) + 0.5) * (maxy - miny) / rows
    xx, yy = np.meshgrid(xs, ys)
    masks = []
    for zone in zones:
        zx0, zy0, zx1, zy1 = zone.bbox()
        if zone.geometry["type"] != "Polygon":
            raise ValueError(
                f"zone {zone.id!r}: non-rectangle geometry needs rasterio"
            )
        coords = zone.geometry["coordinates"][0]
        corners = {(round(float(p[0]), 9), round(float(p[1]), 9)) for p in coords}
        expected = {(zx0, zy0), (zx0, zy1), (zx1, zy1), (zx1, zy0)}
        if corners != expected:
            raise ValueError(
                f"zone {zone.id!r}: without rasterio, zone polygons must be "
                "axis-aligned rectangles"
            )
        masks.append((xx >= zx0) & (xx <= zx1) & (yy >= zy0) & (yy <= zy1))
    return masks
