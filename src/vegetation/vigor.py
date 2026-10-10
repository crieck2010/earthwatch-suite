"""Vigor-class maps: per-pixel classification of a vigor-index raster.

Pixels are binned into vigor bands (water/bare/stressed/moderate/vigorous/
very dense) using configurable thresholds. The class raster is an integer
GeoTIFF that survey-change can diff (transition matrices between classes)
and QGIS can render with the companion .qml from :mod:`vegetation.qml`.
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np

#: Default NDVI vigor bands: (class_id, label, lower_inclusive, upper_exclusive, RGBA).
#: Tuned for NDVI; other indices (EVI/SAVI) share the shape but users should
#: adjust thresholds per crop/index — see docs/VIGOR_CLASSES.md.
DEFAULT_VIGOR_BANDS: Tuple[Tuple[int, str, float, float, Tuple[int, int, int, int]], ...] = (
    (0, "Water / shadow", float("-inf"), 0.0, (12, 44, 132, 255)),
    (1, "Bare soil", 0.0, 0.2, (165, 142, 110, 255)),
    (2, "Stressed", 0.2, 0.4, (222, 214, 120, 255)),
    (3, "Moderate", 0.4, 0.6, (120, 180, 70, 255)),
    (4, "Vigorous", 0.6, 0.8, (34, 120, 40, 255)),
    (5, "Very dense", 0.8, float("inf"), (10, 80, 30, 255)),
)

NODATA_CLASS = 255


def classify(
    index_array: np.ndarray,
    bands: Sequence[Tuple[int, str, float, float, Tuple[int, int, int, int]]] = DEFAULT_VIGOR_BANDS,
) -> np.ndarray:
    """Classify an index raster into vigor classes (uint8, 255 = nodata)."""
    arr = np.asarray(index_array, dtype=np.float64)
    out = np.full(arr.shape, NODATA_CLASS, dtype=np.uint8)
    valid = np.isfinite(arr)
    for class_id, _label, lo, hi, _rgba in bands:
        out[valid & (arr >= lo) & (arr < hi)] = class_id
    return out


def class_histogram(class_array: np.ndarray) -> List[dict]:
    """Pixel counts and fractions per class id."""
    arr = np.asarray(class_array)
    total = int((arr != NODATA_CLASS).sum())
    hist = []
    for class_id, label, _lo, _hi, _rgba in DEFAULT_VIGOR_BANDS:
        n = int((arr == class_id).sum())
        hist.append(
            {
                "class_id": class_id,
                "label": label,
                "pixels": n,
                "fraction": (n / total) if total else 0.0,
            }
        )
    return hist


def write_class_geotiff(
    class_array: np.ndarray,
    path: str,
    transform: Sequence[float] | None = None,
    crs: str | None = None,
) -> str:
    """Write the class raster as a GeoTIFF (needs rasterio + georeferencing)."""
    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError as exc:
        raise ImportError(
            "vigor-map GeoTIFF output needs 'rasterio' "
            "(pip install 'survey-vegetation[raster]')"
        ) from exc
    if transform is None:
        raise ValueError("vigor-map GeoTIFF output needs a transform")
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    rows, cols = np.asarray(class_array).shape
    affine = Affine(*transform)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=affine,
        nodata=NODATA_CLASS,
        compress="deflate",
    ) as dst:
        dst.write(np.asarray(class_array, dtype=np.uint8), 1)
    return path


def write_class_npy(class_array: np.ndarray, path: str) -> str:
    """Write the class raster as .npy (no rasterio needed; demo/testing)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    np.save(path, np.asarray(class_array, dtype=np.uint8))
    return path
