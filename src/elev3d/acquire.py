"""GeoTIFF I/O for DEMs and difference rasters — rasterio is lazy.

survey-3d never *requires* rasterio: every analytic function takes numpy
arrays. This module is the file boundary — ``pip install "survey-3d[raster]"``
to enable it. DEMs themselves are user-supplied (stereo pairs, LiDAR,
drone photogrammetry); this engine starts where two elevation models
already exist.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _rasterio():
    try:
        import rasterio  # type: ignore[import]

        return rasterio
    except ImportError as exc:
        raise ImportError(
            'reading GeoTIFFs needs rasterio: pip install "survey-3d[raster]"'
        ) from exc


def read_dem(path: str) -> Tuple[np.ndarray, Tuple[float, ...], Optional[str], Optional[str]]:
    """Read a single-band DEM. Returns (array, transform, crs_wkt, units_note).

    ``units_note`` is informational only — the engine assumes metres and a
    shared vertical datum (see docs/DATUMS.md); it never converts.
    """
    rio = _rasterio()
    with rio.open(path) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        transform = tuple(src.transform)[:6]
        crs = src.crs.to_wkt() if src.crs else None
        unit = None
        try:
            unit = src.crs.axis_info[2].unit_name if src.crs else None
        except Exception:
            unit = None
    return arr, transform, crs, unit


def write_geotiff(
    path: str,
    arr: np.ndarray,
    transform: Tuple[float, ...],
    crs_wkt: Optional[str] = None,
    nodata: float = float("nan"),
) -> str:
    """Write a single-band float32 GeoTIFF. Returns the path written."""
    rio = _rasterio()
    a, b, c, d, e, f = transform
    out = np.asarray(arr, dtype=np.float32)
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=out.shape[0],
        width=out.shape[1],
        count=1,
        dtype="float32",
        crs=crs_wkt,
        transform=rio.Affine(a, b, c, d, e, f),
        nodata=nodata,
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(out, 1)
    return path


def check_crs_match(path1: str, path2: str) -> None:
    """Raise unless the two DEM files declare the same CRS.

    A cheap guard against differencing DEMs in different datums — the
    single most common silent-corruption mode in DEM differencing.
    """
    rio = _rasterio()
    with rio.open(path1) as s1, rio.open(path2) as s2:
        c1, c2 = s1.crs, s2.crs
    if c1 is None or c2 is None:
        raise ValueError("both DEMs must declare a CRS (one is missing it)")
    if c1 != c2:
        raise ValueError(
            f"DEM CRS mismatch:\n  dem1: {c1}\n  dem2: {c2}\n"
            "Reproject to a common CRS and vertical datum before differencing."
        )
