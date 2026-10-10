"""Water-index computation.

The workhorse index (NDWI) is **reused** from survey-imagery's
:mod:`imagery.indices` registry via a lazy import — it is not reimplemented
here. This module adds only the water-specific index that registry does not
carry (MNDWI), following the same NaN-honest conventions: any input pixel
that is NaN, or whose denominator is zero, yields NaN.

- **NDWI** (McFeeters, 1996): (green − nir) / (green + nir). Water is
  strongly positive; vegetation and soil are near zero or negative.
- **MNDWI** (Xu, 2006): (green − swir1) / (green + swir1). Suppresses
  built-up land noise that can leak into NDWI in urbanised catchments.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

#: Water indices this engine works with. ``ndwi`` resolves through
#: survey-imagery's registry; ``mndwi`` is defined in this module.
WATER_INDICES: Tuple[str, ...] = ("ndwi", "mndwi")

#: Indices defined in this package (not in survey-imagery's registry).
LOCAL_INDICES: Tuple[str, ...] = ("mndwi",)


def _imagery_indices():
    try:
        from imagery import indices as _im
    except ImportError as exc:
        raise ImportError(
            "survey-flood needs the 'survey-imagery' package for "
            "NDWI (pip install 'survey-flood[acquire]')"
        ) from exc
    return _im


def mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Modified NDWI: (green − swir1) / (green + swir1).

    SWIR is absorbed even more strongly by water than NIR, so MNDWI
    separates open water from built-up areas better than NDWI. Needs a
    short-wave infrared band (Sentinel-2 B11, Landsat SR_B6).
    """
    green = np.asarray(green, dtype=np.float32)
    swir1 = np.asarray(swir1, dtype=np.float32)
    denom = green + swir1
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom == 0, np.nan, (green - swir1) / denom)
    out[~np.isfinite(green) | ~np.isfinite(swir1)] = np.nan
    return out.astype(np.float32)


def required_bands(index_name: str) -> Tuple[str, ...]:
    """Band aliases needed to compute a water index."""
    name = index_name.lower()
    if name == "mndwi":
        return ("green", "swir1")
    return tuple(_imagery_indices().required_bands(name))


def compute(index_name: str, bands: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
    """Compute a water index from a dict of alias -> reflectance array."""
    name = index_name.lower()
    if name == "mndwi":
        return mndwi(bands["green"], bands["swir1"])
    return _imagery_indices().compute(name, bands, **kwargs)
