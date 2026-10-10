"""Vigor-index computation.

The workhorse indices (NDVI, EVI, SAVI) are **reused** from survey-imagery's
:mod:`imagery.indices` registry via a lazy import — they are not reimplemented
here. This module adds only the vegetation-specific indices that registry does
not carry (GNDVI, MSAVI), following the same NaN-honest conventions: any
input pixel that is NaN, or whose denominator is zero, yields NaN.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

#: Vigor indices this engine works with. The first three resolve through
#: survey-imagery's registry; the last two are defined in this module.
VIGOR_INDICES: Tuple[str, ...] = ("ndvi", "evi", "savi", "gndvi", "msavi")

_LOCAL_INDICES = ("gndvi", "msavi")

#: Indices defined in this package (not in survey-imagery's registry).
LOCAL_INDICES: Tuple[str, ...] = _LOCAL_INDICES


def _imagery_indices():
    try:
        from imagery import indices as _im
    except ImportError as exc:
        raise ImportError(
            "survey-vegetation needs the 'survey-imagery' package for "
            "NDVI/EVI/SAVI (pip install 'survey-vegetation[acquire]')"
        ) from exc
    return _im


def gndvi(nir: np.ndarray, green: np.ndarray) -> np.ndarray:
    """Green NDVI: (NIR - green) / (NIR + green).

    More sensitive to chlorophyll concentration than NDVI; useful for
    nitrogen-stress scouting in crops.
    """
    nir = np.asarray(nir, dtype=np.float32)
    green = np.asarray(green, dtype=np.float32)
    denom = nir + green
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom == 0, np.nan, (nir - green) / denom)
    out[~np.isfinite(nir) | ~np.isfinite(green)] = np.nan
    return out.astype(np.float32)


def msavi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Modified Soil-Adjusted Vegetation Index (MSAVI2).

    (2*NIR + 1 - sqrt((2*NIR+1)^2 - 8*(NIR - red))) / 2. Self-adjusting soil
    correction — no L parameter to choose, unlike SAVI.
    """
    nir = np.asarray(nir, dtype=np.float32)
    red = np.asarray(red, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = (2.0 * nir + 1.0) ** 2 - 8.0 * (nir - red)
        inner = np.where(inner < 0, np.nan, inner)
        out = (2.0 * nir + 1.0 - np.sqrt(inner)) / 2.0
    out[~np.isfinite(nir) | ~np.isfinite(red)] = np.nan
    return out.astype(np.float32)


def required_bands(index_name: str) -> Tuple[str, ...]:
    """Band aliases needed to compute a vigor index."""
    name = index_name.lower()
    if name == "gndvi":
        return ("nir", "green")
    if name == "msavi":
        return ("nir", "red")
    return tuple(_imagery_indices().required_bands(name))


def compute(index_name: str, bands: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
    """Compute a vigor index from a dict of alias -> reflectance array.

    Registry indices (ndvi/evi/savi) delegate to survey-imagery;
    gndvi/msavi are computed here. Raises ``ValueError`` for unknown names.
    """
    name = index_name.lower()
    if name not in VIGOR_INDICES:
        raise ValueError(
            f"unknown vigor index {index_name!r}; known: {', '.join(VIGOR_INDICES)}"
        )
    if name in _LOCAL_INDICES:
        need = required_bands(name)
        missing = [b for b in need if b not in bands]
        if missing:
            raise ValueError(
                f"index {name!r} needs bands {missing} which were not supplied"
            )
        args = [np.asarray(bands[b], dtype=np.float32) for b in need]
        shapes = {a.shape for a in args}
        if len(shapes) != 1:
            raise ValueError(
                f"band arrays for {name!r} have mismatched shapes: {shapes}"
            )
        func = gndvi if name == "gndvi" else msavi
        return func(*args)
    return _imagery_indices().compute(name, bands, **kwargs)


def list_vigor_indices() -> Sequence[str]:
    """Names of the vigor indices this engine supports."""
    return list(VIGOR_INDICES)
