"""Water spectral indices for shoreline work.

NDWI and MNDWI are **reused** from survey-imagery's index registry (lazy
import) — never reimplemented here. If survey-imagery is not installed, a
local NaN-honest fallback with identical math is used so the engine, tests,
and synthetic demo work standalone.

- NDWI  = (green − nir) / (green + nir)        (McFeeters 1996)
- MNDWI = (green − swir1) / (green + swir1)    (Xu 2006) — better in
  built-up coastal areas because SWIR suppresses built-land confusion.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

#: Indices this engine needs, with the canonical band names it asks for.
SUPPORTED_INDICES: Dict[str, Sequence[str]] = {
    "ndwi": ("green", "nir"),
    "mndwi": ("green", "swir1"),
}


def _local_index(name: str, bands: Dict[str, np.ndarray]) -> np.ndarray:
    """NaN-honest fallback index math (identical to survey-imagery's)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if name == "ndwi":
            num = bands["green"] - bands["nir"]
            den = bands["green"] + bands["nir"]
        elif name == "mndwi":
            num = bands["green"] - bands["swir1"]
            den = bands["green"] + bands["swir1"]
        else:  # pragma: no cover - guarded by SUPPORTED_INDICES
            raise ValueError(f"unsupported index {name!r}")
        out = num / den
    out = np.where(np.isfinite(out), out, np.nan)
    # A pixel NaN in any input band is NaN out — no silent infills.
    mask = np.zeros_like(out, dtype=bool)
    for arr in bands.values():
        mask |= ~np.isfinite(arr)
    return np.where(mask, np.nan, out)


def compute_index(name: str, bands: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute a water index, preferring survey-imagery's registry.

    Falls back to the local NaN-honest implementation when survey-imagery
    is not installed (raises a helpful error only for unknown names).
    """
    key = name.lower()
    if key not in SUPPORTED_INDICES:
        raise ValueError(
            f"unknown index {name!r}; survey-coast supports {sorted(SUPPORTED_INDICES)}"
        )
    try:
        from imagery import indices as _registry  # type: ignore
    except ImportError:
        return _local_index(key, bands)
    try:
        func = _registry.get_index(key)  # registry-style access
    except Exception:
        return _local_index(key, bands)
    return np.asarray(func(bands), dtype=float)


def describe_indices() -> Dict[str, Dict[str, object]]:
    """Human/machine-readable description of the supported indices."""
    return {
        "ndwi": {
            "bands": list(SUPPORTED_INDICES["ndwi"]),
            "formula": "(green - nir) / (green + nir)",
            "reference": "McFeeters (1996)",
            "use": "general open-water mapping",
        },
        "mndwi": {
            "bands": list(SUPPORTED_INDICES["mndwi"]),
            "formula": "(green - swir1) / (green + swir1)",
            "reference": "Xu (2006)",
            "use": "coastal / built-up areas; suppresses built-land confusion",
        },
    }
