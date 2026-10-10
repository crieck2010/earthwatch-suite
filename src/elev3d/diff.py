"""DEM differencing: ``dem2 - dem1`` with NaN-honest handling and a noise floor.

Both DEMs must share the same grid (shape), CRS, and vertical datum — this
module does *not* resample or reproject. Differences are in the DEMs' own
vertical units (metres by convention; see docs/DATUMS.md).

The noise floor (``noise_floor``, metres) marks the minimum detectable
change: ``|Δh| < noise_floor`` is treated as no-change. It exists because
every DEM carries vertical noise; without it, volumetrics happily integrate
noise into phantom cubic metres. The raw (unthresholded) difference is
always available alongside the thresholded one so nothing is hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class DifferenceResult:
    """Result of differencing one DEM pair."""

    diff: np.ndarray  # raw dem2 - dem1, NaN where either input is NaN (metres)
    diff_thresholded: np.ndarray  # |Δh| < noise_floor → 0 (NaN stays NaN)
    noise_floor: float
    valid_pixels: int
    changed_pixels: int  # valid pixels with |Δh| >= noise_floor

    def summary(self) -> Dict[str, float]:
        valid = self.diff[np.isfinite(self.diff)]
        thr = self.diff_thresholded[np.isfinite(self.diff_thresholded)]
        return {
            "valid_pixels": float(self.valid_pixels),
            "changed_pixels": float(self.changed_pixels),
            "changed_fraction": float(self.changed_pixels / self.valid_pixels)
            if self.valid_pixels
            else 0.0,
            "mean_dh": float(np.mean(valid)) if valid.size else float("nan"),
            "median_dh": float(np.median(valid)) if valid.size else float("nan"),
            "std_dh": float(np.std(valid)) if valid.size else float("nan"),
            "min_dh": float(np.min(valid)) if valid.size else float("nan"),
            "max_dh": float(np.max(valid)) if valid.size else float("nan"),
            "noise_floor": self.noise_floor,
        }


def difference(
    dem1: np.ndarray,
    dem2: np.ndarray,
    noise_floor: float = 0.0,
) -> DifferenceResult:
    """Compute ``dem2 - dem1``.

    Parameters
    ----------
    dem1, dem2:
        Elevation arrays in metres, same shape. NaN = nodata.
    noise_floor:
        Minimum detectable change in metres (≥ 0). ``|Δh|`` below it is
        zeroed in ``diff_thresholded``; the raw ``diff`` is untouched.

    Raises
    ------
    ValueError
        On shape mismatch or a negative noise floor.
    """
    a = np.asarray(dem1, dtype=float)
    b = np.asarray(dem2, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"DEM shape mismatch: {a.shape} vs {b.shape}")
    if noise_floor < 0:
        raise ValueError(f"noise_floor must be >= 0 (got {noise_floor})")

    diff = b - a  # NaN propagates: NaN where either input is NaN
    valid = np.isfinite(diff)
    thresholded = np.where(valid & (np.abs(diff) < noise_floor), 0.0, diff)

    return DifferenceResult(
        diff=diff,
        diff_thresholded=thresholded,
        noise_floor=noise_floor,
        valid_pixels=int(np.count_nonzero(valid)),
        changed_pixels=int(np.count_nonzero(valid & (np.abs(diff) >= noise_floor))),
    )


def pixel_area(transform: Tuple[float, float, float, float, float, float]) -> float:
    """Pixel area in square map units from an affine transform tuple.

    ``transform`` is (a, b, c, d, e, f) mapping pixel (col, row) → map
    (x, y): x = c + a*col + b*row, y = f + d*col + e*row. For north-up
    rasters (b == d == 0) this is |a * e|; rotation is rejected.
    """
    a, b, _c, d, e, _f = transform
    if b != 0 or d != 0:
        raise ValueError("rotated transforms are not supported (b and d must be 0)")
    area = abs(a * e)
    if area <= 0:
        raise ValueError(f"degenerate pixel area {area} from transform {transform}")
    return area


def check_grids(
    shape1: Tuple[int, int],
    shape2: Tuple[int, int],
    transform1: Optional[Tuple[float, ...]] = None,
    transform2: Optional[Tuple[float, ...]] = None,
) -> None:
    """Raise unless the two DEM grids are directly comparable."""
    if shape1 != shape2:
        raise ValueError(f"DEM shape mismatch: {shape1} vs {shape2}")
    if transform1 is not None and transform2 is not None:
        if any(abs(x - y) > 1e-9 for x, y in zip(transform1, transform2)):
            raise ValueError(
                "DEM transforms differ — resample to a common grid before "
                "differencing (this engine never resamples silently)."
            )
