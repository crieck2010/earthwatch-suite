"""Coregistration check: is the DEM pair actually comparable?

Differencing two DEMs is only meaningful if they are coregistered — same
grid, same datum, no systematic bias. This module reports the median Δh
over a user-supplied **stable area** (roads, buildings, undisturbed
ground): if the "stable" ground moved, the DEMs are not comparable and
every volume downstream is suspect.

Diagnostic, not a fix: a failed check means *go coregister your DEMs*
(ICP, stable-area shift estimation), not *subtract the bias and carry on*
— although the bias value is reported so the operator can see its size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .zonal import zone_mask
from .zones import Zone


@dataclass
class CoregReport:
    """Bias diagnostic over a stable area."""

    zone_id: str
    bias_m: float  # median Δh over stable pixels
    mad_m: float  # median absolute deviation (robust spread)
    mean_dh: float
    std_dh: float
    stable_pixels: int
    threshold_m: float
    passed: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "zone_id": self.zone_id,
            "bias_m": self.bias_m,
            "mad_m": self.mad_m,
            "mean_dh": self.mean_dh,
            "std_dh": self.std_dh,
            "stable_pixels": self.stable_pixels,
            "threshold_m": self.threshold_m,
            "passed": self.passed,
        }


def coregistration_check(
    diff: np.ndarray,
    stable_zone: Zone,
    transform,
    threshold_m: float = 0.10,
) -> CoregReport:
    """Median Δh over a stable area; ``|bias| <= threshold`` passes.

    Uses the median (not the mean) so a few changed pixels inside the
    stable polygon — a parked truck, a new pothole — don't fail the check.
    ``threshold_m`` should reflect the DEMs' vertical accuracy (a common
    rule: 2× the combined RMSE); the default 0.10 m suits good
    photogrammetry/LiDAR, not 30 m satellite DEMs.
    """
    arr = np.asarray(diff, dtype=float)
    mask = zone_mask(stable_zone, arr.shape, transform)
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError(f"stable zone {stable_zone.id!r} has no valid pixels")

    bias = float(np.median(vals))
    mad = float(np.median(np.abs(vals - bias)))
    return CoregReport(
        zone_id=stable_zone.id,
        bias_m=bias,
        mad_m=mad,
        mean_dh=float(np.mean(vals)),
        std_dh=float(np.std(vals)),
        stable_pixels=int(vals.size),
        threshold_m=threshold_m,
        passed=abs(bias) <= threshold_m,
    )
