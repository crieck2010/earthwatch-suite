"""Cut/fill volumetrics per zone from an elevation-difference raster.

Conventions (documented, not hidden):

* **Cut** = ground removed (Δh < 0), **fill** = ground added (Δh > 0).
  Volumes are positive magnitudes in cubic metres.
* Volume = Σ |Δh| × pixel_area over the zone's valid changed pixels.
* ``noise_floor`` (metres): ``|Δh|`` below it counts as no-change — it is
  the DEM noise floor, and volumetrics without one integrate noise into
  phantom cubic metres.
* **Uncertainty**: given per-DEM vertical RMSEs σ₁, σ₂, the per-pixel
  difference error is σ_Δh = √(σ₁² + σ₂²) and, *assuming independent pixel
  errors*, σ_V = pixel_area × σ_Δh × √N. That independence assumption is
  optimistic — real DEM errors are spatially correlated, so the true
  uncertainty is larger. The reported σ_V is a *lower bound*, and the
  ``confidence`` flag (high/medium/low from the volume signal-to-noise
  ratio) is a screening aid, not a surveyor's tolerance statement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .diff import pixel_area
from .zonal import zone_mask
from .zones import Zone


@dataclass
class ZoneVolumes:
    """Cut/fill volumetrics for one zone."""

    zone_id: str
    zone_name: str
    cut_m3: float  # ground removed, ≥ 0
    fill_m3: float  # ground added, ≥ 0
    net_m3: float  # fill - cut (signed)
    mean_dh: float
    median_dh: float
    max_cut_depth: float  # most negative Δh (≤ 0)
    max_fill_height: float  # most positive Δh (≥ 0)
    changed_pixels: int
    valid_pixels: int
    area_m2: float  # valid zone area
    volume_sigma_m3: Optional[float]  # None when RMSEs not supplied
    confidence: str  # "high" | "medium" | "low"

    def to_dict(self) -> Dict[str, object]:
        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "cut_m3": self.cut_m3,
            "fill_m3": self.fill_m3,
            "net_m3": self.net_m3,
            "mean_dh": self.mean_dh,
            "median_dh": self.median_dh,
            "max_cut_depth": self.max_cut_depth,
            "max_fill_height": self.max_fill_height,
            "changed_pixels": self.changed_pixels,
            "valid_pixels": self.valid_pixels,
            "area_m2": self.area_m2,
            "volume_sigma_m3": self.volume_sigma_m3,
            "confidence": self.confidence,
        }


def _confidence(net_m3: float, sigma_m3: Optional[float]) -> str:
    """Signal-to-noise confidence flag.

    high: |net| ≥ 5σ · medium: |net| ≥ 2σ · low: otherwise (or σ unknown
    and net == 0). Without RMSEs there is no σ, so confidence can only
    ever be "low" — the honest answer when uncertainty is unquantified.
    """
    if sigma_m3 is None or sigma_m3 <= 0:
        return "low"
    snr = abs(net_m3) / sigma_m3
    if snr >= 5.0:
        return "high"
    if snr >= 2.0:
        return "medium"
    return "low"


def zone_volumes(
    diff: np.ndarray,
    zone: Zone,
    transform,
    noise_floor: float = 0.0,
    rmse1: Optional[float] = None,
    rmse2: Optional[float] = None,
) -> ZoneVolumes:
    """Cut/fill volumetrics for one zone from a difference raster (metres).

    Uses the noise-floor-thresholded difference: ``|Δh| < noise_floor``
    counts as no-change. ``rmse1``/``rmse2`` are the per-DEM vertical
    RMSEs in metres; omit both to skip uncertainty (confidence = "low").
    """
    arr = np.asarray(diff, dtype=float)
    px_area = pixel_area(transform)
    mask = zone_mask(zone, arr.shape, transform)

    z = arr[mask]
    valid = z[np.isfinite(z)]
    changed = valid[np.abs(valid) >= noise_floor]

    cut_vals = changed[changed < 0]
    fill_vals = changed[changed > 0]
    cut_m3 = float(np.sum(-cut_vals) * px_area)
    fill_m3 = float(np.sum(fill_vals) * px_area)
    net_m3 = fill_m3 - cut_m3

    sigma_m3: Optional[float] = None
    if rmse1 is not None and rmse2 is not None:
        if rmse1 < 0 or rmse2 < 0:
            raise ValueError("RMSE values must be >= 0")
        sigma_dh = math.sqrt(rmse1**2 + rmse2**2)
        # Independent-pixel-errors lower bound (see module docstring).
        sigma_m3 = px_area * sigma_dh * math.sqrt(max(changed.size, 0))

    return ZoneVolumes(
        zone_id=zone.id,
        zone_name=zone.name,
        cut_m3=cut_m3,
        fill_m3=fill_m3,
        net_m3=net_m3,
        mean_dh=float(np.mean(changed)) if changed.size else 0.0,
        median_dh=float(np.median(changed)) if changed.size else 0.0,
        max_cut_depth=float(np.min(cut_vals)) if cut_vals.size else 0.0,
        max_fill_height=float(np.max(fill_vals)) if fill_vals.size else 0.0,
        changed_pixels=int(changed.size),
        valid_pixels=int(valid.size),
        area_m2=float(valid.size * px_area),
        volume_sigma_m3=sigma_m3,
        confidence=_confidence(net_m3, sigma_m3),
    )
