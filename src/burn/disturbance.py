"""Generalised disturbance detection beyond fire.

The disturbance heuristic flags abrupt vegetation *removal* — clearing,
logging, land conversion — from consecutive-pass index drops:

    disturbed  <=>  (NDVI_now − NDVI_prev) <= −ndvi_drop
                    AND (NBR_now − NBR_prev) <= −nbr_drop

Requiring *both* indices to fall rejects most phenology (seasonal
senescence usually moves NDVI more than NBR) while still catching
mechanical removal, which strips NIR in both. It is a **heuristic**, not a
classifier — documented false-positive modes:

* harvest of an annual crop (a real removal, but not the land-use change
  a compliance user is hunting);
* drought/heat stress browning a whole region at once (large, coherent
  NDVI+NBR drops with no mechanical cause — check the spatial pattern);
* cloud/shadow leakage past the cloud mask (both indices dip together);
* rapid senescence in grasslands after the first frost.

Mitigations built in: minimum mapping unit (``min_pixels`` connected
components), persistence (the drop must survive ``persistence`` passes),
and a ``confidence`` score from the drop magnitudes. Everything the
heuristic needs is in this module; NDVI/NBR series come from the caller
(survey-imagery registry for NDVI, :mod:`burn.indices_fire` for NBR).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .masking import label_components


@dataclass(frozen=True)
class DisturbanceParams:
    """Tunable knobs of the disturbance heuristic."""

    ndvi_drop: float = 0.15  #: NDVI must fall by at least this
    nbr_drop: float = 0.10  #: NBR must fall by at least this
    min_pixels: int = 9  #: smallest reportable patch (3x3 at 10 m = 0.09 ha)
    persistence: int = 1  #: drop must persist this many consecutive passes


@dataclass(frozen=True)
class DisturbanceEvent:
    """One detected disturbance patch."""

    zone_id: str
    pass_id: str  # the pass on which the drop was observed
    date: str
    pixels: int
    area_ha: float
    mean_ndvi_drop: float
    mean_nbr_drop: float
    confidence: float  # 0..1 from drop magnitudes
    persistent: bool


def _confidence(ndvi_drop: float, nbr_drop: float, params: DisturbanceParams) -> float:
    """0..1 score: how far past both thresholds the drops went."""
    s_ndvi = min(1.0, max(0.0, (ndvi_drop - params.ndvi_drop) / params.ndvi_drop))
    s_nbr = min(1.0, max(0.0, (nbr_drop - params.nbr_drop) / params.nbr_drop))
    return float(0.5 * s_ndvi + 0.5 * s_nbr)


def detect_disturbance(
    ndvi_prev: np.ndarray,
    ndvi_now: np.ndarray,
    nbr_prev: np.ndarray,
    nbr_now: np.ndarray,
    valid: np.ndarray | None = None,
    params: DisturbanceParams | None = None,
    pixel_size_m: float = 10.0,
    zone_id: str = "",
    pass_id: str = "",
    date: str = "",
) -> List[DisturbanceEvent]:
    """Flag disturbed pixels between two consecutive passes.

    All four arrays share a shape; NaN anywhere disqualifies the pixel.
    Returns one event per connected disturbed patch of at least
    ``params.min_pixels`` pixels.
    """
    params = params or DisturbanceParams()
    ndvi_prev = np.asarray(ndvi_prev, dtype=np.float32)
    ndvi_now = np.asarray(ndvi_now, dtype=np.float32)
    nbr_prev = np.asarray(nbr_prev, dtype=np.float32)
    nbr_now = np.asarray(nbr_now, dtype=np.float32)
    ok = np.ones(ndvi_prev.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    finite = (
        np.isfinite(ndvi_prev)
        & np.isfinite(ndvi_now)
        & np.isfinite(nbr_prev)
        & np.isfinite(nbr_now)
        & ok
    )
    d_ndvi = np.where(finite, ndvi_now - ndvi_prev, np.nan)
    d_nbr = np.where(finite, nbr_now - nbr_prev, np.nan)
    disturbed = finite & (d_ndvi <= -params.ndvi_drop) & (d_nbr <= -params.nbr_drop)
    labels, sizes = label_components(disturbed, connectivity=8)
    pixel_ha = (pixel_size_m ** 2) / 10_000.0
    events: List[DisturbanceEvent] = []
    for lab, size in sizes.items():
        if size < params.min_pixels:
            continue
        comp = labels == lab
        events.append(
            DisturbanceEvent(
                zone_id=zone_id,
                pass_id=pass_id,
                date=date,
                pixels=int(size),
                area_ha=float(size * pixel_ha),
                mean_ndvi_drop=float(np.nanmean(np.where(comp, -d_ndvi, np.nan))),
                mean_nbr_drop=float(np.nanmean(np.where(comp, -d_nbr, np.nan))),
                confidence=_confidence(
                    float(np.nanmean(np.where(comp, -d_ndvi, np.nan))),
                    float(np.nanmean(np.where(comp, -d_nbr, np.nan))),
                    params,
                ),
                persistent=params.persistence <= 1,
            )
        )
    events.sort(key=lambda e: e.pixels, reverse=True)
    return events
