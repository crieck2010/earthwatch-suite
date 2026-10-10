"""Seeded synthetic data: the whole pipeline runs offline.

Generates demo zones (rectangles), a synthetic per-zone phenology time
series (double-logistic-ish seasonal curve + noise, with one planted
drought anomaly), and synthetic index rasters (spatial gradient + blobs).
Everything is seeded, so demos and tests are deterministic.
"""

from __future__ import annotations

import datetime as _dt
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .indices_ext import VIGOR_INDICES
from .timeseries import ZonePassRecord
from .zones import Zone


def synthetic_zones() -> List[Zone]:
    """Three rectangular demo zones (an orchard block, a pasture, a woodlot)."""
    def rect(x0, y0, x1, y1):
        return {
            "type": "Polygon",
            "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
        }

    return [
        Zone(id="orchard-north", name="Orchard north block",
             geometry=rect(0.0, 0.0, 4.0, 3.0)),
        Zone(id="pasture-east", name="East pasture",
             geometry=rect(4.5, 0.0, 8.0, 3.0)),
        Zone(id="woodlot-south", name="South woodlot",
             geometry=rect(0.0, 3.5, 8.0, 6.5)),
    ]


def phenology_value(
    doy: int,
    greenup_doy: int = 110,
    peak_doy: int = 200,
    senescence_doy: int = 290,
    base: float = 0.15,
    peak: float = 0.85,
) -> float:
    """Piecewise-linear seasonal vigor curve for a day-of-year."""
    if doy <= greenup_doy or doy >= senescence_doy:
        return base
    if doy <= peak_doy:
        frac = (doy - greenup_doy) / (peak_doy - greenup_doy)
        return base + frac * (peak - base)
    frac = (doy - peak_doy) / (senescence_doy - peak_doy)
    return peak - frac * (peak - base)


def synthetic_timeseries(
    zones: Sequence[Zone] | None = None,
    start: str = "2022-03-01",
    end: str = "2024-11-30",
    indices: Sequence[str] = ("ndvi", "evi"),
    step_days: int = 16,
    seed: int = 7,
    anomaly_zone: str = "orchard-north",
    anomaly_year: int = 2024,
    anomaly_doy_range: Tuple[int, int] = (180, 240),
    anomaly_drop: float = 0.25,
) -> List[ZonePassRecord]:
    """Synthetic per-zone vigor series with one planted drought anomaly.

    ``anomaly_zone`` loses ``anomaly_drop`` NDVI through the given DOY window
    in ``anomaly_year`` — the anomaly detector should flag it.
    """
    rng = np.random.default_rng(seed)
    zones = list(zones) if zones else synthetic_zones()
    scale = {"ndvi": 1.0, "evi": 0.75, "savi": 0.9, "gndvi": 0.85, "msavi": 0.9}
    for idx in indices:
        if idx not in VIGOR_INDICES:
            raise ValueError(f"unknown vigor index {idx!r}")
    d0 = _dt.date.fromisoformat(start)
    d1 = _dt.date.fromisoformat(end)
    dates = []
    d = d0
    while d <= d1:
        dates.append(d)
        d += _dt.timedelta(days=step_days)

    records: List[ZonePassRecord] = []
    for zi, zone in enumerate(zones):
        zone_phase = zi * 7  # zones green up a few days apart
        for date in dates:
            doy = int(date.strftime("%j"))
            base_curve = phenology_value(
                doy,
                greenup_doy=110 + zone_phase,
                peak_doy=200 + zone_phase,
                senescence_doy=290 + zone_phase,
            )
            for index in indices:
                val = base_curve * scale.get(index, 1.0)
                if (
                    zone.id == anomaly_zone
                    and date.year == anomaly_year
                    and anomaly_doy_range[0] <= doy <= anomaly_doy_range[1]
                ):
                    val -= anomaly_drop * scale.get(index, 1.0)
                val = float(np.clip(val + rng.normal(0, 0.015), -0.05, 1.0))
                stats = {
                    "mean": val,
                    "median": val,
                    "std": 0.03,
                    "minimum": max(0.0, val - 0.08),
                    "maximum": min(1.0, val + 0.08),
                    "p10": max(0.0, val - 0.05),
                    "p90": min(1.0, val + 0.05),
                    "valid_pixels": 400,
                    "total_pixels": 400,
                }
                pass_id = f"synthetic-{date.isoformat()}"
                records.append(
                    ZonePassRecord.from_zonal_stats(
                        zone.id, zone.name, pass_id, date.isoformat(),
                        index, stats, cloud_cover=5.0,
                    )
                )
    return sorted(records, key=lambda r: (r.zone_id, r.date))


def synthetic_index_raster(
    shape: Tuple[int, int] = (60, 60),
    seed: int = 11,
    stressed_patch: bool = True,
) -> np.ndarray:
    """Synthetic NDVI-like raster: gradient + blobs, optional stressed patch."""
    rng = np.random.default_rng(seed)
    rows, cols = shape
    yy, xx = np.mgrid[0:rows, 0:cols].astype(float)
    base = 0.25 + 0.45 * (yy / rows)  # greener toward the south
    for _ in range(6):
        cy, cx = rng.uniform(0, rows), rng.uniform(0, cols)
        sigma = rng.uniform(4, 10)
        base += 0.25 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    base += rng.normal(0, 0.03, size=shape)
    if stressed_patch:
        base[8:20, 40:54] -= 0.35  # a stressed rectangle
    base[50:58, 2:10] = -0.2  # a water corner
    return np.clip(base, -1, 1).astype(np.float32)


def synthetic_bands(
    shape: Tuple[int, int] = (60, 60), seed: int = 11
) -> Dict[str, np.ndarray]:
    """Synthetic nir/red/green/blue band dict consistent-ish with the raster."""
    ndvi = synthetic_index_raster(shape, seed)
    rng = np.random.default_rng(seed + 1)
    # Invert NDVI approximately: nir = red * (1+ndvi)/(1-ndvi), pick red, add noise.
    red = np.clip(0.08 + 0.05 * rng.random(shape), 0.01, 0.5).astype(np.float32)
    ratio = np.clip((1 + ndvi) / np.clip(1 - ndvi, 0.05, None), 0.2, 8.0)
    nir = np.clip(red * ratio, 0.01, 0.9).astype(np.float32)
    green = np.clip(red * 1.1 + rng.normal(0, 0.01, shape), 0.01, 0.6).astype(np.float32)
    blue = np.clip(red * 0.9 + rng.normal(0, 0.01, shape), 0.01, 0.6).astype(np.float32)
    return {"nir": nir, "red": red, "green": green, "blue": blue}
