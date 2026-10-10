"""Seeded synthetic data: the whole pipeline runs offline.

Generates demo zones (rectangles), synthetic water-index rasters with
coherent water bodies (a river channel whose width grows with the flood
level), and a per-zone monthly time series with **planted flood events**:

- ``river-reach``: major flood on 2024-04-15 (flood level 1.0), smaller
  flood on 2023-05-15 (flood level 0.7)
- ``wetland-north``: seasonal spring high water (no discrete event)
- ``upland-control``: never floods (negative control)

Everything is seeded, so demos and tests are deterministic. The
``demo`` CLI asserts the planted 2024-04-15 flood is detected.
"""

from __future__ import annotations

import datetime as _dt
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .indices_water import WATER_INDICES
from .timeseries import ZonePassRecord
from .watermask import build_water_mask
from .zonal import water_stats
from .zones import Zone

#: The planted flood events: (zone_id, date, flood_level).
#: Dates are the 15th of the month to match the monthly synthetic passes.
PLANTED_FLOODS: Tuple[Tuple[str, str, float], ...] = (
    ("river-reach", "2024-04-15", 1.0),
    ("river-reach", "2023-05-15", 0.7),
)

#: Pixel size (m) of the synthetic grids — Sentinel-2-like 10 m.
SYNTHETIC_PIXEL_SIZE_M = 10.0


def synthetic_zones() -> List[Zone]:
    """Three rectangular demo zones: a river reach, a wetland, an upland control."""

    def rect(x0, y0, x1, y1):
        return {
            "type": "Polygon",
            "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
        }

    return [
        Zone(id="river-reach", name="River reach (floodplain)",
             geometry=rect(0.0, 0.0, 6.0, 4.0)),
        Zone(id="wetland-north", name="North wetland",
             geometry=rect(0.0, 4.5, 6.0, 7.5)),
        Zone(id="upland-control", name="Upland control (never floods)",
             geometry=rect(6.5, 0.0, 10.0, 7.5)),
    ]


def flood_level(zone_id: str, date: str) -> float:
    """Flood level in [0, 1] for a zone on a date (the planted truth)."""
    for zid, d, level in PLANTED_FLOODS:
        if zid == zone_id and d == date:
            return level
    if zone_id == "wetland-north":
        # Seasonal spring high water, peaking mid-April.
        doy = _dt.date.fromisoformat(date).timetuple().tm_yday
        return max(0.0, 0.5 * np.cos(2 * np.pi * (doy - 105) / 365) + 0.25)
    return 0.0


def synthetic_water_mask(
    shape: Tuple[int, int], zone_id: str, level: float, seed: int = 7
) -> np.ndarray:
    """Deterministic coherent water mask for a zone at a flood level.

    The river reach has a diagonal channel whose half-width grows with the
    flood level; the wetland has an elliptical pond that swells seasonally;
    the upland control has no water. ``seed`` is accepted for API symmetry
    (the mask itself is deterministic; noise lives in the index raster).
    """
    rows, cols = shape
    rr, cc = np.mgrid[0:rows, 0:cols]
    if zone_id == "river-reach":
        centre = (cols * 0.25 + rr * 0.25).astype(float)
        half_width = 1.0 + level * 9.0
        return np.abs(cc - centre) <= half_width
    if zone_id == "wetland-north":
        dist = ((rr - rows * 0.45) / (rows * 0.28)) ** 2 + (
            (cc - cols * 0.55) / (cols * 0.22)
        ) ** 2
        return dist < 0.6 + level * 0.9
    return np.zeros(shape, dtype=bool)


def synthetic_index_raster(
    shape: Tuple[int, int],
    zone_id: str,
    level: float,
    seed: int = 7,
    index: str = "ndwi",
) -> np.ndarray:
    """Synthetic water-index raster: water ~ +0.35, land ~ -0.35, plus noise.

    Thresholding at 0.0 recovers approximately the true water mask, so the
    full pipeline (mask -> stats -> series -> delta) can be exercised and
    asserted end to end.
    """
    rng = np.random.default_rng(seed + hash((zone_id, index)) % (2 ** 31))
    mask = synthetic_water_mask(shape, zone_id, level)
    out = np.where(mask, 0.35, -0.35).astype(np.float32)
    out += rng.normal(0.0, 0.09, size=shape).astype(np.float32)
    # A few NaNs (sensor gaps) — the pipeline must treat them as not-water.
    gaps = rng.random(shape) < 0.002
    out[gaps] = np.nan
    return out


def synthetic_cloud_mask(shape: Tuple[int, int], cloud_fraction: float, seed: int) -> np.ndarray:
    """Seeded boolean cloud mask with approximately ``cloud_fraction`` cover."""
    rng = np.random.default_rng(seed)
    return rng.random(shape) < cloud_fraction


def _monthly_passes(start: str, end: str) -> List[str]:
    d = _dt.date.fromisoformat(start)
    last = _dt.date.fromisoformat(end)
    out = []
    while d <= last:
        out.append(d.isoformat())
        # next month, 15th
        year, month = d.year + (d.month == 12), (d.month % 12) + 1
        d = _dt.date(year, month, 15)
    return out


def synthetic_timeseries(
    zones: Sequence[Zone],
    start: str = "2022-01-15",
    end: str = "2024-12-15",
    indices: Sequence[str] = ("ndwi",),
    seed: int = 7,
    pixel_size_m: float = SYNTHETIC_PIXEL_SIZE_M,
    shape: Tuple[int, int] = (40, 60),
) -> List[ZonePassRecord]:
    """Monthly per-zone water records with planted floods, seeded."""
    records: List[ZonePassRecord] = []
    for zi, zone in enumerate(zones):
        for pi, date in enumerate(_monthly_passes(start, end)):
            level = flood_level(zone.id, date)
            # One deliberately cloudy pass to exercise the cloud path.
            cloudy = zone.id == "wetland-north" and date == "2023-06-15"
            cloud_frac = 0.62 if cloudy else 0.05 + 0.15 * ((zi * 13 + pi * 7) % 10) / 10
            for index in indices:
                raster = synthetic_index_raster(
                    shape, zone.id, level, seed=seed + zi * 1000 + pi, index=index
                )
                clouds = synthetic_cloud_mask(shape, cloud_frac, seed + pi)
                water, valid = build_water_mask(raster, cloud_mask=clouds)
                stats = water_stats(water, valid, pixel_size_m)
                pass_id = f"synthetic-{zone.id}-{date}"
                records.append(
                    ZonePassRecord.from_water_stats(
                        zone.id, zone.name, pass_id, date, index, stats,
                        cloud_cover=round(float(1 - valid.mean()) * 100, 1),
                    )
                )
    return records
