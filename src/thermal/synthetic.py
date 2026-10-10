"""Seeded synthetic data for demos and tests — no network, no data files.

The synthetic world has two zones over 2022–2024 (monthly passes):

- ``downtown`` (urban): +4.5 °C mean offset vs. the rural reference —
  the planted UHI the demo recovers.
- ``meadow`` (rural): the reference zone.

A planted heatwave hits ``downtown`` in summer 2024: three consecutive
passes with max LST ≥ 38 °C. The 2024-07-15 downtown pass is also a hot
anomaly (z ≥ 2 vs. prior-year Julys) once 2022–2023 history exists.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

from .timeseries import ZonePassRecord, make_record
from .zones import Zone

#: Planted urban heat-island intensity (°C).
PLANTED_UHI_C = 4.5

#: Planted heatwave: (date, max_c) for downtown, three consecutive passes.
PLANTED_HEATWAVE = [
    ("2024-06-15", 38.5),
    ("2024-07-15", 40.1),
    ("2024-08-15", 39.2),
]

#: 30 m pixels — honest to Landsat thermal's resampled resolution.
PIXEL_M = 30.0


def synthetic_zones() -> List[Zone]:
    """Two rectangle zones: urban downtown and rural meadow."""
    urban = Zone(
        id="downtown",
        name="Downtown district",
        kind="urban",
        geometry={
            "type": "Polygon",
            "coordinates": [[[0, 0], [300, 0], [300, 300], [0, 300], [0, 0]]],
        },
    )
    rural = Zone(
        id="meadow",
        name="Meadow reference",
        kind="rural",
        geometry={
            "type": "Polygon",
            "coordinates": [[[600, 0], [900, 0], [900, 300], [600, 300], [600, 0]]],
        },
    )
    return [urban, rural]


def _seasonal_mean_c(year: int, month: int) -> float:
    """Rural baseline mean LST: annual cycle peaking in mid-July."""
    doy = (month - 1) * 30 + 15
    return 15.0 + 12.0 * math.cos(2 * math.pi * (doy - 200) / 365.0)


def synthetic_timeseries(
    zones: List[Zone],
    start: str = "2022-01-15",
    end: str = "2024-12-15",
    seed: int = 7,
) -> List[ZonePassRecord]:
    """Seeded per-zone per-pass LST records (deterministic for a seed)."""
    rng = random.Random(seed)
    heatwave_max = dict(PLANTED_HEATWAVE)
    records: List[ZonePassRecord] = []
    y0, m0 = int(start[:4]), int(start[5:7])
    y1, m1 = int(end[:4]), int(end[5:7])
    months = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    for zone in zones:
        for (y, m) in months:
            date = f"{y:04d}-{m:02d}-15"
            base = _seasonal_mean_c(y, m)
            urban_offset = PLANTED_UHI_C if zone.kind == "urban" else 0.0
            mean_c = base + urban_offset + rng.uniform(-1.0, 1.0)
            hw_max = heatwave_max.get(date) if zone.kind == "urban" else None
            if hw_max is not None:
                mean_c = max(mean_c, hw_max - 5.5)  # heatwave lifts the mean too
                max_c = hw_max
            else:
                max_c = mean_c + 5.0 + rng.uniform(0.0, 2.0)
            stats = {
                "mean": round(mean_c, 2),
                "median": round(mean_c - 0.4 + rng.uniform(-0.3, 0.3), 2),
                "std": round(2.2 + rng.uniform(0.0, 0.8), 2),
                "minimum": round(mean_c - 6.0 + rng.uniform(-1.0, 1.0), 2),
                "maximum": round(max_c, 2),
                "p10": round(mean_c - 3.5, 2),
                "p90": round(mean_c + 3.5, 2),
                "valid_pixels": 100.0,
                "total_pixels": 100.0,
            }
            records.append(
                make_record(
                    zone_id=zone.id,
                    zone_name=zone.name,
                    pass_id=f"synthetic-{zone.id}-{date}",
                    date=date,
                    stats=stats,
                    cloud_cover=round(rng.uniform(0.0, 15.0), 1),
                )
            )
    return records


def synthetic_lst_raster(
    shape: Tuple[int, int] = (20, 40),
    seed: int = 7,
) -> Tuple["object", Tuple[float, float, float, float, float, float]]:
    """One synthetic LST raster (°C): cool rural west, hot urban east.

    Returns (array, transform) with 30 m north-up pixels. The east half
    runs ~4.5 °C hotter — the planted UHI in raster form.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    rows, cols = shape
    base = 24.0 + rng.normal(0, 0.8, shape)
    hotspot = np.zeros(shape)
    hotspot[:, cols // 2 :] = PLANTED_UHI_C + rng.normal(0, 0.5, (rows, cols - cols // 2))
    arr = base + hotspot
    transform = (PIXEL_M, 0.0, 0.0, 0.0, -PIXEL_M, rows * PIXEL_M)
    return arr, transform


def demo_report(out_dir: str, seed: int = 7) -> Dict[str, str]:
    """Run the full pipeline on synthetic data; return artifact paths."""
    import os

    from . import anomalies as an
    from . import heatwave as hw
    from . import timeseries as ts
    from . import uhi

    os.makedirs(out_dir, exist_ok=True)
    zones = synthetic_zones()
    records = synthetic_timeseries(zones, seed=seed)
    ts_path, n_ts = ts.append_csv(records, os.path.join(out_dir, "lst_timeseries.csv"))

    uhi_records = uhi.compute_uhi(records)
    uhi_path = uhi.write_uhi_csv(uhi_records, os.path.join(out_dir, "uhi.csv"))

    anom = an.detect_anomalies(records, field="mean")
    anom_path = an.write_anomalies_csv(anom, os.path.join(out_dir, "anomalies.csv"))

    events = hw.detect_heatwaves(records)
    hw_path = hw.write_heatwaves_csv(events, os.path.join(out_dir, "heatwaves.csv"))

    return {
        "timeseries": ts_path,
        "uhi": uhi_path,
        "anomalies": anom_path,
        "heatwaves": hw_path,
        "n_timeseries": str(n_ts),
        "n_uhi": str(len(uhi_records)),
        "n_hot": str(sum(1 for a in anom if a.hot)),
        "n_heatwaves": str(len(events)),
    }
