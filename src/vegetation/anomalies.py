"""Phenology-aware anomaly detection.

A June NDVI of 0.45 means something very different from a January NDVI of
0.45, so anomalies are judged against the zone's own history for the *same
day-of-year window* — June is compared to Junes, not to January. For each
(zone, index, pass), the baseline is prior years' values within
``window_days`` of the pass's day-of-year; the anomaly is the z-score of the
current value against that baseline. Negative z beyond ``z_threshold`` flags
a stressed zone.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

from .timeseries import ZonePassRecord, select

ANOMALY_COLUMNS = [
    "zone_id",
    "zone_name",
    "index",
    "date",
    "doy",
    "year",
    "value",
    "baseline_mean",
    "baseline_std",
    "baseline_n",
    "baseline_years",
    "z_score",
    "stressed",
    "window_days",
]


@dataclass
class AnomalyRecord:
    zone_id: str
    zone_name: str
    index: str
    date: str
    doy: int
    year: int
    value: float
    baseline_mean: float | None
    baseline_std: float | None
    baseline_n: int
    baseline_years: int
    z_score: float | None
    stressed: bool
    window_days: int
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _doy_distance(a: int, b: int) -> int:
    """Circular day-of-year distance (Dec 30 is 3 days from Jan 2)."""
    return min(abs(a - b), 365 - abs(a - b))


def _z_score(value: float, mean: float, std: float) -> float:
    if not math.isfinite(value) or not math.isfinite(mean):
        return float("nan")
    if std == 0:
        if abs(value - mean) < 1e-12:
            return 0.0
        return 999.0 if value > mean else -999.0
    return (value - mean) / std


def detect_anomalies(
    records: Sequence[ZonePassRecord],
    index: str,
    window_days: int = 15,
    z_threshold: float = -2.0,
    min_years: int = 1,
    zone_id: str | None = None,
) -> List[AnomalyRecord]:
    """Phenology-aware z-score anomalies for one index.

    Baseline for each pass: same zone, same index, *strictly prior* years,
    day-of-year within ``window_days``. ``z_threshold`` is the (negative)
    z-score at or below which a zone is flagged stressed. Passes without
    enough history get ``z_score=None`` and a note instead of a verdict.
    """
    recs = select(records, zone_id=zone_id, index=index)
    by_zone: Dict[str, List[ZonePassRecord]] = {}
    for r in recs:
        by_zone.setdefault(r.zone_id, []).append(r)

    out: List[AnomalyRecord] = []
    for zid, zrecs in sorted(by_zone.items()):
        for target in zrecs:
            base = [
                r
                for r in zrecs
                if r.year < target.year
                and _doy_distance(r.doy, target.doy) <= window_days
                and math.isfinite(r.mean)
            ]
            years = len({r.year for r in base})
            vals = [r.mean for r in base]
            note = ""
            mean = std = z = None
            stressed = False
            if years >= min_years and len(vals) >= 2:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                std = math.sqrt(var)
                z = _z_score(target.mean, mean, std)
                stressed = math.isfinite(z) and z <= z_threshold
            elif years >= min_years:
                note = "baseline has <2 samples; z-score undefined"
            else:
                note = (
                    f"needs >= {min_years} prior year(s) of history, "
                    f"has {years}"
                )
            out.append(
                AnomalyRecord(
                    zone_id=target.zone_id,
                    zone_name=target.zone_name,
                    index=index,
                    date=target.date,
                    doy=target.doy,
                    year=target.year,
                    value=target.mean,
                    baseline_mean=mean,
                    baseline_std=std,
                    baseline_n=len(vals),
                    baseline_years=years,
                    z_score=z,
                    stressed=stressed,
                    window_days=window_days,
                    note=note,
                )
            )
    return sorted(out, key=lambda r: (r.zone_id, r.date))


def write_anomalies_csv(records: Sequence[AnomalyRecord], path: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ANOMALY_COLUMNS)
        writer.writeheader()
        for rec in records:
            row = rec.to_dict()
            writer.writerow(
                {
                    c: ("" if row.get(c) is None else str(row.get(c)))
                    for c in ANOMALY_COLUMNS
                }
            )
    return path
