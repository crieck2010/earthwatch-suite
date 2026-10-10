"""Phenology-style heat-anomaly detection.

A July LST of 34 °C means something very different from a January LST of
34 °C, so anomalies are judged against the zone's own history for the *same
day-of-year window* — July is compared to Julys, not to January. This
mirrors :mod:`vegetation.anomalies` (the sibling engine); here the flagged
direction is *hot* (positive z beyond ``z_threshold``).

For each (zone, pass), the baseline is prior years' mean-LST values within
``window_days`` of the pass's day-of-year; the anomaly is the z-score of the
current value against that baseline.
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
    "hot",
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
    hot: bool
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
        return 0.0 if value == mean else float("inf") * (1 if value > mean else -1)
    return (value - mean) / std


def detect_anomalies(
    records: Sequence[ZonePassRecord],
    index: str = "lst",
    window_days: int = 15,
    z_threshold: float = 2.0,
    min_years: int = 1,
    field: str = "mean",
) -> List[AnomalyRecord]:
    """Flag hot passes: z of ``field`` vs. prior-year same-doy baseline.

    ``field`` is ``"mean"`` or ``"maximum"``. Only *prior* years form the
    baseline (no peeking at the current year), and a pass needs at least
    ``min_years`` distinct baseline years to get a z-score.
    """
    zone_ids = sorted({r.zone_id for r in records if r.index == index})
    out: List[AnomalyRecord] = []
    for zid in zone_ids:
        series = select(records, zone_id=zid, index=index)
        zone_name = series[0].zone_name if series else zid
        for rec in series:
            value = getattr(rec, field)
            baseline_vals = [
                getattr(o, field)
                for o in series
                if o.year < rec.year
                and _doy_distance(o.doy, rec.doy) <= window_days
                and math.isfinite(getattr(o, field))
            ]
            years = sorted({o.year for o in series if o.year < rec.year
                            and _doy_distance(o.doy, rec.doy) <= window_days})
            if len(years) >= min_years and len(baseline_vals) >= 2:
                mean = sum(baseline_vals) / len(baseline_vals)
                var = sum((v - mean) ** 2 for v in baseline_vals) / len(baseline_vals)
                std = math.sqrt(var)
                z = _z_score(value, mean, std)
                hot = math.isfinite(z) and z >= z_threshold
                out.append(AnomalyRecord(
                    zone_id=zid, zone_name=zone_name, index=index,
                    date=rec.date, doy=rec.doy, year=rec.year,
                    value=value, baseline_mean=mean, baseline_std=std,
                    baseline_n=len(baseline_vals), baseline_years=len(years),
                    z_score=z, hot=hot, window_days=window_days,
                ))
            else:
                out.append(AnomalyRecord(
                    zone_id=zid, zone_name=zone_name, index=index,
                    date=rec.date, doy=rec.doy, year=rec.year,
                    value=value, baseline_mean=None, baseline_std=None,
                    baseline_n=len(baseline_vals), baseline_years=len(years),
                    z_score=None, hot=False, window_days=window_days,
                    note="insufficient baseline history",
                ))
    return out


def write_anomalies_csv(records: Sequence[AnomalyRecord], path: str) -> str:
    """Write anomaly records to CSV. Returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ANOMALY_COLUMNS)
        writer.writeheader()
        for record in records:
            d = record.to_dict()
            writer.writerow({c: d.get(c, "") for c in ANOMALY_COLUMNS})
    return path
