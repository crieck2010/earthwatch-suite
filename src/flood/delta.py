"""Flood deltas: per-pass water area vs. a dry-season baseline.

A single pass's water area is hard to interpret — rivers swell and shrink
seasonally. The *delta* is the decision variable: how much more water is
there than the catchment normally holds at this time of year?

The baseline is the same calendar months in prior years (the "dry-season"
window is configurable — it should be the months the catchment is normally
at low water). For each pass we report:

- ``delta_ha`` — absolute change vs. baseline mean (hectares)
- ``rel_change`` — delta as a fraction of the baseline mean (None when the
  baseline mean is zero, i.e. a normally-dry catchment seeing its first water)
- ``z_score`` — delta in baseline standard deviations (None when the
  baseline has fewer than 2 passes or zero variance)

Passes with ``cloud_cover`` above ``max_cloud_cover`` are excluded from both
the baseline and the evaluation — a clouded pass underestimates extent and
would corrupt the comparison.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

from .timeseries import ZonePassRecord

#: Default dry-season window (months) for the baseline. Tune per catchment:
#: pick the months the zone is normally at low water.
DEFAULT_BASELINE_MONTHS = (6, 7, 8)


@dataclass
class FloodDelta:
    """Flood delta of one zone-pass against its dry-season baseline."""

    zone_id: str
    zone_name: str
    pass_id: str
    date: str
    index: str
    water_ha: float
    baseline_mean_ha: float | None
    baseline_std_ha: float | None
    baseline_passes: int
    delta_ha: float | None
    rel_change: float | None
    z_score: float | None
    flooded: bool  # delta_ha > flood_threshold_ha (and baseline exists)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DELTA_COLUMNS = [
    "zone_id",
    "zone_name",
    "pass_id",
    "date",
    "index",
    "water_ha",
    "baseline_mean_ha",
    "baseline_std_ha",
    "baseline_passes",
    "delta_ha",
    "rel_change",
    "z_score",
    "flooded",
]


def _month(date_iso: str) -> int:
    return int(date_iso[5:7])


def baseline_records(
    records: Sequence[ZonePassRecord],
    zone_id: str,
    index: str,
    pass_date: str,
    baseline_months: Sequence[int] = DEFAULT_BASELINE_MONTHS,
    baseline_years: int = 1,
    max_cloud_cover: float = 40.0,
) -> List[ZonePassRecord]:
    """Prior-year records in the baseline months for one zone/index.

    Only passes strictly *before* ``pass_date`` are eligible, and only in
    ``baseline_months`` of the ``baseline_years`` years preceding the pass
    year. Clouded passes are excluded.
    """
    year = int(pass_date[:4])
    eligible_years = {year - k for k in range(1, baseline_years + 1)}
    months = set(baseline_months)
    out = []
    for r in records:
        if r.zone_id != zone_id or r.index != index:
            continue
        if r.date >= pass_date:
            continue
        if r.year not in eligible_years or _month(r.date) not in months:
            continue
        if r.cloud_cover is not None and r.cloud_cover > max_cloud_cover:
            continue
        out.append(r)
    return sorted(out, key=lambda r: r.date)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / (n - 1)) ** 0.5


def compute_delta(
    record: ZonePassRecord,
    records: Sequence[ZonePassRecord],
    baseline_months: Sequence[int] = DEFAULT_BASELINE_MONTHS,
    baseline_years: int = 1,
    flood_threshold_ha: float = 0.0,
    max_cloud_cover: float = 40.0,
) -> FloodDelta:
    """Flood delta of one pass record against its baseline."""
    base = baseline_records(
        records, record.zone_id, record.index, record.date,
        baseline_months, baseline_years, max_cloud_cover,
    )
    n = len(base)
    if n == 0:
        return FloodDelta(
            zone_id=record.zone_id, zone_name=record.zone_name,
            pass_id=record.pass_id, date=record.date, index=record.index,
            water_ha=record.water_ha, baseline_mean_ha=None,
            baseline_std_ha=None, baseline_passes=0, delta_ha=None,
            rel_change=None, z_score=None, flooded=False,
        )
    vals = [r.water_ha for r in base]
    mean = _mean(vals)
    std = _std(vals)
    delta = record.water_ha - mean
    rel = delta / mean if mean > 0 else None
    z = delta / std if std > 0 else None
    return FloodDelta(
        zone_id=record.zone_id, zone_name=record.zone_name,
        pass_id=record.pass_id, date=record.date, index=record.index,
        water_ha=record.water_ha, baseline_mean_ha=mean,
        baseline_std_ha=std if n >= 2 else None, baseline_passes=n,
        delta_ha=delta, rel_change=rel, z_score=z,
        flooded=delta > flood_threshold_ha,
    )


def compute_all_deltas(
    records: Sequence[ZonePassRecord],
    index: str,
    baseline_months: Sequence[int] = DEFAULT_BASELINE_MONTHS,
    baseline_years: int = 1,
    flood_threshold_ha: float = 0.0,
    max_cloud_cover: float = 40.0,
) -> List[FloodDelta]:
    """Deltas for every pass of one index, sorted by date."""
    selected = [r for r in records if r.index == index]
    selected.sort(key=lambda r: (r.zone_id, r.date))
    return [
        compute_delta(
            r, records, baseline_months, baseline_years,
            flood_threshold_ha, max_cloud_cover,
        )
        for r in selected
    ]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def write_deltas_csv(deltas: Sequence[FloodDelta], path: str) -> str:
    """Write deltas to CSV (overwrite). Returns the path."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DELTA_COLUMNS)
        writer.writeheader()
        for d in deltas:
            row = d.to_dict()
            writer.writerow({c: _fmt(row.get(c)) for c in DELTA_COLUMNS})
    return path


def write_deltas_json(deltas: Sequence[FloodDelta], path: str) -> str:
    """Write deltas to JSON (overwrite). Returns the path."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([d.to_dict() for d in deltas], fh, indent=2)
    return path
