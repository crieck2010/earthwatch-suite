"""Per-zone per-pass LST time-series records and CSV persistence.

One row per (zone, pass): the zonal statistics of land surface temperature
(°C) for a zone on one satellite pass. Series are **appended incrementally**
— re-running with new passes never recomputes old ones, and re-appending
the same records is a no-op (dedupe on zone_id + pass_id).

The CSV columns mirror the sibling analytics engines (survey-vegetation,
survey-flood, survey-burn, survey-coast): ``index`` is ``"lst"`` here, and
the ``mean``/``maximum`` columns are the contract survey-monitor watches —
a monitor site config per zone can threshold them.
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Sequence, Tuple

#: Column order of the time-series CSV. Append-only: new columns go at the end.
TIMESERIES_COLUMNS = [
    "zone_id",
    "zone_name",
    "pass_id",
    "date",
    "doy",
    "year",
    "index",
    "mean",
    "median",
    "std",
    "minimum",
    "maximum",
    "p10",
    "p90",
    "valid_pixels",
    "total_pixels",
    "cloud_cover",
]

_RECORD_KEY = ("zone_id", "pass_id", "index")


@dataclass
class ZonePassRecord:
    """Zonal LST statistics for one zone on one pass (°C)."""

    zone_id: str
    zone_name: str
    pass_id: str
    date: str  # ISO YYYY-MM-DD
    doy: int
    year: int
    index: str  # "lst"
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    p10: float
    p90: float
    valid_pixels: float
    total_pixels: float
    cloud_cover: float = 0.0  # percent 0–100, suite convention

    def key(self) -> Tuple[str, str, str]:
        return (self.zone_id, self.pass_id, self.index)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_record(
    zone_id: str,
    zone_name: str,
    pass_id: str,
    date: str,
    stats: Dict[str, float],
    cloud_cover: float = 0.0,
    index: str = "lst",
) -> ZonePassRecord:
    """Build a record from :func:`zonal.zonal_lst_stats` output."""
    dt = _dt.date.fromisoformat(date)
    return ZonePassRecord(
        zone_id=zone_id,
        zone_name=zone_name,
        pass_id=pass_id,
        date=date,
        doy=dt.timetuple().tm_yday,
        year=dt.year,
        index=index,
        mean=stats["mean"],
        median=stats["median"],
        std=stats["std"],
        minimum=stats["minimum"],
        maximum=stats["maximum"],
        p10=stats["p10"],
        p90=stats["p90"],
        valid_pixels=stats["valid_pixels"],
        total_pixels=stats["total_pixels"],
        cloud_cover=float(cloud_cover),
    )


def _read_existing_keys(path: str) -> set:
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            keys.add((row["zone_id"], row["pass_id"], row["index"]))
    return keys


def append_csv(records: Sequence[ZonePassRecord], path: str) -> Tuple[str, int]:
    """Append records to a time-series CSV, skipping existing keys.

    Returns (path, number of new records written).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    seen = _read_existing_keys(path)
    new = [r for r in records if r.key() not in seen]
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TIMESERIES_COLUMNS)
        if write_header:
            writer.writeheader()
        for record in new:
            d = record.to_dict()
            writer.writerow({c: d.get(c, "") for c in TIMESERIES_COLUMNS})
    return path, len(new)


def read_csv(path: str) -> List[ZonePassRecord]:
    """Read a time-series CSV back into records."""
    records = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            kwargs = {}
            for f in fields(ZonePassRecord):
                raw = row.get(f.name, "")
                if f.type == "str":
                    kwargs[f.name] = raw
                elif f.type == "int":
                    kwargs[f.name] = int(float(raw)) if raw not in ("", None) else 0
                else:
                    kwargs[f.name] = float(raw) if raw not in ("", None) else float("nan")
            records.append(ZonePassRecord(**kwargs))
    return records


def select(
    records: Sequence[ZonePassRecord],
    zone_id: str | None = None,
    index: str = "lst",
) -> List[ZonePassRecord]:
    """Filter records by zone and index, sorted by date."""
    out = [r for r in records if r.index == index]
    if zone_id is not None:
        out = [r for r in out if r.zone_id == zone_id]
    return sorted(out, key=lambda r: r.date)
