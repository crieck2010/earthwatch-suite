"""Per-zone per-pass water-area time-series records and CSV persistence.

One row per (zone, pass, index): the water extent of a zone on one
satellite pass. Series are **appended incrementally** — re-running with new
passes never recomputes old ones, and re-appending the same records is a
no-op (dedupe on zone_id + pass_id + index).

The CSV columns are the contract survey-monitor watches: a monitor site
config per zone can threshold the ``water_ha`` or ``water_fraction`` column
of any index row. The column *names* mirror survey-vegetation's time-series
schema where the concepts overlap (zone_id, pass_id, date, doy, year,
index, valid_pixels, total_pixels, cloud_cover) so the two engines' CSVs
feel like one family.
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
    "water_pixels",
    "valid_pixels",
    "total_pixels",
    "water_fraction",
    "water_ha",
    "pixel_size_m",
    "cloud_cover",
]

_RECORD_KEY = ("zone_id", "pass_id", "index")


@dataclass
class ZonePassRecord:
    """Water extent for one zone on one pass."""

    zone_id: str
    zone_name: str
    pass_id: str
    date: str  # ISO YYYY-MM-DD
    doy: int
    year: int
    index: str
    water_pixels: int
    valid_pixels: int
    total_pixels: int
    water_fraction: float
    water_ha: float
    pixel_size_m: float
    cloud_cover: float | None = None  # percent 0-100, suite convention

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ZonePassRecord":
        """Rebuild from :meth:`to_dict` output (CSV rows come back as strings)."""
        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            raw = data[f.name]
            if raw is None or raw == "":
                kwargs[f.name] = None if f.name == "cloud_cover" else raw
                continue
            if f.type in ("int",):
                kwargs[f.name] = int(float(raw))
            elif f.type in ("float", "float | None"):
                kwargs[f.name] = float(raw)
            else:
                kwargs[f.name] = raw
        return cls(**kwargs)

    def key(self) -> Tuple[str, str, str]:
        return (self.zone_id, self.pass_id, self.index)

    @classmethod
    def from_water_stats(
        cls,
        zone_id: str,
        zone_name: str,
        pass_id: str,
        date: str,
        index: str,
        stats: Dict[str, float],
        cloud_cover: float | None = None,
    ) -> "ZonePassRecord":
        d = _dt.date.fromisoformat(date[:10])
        return cls(
            zone_id=zone_id,
            zone_name=zone_name,
            pass_id=pass_id,
            date=d.isoformat(),
            doy=int(d.strftime("%j")),
            year=d.year,
            index=index,
            water_pixels=int(stats["water_pixels"]),
            valid_pixels=int(stats["valid_pixels"]),
            total_pixels=int(stats["total_pixels"]),
            water_fraction=float(stats["water_fraction"]),
            water_ha=float(stats["water_ha"]),
            pixel_size_m=float(stats["pixel_size_m"]),
            cloud_cover=cloud_cover,
        )


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def write_csv(records: Sequence[ZonePassRecord], path: str) -> str:
    """Write records to CSV (overwrite). Returns the path."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TIMESERIES_COLUMNS)
        writer.writeheader()
        for rec in records:
            row = rec.to_dict()
            writer.writerow({c: _fmt(row.get(c)) for c in TIMESERIES_COLUMNS})
    return path


def read_csv(path: str) -> List[ZonePassRecord]:
    """Read records from a CSV written by :func:`write_csv`."""
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in TIMESERIES_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"time-series CSV missing columns: {missing}")
        return [ZonePassRecord.from_dict(row) for row in reader]


def append_csv(records: Sequence[ZonePassRecord], path: str) -> Tuple[str, int]:
    """Append records, skipping ones already present (by zone/pass/index).

    Returns (path, number_of_new_records). Creates the file if missing.
    """
    existing: List[ZonePassRecord] = []
    if os.path.exists(path):
        existing = read_csv(path)
    seen = {r.key() for r in existing}
    new = [r for r in records if r.key() not in seen]
    write_csv(existing + new, path)
    return path, len(new)


def select(
    records: Sequence[ZonePassRecord],
    zone_id: str | None = None,
    index: str | None = None,
    year: int | None = None,
) -> List[ZonePassRecord]:
    """Filter records by zone/index/year, sorted by date."""
    out = [
        r
        for r in records
        if (zone_id is None or r.zone_id == zone_id)
        and (index is None or r.index == index)
        and (year is None or r.year == year)
    ]
    return sorted(out, key=lambda r: r.date)
