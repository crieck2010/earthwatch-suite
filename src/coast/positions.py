"""Per-transect shoreline-position time series and CSV persistence.

One row per (transect, pass): where the shoreline crossed the transect on
one satellite pass. Series are **appended incrementally** — re-running with
new passes never recomputes old ones, and re-appending the same records is
a no-op (dedupe on transect_id + pass_id).

Shared column names mirror the sibling engines' time-series schemas where
the concepts overlap (``*_id``, ``pass_id``, ``date``, ``doy``, ``year``,
``index``, ``cloud_cover``, ``pixel_size_m``) so the analytics engines'
CSVs feel like one family. See ``docs/POSITIONS_SCHEMA.md``.
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Sequence, Tuple

#: Column order of the positions CSV. Append-only: new columns go at the end.
POSITIONS_COLUMNS = [
    "transect_id",
    "transect_name",
    "site_id",
    "site_name",
    "pass_id",
    "date",
    "doy",
    "year",
    "index",
    "position_m",
    "shoreline_length_m",
    "valid",
    "cloud_cover",
    "pixel_size_m",
]

_RECORD_KEY = ("transect_id", "pass_id")


@dataclass
class TransectPassRecord:
    """Shoreline position for one transect on one pass."""

    transect_id: str
    transect_name: str
    site_id: str
    site_name: str
    pass_id: str
    date: str  # ISO YYYY-MM-DD
    doy: int
    year: int
    index: str  # water index used, e.g. "ndwi"
    position_m: float  # NaN when no crossing
    shoreline_length_m: float
    valid: bool
    cloud_cover: float  # percent 0-100
    pixel_size_m: float


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return repr(value)
    return str(value)


def _parse(row: Dict[str, str]) -> TransectPassRecord:
    def f(key: str) -> float:
        text = (row.get(key) or "").strip()
        return float(text) if text else float("nan")

    return TransectPassRecord(
        transect_id=row["transect_id"],
        transect_name=row.get("transect_name", ""),
        site_id=row.get("site_id", ""),
        site_name=row.get("site_name", ""),
        pass_id=row["pass_id"],
        date=row["date"],
        doy=int(row["doy"]) if row.get("doy") else 0,
        year=int(row["year"]) if row.get("year") else 0,
        index=row.get("index", ""),
        position_m=f("position_m"),
        shoreline_length_m=f("shoreline_length_m"),
        valid=(row.get("valid", "").strip().lower() == "true"),
        cloud_cover=f("cloud_cover"),
        pixel_size_m=f("pixel_size_m"),
    )


def read_positions(path: str) -> List[TransectPassRecord]:
    """Read a positions CSV (empty list when the file is missing)."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [_parse(row) for row in reader]


def append_records(
    path: str, records: Sequence[TransectPassRecord]
) -> Tuple[int, int]:
    """Append records, skipping (transect_id, pass_id) already present.

    Returns (added, skipped).
    """
    existing = {
        (r.transect_id, r.pass_id) for r in read_positions(path)
    }
    new_rows: List[Dict[str, str]] = []
    skipped = 0
    for rec in records:
        key = (rec.transect_id, rec.pass_id)
        if key in existing:
            skipped += 1
            continue
        existing.add(key)
        new_rows.append({c: _fmt(getattr(rec, c)) for c in POSITIONS_COLUMNS})
    if new_rows:
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=POSITIONS_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerows(new_rows)
    return len(new_rows), skipped
