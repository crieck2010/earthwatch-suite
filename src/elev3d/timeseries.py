"""Per-zone volumetric time-series records and CSV persistence.

One row per (zone, DEM pair): the cut/fill volumetrics of a zone between
two elevation models. Series are **appended incrementally** — re-running
with new DEM pairs never recomputes old ones, and re-appending the same
records is a no-op (dedupe on zone_id + pass_id + index).

The CSV columns mirror the sibling analytics engines (survey-vegetation,
survey-flood, survey-burn, survey-coast, survey-thermal): ``index`` is
``"dem-diff"`` here. The ``mean``/``maximum`` columns are the contract
survey-monitor watches — a monitor site config per zone can threshold
mean Δh or max fill height — and ``net_m3``/``cut_m3``/``fill_m3`` are
documented for monitor metric configs that read them directly.
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Tuple

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
    # survey-3d extensions (volumetrics):
    "cut_m3",
    "fill_m3",
    "net_m3",
    "max_cut_depth",
    "max_fill_height",
    "area_m2",
    "volume_sigma_m3",
    "confidence",
]

_RECORD_KEY = ("zone_id", "pass_id", "index")


@dataclass
class ZoneVolumeRecord:
    """Volumetrics for one zone between one DEM pair."""

    zone_id: str
    zone_name: str
    pass_id: str  # DEM-pair id, e.g. "dem-2024_vs_dem-2025"
    date: str  # ISO YYYY-MM-DD of the *later* DEM
    doy: int
    year: int
    index: str  # "dem-diff"
    mean: float = 0.0  # mean Δh (m) — the survey-monitor contract column
    median: float = 0.0
    std: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0  # max Δh (m) — the survey-monitor contract column
    p10: float = 0.0
    p90: float = 0.0
    valid_pixels: float = 0.0
    total_pixels: float = 0.0
    cloud_cover: float = 0.0  # percent 0–100, suite convention (DEMs: usually 0)
    cut_m3: float = 0.0
    fill_m3: float = 0.0
    net_m3: float = 0.0
    max_cut_depth: float = 0.0
    max_fill_height: float = 0.0
    area_m2: float = 0.0
    volume_sigma_m3: float = float("nan")
    confidence: str = "low"

    def key(self) -> Tuple[str, str, str]:
        return (self.zone_id, self.pass_id, self.index)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_record(
    zone_id: str,
    zone_name: str,
    pass_id: str,
    date: str,
    zonal: Dict[str, float],
    volumes: Dict[str, Any],
) -> ZoneVolumeRecord:
    """Build a record from zonal Δh stats and volumetrics.

    ``zonal`` keys: mean, median, std, minimum, maximum, p10, p90,
    valid_pixels, total_pixels, cloud_cover. ``volumes`` is
    ``ZoneVolumes.to_dict()`` (see volumes.py).
    """
    import datetime as _dt

    dt = _dt.date.fromisoformat(date)
    field_names = {f.name for f in fields(ZoneVolumeRecord)}
    kwargs: Dict[str, Any] = {
        "zone_id": zone_id,
        "zone_name": zone_name,
        "pass_id": pass_id,
        "date": date,
        "doy": dt.timetuple().tm_yday,
        "year": dt.year,
        "index": "dem-diff",
    }
    for k, v in zonal.items():
        if k in field_names:
            kwargs[k] = v
    for k, v in volumes.items():
        if k in field_names and k not in ("zone_id", "zone_name"):
            kwargs[k] = v if v is not None else float("nan")
    return ZoneVolumeRecord(**kwargs)


def _read_existing(path: str) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    existing: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    if not os.path.exists(path):
        return existing
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = tuple(row.get(k, "") for k in _RECORD_KEY)  # type: ignore[misc]
            existing[key] = row
    return existing


def append_records(path: str, records: List[ZoneVolumeRecord]) -> int:
    """Append records to a CSV, skipping keys already present.

    Returns the number of records actually appended. Creates parent
    directories as needed. Never rewrites existing rows.
    """
    existing = _read_existing(path)
    new = [r for r in records if r.key() not in existing]
    if not new:
        return 0
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TIMESERIES_COLUMNS)
        if write_header:
            writer.writeheader()
        for r in new:
            d = r.to_dict()
            writer.writerow({k: d.get(k, "") for k in TIMESERIES_COLUMNS})
    return len(new)


def read_records(path: str) -> List[Dict[str, str]]:
    """Read all rows of a time-series CSV as dicts."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
