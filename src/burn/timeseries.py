"""Per-zone burned-area time-series records and CSV persistence.

One row per (zone, pass, index): the burn/disturbance state of a zone on
one satellite pass. Series are **appended incrementally** — re-running with
new passes never recomputes old ones, and re-appending the same records is
a no-op (dedupe on zone_id + pass_id + index).

The CSV columns are the contract survey-monitor watches: a monitor site
config per zone can threshold the ``burned_ha`` or ``burned_fraction``
column. The shared column *names* mirror survey-vegetation's and
survey-flood's time-series schemas where the concepts overlap
(zone_id, zone_name, pass_id, date, doy, year, index, valid_pixels,
total_pixels, pixel_size_m, cloud_cover) so the three engines' CSVs feel
like one family.
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
    "valid_pixels",
    "total_pixels",
    "burned_pixels",
    "burned_fraction",
    "burned_ha",
    "unburned_ha",
    "low_severity_ha",
    "moderate_low_severity_ha",
    "moderate_high_severity_ha",
    "high_severity_ha",
    "enhanced_regrowth_ha",
    "pixel_size_m",
    "cloud_cover",
]

_RECORD_KEY = ("zone_id", "pass_id", "index")


@dataclass
class ZonePassRecord:
    """Burned area for one zone on one pass."""

    zone_id: str
    zone_name: str
    pass_id: str
    date: str  # ISO YYYY-MM-DD
    doy: int
    year: int
    index: str  # "dnbr" | "disturbance"
    valid_pixels: int
    total_pixels: int
    burned_pixels: int
    burned_fraction: float
    burned_ha: float
    unburned_ha: float = 0.0
    low_severity_ha: float = 0.0
    moderate_low_severity_ha: float = 0.0
    moderate_high_severity_ha: float = 0.0
    high_severity_ha: float = 0.0
    enhanced_regrowth_ha: float = 0.0
    pixel_size_m: float = 10.0
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
            ftype = f.type
            if isinstance(ftype, str):
                if ftype == "int":
                    kwargs[f.name] = int(float(raw))
                elif ftype in ("float", "float | None"):
                    kwargs[f.name] = float(raw)
                else:
                    kwargs[f.name] = raw
            else:
                kwargs[f.name] = raw
        return cls(**kwargs)

    def key(self) -> Tuple[str, str, str]:
        return (self.zone_id, self.pass_id, self.index)

    @classmethod
    def from_class_areas(
        cls,
        zone_id: str,
        zone_name: str,
        pass_id: str,
        date: str,
        index: str,
        areas: Dict[str, float],
        total_pixels: int,
        pixel_size_m: float,
        cloud_cover: float | None = None,
    ) -> "ZonePassRecord":
        """Build from :func:`burn.severity.class_areas` output."""
        d = _dt.date.fromisoformat(date[:10])
        valid = int(areas.get("valid_pixels", 0))
        return cls(
            zone_id=zone_id,
            zone_name=zone_name,
            pass_id=pass_id,
            date=d.isoformat(),
            doy=int(d.strftime("%j")),
            year=d.year,
            index=index,
            valid_pixels=valid,
            total_pixels=int(total_pixels),
            burned_pixels=int(round(areas.get("burned_ha", 0.0) * 10_000.0 / (pixel_size_m ** 2))),
            burned_fraction=float(areas.get("burned_fraction", 0.0)),
            burned_ha=float(areas.get("burned_ha", 0.0)),
            unburned_ha=float(areas.get("unburned_ha", 0.0)),
            low_severity_ha=float(areas.get("low_severity_ha", 0.0)),
            moderate_low_severity_ha=float(areas.get("moderate_low_severity_ha", 0.0)),
            moderate_high_severity_ha=float(areas.get("moderate_high_severity_ha", 0.0)),
            high_severity_ha=float(areas.get("high_severity_ha", 0.0)),
            enhanced_regrowth_ha=float(areas.get("enhanced_regrowth_ha", 0.0)),
            pixel_size_m=float(pixel_size_m),
            cloud_cover=cloud_cover,
        )


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _existing_keys(path: str) -> set:
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            keys.add((row.get("zone_id"), row.get("pass_id"), row.get("index")))
    return keys


def append_records(path: str, records: Sequence[ZonePassRecord]) -> int:
    """Append records to the CSV, skipping ones already present.

    Returns the number of rows actually written. Creates parent dirs.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    seen = _existing_keys(path)
    new = [r for r in records if r.key() not in seen]
    if not new:
        return 0
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TIMESERIES_COLUMNS)
        if write_header:
            writer.writeheader()
        for r in new:
            d = r.to_dict()
            writer.writerow({c: _fmt(d.get(c)) for c in TIMESERIES_COLUMNS})
    return len(new)


def read_records(path: str) -> List[ZonePassRecord]:
    """Read all records from a time-series CSV."""
    records: List[ZonePassRecord] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(ZonePassRecord.from_dict(row))
    return records
