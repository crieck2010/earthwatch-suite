"""Urban heat-island (UHI) intensity per zone per pass.

UHI intensity is the zone's LST minus a *rural reference* zone's LST on the
same pass, in °C. The reference can be user-supplied (``--reference``) or
auto-selected as the coolest zone by mean LST over the series — documented
below with its sensitivity.

Reference-zone methodology and sensitivity
-----------------------------------------
Auto-selection picks the zone with the lowest mean LST across the whole
time series. This is a heuristic: it assumes the coolest zone is
representative rural background. It is sensitive to (a) cloud-contaminated
passes dragging a zone's mean down, (b) water bodies in a zone (water is
cool but not "rural land"), and (c) series that don't overlap in time —
zones are compared pass-by-pass, so a reference is only valid on passes
where it has a record. For defensible reporting, supply an explicit
reference zone you have vetted (vegetated, cloud-free, same overpass
conditions) rather than relying on auto-selection.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

from .temperature import lst_anomaly_c
from .timeseries import ZonePassRecord, select

UHI_COLUMNS = [
    "zone_id",
    "zone_name",
    "reference_id",
    "reference_name",
    "date",
    "doy",
    "year",
    "zone_mean_c",
    "reference_mean_c",
    "uhi_c",
    "cloud_cover",
]


@dataclass
class UhiRecord:
    zone_id: str
    zone_name: str
    reference_id: str
    reference_name: str
    date: str
    doy: int
    year: int
    zone_mean_c: float
    reference_mean_c: float
    uhi_c: float
    cloud_cover: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def auto_reference(records: Sequence[ZonePassRecord], index: str = "lst") -> str:
    """Pick the coolest zone by mean LST as the rural reference.

    Zones with fewer than 3 finite records are ignored so a single
    cloud-contaminated pass can't win the contest.
    """
    zone_ids = sorted({r.zone_id for r in records if r.index == index})
    best_id, best_mean = "", float("inf")
    for zid in zone_ids:
        vals = [r.mean for r in select(records, zone_id=zid, index=index)
                if math.isfinite(r.mean)]
        if len(vals) >= 3:
            mean = sum(vals) / len(vals)
            if mean < best_mean:
                best_mean, best_id = mean, zid
    if not best_id:
        raise ValueError("no zone with >= 3 finite records to auto-select a reference")
    return best_id


def compute_uhi(
    records: Sequence[ZonePassRecord],
    reference_id: str | None = None,
    index: str = "lst",
) -> List[UhiRecord]:
    """Per-zone per-pass UHI intensity vs. the reference zone (°C).

    Only passes where *both* the zone and the reference have finite means
    produce a record. The reference zone itself is skipped (its UHI is
    trivially 0).
    """
    ref_id = reference_id or auto_reference(records, index)
    ref_series = {r.date: r for r in select(records, zone_id=ref_id, index=index)}
    if not ref_series:
        raise ValueError(f"reference zone {ref_id!r} has no {index} records")
    ref_name = next(iter(ref_series.values())).zone_name
    out: List[UhiRecord] = []
    for zid in sorted({r.zone_id for r in records if r.index == index}):
        if zid == ref_id:
            continue
        for rec in select(records, zone_id=zid, index=index):
            ref = ref_series.get(rec.date)
            if ref is None:
                continue
            if not (math.isfinite(rec.mean) and math.isfinite(ref.mean)):
                continue
            out.append(UhiRecord(
                zone_id=zid,
                zone_name=rec.zone_name,
                reference_id=ref_id,
                reference_name=ref_name,
                date=rec.date,
                doy=rec.doy,
                year=rec.year,
                zone_mean_c=rec.mean,
                reference_mean_c=ref.mean,
                uhi_c=lst_anomaly_c(rec.mean, ref.mean),
                cloud_cover=max(rec.cloud_cover, ref.cloud_cover),
            ))
    return sorted(out, key=lambda r: (r.date, r.zone_id))


def write_uhi_csv(records: Sequence[UhiRecord], path: str) -> str:
    """Write UHI records to CSV. Returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=UHI_COLUMNS)
        writer.writeheader()
        for record in records:
            d = record.to_dict()
            writer.writerow({c: d.get(c, "") for c in UHI_COLUMNS})
    return path
