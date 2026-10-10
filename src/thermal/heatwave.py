"""Heatwave flagging: N consecutive passes above a max-LST threshold.

A heatwave event for a zone starts when its per-pass *maximum* LST meets or
exceeds ``threshold_c`` for ``consecutive`` passes in a row, and ends at
the first pass below the threshold. Each event records its start/end dates,
duration in passes, and peak maximum LST.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

from .timeseries import ZonePassRecord, select

HEATWAVE_COLUMNS = [
    "zone_id",
    "zone_name",
    "start_date",
    "end_date",
    "duration_passes",
    "peak_max_c",
    "threshold_c",
    "consecutive",
]


@dataclass
class HeatwaveEvent:
    zone_id: str
    zone_name: str
    start_date: str
    end_date: str
    duration_passes: int
    peak_max_c: float
    threshold_c: float
    consecutive: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_heatwaves(
    records: Sequence[ZonePassRecord],
    threshold_c: float = 38.0,
    consecutive: int = 3,
    index: str = "lst",
) -> List[HeatwaveEvent]:
    """Flag heatwave events per zone.

    ``threshold_c`` is compared against each pass's *maximum* LST (the
    hottest pixel in the zone), not the mean — heat stress lives in the
    extremes. Passes with non-finite maxima break a streak (missing data
    is not evidence of heat).
    """
    if consecutive < 1:
        raise ValueError("consecutive must be >= 1")
    events: List[HeatwaveEvent] = []
    for zid in sorted({r.zone_id for r in records if r.index == index}):
        series = select(records, zone_id=zid, index=index)
        if not series:
            continue
        zone_name = series[0].zone_name
        streak: List[ZonePassRecord] = []
        for rec in series:
            hot = math.isfinite(rec.maximum) and rec.maximum >= threshold_c
            if hot:
                streak.append(rec)
            else:
                if len(streak) >= consecutive:
                    events.append(_event(zid, zone_name, streak, threshold_c, consecutive))
                streak = []
        if len(streak) >= consecutive:
            events.append(_event(zid, zone_name, streak, threshold_c, consecutive))
    return events


def _event(
    zid: str,
    zone_name: str,
    streak: List[ZonePassRecord],
    threshold_c: float,
    consecutive: int,
) -> HeatwaveEvent:
    return HeatwaveEvent(
        zone_id=zid,
        zone_name=zone_name,
        start_date=streak[0].date,
        end_date=streak[-1].date,
        duration_passes=len(streak),
        peak_max_c=max(r.maximum for r in streak),
        threshold_c=threshold_c,
        consecutive=consecutive,
    )


def write_heatwaves_csv(events: Sequence[HeatwaveEvent], path: str) -> str:
    """Write heatwave events to CSV. Returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEATWAVE_COLUMNS)
        writer.writeheader()
        for event in events:
            d = event.to_dict()
            writer.writerow({c: d.get(c, "") for c in HEATWAVE_COLUMNS})
    return path
