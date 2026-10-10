"""Storm-event deltas: shoreline displacement around a named event date.

For each transect, take the last valid position *before* the event date and
the first valid position *after* it:

    displacement_m = position_after − position_before

Sign follows the transect convention (landward→seaward drawing):
negative = the shoreline moved landward across the event (erosion),
positive = seaward (accretion / recovery).

This is deliberately simple — one number per transect per event, no model.
Pair it with the long-term rate from :mod:`coast.rates` to separate a
storm's bite from the background trend.
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
from dataclasses import dataclass
from typing import List, Sequence

#: Column order of the storm-delta CSV. Append-only.
STORM_COLUMNS = [
    "transect_id",
    "transect_name",
    "site_id",
    "event_date",
    "before_date",
    "after_date",
    "before_position_m",
    "after_position_m",
    "displacement_m",
]


@dataclass
class StormDelta:
    transect_id: str
    transect_name: str
    site_id: str
    event_date: str
    before_date: str
    after_date: str
    before_position_m: float
    after_position_m: float
    displacement_m: float


def storm_deltas(
    records: Sequence[object], event_date: str
) -> List[StormDelta]:
    """Compute per-transect displacement around ``event_date`` (ISO).

    Transects with no valid position on *both* sides of the event are
    skipped (not zero-filled — a missing delta is honest).
    """
    event = _dt.date.fromisoformat(event_date)
    groups: dict = {}
    for r in records:  # type: ignore[union-attr]
        if not r.valid:
            continue
        key = (r.transect_id, r.site_id)
        groups.setdefault(key, []).append(r)
    deltas: List[StormDelta] = []
    for (tid, sid), recs in sorted(groups.items()):
        recs = sorted(recs, key=lambda r: r.date)
        before = [r for r in recs if _dt.date.fromisoformat(r.date) < event]
        after = [r for r in recs if _dt.date.fromisoformat(r.date) > event]
        if not before or not after:
            continue
        b, a = before[-1], after[0]
        deltas.append(
            StormDelta(
                transect_id=tid,
                transect_name=b.transect_name,
                site_id=sid,
                event_date=event_date,
                before_date=b.date,
                after_date=a.date,
                before_position_m=float(b.position_m),
                after_position_m=float(a.position_m),
                displacement_m=float(a.position_m) - float(b.position_m),
            )
        )
    return deltas


def write_storm_csv(path: str, deltas: Sequence[StormDelta]) -> str:
    """Write storm deltas (overwrite). Returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=STORM_COLUMNS)
        writer.writeheader()
        for d in deltas:
            writer.writerow(
                {
                    "transect_id": d.transect_id,
                    "transect_name": d.transect_name,
                    "site_id": d.site_id,
                    "event_date": d.event_date,
                    "before_date": d.before_date,
                    "after_date": d.after_date,
                    "before_position_m": f"{d.before_position_m:.3f}",
                    "after_position_m": f"{d.after_position_m:.3f}",
                    "displacement_m": f"{d.displacement_m:.3f}",
                }
            )
    return path
