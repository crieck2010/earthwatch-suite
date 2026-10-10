"""Report model: alert records in, a site report out.

A :class:`SiteReport` is the single in-memory representation that every
renderer (HTML, plain text, email) and every notifier (SMTP, file) consumes.
Building the model once and rendering many times keeps all deliverables
consistent — the email body and the PDF always say the same thing.

Reports are deterministic: the same input records always produce the same
report, and the ``report_id`` (SHA-1 over the sorted alert ids) makes
idempotent delivery checkable downstream.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .reader import AlertRecord, SEVERITY_RANK

_SEVERITY_ORDER = ("info", "warning", "critical")

#: Human-friendly index names for the plain-English summary lines.
INDEX_LABELS = {
    "ndvi": "vegetation vigor (NDVI)",
    "evi": "vegetation vigor (EVI)",
    "savi": "vegetation vigor (SAVI)",
    "ndwi": "surface water (NDWI)",
    "ndmi": "canopy moisture (NDMI)",
    "ndbi": "built-up area (NDBI)",
    "nbr": "burn/disturbance (NBR)",
    "bsi": "bare soil (BSI)",
}


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _describe(record: AlertRecord) -> str:
    """One plain-English line for a single alert record."""
    label = INDEX_LABELS.get(record.index.lower(), record.index.upper() or "index")
    delta = record.delta
    if delta is None:
        core = f"{label} changed ({record.metric or 'metric'})"
    else:
        verb = "dropped" if delta < 0 else "rose"
        core = f"{label} {verb} {_fmt(abs(delta))}"
    if record.z_score is not None:
        core += f" ({record.z_score:+.1f}\u03c3)"
    when = record.pass_date or record.timestamp[:10]
    if when:
        core += f" since {when}"
    if record.threshold is not None:
        kind = "warning" if record.severity == "warning" else record.severity
        core += f" \u2014 exceeds {kind} threshold ({_fmt(record.threshold)})"
    return core[0].upper() + core[1:] + "."


@dataclass
class SiteReport:
    """Everything a renderer needs for one site."""

    site_id: str
    site_name: str
    generated_at: str
    events: List[AlertRecord] = field(default_factory=list)
    by_severity: Dict[str, int] = field(default_factory=dict)
    by_index: Dict[str, int] = field(default_factory=dict)
    worst: Optional[str] = None
    report_id: str = ""
    headline: str = ""
    summary_lines: List[str] = field(default_factory=list)

    def total(self) -> int:
        return len(self.events)


def build_report(records: Iterable[AlertRecord],
                 generated_at: Optional[str] = None) -> SiteReport:
    """Build a deterministic :class:`SiteReport` from alert records.

    Records may span several sites — they are grouped, and the report covers
    the union (multi-site digests use this and render per site; the CLI's
    ``render`` command expects one site per file and the report reflects it).
    Events are sorted worst-severity-first, then by timestamp.
    """
    records = list(records)
    ordered = sorted(
        records,
        key=lambda r: (-r.severity_rank(), r.timestamp, r.alert_id),
    )
    by_severity: Dict[str, int] = {s: 0 for s in _SEVERITY_ORDER}
    by_index: Dict[str, int] = {}
    worst: Optional[str] = None
    for record in ordered:
        by_severity[record.severity] = by_severity.get(record.severity, 0) + 1
        by_index[record.index] = by_index.get(record.index, 0) + 1
        if worst is None or SEVERITY_RANK[record.severity] > SEVERITY_RANK[worst]:
            worst = record.severity

    first = ordered[0] if ordered else None
    site_id = first.site_id if first else ""
    site_name = (first.site_name if first else "") or site_id or "Unknown site"

    digest = hashlib.sha1(
        "|".join(sorted(r.alert_id for r in ordered)).encode("utf-8")
    ).hexdigest()[:12]
    generated = generated_at or _dt.datetime.now(
        _dt.timezone.utc).isoformat(timespec="seconds")

    total = len(ordered)
    if total == 0:
        headline = "No alerts to report."
    elif worst == "critical":
        headline = (f"{total} alert{'s' if total != 1 else ''} \u2014 "
                    f"CRITICAL change detected at {site_name}.")
    elif worst == "warning":
        headline = (f"{total} alert{'s' if total != 1 else ''} \u2014 "
                    f"review recommended at {site_name}.")
    else:
        headline = (f"{total} watch-level event{'s' if total != 1 else ''} "
                    f"at {site_name}.")

    return SiteReport(
        site_id=site_id,
        site_name=site_name,
        generated_at=generated,
        events=ordered,
        by_severity=by_severity,
        by_index=by_index,
        worst=worst,
        report_id=digest,
        headline=headline,
        summary_lines=[_describe(r) for r in ordered],
    )
