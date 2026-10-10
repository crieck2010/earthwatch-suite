"""Alert-event reader: tolerant ingestion of survey-monitor ``alerts.jsonl``.

The reader pins to the stable contract in survey-monitor's
``docs/ALERT_SCHEMA.md``:

* fields are only ever *added* within a major version, so unknown extra
  fields are kept verbatim in ``extras`` and never break parsing;
* optional fields may be absent — the record degrades gracefully;
* a corrupt line never hides the rest of the log: it is skipped and counted.

``AlertRecord`` is intentionally independent of survey-monitor's
``AlertEvent`` class: survey-alerts must also work standalone on a bare path
to an ``alerts.jsonl`` file, with survey-monitor not installed at all.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .errors import AlertsError

#: Severity ordering, matching survey-monitor's info < warning < critical.
SEVERITY_RANK: Dict[str, int] = {"info": 0, "warning": 1, "critical": 2}

#: Fields the ALERT_SCHEMA guarantees. Everything else lands in ``extras``.
KNOWN_FIELDS = (
    "alert_id", "timestamp", "site_id", "site_name", "index", "metric",
    "value", "baseline_mean", "baseline_std", "delta", "z_score",
    "threshold_kind", "threshold", "severity", "direction", "pass_id",
    "pass_date",
)


@dataclass
class AlertRecord:
    """One alert event, read tolerantly from JSONL.

    Unknown extra fields are preserved in ``extras`` so future schema
    additions (or site-specific annotations) survive a round trip through
    this engine and into reports.
    """

    alert_id: str
    timestamp: str = ""
    site_id: str = ""
    site_name: str = ""
    index: str = ""
    metric: str = ""
    value: Optional[float] = None
    baseline_mean: Optional[float] = None
    baseline_std: Optional[float] = None
    delta: Optional[float] = None
    z_score: Optional[float] = None
    threshold_kind: str = ""
    threshold: Optional[float] = None
    severity: str = "info"
    direction: str = ""
    pass_id: str = ""
    pass_date: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlertRecord":
        if not isinstance(data, dict):
            raise AlertsError(f"alert event is not a JSON object: {data!r}")
        if "alert_id" not in data or not data["alert_id"]:
            raise AlertsError("alert event missing required field 'alert_id'")
        known = {k: data.get(k) for k in KNOWN_FIELDS if k in data}
        severity = str(data.get("severity") or "info").lower()
        known["severity"] = severity if severity in SEVERITY_RANK else "info"
        extras = {k: v for k, v in data.items() if k not in KNOWN_FIELDS}
        return cls(extras=extras, **known)  # type: ignore[arg-type]

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "site_id": self.site_id,
            "site_name": self.site_name,
            "index": self.index,
            "metric": self.metric,
            "value": self.value,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "delta": self.delta,
            "z_score": self.z_score,
            "threshold_kind": self.threshold_kind,
            "threshold": self.threshold,
            "severity": self.severity,
            "direction": self.direction,
            "pass_id": self.pass_id,
            "pass_date": self.pass_date,
        }
        out.update(self.extras)
        return out

    def severity_rank(self) -> int:
        return SEVERITY_RANK.get(self.severity, 0)


def read_alerts(path: str,
                min_severity: Optional[str] = None,
                since: Optional[str] = None) -> Tuple[List[AlertRecord], int]:
    """Read an ``alerts.jsonl`` file (oldest first).

    Returns ``(records, skipped)`` where ``skipped`` counts corrupt or
    invalid lines. ``min_severity`` keeps only events at or above that
    severity (e.g. ``"warning"`` keeps warning + critical). ``since`` is an
    ISO-8601 timestamp cutoff (string comparison — timestamps are UTC).
    """
    if not os.path.isfile(path):
        raise AlertsError(f"alerts file not found: {path}")
    min_rank = SEVERITY_RANK.get((min_severity or "info").lower(), 0)
    records: List[AlertRecord] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = AlertRecord.from_dict(json.loads(line))
            except (ValueError, AlertsError):
                skipped += 1  # one bad line never hides the rest
                continue
            if record.severity_rank() < min_rank:
                continue
            if since is not None and record.timestamp < since:
                continue
            records.append(record)
    return records, skipped


def dedupe(records: Iterable[AlertRecord]) -> List[AlertRecord]:
    """Deduplicate on ``alert_id`` (append-only logs can contain replays).

    Keeps the *last* occurrence of each id and preserves first-seen order.
    """
    seen: Dict[str, AlertRecord] = {}
    order: List[str] = []
    for record in records:
        if record.alert_id not in seen:
            order.append(record.alert_id)
        seen[record.alert_id] = record
    return [seen[aid] for aid in order]
