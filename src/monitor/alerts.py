"""Alert events: the machine-readable output of survey-monitor.

An :class:`AlertEvent` is a plain, JSON-serialisable record — timestamped,
self-describing, and stable. ``survey-alerts`` (the future delivery module)
consumes these files without any change here:

* ``<store>/<site_id>/alerts.jsonl`` — one JSON object per line, appended.
* ``<store>/<site_id>/alerts.geojson`` — same events as point/polygon
  features carrying the alert properties (QGIS-draggable).
* ``<store>/<site_id>/summary.json`` — per-run summary.

The alert-event field contract is documented in ``docs/ALERT_SCHEMA.md``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .config import MonitorError
from .evaluate import AlertCandidate


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


@dataclass
class AlertEvent:
    """One threshold breach that survived persistence gating.

    ``threshold_kind``/``threshold`` say *which* rule fired;
    ``severity`` is one of ``info`` / ``warning`` / ``critical``.
    """

    alert_id: str
    timestamp: str            # UTC ISO-8601, when the event was written
    site_id: str
    site_name: str
    index: str
    metric: str               # watched zonal statistic, e.g. "mean"
    value: float              # this pass's metric value
    baseline_mean: float
    baseline_std: float
    delta: float              # value - baseline_mean
    z_score: float
    threshold_kind: str       # "absolute_delta" | "z_score"
    threshold: float
    severity: str             # info | warning | critical
    direction: str            # up | down | either
    pass_id: str              # the pass that triggered the alert
    pass_date: str            # ISO acquisition date of that pass

    def to_dict(self) -> Dict[str, Any]:
        return {
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlertEvent":
        try:
            return cls(**{f: data[f] for f in (
                "alert_id", "timestamp", "site_id", "site_name", "index",
                "metric", "value", "baseline_mean", "baseline_std", "delta",
                "z_score", "threshold_kind", "threshold", "severity",
                "direction", "pass_id", "pass_date")})
        except KeyError as exc:
            raise MonitorError(f"alert event missing field: {exc}") from exc


def make_event(site_id: str, site_name: str, metric: str,
               candidate: AlertCandidate, pass_id: str, pass_date: str,
               seq: int = 0, timestamp: Optional[str] = None) -> AlertEvent:
    """Build an :class:`AlertEvent` from an :class:`AlertCandidate`.

    ``alert_id`` is deterministic: ``site_id:pass_id:index:severity:seq``.
    """
    alert_id = f"{site_id}:{pass_id}:{candidate.index}:{candidate.severity}:{seq}"
    return AlertEvent(
        alert_id=alert_id,
        timestamp=timestamp or _utcnow(),
        site_id=site_id,
        site_name=site_name,
        index=candidate.index,
        metric=metric,
        value=candidate.value,
        baseline_mean=candidate.baseline_mean,
        baseline_std=candidate.baseline_std,
        delta=candidate.delta,
        z_score=candidate.z_score,
        threshold_kind=candidate.threshold_kind,
        threshold=candidate.threshold,
        severity=candidate.severity,
        direction=candidate.direction,
        pass_id=pass_id,
        pass_date=pass_date,
    )


def alerts_path(store_dir: str, site_id: str) -> str:
    return os.path.join(store_dir, site_id, "alerts.jsonl")


def append_alerts(store_dir: str, site_id: str,
                  events: Iterable[AlertEvent]) -> str:
    """Append events to the site's JSONL alert log. Returns the path."""
    path = alerts_path(store_dir, site_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event.to_dict()) + "\n")
    return path


def read_alerts(store_dir: str, site_id: str,
                severity: Optional[str] = None,
                since: Optional[str] = None) -> List[AlertEvent]:
    """Read a site's alert log (oldest first), with optional filters."""
    path = alerts_path(store_dir, site_id)
    events: List[AlertEvent] = []
    if not os.path.isfile(path):
        return events
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = AlertEvent.from_dict(json.loads(line))
            except (ValueError, MonitorError):
                continue  # never let one corrupt line hide the rest
            if severity is not None and event.severity != severity:
                continue
            if since is not None and event.timestamp < since:
                continue
            events.append(event)
    return events


def summarize_alerts(events: Iterable[AlertEvent]) -> Dict[str, Any]:
    """Count events by severity/index for run summaries."""
    by_severity: Dict[str, int] = {"info": 0, "warning": 0, "critical": 0}
    by_index: Dict[str, int] = {}
    worst: Optional[str] = None
    rank = {"info": 0, "warning": 1, "critical": 2}
    for event in events:
        by_severity[event.severity] = by_severity.get(event.severity, 0) + 1
        by_index[event.index] = by_index.get(event.index, 0) + 1
        if worst is None or rank[event.severity] > rank[worst]:
            worst = event.severity
    return {"by_severity": by_severity, "by_index": by_index,
            "worst": worst, "total": sum(by_severity.values())}


@dataclass
class RunSummary:
    """Machine-readable summary written after every run."""

    site_id: str
    site_name: str
    timestamp: str = field(default_factory=_utcnow)
    passes_processed: int = 0
    passes_skipped: int = 0
    alerts_fired: int = 0
    alerts: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "site_id": self.site_id,
            "site_name": self.site_name,
            "timestamp": self.timestamp,
            "passes_processed": self.passes_processed,
            "passes_skipped": self.passes_skipped,
            "alerts_fired": self.alerts_fired,
            "alerts": self.alerts,
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunSummary":
        return cls(
            site_id=data.get("site_id", ""),
            site_name=data.get("site_name", ""),
            timestamp=data.get("timestamp", ""),
            passes_processed=data.get("passes_processed", 0),
            passes_skipped=data.get("passes_skipped", 0),
            alerts_fired=data.get("alerts_fired", 0),
            alerts=dict(data.get("alerts") or {}),
            errors=list(data.get("errors") or []),
        )


def write_summary(store_dir: str, summary: RunSummary) -> str:
    path = os.path.join(store_dir, summary.site_id, "summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary.to_dict(), fh, indent=2)
        fh.write("\n")
    return path


def _site_centroid(geometry: Dict[str, Any]) -> Optional[List[float]]:
    """Best-effort [lon, lat] for a GeoJSON polygon or bbox dict."""
    if "bbox" in geometry:
        minx, miny, maxx, maxy = geometry["bbox"]
        return [(minx + maxx) / 2.0, (miny + maxy) / 2.0]
    coords = geometry.get("coordinates") or []
    try:
        ring = coords[0][0] if geometry.get("type") == "MultiPolygon" \
            else coords[0]
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return [sum(xs) / len(xs), sum(ys) / len(ys)]
    except (IndexError, TypeError, ZeroDivisionError):
        return None


def alerts_to_geojson(events: Iterable[AlertEvent],
                      geometry: Optional[Dict[str, Any]] = None
                      ) -> Dict[str, Any]:
    """Render alert events as a GeoJSON FeatureCollection.

    Each feature carries the site footprint (or its centroid point as a
    fallback) plus every alert field as properties — drag straight into QGIS.
    """
    features = []
    for event in events:
        props = event.to_dict()
        geom: Dict[str, Any]
        if geometry and geometry.get("type") in ("Polygon", "MultiPolygon"):
            geom = {"type": geometry["type"],
                    "coordinates": geometry["coordinates"]}
        else:
            centroid = _site_centroid(geometry or {})
            geom = {"type": "Point",
                    "coordinates": centroid or [0.0, 0.0]}
        features.append({"type": "Feature", "geometry": geom,
                         "properties": props})
    return {"type": "FeatureCollection", "features": features}


def write_alerts_geojson(store_dir: str, site_id: str,
                         events: Iterable[AlertEvent],
                         geometry: Optional[Dict[str, Any]] = None) -> str:
    path = os.path.join(store_dir, site_id, "alerts.geojson")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(alerts_to_geojson(events, geometry), fh, indent=2)
        fh.write("\n")
    return path
