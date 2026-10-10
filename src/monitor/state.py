"""Persistent per-site state for survey-monitor.

State is a small JSON sidecar per site, ``<store_dir>/<site_id>/state.json``,
so re-runs are incremental: the engine remembers every processed pass, the
baseline windows, and the alert streaks. Deleting a site's folder resets it.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import BaselineConfig, MonitorError


@dataclass
class PassRecord:
    """One processed satellite pass: pass_id, acquisition date, per-index metric."""

    pass_id: str
    date: str                      # ISO date, e.g. "2026-09-15"
    metrics: Dict[str, float]      # index -> watched zonal statistic

    def to_dict(self) -> Dict[str, Any]:
        return {"pass_id": self.pass_id, "date": self.date,
                "metrics": dict(self.metrics)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PassRecord":
        return cls(pass_id=data["pass_id"], date=data["date"],
                   metrics=dict(data.get("metrics") or {}))


@dataclass
class SiteState:
    """Everything survey-monitor remembers about one site."""

    site_id: str
    passes: List[PassRecord] = field(default_factory=list)  # oldest first
    last_run: Optional[str] = None          # ISO timestamp of last run
    # consecutive breaching passes per index (drives ``persistence``)
    streaks: Dict[str, int] = field(default_factory=dict)
    # pass_id of the last fired alert per index (drives ``cooldown_passes``)
    last_alert_pass: Dict[str, str] = field(default_factory=dict)
    # alert fired per index (last severity, for summaries)
    last_severity: Dict[str, str] = field(default_factory=dict)

    def known_pass_ids(self) -> set:
        return {p.pass_id for p in self.passes}

    def record_pass(self, pass_id: str, date: str,
                    metrics: Dict[str, float]) -> None:
        """Append a processed pass. Re-recording the same pass_id replaces it."""
        clean = {k: float(v) for k, v in metrics.items()
                 if isinstance(v, (int, float)) and math.isfinite(v)}
        existing = [i for i, p in enumerate(self.passes)
                    if p.pass_id == pass_id]
        record = PassRecord(pass_id=pass_id, date=date, metrics=clean)
        if existing:
            self.passes[existing[0]] = record
        else:
            self.passes.append(record)
        self.passes.sort(key=lambda p: (p.date, p.pass_id))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "site_id": self.site_id,
            "passes": [p.to_dict() for p in self.passes],
            "last_run": self.last_run,
            "streaks": dict(self.streaks),
            "last_alert_pass": dict(self.last_alert_pass),
            "last_severity": dict(self.last_severity),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SiteState":
        state = cls(site_id=data.get("site_id", ""))
        state.passes = [PassRecord.from_dict(p)
                        for p in data.get("passes") or []]
        state.passes.sort(key=lambda p: (p.date, p.pass_id))
        state.last_run = data.get("last_run")
        state.streaks = dict(data.get("streaks") or {})
        state.last_alert_pass = dict(data.get("last_alert_pass") or {})
        state.last_severity = dict(data.get("last_severity") or {})
        return state


def site_dir(store_dir: str, site_id: str) -> str:
    return os.path.join(store_dir, site_id)


def state_path(store_dir: str, site_id: str) -> str:
    return os.path.join(site_dir(store_dir, site_id), "state.json")


def load_state(store_dir: str, site_id: str) -> SiteState:
    """Load a site's state; returns an empty state if none exists yet."""
    path = state_path(store_dir, site_id)
    if not os.path.isfile(path):
        return SiteState(site_id=site_id)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise MonitorError(f"cannot read state file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise MonitorError(f"state file {path} is not a JSON object")
    state = SiteState.from_dict(data)
    state.site_id = site_id
    return state


def save_state(store_dir: str, state: SiteState) -> str:
    """Persist a site's state; returns the path written."""
    path = state_path(store_dir, state.site_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state.to_dict(), fh, indent=2)
        fh.write("\n")
    return path


def baseline_stats(state: SiteState, index: str,
                   baseline: BaselineConfig) -> Optional[Tuple[float, float]]:
    """Return ``(mean, std)`` of a watched metric over the baseline window.

    Rolling: trailing ``window_passes`` passes. Fixed: statistics anchored to
    the window ending at ``reference_pass`` (i.e. the ``window_passes``
    passes up to and including the reference). Returns ``None`` when fewer
    than ``min_passes`` baseline passes carry this index.
    """
    passes = [p for p in state.passes if index in p.metrics]
    if baseline.strategy == "fixed" and baseline.reference_pass:
        ref = [i for i, p in enumerate(passes)
               if p.pass_id == baseline.reference_pass]
        if not ref:
            return None
        passes = passes[max(0, ref[0] - baseline.window_passes + 1):ref[0] + 1]
    else:
        passes = passes[-baseline.window_passes:]
    if len(passes) < baseline.min_passes:
        return None
    values = [p.metrics[index] for p in passes]
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)
