"""Digest state: "what have we already told the customer about?"

survey-monitor fires alert events; survey-alerts decides what is *new* since
the last digest. A small JSON state file records, per site, the alert ids
already delivered and the last digest timestamp. Digest selection is:

* keep only events not in the delivered set (idempotent re-runs);
* keep only events at/above ``min_severity``;
* optionally keep only events newer than ``since`` (quiet-hours / scheduling
  windows are the caller's job — this engine just filters on time).

Re-running a digest on unchanged input selects nothing new, so repeated cron
runs never re-send the same alerts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .errors import AlertsError
from .reader import AlertRecord, SEVERITY_RANK


@dataclass
class DigestState:
    """Per-site "already delivered" bookkeeping, persisted as JSON."""

    path: str
    delivered: Dict[str, List[str]] = field(default_factory=dict)
    last_digest: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "DigestState":
        state = cls(path=path)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (ValueError, OSError) as exc:
                raise AlertsError(
                    f"cannot read digest state {path}: {exc}") from exc
            if isinstance(data, dict):
                delivered = data.get("delivered") or {}
                last = data.get("last_digest") or {}
                if isinstance(delivered, dict):
                    state.delivered = {
                        str(k): [str(i) for i in v] for k, v in
                        delivered.items() if isinstance(v, list)}
                if isinstance(last, dict):
                    state.last_digest = {str(k): str(v)
                                         for k, v in last.items()}
        return state

    def save(self) -> str:
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"delivered": self.delivered,
                       "last_digest": self.last_digest}, fh, indent=2)
            fh.write("\n")
        return self.path

    def mark_delivered(self, site_id: str,
                       records: Iterable[AlertRecord],
                       timestamp: str) -> None:
        known = set(self.delivered.get(site_id, []))
        for record in records:
            known.add(record.alert_id)
        self.delivered[site_id] = sorted(known)
        self.last_digest[site_id] = timestamp


def select_new(records: Iterable[AlertRecord],
               state: DigestState,
               min_severity: str = "warning",
               since: Optional[str] = None) -> List[AlertRecord]:
    """Select the undelivered records for a digest.

    A record is new when its ``alert_id`` is not in the delivered set for its
    site, its severity is at/above ``min_severity``, and (if ``since`` is
    given) its timestamp is not older than ``since``. Result is grouped by
    site in first-seen order, each site's events worst-severity-first.
    """
    min_rank = SEVERITY_RANK.get(min_severity.lower(), 0)
    fresh: List[AlertRecord] = []
    for record in records:
        if record.severity_rank() < min_rank:
            continue
        if since is not None and record.timestamp < since:
            continue
        if record.alert_id in set(state.delivered.get(record.site_id, [])):
            continue
        fresh.append(record)

    by_site: Dict[str, List[AlertRecord]] = {}
    order: List[str] = []
    for record in fresh:
        if record.site_id not in by_site:
            by_site[record.site_id] = []
            order.append(record.site_id)
        by_site[record.site_id].append(record)

    out: List[AlertRecord] = []
    for site_id in order:
        site_records = sorted(
            by_site[site_id],
            key=lambda r: (-r.severity_rank(), r.timestamp, r.alert_id))
        out.extend(site_records)
    return out


def split_by_site(records: Iterable[AlertRecord]
                  ) -> List[Tuple[str, List[AlertRecord]]]:
    """Group records by site, preserving first-seen site order."""
    by_site: Dict[str, List[AlertRecord]] = {}
    order: List[str] = []
    for record in records:
        if record.site_id not in by_site:
            by_site[record.site_id] = []
            order.append(record.site_id)
        by_site[record.site_id].append(record)
    return [(site_id, by_site[site_id]) for site_id in order]
