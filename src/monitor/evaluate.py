"""Threshold evaluation: compare one pass's metrics against a baseline.

Pure math over dicts — no I/O, no network. Given a pass's per-index metric
values and the baseline statistics, this module computes absolute deltas and
z-scores and assigns severities:

* ``critical`` — breached a critical level (if configured),
* ``warning`` — breached a warning level,
* ``info`` — approaching a warning level (at least half the warning
  threshold) but not breaching it; the "watch" band.

Persistence (consecutive breaching passes) and cooldown are handled by the
pipeline; here each pass is judged on its own numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import AlertLevel, IndexThreshold

#: Severities, ordered from mildest to most severe.
SEVERITIES = ("info", "warning", "critical")

#: Fraction of the warning threshold at which a change becomes "approaching"
#: and earns an ``info`` watch event.
INFO_FRACTION = 0.5


@dataclass
class AlertCandidate:
    """A breach (or near-breach) found in one pass, before persistence gating."""

    index: str
    severity: str
    delta: float
    z_score: float
    threshold_kind: str     # "absolute_delta" | "z_score"
    threshold: float        # the breached (or approached) threshold value
    direction: str
    value: float
    baseline_mean: float
    baseline_std: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "severity": self.severity,
            "delta": self.delta,
            "z_score": self.z_score,
            "threshold_kind": self.threshold_kind,
            "threshold": self.threshold,
            "direction": self.direction,
            "value": self.value,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
        }


def _signed_delta(value: float, mean: float) -> float:
    return value - mean


def _z_score(delta: float, std: float) -> float:
    return delta / std if std > 0 and math.isfinite(std) else 0.0


def _direction_ok(direction: str, delta: float) -> bool:
    if direction == "either":
        return True
    if direction == "up":
        return delta > 0
    return delta < 0  # "down"


def _level_breach(level: AlertLevel, delta: float, z: float,
                  direction: str) -> Optional[Tuple[str, float]]:
    """Return ``(threshold_kind, threshold)`` if the level is breached."""
    if not _direction_ok(direction, delta):
        return None
    mag, zmag = abs(delta), abs(z)
    breached: Optional[Tuple[str, float]] = None
    if level.absolute_delta is not None and mag >= level.absolute_delta:
        breached = ("absolute_delta", level.absolute_delta)
    if level.z_score is not None and zmag >= level.z_score:
        breached = ("z_score", level.z_score)
    return breached


def evaluate_pass(
    metrics: Dict[str, float],
    baselines: Dict[str, Tuple[float, float]],
    thresholds: Dict[str, IndexThreshold],
) -> List[AlertCandidate]:
    """Evaluate one pass's metrics against baselines.

    ``metrics`` maps index -> watched zonal statistic; ``baselines`` maps
    index -> ``(mean, std)``. Indices missing from either side are skipped.
    Returns candidates ordered: critical first, then warning, then info,
    indices alphabetical within a severity.
    """
    candidates: List[AlertCandidate] = []
    for index in sorted(metrics):
        if index not in baselines or index not in thresholds:
            continue
        value = metrics[index]
        if not math.isfinite(value):
            continue
        mean, std = baselines[index]
        threshold = thresholds[index]
        delta = _signed_delta(value, mean)
        z = _z_score(delta, std)

        # Critical first: only when a critical level exists and is breached.
        if threshold.critical is not None:
            hit = _level_breach(threshold.critical, delta, z,
                                threshold.direction)
            if hit is not None:
                kind, level = hit
                candidates.append(AlertCandidate(
                    index=index, severity="critical", delta=delta, z_score=z,
                    threshold_kind=kind, threshold=level,
                    direction=threshold.direction, value=value,
                    baseline_mean=mean, baseline_std=std))
                continue  # critical subsumes warning/info for this index

        hit = _level_breach(threshold.warning, delta, z, threshold.direction)
        if hit is not None:
            kind, level = hit
            candidates.append(AlertCandidate(
                index=index, severity="warning", delta=delta, z_score=z,
                threshold_kind=kind, threshold=level,
                direction=threshold.direction, value=value,
                baseline_mean=mean, baseline_std=std))
            continue

        # Watch band: at least half the warning threshold, right direction.
        if _direction_ok(threshold.direction, delta):
            approached: Optional[Tuple[str, float]] = None
            if (threshold.warning.absolute_delta is not None
                    and abs(delta) >= INFO_FRACTION * threshold.warning.absolute_delta):
                approached = ("absolute_delta",
                              threshold.warning.absolute_delta)
            if (threshold.warning.z_score is not None
                    and abs(z) >= INFO_FRACTION * threshold.warning.z_score):
                approached = ("z_score", threshold.warning.z_score)
            if approached is not None:
                kind, level = approached
                candidates.append(AlertCandidate(
                    index=index, severity="info", delta=delta, z_score=z,
                    threshold_kind=kind, threshold=level,
                    direction=threshold.direction, value=value,
                    baseline_mean=mean, baseline_std=std))
    order = {"critical": 0, "warning": 1, "info": 2}
    candidates.sort(key=lambda c: (order[c.severity], c.index))
    return candidates
