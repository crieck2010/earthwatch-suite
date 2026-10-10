"""Data QA gate: inspect a field's grids before any render runs.

Why this exists: renders are downstream of data fetches, and data fetches
fail in boring, repeatable ways — a timestep that is entirely NaN, a variable
that drifted out of its plausible physical range, a gap in the time axis that
turns a smooth animation into a jump cut. Catching those before the render
saves compute and, more importantly, saves shipping a broken frame.

Design rules:
- Detection and repair are separate steps. qa_field never modifies data.
- repair never silently alters data: every dropped or interpolated index is
  appended to report.actions with a reason.
- Checks are conservative. The plausible-range table is documented with its
  units assumptions; the caller can override it per variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from datetime import datetime

import numpy as np

#: Default plausible physical ranges. Units assumptions: u10/v10 in m/s,
#: t2m in degC, sst in degC, speed in m/s. These are sanity bounds, not
#: climatology — override per variable via the ``plausible`` argument when
#: the data's units or domain differ.
DEFAULT_PLAUSIBLE = {
    "u10": (-150.0, 150.0),
    "v10": (-150.0, 150.0),
    "t2m": (-100.0, 60.0),
    "sst": (-5.0, 40.0),
    "speed": (0.0, 150.0),
}


class QAError(Exception):
    """Raised by repair(..., policy="fail") when any QA issue is present."""


@dataclass
class QAReport:
    """Result of qa_field: what was checked and what was found.

    - ok: True when no issues were found.
    - issues: list of dicts, each with a "type" key ("all_nan",
      "out_of_range", "temporal_gap", "shape_mismatch") plus detail keys.
    - actions: list of dicts recording every repair action taken (empty until
      repair() runs).
    - n_timesteps: number of time steps checked.
    - checked_vars: variable names that were checked.
    """

    ok: bool
    issues: list = _field(default_factory=list)
    actions: list = _field(default_factory=list)
    n_timesteps: int = 0
    checked_vars: list = _field(default_factory=list)


def _parse_time(value) -> datetime:
    """Parse an ISO-8601 time string, tolerating a trailing 'Z'."""
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _as_field_dict(field) -> dict:
    """Duck-typed field access: objects with .to_dict() are converted."""
    if hasattr(field, "to_dict") and callable(field.to_dict):
        field = field.to_dict()
    if not isinstance(field, dict):
        raise TypeError(
            f"field must be a dict or expose .to_dict(), got {type(field).__name__}"
        )
    if "grids" not in field or "times" not in field:
        raise KeyError("field must contain 'grids' and 'times'")
    return field


def qa_field(
    field,
    plausible=None,
    expected_cadence_hours=None,
    outlier_fraction_threshold=0.01,
) -> QAReport:
    """Run the QA checks on a field's grids and return a QAReport.

    Args:
        field: dict with "grids": {varname: array-like, time as first axis},
            "times": [ISO-8601 strings], optionally "lats", "lons". Objects
            exposing .to_dict() are accepted (duck-typed).
        plausible: dict varname -> (lo, hi) physical bounds. Defaults to
            DEFAULT_PLAUSIBLE; variables not listed are not range-checked.
        expected_cadence_hours: expected hours between timesteps for gap
            detection. Defaults to the median observed diff. Diffs larger
            than 1.5x the expected cadence are flagged as temporal gaps.
        outlier_fraction_threshold: a timestep is flagged "out_of_range"
            when the fraction of its cells outside the plausible range
            exceeds this (default 0.01).

    Checks:
        (a) all-NaN timestep per variable -> issue {type, var, index, time}.
        (b) out-of-plausible-range cells -> issue {type, var, index, fraction}.
        (c) temporal gaps -> issue {type, index, gap_hours, expected_hours};
            "index" is the timestep that follows the gap.
    """
    f = _as_field_dict(field)
    grids = f["grids"]
    times = list(f["times"])
    n = len(times)
    bounds = dict(DEFAULT_PLAUSIBLE)
    if plausible:
        bounds.update(plausible)

    issues = []
    checked = []

    for var, raw in grids.items():
        checked.append(var)
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 0 or arr.shape[0] != n:
            issues.append(
                {
                    "type": "shape_mismatch",
                    "var": var,
                    "expected_first_axis": n,
                    "got_shape": tuple(arr.shape),
                }
            )
            continue
        lo_hi = bounds.get(var)
        for i in range(n):
            step = arr[i]
            step = step.ravel()
            if step.size == 0:
                continue
            if np.all(~np.isfinite(step)):
                issues.append(
                    {"type": "all_nan", "var": var, "index": i, "time": times[i]}
                )
                continue
            if lo_hi is not None:
                lo, hi = lo_hi
                finite = step[np.isfinite(step)]
                if finite.size:
                    bad = np.mean((finite < lo) | (finite > hi))
                    if bad > outlier_fraction_threshold:
                        issues.append(
                            {
                                "type": "out_of_range",
                                "var": var,
                                "index": i,
                                "fraction": float(bad),
                            }
                        )

    # Temporal gaps.
    if n >= 2:
        parsed = [_parse_time(t) for t in times]
        diffs_hours = [
            (b - a).total_seconds() / 3600.0 for a, b in zip(parsed[:-1], parsed[1:])
        ]
        positive = [d for d in diffs_hours if d > 0]
        expected = (
            float(expected_cadence_hours)
            if expected_cadence_hours is not None
            else (float(np.median(positive)) if positive else None)
        )
        if expected and expected > 0:
            for i, d in enumerate(diffs_hours):
                if d > 1.5 * expected:
                    issues.append(
                        {
                            "type": "temporal_gap",
                            "index": i + 1,
                            "gap_hours": float(d),
                            "expected_hours": float(expected),
                        }
                    )

    return QAReport(
        ok=len(issues) == 0,
        issues=issues,
        actions=[],
        n_timesteps=n,
        checked_vars=checked,
    )


def repair(report: QAReport, field, policy="drop") -> tuple:
    """Apply a repair policy to the flagged timesteps. Never silent.

    Args:
        report: QAReport from qa_field.
        field: same field passed to qa_field (dict or .to_dict() object).
        policy:
            - "drop": remove flagged timesteps (all_nan + out_of_range
              indices, unioned across variables).
            - "interpolate": linearly interpolate isolated single bad steps
              (bad step with good neighbors on both sides, and no temporal
              gap adjacent). Runs of bad steps are left untouched.
            - "fail": raise QAError if any issues are present.

    Returns:
        (new_field_dict, updated_report). The field is a plain dict with
        "grids"/"times"/"lats"/"lons" as numpy arrays. Every dropped or
        interpolated index is appended to report.actions with its reason.

    Unknown policies raise ValueError.
    """
    if policy not in ("drop", "interpolate", "fail"):
        raise ValueError(
            f"unknown repair policy {policy!r}; valid: 'drop', 'interpolate', 'fail'"
        )
    if policy == "fail" and report.issues:
        raise QAError(
            f"QA failed with {len(report.issues)} issue(s); refusing to repair"
        )

    f = _as_field_dict(field)
    times = list(f["times"])
    n = len(times)

    def bad_indices_for(var):
        return sorted(
            {
                issue["index"]
                for issue in report.issues
                if issue["var"] == var
                if issue["type"] in ("all_nan", "out_of_range")
            }
        )

    gap_indices = {issue["index"] for issue in report.issues if issue["type"] == "temporal_gap"}
    gap_adjacent = set()
    for g in gap_indices:
        gap_adjacent.update({g - 1, g, g + 1})

    new_grids = {}
    if policy == "drop":
        drop = sorted(
            {
                issue["index"]
                for issue in report.issues
                if issue["type"] in ("all_nan", "out_of_range")
            }
        )
        keep = [i for i in range(n) if i not in set(drop)]
        for var, raw in f["grids"].items():
            arr = np.asarray(raw, dtype=np.float64)
            new_grids[var] = arr[keep] if arr.ndim and arr.shape[0] == n else arr
        for i in drop:
            reasons = sorted(
                {
                    issue["type"]
                    for issue in report.issues
                    if issue.get("index") == i
                    and issue["type"] in ("all_nan", "out_of_range")
                }
            )
            report.actions.append(
                {
                    "action": "dropped_timestep",
                    "index": i,
                    "time": times[i],
                    "reason": ",".join(reasons),
                }
            )
        new_times = [times[i] for i in keep]
    else:  # interpolate
        for var, raw in f["grids"].items():
            arr = np.asarray(raw, dtype=np.float64).copy()
            bad = set(bad_indices_for(var))
            if arr.ndim and arr.shape[0] == n:
                for i in sorted(bad):
                    isolated = (
                        0 < i < n - 1
                        and (i - 1) not in bad
                        and (i + 1) not in bad
                        and i not in gap_adjacent
                    )
                    if isolated:
                        arr[i] = 0.5 * (arr[i - 1] + arr[i + 1])
                        report.actions.append(
                            {
                                "action": "interpolated_timestep",
                                "var": var,
                                "index": i,
                                "time": times[i],
                                "reason": "isolated single bad step; linear interpolation from neighbors",
                            }
                        )
            new_grids[var] = arr
        new_times = list(times)

    out = {
        "grids": new_grids,
        "times": np.asarray(new_times),
        "lats": np.asarray(f["lats"]) if f.get("lats") is not None else None,
        "lons": np.asarray(f["lons"]) if f.get("lons") is not None else None,
    }
    return out, report
