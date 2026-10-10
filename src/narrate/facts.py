"""Descriptive statistics for wind/current fields: :func:`story_facts`.

Everything here is NaN-aware: masked/NaN cells are excluded from every
statistic, and gaps are never filled or interpolated. A field with no
valid cells at all raises :class:`NoValidDataError` instead of returning
zeros that could read as a real calm.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .shapes import NormalizedField, UnrecognizedFieldError, normalize_field


class NoValidDataError(ValueError):
    """The field has no valid (non-NaN, unmasked) cells to describe."""


#: Exact conversion: 1 m/s = 1.9438444924 knots.
MS_TO_KNOTS = 1.9438444924

#: Standard cadences the cadence guesser matches against (seconds, label).
_CADENCES: List[Tuple[float, str]] = [
    (60, "minute"),
    (300, "5-minute"),
    (600, "10-minute"),
    (900, "15-minute"),
    (1800, "30-minute"),
    (3600, "hourly"),
    (7200, "2-hourly"),
    (10800, "3-hourly"),
    (21600, "6-hourly"),
    (43200, "12-hourly"),
    (86400, "daily"),
    (172800, "2-daily"),
    (604800, "weekly"),
]


def _valid_mask(*arrays: np.ndarray) -> np.ndarray:
    """Boolean mask of cells valid in every array (NaN-safe, masked-safe)."""
    mask = None
    for arr in arrays:
        if arr is None:
            continue
        m = ~np.isnan(np.ma.filled(np.ma.asarray(arr, dtype=float), np.nan))
        mask = m if mask is None else mask & m
    return mask


def _fmt_iso(value: Any) -> str:
    """Best-effort ISO string for a timestamp (str/datetime passthrough)."""
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    return str(value)


def _guess_cadence(times: List[str]) -> Tuple[Optional[float], str]:
    """Guess the sampling cadence from the median inter-step gap.

    Returns ``(median_seconds, label)``. The label is the nearest standard
    cadence within 5% of the median gap, else ``"irregular"``; a single
    timestep yields ``(None, "single")``.
    """
    parsed: List[_dt.datetime] = []
    for t in times:
        try:
            parsed.append(_dt.datetime.fromisoformat(str(t).replace("Z", "+00:00")))
        except ValueError:
            return None, "irregular"
    if len(parsed) < 2:
        return None, "single"
    gaps = sorted((b - a).total_seconds() for a, b in zip(parsed, parsed[1:]))
    median = gaps[len(gaps) // 2]
    for seconds, label in _CADENCES:
        if abs(median - seconds) / seconds <= 0.05:
            return float(median), label
    return float(median), "irregular"


#: 8-point compass labels for a flow bearing (direction the vector heads).
_COMPASS = [
    (0.0, "northerly"), (45.0, "northeasterly"), (90.0, "easterly"),
    (135.0, "southeasterly"), (180.0, "southerly"), (225.0, "southwesterly"),
    (270.0, "westerly"), (315.0, "northwesterly"),
]


def compass_label(bearing_deg: float) -> str:
    """Nearest 8-point compass label for a bearing (0°=N, 90°=E, 360°=N)."""
    bearing = bearing_deg % 360.0
    return min(_COMPASS, key=lambda item: abs(bearing - item[0]))[1]


def dominant_direction(u: np.ndarray, v: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    """Dominant flow direction from the vector mean of (u, v), NaN-aware.

    The bearing is the direction the vector-mean flow *heads*, measured
    clockwise from true north (0° = toward north, 90° = toward east) —
    i.e. the compass bearing of the resultant vector. This is **not** the
    meteorological "wind from" convention: a wind field blowing from the
    west (westerlies) is reported here as bearing ~90°, "easterly".
    The distinction is documented on every output (see ``direction_note``).

    Returns ``(None, None)`` when the vector mean is zero (no dominant
    direction) or there are no valid cells.
    """
    mask = _valid_mask(u, v)
    if not bool(np.any(mask)):
        return None, None
    u_mean = float(np.mean(np.ma.filled(np.ma.asarray(u, dtype=float), np.nan)[mask]))
    v_mean = float(np.mean(np.ma.filled(np.ma.asarray(v, dtype=float), np.nan)[mask]))
    if math.hypot(u_mean, v_mean) == 0.0:
        return None, None
    bearing = (90.0 - math.degrees(math.atan2(v_mean, u_mean))) % 360.0
    return bearing, compass_label(bearing)


def story_facts(field: Any, region_name: str = "",
                temperature_unit: Optional[str] = None) -> Dict[str, Any]:
    """Compute narrative-ready facts from a wind or current field.

    Parameters
    ----------
    field:
        A GFS-wind-shaped field (``grids["u10"]/grids["v10"]``) or a
        CurrentField-shaped field (``u`` / ``v``), as a dict or object.
        See :mod:`narrate.shapes` for the accepted layouts.
    region_name:
        Caller-supplied place label (e.g. "the North Sound"). Recorded
        verbatim in the facts; never invented when empty.
    temperature_unit:
        Optional override for the temperature unit ("°C" / "°F"-like).
        Overrides any unit declared on the field.

    Returns
    -------
    dict
        All values JSON-serializable (plain floats/ints/strings/None):

        ``field_kind`` ("wind" | "current"), ``region_name``,
        ``peak``: {value_m_s, value_knots, lat, lon, time},
        ``mean_speed_m_s``, ``mean_speed_knots``,
        ``p95_speed_m_s``, ``p95_speed_knots``,
        ``direction``: {bearing_deg, label, note},
        ``temperature``: {min, max, unit} or None,
        ``time_span``: {start, end, n_timesteps, cadence, median_step_s},
        ``n_valid_cells``, ``n_total_cells``, ``coverage_fraction``.

    Notes
    -----
    Every statistic is descriptive of the sampled window only — no
    causal claims, no extrapolation beyond the grid/times given.
    NaN/masked cells are excluded from all statistics; they are counted
    (``coverage_fraction``) but never filled.
    """
    nf: NormalizedField = normalize_field(field, temperature_unit=temperature_unit)

    speed = np.ma.sqrt(nf.u ** 2 + nf.v ** 2)
    speed_f = np.ma.filled(np.ma.asarray(speed, dtype=float), np.nan)
    valid = _valid_mask(nf.u, nf.v)
    n_valid = int(np.count_nonzero(valid))
    n_total = int(speed_f.size)
    if n_valid == 0:
        raise NoValidDataError(
            "field has no valid (non-NaN, unmasked) u/v cells; "
            "nothing to describe.")

    speed_valid = speed_f[valid]
    peak_idx = int(np.argmax(speed_valid))
    peak_speed = float(speed_valid[peak_idx])
    flat = np.flatnonzero(valid)[peak_idx]
    it, iy, ix = np.unravel_index(flat, speed_f.shape)

    bearing, label = dominant_direction(nf.u, nf.v)

    temperature_facts: Optional[Dict[str, Any]] = None
    if nf.temperature is not None:
        t_valid = np.ma.filled(np.ma.asarray(nf.temperature, dtype=float), np.nan)[valid]
        t_valid = t_valid[~np.isnan(t_valid)]
        if t_valid.size:
            temperature_facts = {
                "min": float(np.min(t_valid)),
                "max": float(np.max(t_valid)),
                "unit": nf.temperature_unit,
            }

    median_step_s, cadence = _guess_cadence(nf.times)

    facts: Dict[str, Any] = {
        "field_kind": nf.kind,
        "region_name": str(region_name),
        "peak": {
            "value_m_s": peak_speed,
            "value_knots": peak_speed * MS_TO_KNOTS,
            "lat": float(nf.lats[iy]),
            "lon": float(nf.lons[ix]),
            "time": _fmt_iso(nf.times[it]),
        },
        "mean_speed_m_s": float(np.mean(speed_valid)),
        "mean_speed_knots": float(np.mean(speed_valid)) * MS_TO_KNOTS,
        "p95_speed_m_s": float(np.percentile(speed_valid, 95)),
        "p95_speed_knots": float(np.percentile(speed_valid, 95)) * MS_TO_KNOTS,
        "direction": {
            "bearing_deg": None if bearing is None else round(bearing, 1),
            "label": label,
            "note": ("Bearing is the compass direction the flow heads "
                     "(0=N, 90=E); NOT the meteorological 'wind from' "
                     "convention."),
        },
        "temperature": temperature_facts,
        "time_span": {
            "start": _fmt_iso(nf.times[0]),
            "end": _fmt_iso(nf.times[-1]),
            "n_timesteps": int(len(nf.times)),
            "cadence": cadence,
            "median_step_s": median_step_s,
        },
        "n_valid_cells": n_valid,
        "n_total_cells": n_total,
        "coverage_fraction": round(n_valid / n_total, 4),
    }
    return facts
