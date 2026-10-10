"""Day-of-year climatology builder.

A climatology is the long-term mean annual cycle: for each day of year,
the average (and spread) of what the variable looked like on that date
across the baseline years. Anomalies are then "today minus normal",
the quantity behind every climatological anomaly map.

See docs/METHODS.md for the science: why a ±15-day window, which
baseline periods to use, and how leap days are handled.
"""

from __future__ import annotations

import datetime as _dt
import warnings as _warnings
from dataclasses import dataclass, field as _dc_field
from typing import Any, Dict

import numpy as _np

from ._coerce import FieldView, coerce_field

#: Day-of-year axis length. doy 366 exists only in leap years; every other
#: doy pools samples from all baseline years (see docs/METHODS.md).
_N_DOY = 366

#: Cycle length for year-boundary distances. The axis runs 1..366, so a
#: 366-day cycle keeps window_days=0 as exact day-of-year pooling: doy 366
#: (Dec 31 of leap years) sits 1 day from doy 1, and only leap-year
#: samples ever land on doy 366.
_WRAP = 366


@dataclass
class Climatology:
    """Day-of-year climatology on a fixed grid.

    ``doy_means[d]`` is the NaN-aware mean of all baseline samples whose
    date falls within ``±window_days`` of day-of-year ``d + 1``
    (wrapping the year boundary). ``doy_stds`` is the sample standard
    deviation (ddof=1) of the same pool; ``doy_counts`` is the number of
    finite samples pooled (NaN inputs are excluded, never zero-filled).
    """

    doy_means: _np.ndarray   # (366, ny, nx) float64, NaN where no samples
    doy_stds: _np.ndarray    # (366, ny, nx) float64, NaN where < 2 samples
    doy_counts: _np.ndarray  # (366, ny, nx) int64
    baseline_start: _dt.date
    baseline_end: _dt.date
    window_days: int
    min_samples: int
    lats: _np.ndarray        # (ny,)
    lons: _np.ndarray        # (nx,)
    source_variable: str
    source_units: str
    provenance: Dict[str, Any] = _dc_field(default_factory=dict)


def _circular_within(doys: _np.ndarray, center: int, window_days: int) -> _np.ndarray:
    """Boolean mask of samples within ±window_days of ``center`` (1..366)."""
    raw = _np.abs(doys - center)
    dist = _np.minimum(raw, _WRAP - raw)
    return dist <= window_days


def climatology(field: Any, window_days: int = 15, min_samples: int = 10) -> Climatology:
    """Build a day-of-year climatology from a multi-year baseline field.

    Parameters
    ----------
    field:
        Render-dict or duck-typed field object (see :mod:`derive._coerce`).
        Should span several full years; a 30-year baseline (e.g.
        1991–2020) is the WMO standard for climate normals.
    window_days:
        Half-width of the day-of-year pooling window. Each doy pools all
        samples within ±``window_days`` (wrapping Dec/Jan). ``0`` pools
        only exact day-of-year matches.
    min_samples:
        Advisory floor on pooled finite samples per doy/cell. It is
        recorded in the climatology and enforced by
        :func:`derive.standardized_anomaly`; the means/stds themselves
        are always computed from whatever finite samples exist.

    Returns
    -------
    Climatology
        With ``doy_means`` / ``doy_stds`` / ``doy_counts`` on a
        ``(366, ny, nx)`` axis. Day-of-year follows
        ``date.timetuple().tm_yday``: 1..366, Feb 29 = 60 in leap years,
        doy 366 only in leap years.

    Raises
    ------
    ValueError
        On invalid input (see :func:`derive._coerce.coerce_field`),
        negative ``window_days``, or ``min_samples < 1``.
    """
    if not isinstance(window_days, (int, _np.integer)) or int(window_days) < 0:
        raise ValueError(
            f"window_days must be a non-negative integer, got {window_days!r}"
        )
    if not isinstance(min_samples, (int, _np.integer)) or int(min_samples) < 1:
        raise ValueError(
            f"min_samples must be a positive integer, got {min_samples!r}"
        )
    window_days, min_samples = int(window_days), int(min_samples)

    fv: FieldView = coerce_field(field)
    nt, ny, nx = fv.values.shape
    doys = _np.array([t.timetuple().tm_yday for t in fv.times], dtype=_np.int64)

    doy_means = _np.full((_N_DOY, ny, nx), _np.nan, dtype=_np.float64)
    doy_stds = _np.full((_N_DOY, ny, nx), _np.nan, dtype=_np.float64)
    doy_counts = _np.zeros((_N_DOY, ny, nx), dtype=_np.int64)

    with _warnings.catch_warnings():
        # All-NaN pools and single-sample stds are expected; not warnings.
        _warnings.simplefilter("ignore", RuntimeWarning)
        for d in range(1, _N_DOY + 1):
            mask = _circular_within(doys, d, window_days)
            if not mask.any():
                continue
            block = fv.values[mask]                      # (k, ny, nx)
            finite = ~_np.isnan(block)
            doy_counts[d - 1] = finite.sum(axis=0)
            doy_means[d - 1] = _np.nanmean(block, axis=0)
            doy_stds[d - 1] = _np.nanstd(block, axis=0, ddof=1)

    provenance: Dict[str, Any] = {
        "method": "doy_window",
        "window_days": window_days,
        "min_samples": min_samples,
        "baseline_start": fv.times[0].isoformat(),
        "baseline_end": fv.times[-1].isoformat(),
        "n_input_times": nt,
        "derive_version": _derive_version(),
    }
    return Climatology(
        doy_means=doy_means,
        doy_stds=doy_stds,
        doy_counts=doy_counts,
        baseline_start=fv.times[0],
        baseline_end=fv.times[-1],
        window_days=window_days,
        min_samples=min_samples,
        lats=fv.lats,
        lons=fv.lons,
        source_variable=fv.variable,
        source_units=fv.units,
        provenance=provenance,
    )


def _derive_version() -> str:
    from . import __version__

    return __version__
