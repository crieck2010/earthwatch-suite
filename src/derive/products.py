"""Derived products computed from fields and climatologies.

Every product returns a plain dict in the render-dict form survey-viz
consumes::

    {"times": [datetime.date, ...], "lats": np.ndarray, "lons": np.ndarray,
     "values": np.ndarray (nt, ny, nx), "variable": str, "units": str,
     "long_name": str, "provenance": dict}

The one exception is :func:`trend`, which is a static map (no time axis);
see its docstring for the single-frame wrapping recipe.

NaN means missing everywhere: NaN inputs produce NaN outputs, and pooled
statistics (means, counts) exclude NaN samples.
"""

from __future__ import annotations

import datetime as _dt
import warnings as _warnings
from typing import Any, Dict, List, Optional

import numpy as _np

from ._coerce import FieldView, coerce_field
from .climatology import Climatology

#: Absolute floor below which a climatological mean is treated as zero
#: for :func:`percent_of_normal` (division guard; see docs/METHODS.md).
_ZERO_MEAN_TOL = 1e-9

#: Days per year used to convert time deltas to fractional years.
_DAYS_PER_YEAR = 365.25


def _derive_version() -> str:
    from . import __version__

    return __version__


def _check_grid(fv: FieldView, clim: Climatology, product: str) -> None:
    if fv.lats.shape != clim.lats.shape or not _np.array_equal(fv.lats, clim.lats):
        raise ValueError(
            f"{product}: field lats do not exactly match the climatology "
            "grid; anomaly products require the analysis field and the "
            "baseline climatology to share the same grid"
        )
    if fv.lons.shape != clim.lons.shape or not _np.array_equal(fv.lons, clim.lons):
        raise ValueError(
            f"{product}: field lons do not exactly match the climatology "
            "grid; anomaly products require the analysis field and the "
            "baseline climatology to share the same grid"
        )


def _long_name(fv: FieldView, suffix: str) -> str:
    if fv.long_name:
        return f"{fv.long_name}{suffix}"
    readable = fv.variable.replace("-", " ").replace("_", " ").title()
    return f"{readable}{suffix}"


def _render_dict(
    fv: FieldView,
    values: _np.ndarray,
    variable: str,
    units: str,
    long_name: str,
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "times": list(fv.times),
        "lats": fv.lats,
        "lons": fv.lons,
        "values": _np.asarray(values, dtype=float),
        "variable": variable,
        "units": units,
        "long_name": long_name,
        "provenance": provenance,
    }


def _doy_index(fv: FieldView) -> _np.ndarray:
    return _np.array([t.timetuple().tm_yday for t in fv.times], dtype=_np.int64) - 1


def _base_provenance(method: str, fv: FieldView, clim: Climatology) -> Dict[str, Any]:
    return {
        "method": method,
        "baseline_start": clim.baseline_start.isoformat(),
        "baseline_end": clim.baseline_end.isoformat(),
        "window_days": clim.window_days,
        "min_samples": clim.min_samples,
        "derive_version": _derive_version(),
        "source_variable": fv.variable,
    }


def anomaly(field: Any, clim: Climatology) -> Dict[str, Any]:
    """Absolute anomaly: value minus the day-of-year climatological mean.

    ``values[t] = field[t] - clim.doy_means[doy(t)]``. Units are unchanged
    from the source. Use for unbounded continuous variables (temperature,
    sea-surface height, ...); see docs/METHODS.md for product-choice
    guidance.

    Raises
    ------
    ValueError
        If the field's grid does not exactly match the climatology grid.
    """
    fv = coerce_field(field)
    _check_grid(fv, clim, "anomaly")
    idx = _doy_index(fv)
    out = fv.values - clim.doy_means[idx]
    return _render_dict(
        fv,
        out,
        variable=f"{fv.variable}-anomaly",
        units=fv.units,
        long_name=_long_name(fv, " anomaly"),
        provenance=_base_provenance("anomaly", fv, clim),
    )


def standardized_anomaly(field: Any, clim: Climatology) -> Dict[str, Any]:
    """Standardized anomaly (z-score): anomaly / climatological std dev.

    Cells are masked to NaN where the climatological std is ``<= 0``
    (no interannual spread) or where the pooled sample count is below
    the climatology's ``min_samples``. Shares the ``"-anomaly"`` variable
    family with :func:`anomaly`; it is distinguished by ``units == "σ"``
    and ``provenance["method"] == "standardized_anomaly"`.

    Raises
    ------
    ValueError
        If the field's grid does not exactly match the climatology grid.
    """
    fv = coerce_field(field)
    _check_grid(fv, clim, "standardized_anomaly")
    idx = _doy_index(fv)
    anom = fv.values - clim.doy_means[idx]
    std = clim.doy_stds[idx]
    cnt = clim.doy_counts[idx]
    valid = (std > 0) & (cnt >= clim.min_samples)
    with _np.errstate(divide="ignore", invalid="ignore"):
        out = _np.where(valid, anom / std, _np.nan)
    return _render_dict(
        fv,
        out,
        variable=f"{fv.variable}-anomaly",
        units="σ",
        long_name=_long_name(fv, " anomaly"),
        provenance=_base_provenance("standardized_anomaly", fv, clim),
    )


def percent_of_normal(field: Any, clim: Climatology) -> Dict[str, Any]:
    """Percent of normal: ``100 * value / climatological mean``.

    Cells are masked to NaN where ``|doy_means| < 1e-9`` (division guard).
    Intended for variables bounded at zero with skewed distributions —
    precipitation, chlorophyll concentration — where a ratio is more
    informative than a difference. See docs/METHODS.md.

    Raises
    ------
    ValueError
        If the field's grid does not exactly match the climatology grid.
    """
    fv = coerce_field(field)
    _check_grid(fv, clim, "percent_of_normal")
    idx = _doy_index(fv)
    mean = clim.doy_means[idx]
    with _np.errstate(divide="ignore", invalid="ignore"):
        out = _np.where(_np.abs(mean) >= _ZERO_MEAN_TOL, 100.0 * fv.values / mean, _np.nan)
    return _render_dict(
        fv,
        out,
        variable=f"{fv.variable}-percent-of-normal",
        units="%",
        long_name=_long_name(fv, " percent of normal"),
        provenance=_base_provenance("percent_of_normal", fv, clim),
    )


def monthly_means(field: Any) -> Dict[str, Any]:
    """NaN-aware calendar-month aggregation.

    Returns one time step per month present in the input, stamped with
    the first day of the month (``datetime.date``). Variable and units
    are unchanged; this is a resampling, not a new physical quantity.
    """
    fv = coerce_field(field)
    order: Dict[tuple, List[int]] = {}
    for i, t in enumerate(fv.times):
        order.setdefault((t.year, t.month), []).append(i)
    months = sorted(order)
    nt_out = len(months)
    ny, nx = fv.values.shape[1], fv.values.shape[2]
    out = _np.full((nt_out, ny, nx), _np.nan, dtype=_np.float64)
    times: List[_dt.date] = []
    for k, (year, month) in enumerate(months):
        idx = order[(year, month)]
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            out[k] = _np.nanmean(fv.values[idx], axis=0)
        times.append(_dt.date(year, month, 1))
    result = _render_dict(
        fv,
        out,
        variable=fv.variable,
        units=fv.units,
        long_name=fv.long_name or fv.variable,
        provenance={
            "method": "monthly_means",
            "n_input_times": len(fv.times),
            "n_months": nt_out,
            "derive_version": _derive_version(),
        },
    )
    result["times"] = times
    return result


def trend(field: Any) -> Dict[str, Any]:
    """Per-cell ordinary least-squares linear trend over the time span.

    Time is fractional years since the first time step
    (``days / 365.25``). Cells with fewer than 2 finite samples get NaN
    slope/intercept. Honest limits: this is a descriptive slope only —
    no p-values, no significance mask (that would need scipy), and OLS
    assumes independent, homoskedastic errors, which geophysical series
    rarely satisfy. See docs/METHODS.md.

    Returns a static-map dict (no time axis)::

        {"lats", "lons", "slope_per_year" (ny, nx), "intercept" (ny, nx),
         "n_obs" (ny, nx) int, "time_start", "time_end" (dates),
         "variable": "<var>-trend", "units": "<units>/yr", "provenance": {...}}

    ``intercept`` is the fitted value at ``time_start``. To render with
    survey-viz, wrap as a single-frame field::

        {"times": [out["time_end"]], "lats": out["lats"], "lons": out["lons"],
         "values": out["slope_per_year"][None, ...], ...}
    """
    fv = coerce_field(field)
    t0 = fv.times[0]
    x = _np.array([(t - t0).days for t in fv.times], dtype=_np.float64) / _DAYS_PER_YEAR
    y = fv.values
    finite = _np.isfinite(y).astype(_np.float64)
    y0 = _np.where(finite, y, 0.0)

    n = finite.sum(axis=0)
    sx = (x[:, None, None] * finite).sum(axis=0)
    sy = (y0).sum(axis=0)
    sxx = ((x ** 2)[:, None, None] * finite).sum(axis=0)
    sxy = (x[:, None, None] * y0).sum(axis=0)

    denom = n * sxx - sx ** 2
    with _np.errstate(divide="ignore", invalid="ignore"):
        slope = _np.where((n >= 2) & (denom > 0), (n * sxy - sx * sy) / denom, _np.nan)
        intercept = _np.where(
            (n >= 2) & (denom > 0), (sy - slope * sx) / _np.maximum(n, 1), _np.nan
        )
    return {
        "lats": fv.lats,
        "lons": fv.lons,
        "slope_per_year": slope,
        "intercept": intercept,
        "n_obs": n.astype(_np.int64),
        "time_start": t0,
        "time_end": fv.times[-1],
        "variable": f"{fv.variable}-trend",
        "units": f"{fv.units}/yr" if fv.units else "per year",
        "provenance": {
            "method": "ols_trend",
            "time_start": t0.isoformat(),
            "time_end": fv.times[-1].isoformat(),
            "n_input_times": len(fv.times),
            "derive_version": _derive_version(),
            "note": "descriptive OLS slope only; no significance test",
        },
    }


def suggest_symmetric_limits(values: Any, quantile: float = 0.99) -> tuple:
    """Symmetric color limits for diverging anomaly maps.

    Returns ``(-q, +q)`` where ``q`` is the ``quantile`` of
    ``|finite values|``. Degenerate input (all-NaN, all-zero, or
    non-finite quantile) returns ``(-1.0, 1.0)`` so a colorbar always
    has a sane range.

    Raises
    ------
    ValueError
        If ``quantile`` is not in the open interval (0, 1).
    """
    q = float(quantile)
    if not 0.0 < q < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile!r}")
    arr = _np.asarray(values, dtype=float)
    finite = arr[_np.isfinite(arr)]
    if finite.size == 0:
        return (-1.0, 1.0)
    level = float(_np.quantile(_np.abs(finite), q))
    if not _np.isfinite(level) or level <= 0.0:
        return (-1.0, 1.0)
    return (-level, level)
