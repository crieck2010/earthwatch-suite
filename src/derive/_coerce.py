"""Field coercion: accept the two shapes survey-derive consumes.

1. The plain render-dict form shared with survey-viz::

      {"times": [...], "lats": [...], "lons": [...], "values": np.ndarray,
       "variable": str, "units": str, "long_name": str (optional),
       "provenance": dict (optional)}

2. Duck-typed field objects exposing ``.times`` / ``.lats`` / ``.lons`` /
   ``.values`` attributes (the shape survey-currents field objects use).

``times`` entries may be ``datetime.date``, ``datetime.datetime``,
``numpy.datetime64``, or ISO-8601 strings (``"YYYY-MM-DD"`` or a full
timestamp — only the calendar date is used). NaN in ``values`` means
missing; it is never treated as zero.
"""

from __future__ import annotations

import datetime as _dt
import warnings as _warnings
from dataclasses import dataclass, field as _dc_field
from typing import Any, Dict, List, Optional

import numpy as _np


@dataclass
class FieldView:
    """A validated, canonical view of an input field."""

    times: List[_dt.date]
    lats: _np.ndarray            # (ny,) float64
    lons: _np.ndarray            # (nx,) float64
    values: _np.ndarray         # (nt, ny, nx) float64, NaN = missing
    variable: str
    units: str
    long_name: Optional[str] = None
    provenance: Dict[str, Any] = _dc_field(default_factory=dict)


def _coerce_date(t: Any) -> _dt.date:
    if isinstance(t, _dt.datetime):
        return t.date()
    if isinstance(t, _dt.date):
        return t
    if isinstance(t, _np.datetime64):
        return _coerce_date(str(t)[:10])
    if isinstance(t, str):
        s = t.strip()
        for sep in ("T", " "):
            if sep in s:
                s = s.split(sep)[0]
                break
        try:
            return _dt.date.fromisoformat(s)
        except ValueError:
            raise ValueError(
                f"unparseable time value {t!r}; expected an ISO-8601 date "
                "like '2020-01-15'"
            ) from None
    raise ValueError(
        f"unsupported time value {t!r} (type {type(t).__name__}); expected "
        "datetime.date, datetime.datetime, numpy.datetime64, or an "
        "ISO-8601 string"
    )


def _get(mapping: Dict[str, Any], key: str, what: str) -> Any:
    try:
        return mapping[key]
    except KeyError:
        raise ValueError(
            f"{what} is missing required key {key!r}; expected keys "
            "'times', 'lats', 'lons', 'values'"
        ) from None


def coerce_field(field: Any) -> FieldView:
    """Validate ``field`` and return a canonical :class:`FieldView`.

    Raises
    ------
    ValueError
        On empty ``times``, non-3-D ``values``, or any shape inconsistency
        between ``times`` / ``lats`` / ``lons`` / ``values``.
    """
    if isinstance(field, dict):
        times_raw = _get(field, "times", "field dict")
        lats_raw = _get(field, "lats", "field dict")
        lons_raw = _get(field, "lons", "field dict")
        values_raw = _get(field, "values", "field dict")
        variable = str(field.get("variable", "unknown"))
        units = str(field.get("units", ""))
        long_name = field.get("long_name")
        provenance = dict(field.get("provenance") or {})
    else:
        for attr in ("times", "lats", "lons", "values"):
            if not hasattr(field, attr):
                raise ValueError(
                    "field object is missing attribute "
                    f"{attr!r}; expected an object with .times/.lats/.lons/"
                    ".values or a dict with 'times'/'lats'/'lons'/'values'"
                )
        times_raw, lats_raw = field.times, field.lats
        lons_raw, values_raw = field.lons, field.values
        variable = str(getattr(field, "variable", "unknown") or "unknown")
        units = str(getattr(field, "units", "") or "")
        long_name = getattr(field, "long_name", None)
        provenance = dict(getattr(field, "provenance", None) or {})

    times = [_coerce_date(t) for t in times_raw]
    if not times:
        raise ValueError("field has no times: 'times' is empty")

    lats = _np.asarray(lats_raw, dtype=float).ravel()
    lons = _np.asarray(lons_raw, dtype=float).ravel()
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        values = _np.asarray(values_raw, dtype=float)
    if lats.ndim != 1 or lats.size == 0:
        raise ValueError("'lats' must be a non-empty 1-D coordinate vector")
    if lons.ndim != 1 or lons.size == 0:
        raise ValueError("'lons' must be a non-empty 1-D coordinate vector")
    if values.ndim != 3:
        raise ValueError(
            f"'values' must be a 3-D (nt, ny, nx) array, got shape "
            f"{values.shape}"
        )
    nt, ny, nx = values.shape
    if nt != len(times):
        raise ValueError(
            f"'values' has {nt} time steps but 'times' has {len(times)} entries"
        )
    if (ny, nx) != (lats.size, lons.size):
        raise ValueError(
            f"'values' grid shape {(ny, nx)} is inconsistent with "
            f"len(lats)={lats.size}, len(lons)={lons.size}"
        )

    return FieldView(
        times=times,
        lats=lats,
        lons=lons,
        values=values,
        variable=variable,
        units=units,
        long_name=long_name,
        provenance=provenance,
    )
