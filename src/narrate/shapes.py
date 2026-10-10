"""Field-shape detection and normalization for ``survey-narrate``.

This engine never requires a specific class from the peer repos
(``survey-currents``, ``survey-viz``). It detects two duck-typed shapes,
normalizes them to a common internal record, and raises a clear
:class:`UnrecognizedFieldError` for anything else — never guessing.

Shapes
------
``"gfs-wind"`` (GfsWindField-like; winds):
    Dicts or objects with ``grids["u10"]`` / ``grids["v10"]`` in m/s,
    plus ``times`` / ``lats`` / ``lons``. Temperature comes from
    ``air_temperature`` (3-D) with its unit read from ``temperature_unit``
    ("°F" / "degF" style strings accepted, else assumed °C), falling back
    to ``grids["t2m"]`` (assumed °C, the GFS on-disk convention).

``"current"`` (CurrentField-like; ocean/river currents):
    Dicts or objects with ``u`` / ``v`` in m/s (east/north positive),
    plus ``times`` / ``lats`` / ``lons``. ``temperature`` is optional.

Both attribute access (``field.u``) and mapping access
(``field["u"]``) are honored, so plain dicts work without wrappers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class UnrecognizedFieldError(ValueError):
    """The field's shape was not recognized; no statistics were computed."""


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute-then-mapping lookup so dicts and objects both work."""
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    return default


def _has(obj: Any, name: str) -> bool:
    return hasattr(obj, name) or (isinstance(obj, dict) and name in obj)


def _as_float_1d(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D coordinate vector, got shape {arr.shape}")
    return arr


def _as_float_3d(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be (nt, ny, nx), got shape {arr.shape}")
    return arr


def _shape_of(grid_map: Any) -> bool:
    return isinstance(grid_map, dict) or _has(grid_map, "__getitem__")


@dataclass(frozen=True)
class NormalizedField:
    """One internal representation both shapes normalize into."""

    u: np.ndarray                    # (nt, ny, nx) eastward component, m/s
    v: np.ndarray                    # (nt, ny, nx) northward component, m/s
    temperature: Optional[np.ndarray]  # (nt, ny, nx) or None
    temperature_unit: Optional[str]  # canonical "°C" or "°F" (None when no temp)
    times: List[str]                 # ISO-8601 timestamps, one per step
    lats: np.ndarray                 # (ny,) degrees north
    lons: np.ndarray                 # (nx,) degrees east
    kind: str                        # "wind" or "current"


def _canonical_temperature_unit(raw: Any) -> Optional[str]:
    """Map loose unit strings to "°C" / "°F"; None means unparseable."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s in ("°C", "degC", "C", "celsius", "Celsius"):
        return "°C"
    if s in ("°F", "degF", "F", "fahrenheit", "Fahrenheit"):
        return "°F"
    return None


def _coerce_times(field: Any) -> List[str]:
    raw = _get(field, "times")
    if raw is None:
        return []
    return [str(t) for t in raw]


def _coerce_coord(field: Any, name: str) -> np.ndarray:
    raw = _get(field, "lats") if name == "lats" else _get(field, "lons")
    if raw is None:
        raw = []
    return _as_float_1d(raw, name)


def _validate_coords(times: List[str], lats: np.ndarray, lons: np.ndarray,
                     u: np.ndarray, kind: str) -> None:
    nt, ny, nx = u.shape
    if len(times) != nt:
        raise ValueError(
            f"{kind} field: len(times)={len(times)} does not match grid nt={nt}")
    if lats.shape[0] != ny or lons.shape[0] != nx:
        raise ValueError(
            f"{kind} field: lats/lons ({lats.shape[0]}, {lons.shape[0]}) "
            f"do not match grid (ny, nx)=({ny}, {nx})")


def normalize_field(field: Any, temperature_unit: Optional[str] = None) -> NormalizedField:
    """Detect the field's shape and normalize it to :class:`NormalizedField`.

    Parameters
    ----------
    field:
        A GFS-wind-shaped field (``grids["u10"]/grids["v10"]``) or a
        CurrentField-shaped field (``u`` / ``v``), as a dict or an object.
    temperature_unit:
        Optional override for the temperature unit ("°C" / "°F"). When
        given, it wins over any unit declared on the field. When omitted,
        the field's declared ``temperature_unit`` (or the GFS ``t2m`` = °C
        convention) is used.

    Raises
    ------
    UnrecognizedFieldError
        The field matches neither supported shape.
    ValueError
        Coordinate/array dimensions are inconsistent.
    """
    # --- gfs-wind shape: grids["u10"]/grids["v10"] ---------------------
    grids = _get(field, "grids")
    if grids is not None and _shape_of(grids):
        u10, v10 = None, None
        if _has(grids, "__contains__"):
            try:
                u10 = grids["u10"]
                v10 = grids["v10"]
            except (KeyError, TypeError):
                u10 = v10 = None
        if u10 is not None and v10 is not None:
            return _normalize_gfs(field, grids, u10, v10, temperature_unit)
        # grids exists but lacks u10/v10 -> unrecognized, not "guessed".
        raise UnrecognizedFieldError(
            "unrecognized field shape: 'grids' is present but lacks "
            "'u10'/'v10' wind components (GFS-wind shape) and the field "
            "has no 'u'/'v' attributes (CurrentField shape). "
            "survey-narrate accepts only: grids['u10']/grids['v10'] "
            "+ times/lats/lons (winds), or u/v + times/lats/lons (currents)."
        )

    # --- CurrentField shape: u / v attributes or keys ------------------
    u = _get(field, "u")
    v = _get(field, "v")
    if u is not None and v is not None:
        return _normalize_current(field, u, v, temperature_unit)

    raise UnrecognizedFieldError(
        "unrecognized field shape: expected either "
        "grids['u10']/grids['v10'] + times/lats/lons (GFS-wind shape) or "
        "u/v + times/lats/lons (CurrentField shape); got "
        f"{type(field).__name__}. survey-narrate does not guess field "
        "layouts."
    )


def _normalize_gfs(field: Any, grids: Any, u10: Any, v10: Any,
                   temperature_unit: Optional[str]) -> NormalizedField:
    u = _as_float_3d(u10, "grids['u10']")
    v = _as_float_3d(v10, "grids['v10']")
    if u.shape != v.shape:
        raise ValueError(
            f"gfs-wind field: grids['u10'] shape {u.shape} != "
            f"grids['v10'] shape {v.shape}")

    # Temperature: air_temperature (3-D, unit declared on the field) first,
    # then grids["t2m"] (GFS convention: °C), then None.
    temperature = None
    unit: Optional[str] = None
    air_temp = _get(field, "air_temperature")
    if air_temp is not None:
        temperature = _as_float_3d(air_temp, "air_temperature")
        unit = _canonical_temperature_unit(_get(field, "temperature_unit"))
        if unit is None:
            unit = "°C"  # GFS store convention when nothing is declared
    else:
        t2m = grids.get("t2m") if isinstance(grids, dict) else None
        if t2m is not None:
            temperature = _as_float_3d(t2m, "grids['t2m']")
            unit = "°C"
    if temperature_unit is not None:
        hinted = _canonical_temperature_unit(temperature_unit)
        if hinted is None:
            raise ValueError(
                f"temperature_unit hint must be °C/°F-like, got {temperature_unit!r}")
        unit = hinted

    times = _coerce_times(field)
    lats = _coerce_coord(field, "lats")
    lons = _coerce_coord(field, "lons")
    _validate_coords(times, lats, lons, u, "gfs-wind")
    return NormalizedField(u=u, v=v, temperature=temperature,
                           temperature_unit=unit, times=times,
                           lats=lats, lons=lons, kind="wind")


def _normalize_current(field: Any, u_raw: Any, v_raw: Any,
                       temperature_unit: Optional[str]) -> NormalizedField:
    u = _as_float_3d(u_raw, "u")
    v = _as_float_3d(v_raw, "v")
    if u.shape != v.shape:
        raise ValueError(
            f"current field: u shape {u.shape} != v shape {v.shape}")
    temperature = None
    t_raw = _get(field, "temperature")
    if t_raw is not None:
        temperature = _as_float_3d(t_raw, "temperature")
        if temperature.shape != u.shape:
            raise ValueError(
                f"current field: temperature shape {temperature.shape} "
                f"does not match u/v shape {u.shape}")
    unit = _canonical_temperature_unit(temperature_unit)
    if temperature is not None and unit is None:
        unit = "°C"  # CurrentField convention (survey-currents): degC
    if temperature_unit is not None and unit is None:
        raise ValueError(
            f"temperature_unit hint must be °C/°F-like, got {temperature_unit!r}")

    times = _coerce_times(field)
    lats = _coerce_coord(field, "lats")
    lons = _coerce_coord(field, "lons")
    _validate_coords(times, lats, lons, u, "current")
    return NormalizedField(u=u, v=v, temperature=temperature,
                           temperature_unit=unit, times=times,
                           lats=lats, lons=lons, kind="current")
