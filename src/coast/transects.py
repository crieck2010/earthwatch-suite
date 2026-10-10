"""Transect-based shoreline positions.

A *transect* is a user-supplied LineString, drawn roughly perpendicular to
the shore. For each pass's shoreline polylines we compute where the
transect crosses the shoreline and record the distance from the transect's
**first vertex** to the crossing, in meters.

Orientation convention (documented, not enforced): draw transects from the
**landward** end to the **seaward** end. Then:

- position increases → shoreline advanced seaward (accretion)
- position decreases → shoreline retreated landward (erosion)

If a transect crosses several shoreline fragments, the median crossing is
used (robust to pond/lagoon edges behind the beach). No crossing → NaN
position (cloud gap, dry pass) and ``valid=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class Transect:
    """A named transect line in map meters."""

    id: str
    name: str
    coords: np.ndarray  # (N, 2) in meters

    @property
    def origin(self) -> np.ndarray:
        return self.coords[0]

    @property
    def direction(self) -> np.ndarray:
        """Unit vector from first to last vertex."""
        d = self.coords[-1] - self.coords[0]
        n = float(np.hypot(d[0], d[1]))
        if n == 0:
            raise ValueError(f"transect {self.id!r}: first == last vertex")
        return d / n

    @property
    def length_m(self) -> float:
        d = self.coords[-1] - self.coords[0]
        return float(np.hypot(d[0], d[1]))


def _transect_id(props: Dict[str, Any], index: int) -> str:
    for key in ("id", "name", "transect_id"):
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"transect-{index + 1}"


def load_transects_geojson(path: str) -> List[Transect]:
    """Load transects from a GeoJSON FeatureCollection of LineStrings."""
    from .geojson import read_geojson

    obj = read_geojson(path)
    transects: List[Transect] = []
    for i, feat in enumerate(obj["features"]):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "LineString":
            raise ValueError(
                f"{path}: feature {i} must be a LineString "
                f"(got {geom.get('type')!r})"
            )
        coords = np.asarray(geom.get("coordinates"), dtype=float)
        if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] < 2:
            raise ValueError(f"{path}: feature {i} needs >= 2 coordinates")
        props = feat.get("properties") or {}
        tid = _transect_id(props, i)
        transects.append(
            Transect(id=tid, name=str(props.get("name", tid)), coords=coords[:, :2])
        )
    if not transects:
        raise ValueError(f"{path}: no transect features found")
    return transects


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def transect_shoreline_crossings(
    transect: Transect, shorelines: Sequence[np.ndarray]
) -> List[float]:
    """Distances (m) from the transect origin to each shoreline crossing."""
    o = transect.origin
    u = transect.direction
    tmax = transect.length_m
    crossings: List[float] = []
    for line in shorelines:
        pts = np.asarray(line, dtype=float)
        if len(pts) < 2:
            continue
        for p1, p2 in zip(pts[:-1], pts[1:]):
            d = p2 - p1
            denom = _cross2(u, d)
            if abs(denom) < 1e-12:
                continue  # parallel
            w = p1 - o
            t = _cross2(w, d) / denom
            s = _cross2(w, u) / denom
            if 0.0 <= s <= 1.0 and 0.0 <= t <= tmax:
                crossings.append(t)
    return crossings


def transect_position(
    transect: Transect, shorelines: Sequence[np.ndarray]
) -> float:
    """Median crossing distance in meters, or NaN when there is none."""
    crossings = transect_shoreline_crossings(transect, shorelines)
    if not crossings:
        return float("nan")
    return float(np.median(crossings))
