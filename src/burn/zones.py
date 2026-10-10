"""Zone loading: management units / fire perimeters the analytics run over.

A *zone* is a named polygon (a timber compartment, a burn-perimeter
candidate, a construction site, a monitoring block). Zones arrive as a
GeoJSON FeatureCollection; each feature must be a Polygon or MultiPolygon
with a string ``id`` (or ``name``) property. This is the "GeoJSON in" side
of the cross-suite interchange contract — the same zone files work with
survey-vegetation and survey-flood.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

_VALID_TYPES = ("Polygon", "MultiPolygon")


@dataclass(frozen=True)
class Zone:
    """A named polygon zone."""

    id: str
    name: str
    geometry: Dict[str, Any]  # GeoJSON geometry dict

    def bbox(self) -> tuple:
        """(minx, miny, maxx, maxy) over all rings."""
        xs: List[float] = []
        ys: List[float] = []

        def walk(coords: Any) -> None:
            if isinstance(coords[0], (int, float)):
                xs.append(float(coords[0]))
                ys.append(float(coords[1]))
            else:
                for part in coords:
                    walk(part)

        walk(self.geometry["coordinates"])
        return (min(xs), min(ys), max(xs), max(ys))


def _zone_id(props: Dict[str, Any], index: int) -> str:
    for key in ("id", "name", "zone_id"):
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"zone-{index + 1}"


def load_zones_geojson(path: str) -> List[Zone]:
    """Load zones from a GeoJSON FeatureCollection file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return zones_from_feature_collection(data)


def zones_from_feature_collection(data: Dict[str, Any]) -> List[Zone]:
    """Build zones from an already-parsed GeoJSON FeatureCollection."""
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        raise ValueError("zones must be a GeoJSON FeatureCollection")
    features = data.get("features", [])
    if not features:
        raise ValueError("zone FeatureCollection has no features")
    zones: List[Zone] = []
    seen = set()
    for i, feat in enumerate(features):
        if not isinstance(feat, dict) or feat.get("type") != "Feature":
            raise ValueError(f"feature {i} is not a GeoJSON Feature")
        geom = feat.get("geometry") or {}
        if geom.get("type") not in _VALID_TYPES:
            raise ValueError(
                f"feature {i}: zone geometry must be Polygon/MultiPolygon, "
                f"got {geom.get('type')!r}"
            )
        props = feat.get("properties") or {}
        zid = _zone_id(props, i)
        if zid in seen:
            raise ValueError(f"duplicate zone id {zid!r}")
        seen.add(zid)
        zones.append(
            Zone(id=zid, name=str(props.get("name", zid)), geometry=geom)
        )
    return zones
