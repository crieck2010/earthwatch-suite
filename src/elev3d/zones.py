"""Zone loading from GeoJSON.

A *zone* is a named polygon (construction site, quarry bench, stockpile
area, stable road used for coregistration checks). Zones are the unit of
the volumetric records: one row per (zone, DEM pair). Mirrors the zone
model of the sibling analytics engines (survey-vegetation, survey-flood,
survey-burn, survey-coast, survey-thermal) so zone files are
interchangeable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Zone:
    """One named polygon zone."""

    id: str
    name: str
    geometry: dict  # GeoJSON Polygon geometry
    kind: str = ""  # e.g. "pit", "stockpile", "stable" — informational only

    @property
    def bbox(self):
        xs = [p[0] for ring in self.geometry["coordinates"] for p in ring]
        ys = [p[1] for ring in self.geometry["coordinates"] for p in ring]
        return (min(xs), min(ys), max(xs), max(ys))


def _zone_from_feature(feature: dict, idx: int) -> Zone:
    props = feature.get("properties") or {}
    geom = feature.get("geometry") or {}
    if geom.get("type") != "Polygon":
        raise ValueError(
            f"feature {idx}: only Polygon geometries are supported "
            f"(got {geom.get('type')})"
        )
    zone_id = str(props.get("id") or props.get("zone_id") or f"zone-{idx}")
    name = str(props.get("name") or zone_id)
    kind = str(props.get("kind") or "")
    return Zone(id=zone_id, name=name, geometry=geom, kind=kind)


def load_zones_geojson(path: str) -> List[Zone]:
    """Load zones from a GeoJSON FeatureCollection file."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("type") != "FeatureCollection":
        raise ValueError("zones file must be a GeoJSON FeatureCollection")
    features = data.get("features") or []
    if not features:
        raise ValueError("zones file contains no features")
    return [_zone_from_feature(f, i) for i, f in enumerate(features)]


def zones_to_geojson(zones: List[Zone]) -> Dict:
    """Back to a GeoJSON FeatureCollection (round-trip helper)."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": z.id, "name": z.name, "kind": z.kind},
                "geometry": z.geometry,
            }
            for z in zones
        ],
    }
