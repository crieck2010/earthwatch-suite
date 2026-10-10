"""Zone loading from GeoJSON.

A *zone* is a named polygon (urban district, rural reference field, study
area). Zones are the unit of the time series: one row per (zone, pass).
Mirrors the zone model of the sibling analytics engines (survey-vegetation,
survey-flood, survey-burn, survey-coast) so zone files are interchangeable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class Zone:
    """One named polygon zone."""

    id: str
    name: str
    geometry: dict  # GeoJSON Polygon geometry
    kind: str = ""  # e.g. "urban", "rural" — informational only

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
    features = data.get("features", [])
    if not features:
        raise ValueError(f"no features in {path}")
    return [_zone_from_feature(f, i) for i, f in enumerate(features)]


def load_zones_directory(directory: str) -> Dict[str, List[Zone]]:
    """Load every ``*.geojson`` file in a directory → {stem: zones}."""
    out: Dict[str, List[Zone]] = {}
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(".geojson"):
            out[os.path.splitext(fname)[0]] = load_zones_geojson(
                os.path.join(directory, fname)
            )
    return out
