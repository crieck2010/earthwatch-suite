"""Small GeoJSON read/write helpers (stdlib only).

The cross-suite interchange contract is GeoJSON in / GeoJSON out; this
module keeps that dependency-free.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def read_geojson(path: str) -> Dict[str, Any]:
    """Read a GeoJSON file; must be a FeatureCollection."""
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict) or obj.get("type") != "FeatureCollection":
        raise ValueError(f"{path}: expected a GeoJSON FeatureCollection")
    if not isinstance(obj.get("features"), list):
        raise ValueError(f"{path}: FeatureCollection has no 'features' list")
    return obj


def write_geojson(path: str, features: List[Dict[str, Any]], **foreign: Any) -> str:
    """Write features as a FeatureCollection. Returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    obj: Dict[str, Any] = {"type": "FeatureCollection", "features": features}
    obj.update(foreign)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
    return path


def linestring_feature(
    coords: List[List[float]], properties: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": properties,
    }
