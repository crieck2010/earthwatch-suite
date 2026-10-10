"""Site loading: coastal sites the analytics run over.

A *site* is a named polygon (a beach compartment, a municipal shoreline
reach, an island). Sites arrive as a GeoJSON FeatureCollection; each
feature must be a Polygon or MultiPolygon with a string ``id`` (or
``name``) property. The same files work with the sibling engines'
``zones`` loaders — the cross-suite interchange contract is GeoJSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

_VALID_TYPES = ("Polygon", "MultiPolygon")


@dataclass(frozen=True)
class Site:
    """A named polygon site."""

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


def _site_id(props: Dict[str, Any], index: int) -> str:
    for key in ("id", "name", "site_id"):
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"site-{index + 1}"


def load_sites_geojson(path: str) -> List[Site]:
    """Load sites from a GeoJSON FeatureCollection file."""
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    if obj.get("type") != "FeatureCollection" or not isinstance(
        obj.get("features"), list
    ):
        raise ValueError(f"{path}: expected a GeoJSON FeatureCollection")
    sites: List[Site] = []
    for i, feat in enumerate(obj["features"]):
        geom = feat.get("geometry") or {}
        if geom.get("type") not in _VALID_TYPES:
            raise ValueError(
                f"{path}: feature {i} must be a Polygon/MultiPolygon "
                f"(got {geom.get('type')!r})"
            )
        props = feat.get("properties") or {}
        sid = _site_id(props, i)
        sites.append(
            Site(id=sid, name=str(props.get("name", sid)), geometry=geom)
        )
    if not sites:
        raise ValueError(f"{path}: no site features found")
    return sites
