"""Disturbance polygons: vectorise burn/severity masks to GeoJSON.

Each connected burned component becomes one GeoJSON Polygon feature with
pixel-exact boundaries — no shapely, no rasterio. The boundary is built by
collecting the edges between burned and non-burned pixels and chaining them
into closed rings, so the polygon area equals the component's pixel count
times the pixel area exactly. Interior holes (unburned islands inside a
scar) are preserved as interior rings.

Georeferencing comes from a 6-tuple affine transform
``(a, b, c, d, e, f)`` with ``x = a*col + b*row + c``,
``y = d*col + e*row + f`` (the GDAL/rasterio convention). For a plain
north-up grid with pixel size ``s`` and top-left corner ``(ox, oy)`` that
is ``(s, 0, ox, 0, -s, oy)``.

Each feature carries ``area_ha``, ``pixels``, and ``dominant_severity``
(the most common severity class id in the component, from
:mod:`burn.severity`, or ``null`` when no severity raster was supplied).
``min_area_ha`` drops noise speckles before vectorisation.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .masking import label_components
from .severity import CLASS_BY_ID

#: Square metres per hectare.
_M2_PER_HA = 10_000.0

#: Identity-ish demo transform: pixel coordinates, y pointing down like rows.
PIXEL_TRANSFORM: Tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)


def north_up_transform(origin_x: float, origin_y: float, pixel_size_m: float) -> Tuple[float, ...]:
    """Affine 6-tuple for a north-up grid."""
    return (pixel_size_m, 0.0, origin_x, 0.0, -pixel_size_m, origin_y)


def _component_edges(comp: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Directed boundary edges of one component, in pixel-corner coords.

    For pixel (r, c), corners are (x=c, y=r) with y growing *down* (row
    direction). Each side shared with a non-component pixel emits one edge.
    """
    rows, cols = comp.shape
    edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if not comp[r, c]:
                continue
            if r == 0 or not comp[r - 1, c]:
                edges.append(((c, r), (c + 1, r)))
            if c == cols - 1 or not comp[r, c + 1]:
                edges.append(((c + 1, r), (c + 1, r + 1)))
            if r == rows - 1 or not comp[r + 1, c]:
                edges.append(((c + 1, r + 1), (c, r + 1)))
            if c == 0 or not comp[r, c - 1]:
                edges.append(((c, r + 1), (c, r)))
    return edges


def _chain_edges(edges: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """Chain directed edges into closed rings."""
    start_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for a, b in edges:
        start_map.setdefault(a, []).append(b)
    rings: List[List[Tuple[int, int]]] = []
    while start_map:
        start = next(iter(start_map))
        ring = [start]
        cur = start
        while True:
            nxt_list = start_map.get(cur)
            if not nxt_list:
                break
            nxt = nxt_list.pop()
            if not nxt_list:
                del start_map[cur]
            if nxt == start:
                break
            ring.append(nxt)
            cur = nxt
        if len(ring) >= 3:
            rings.append(ring)
    return rings


def _ring_area(ring: List[Tuple[int, int]]) -> float:
    """Signed shoelace area in pixel-corner units (y grows down)."""
    s = 0.0
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return s / 2.0


def _to_world(pt: Tuple[int, int], transform: Tuple[float, ...]) -> List[float]:
    a, b, c, d, e, f = transform
    x, y = pt
    return [a * x + b * y + c, d * x + e * y + f]


def vectorize(
    burn_mask: np.ndarray,
    transform: Tuple[float, ...] = PIXEL_TRANSFORM,
    severity: np.ndarray | None = None,
    min_area_ha: float = 0.0,
    pixel_size_m: float = 10.0,
) -> List[Dict[str, Any]]:
    """Vectorise a burn mask to GeoJSON Polygon features.

    ``min_area_ha`` filters speckles before vectorisation. Returns a list of
    GeoJSON Feature dicts (not a full FeatureCollection).
    """
    mask = np.asarray(burn_mask, dtype=bool)
    sev = np.asarray(severity, dtype=np.uint8) if severity is not None else None
    pixel_ha = (pixel_size_m ** 2) / _M2_PER_HA
    min_pixels = int(min_area_ha / pixel_ha) if pixel_ha > 0 else 0
    labels, sizes = label_components(mask, connectivity=8)
    features: List[Dict[str, Any]] = []
    for lab, size in sorted(sizes.items(), key=lambda kv: kv[1], reverse=True):
        if size < min_pixels:
            continue
        comp = labels == lab
        edges = _component_edges(comp)
        rings = _chain_edges(edges)
        if not rings:
            continue
        # Outer ring = largest absolute area; the rest are holes.
        rings.sort(key=lambda r: abs(_ring_area(r)), reverse=True)
        world_rings = [[_to_world(pt, transform) for pt in ring] for ring in rings]
        # Close each ring explicitly for valid GeoJSON.
        world_rings = [ring + [ring[0]] for ring in world_rings]
        dominant = None
        if sev is not None:
            vals, counts = np.unique(sev[comp & (sev != 255)], return_counts=True)
            if len(vals):
                dominant = int(vals[int(np.argmax(counts))])
        props: Dict[str, Any] = {
            "pixels": int(size),
            "area_ha": float(size * pixel_ha),
            "dominant_severity": dominant,
            "dominant_severity_name": CLASS_BY_ID[dominant][0] if dominant is not None else None,
        }
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Polygon", "coordinates": world_rings},
            }
        )
    return features


def write_geojson(features: List[Dict[str, Any]], path: str) -> str:
    """Write features as a GeoJSON FeatureCollection."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fc = {"type": "FeatureCollection", "features": features}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fc, fh, indent=2)
    return path
