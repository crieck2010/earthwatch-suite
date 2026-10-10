"""Flood-extent polygons: vectorise water masks to GeoJSON.

Each connected water component becomes one GeoJSON Polygon feature with
pixel-exact boundaries — no shapely, no rasterio. The boundary is built by
collecting the edges between water and non-water pixels and chaining them
into closed rings, so the polygon area equals the component's pixel count
times the pixel area exactly.

Georeferencing comes from a 6-tuple affine transform
``(a, b, c, d, e, f)`` with ``x = a*col + b*row + c``,
``y = d*col + e*row + f`` (the GDAL/rasterio convention). For a plain
north-up grid with pixel size ``s`` and top-left corner ``(ox, oy)`` that
is ``(s, 0, ox, 0, -s, oy)``. In ``--synthetic`` mode the demo uses pixel
coordinates (``(1, 0, 0, 0, -1, 0)``) and says so.

``min_area_ha`` (or equivalently ``min_pixels``) drops noise speckles before
vectorisation — the same despeckling idea as the raster path, applied at the
component level.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .watermask import component_sizes, label_components

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
    direction). Each side shared with a non-component pixel emits one edge,
    wound so the component interior stays on the right of travel.
    """
    rows, cols = comp.shape
    edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if not comp[r, c]:
                continue
            # top
            if r == 0 or not comp[r - 1, c]:
                edges.append(((c, r), (c + 1, r)))
            # right
            if c == cols - 1 or not comp[r, c + 1]:
                edges.append(((c + 1, r), (c + 1, r + 1)))
            # bottom
            if r == rows - 1 or not comp[r + 1, c]:
                edges.append(((c + 1, r + 1), (c, r + 1)))
            # left
            if c == 0 or not comp[r, c - 1]:
                edges.append(((c, r + 1), (c, r)))
    return edges


def _chain_rings(
    edges: List[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> List[List[Tuple[int, int]]]:
    """Chain directed edges into closed rings."""
    start_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for a, b in edges:
        start_map.setdefault(a, []).append(b)
    rings: List[List[Tuple[int, int]]] = []
    while start_map:
        start = next(iter(start_map))
        ring = [start]
        node = start
        while True:
            nxt_list = start_map.get(node)
            if not nxt_list:
                raise RuntimeError("dangling boundary edge while chaining rings")
            nxt = nxt_list.pop()
            if not nxt_list:
                del start_map[node]
            if nxt == start:
                break
            ring.append(nxt)
            node = nxt
        rings.append(ring)
    return rings


def _ring_area(ring: Sequence[Tuple[float, float]]) -> float:
    """Shoelace area (signed) of a ring."""
    s = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def _to_map(
    ring: Sequence[Tuple[int, int]], transform: Sequence[float]
) -> List[List[float]]:
    a, b, c, d, e, f = transform
    return [[a * x + b * y + c, d * x + e * y + f] for x, y in ring]


def vectorize(
    water: np.ndarray,
    transform: Sequence[float] = PIXEL_TRANSFORM,
    min_pixels: int = 4,
) -> List[Dict[str, Any]]:
    """Vectorise a water mask to GeoJSON Polygon features.

    Returns a list of feature dicts (no FeatureCollection wrapper — see
    :func:`write_polygons_geojson`). Each feature carries ``component_id``,
    ``pixel_count`` and ``area_ha`` (from the pixel size embedded in the
    transform: ``sqrt(|a*e - b*d|)``).
    """
    water = np.asarray(water, dtype=bool)
    if len(transform) != 6:
        raise ValueError("transform must be a 6-tuple affine")
    labels, n = label_components(water)
    if n == 0:
        return []
    sizes = component_sizes(labels, n)
    a, b, _c, d, e, _f = transform
    pixel_size = abs(a * e - b * d) ** 0.5
    pixel_ha = (pixel_size ** 2) / _M2_PER_HA
    features: List[Dict[str, Any]] = []
    for comp_id in range(1, n + 1):
        if sizes[comp_id] < min_pixels:
            continue
        comp = labels == comp_id
        rings = _chain_rings(_component_edges(comp))
        # Exterior = largest ring by absolute area; the rest are holes.
        rings.sort(key=lambda r: abs(_ring_area(r)), reverse=True)
        map_rings = []
        for ring in rings:
            mapped = _to_map(ring, transform)
            mapped.append(mapped[0])  # GeoJSON rings are closed
            map_rings.append(mapped)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "component_id": comp_id,
                    "pixel_count": sizes[comp_id],
                    "area_ha": sizes[comp_id] * pixel_ha,
                },
                "geometry": {"type": "Polygon", "coordinates": map_rings},
            }
        )
    return features


def write_polygons_geojson(
    features: Sequence[Dict[str, Any]],
    path: str,
    extra_properties: Dict[str, Any] | None = None,
) -> str:
    """Write polygon features as a GeoJSON FeatureCollection.

    ``extra_properties`` (e.g. zone_id, pass_id, date) are merged into every
    feature's properties. Returns the path.
    """
    extra = extra_properties or {}
    for feat in features:
        feat["properties"] = {**extra, **feat["properties"]}
    collection = {"type": "FeatureCollection", "features": list(features)}
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(collection, fh, indent=2)
    return path
