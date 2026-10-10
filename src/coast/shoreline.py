"""Shoreline extraction: the water/land boundary as polylines.

Given a boolean water mask, every water pixel adjacent to a non-water
pixel contributes its shared grid edge; edges are chained into continuous
polylines and converted to map meters with the raster transform.

Pure numpy + dict chaining — no skimage/scipy. Output is GeoJSON
LineStrings (see :mod:`coast.geojson`), which open directly in QGIS.

Edge conventions
----------------
- Edges are emitted only between *valid* water and *valid* land. Invalid
  pixels (clouds, NaN) and out-of-bounds neighbours emit nothing: clouds
  hide the shoreline, they don't create one, and a coast running off-tile
  is a real coast, not a boundary.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def _boundary_edges(
    water: np.ndarray, valid: np.ndarray
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Grid edges (pixel coords) between valid water and valid land cells."""
    w = np.asarray(water, dtype=bool) & np.asarray(valid, dtype=bool)
    v = np.asarray(valid, dtype=bool)
    # Pad with invalid (False): tile edges and the great beyond emit nothing.
    pw = np.pad(w, 1, mode="constant", constant_values=False)
    pv = np.pad(v, 1, mode="constant", constant_values=False)
    rows, cols = w.shape
    edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    def is_land(pr: int, pc: int) -> bool:
        return bool(pv[pr, pc] and not pw[pr, pc])

    for r in range(rows):
        for c in range(cols):
            if not w[r, c]:
                continue
            pr, pc = r + 1, c + 1  # padded coords
            if is_land(pr - 1, pc):  # north neighbour is valid land
                edges.append(((c, r), (c + 1, r)))
            if is_land(pr, pc + 1):  # east
                edges.append(((c + 1, r), (c + 1, r + 1)))
            if is_land(pr + 1, pc):  # south
                edges.append(((c, r + 1), (c + 1, r + 1)))
            if is_land(pr, pc - 1):  # west
                edges.append(((c, r), (c, r + 1)))
    return edges


def _chain_edges(
    edges: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> List[List[Tuple[int, int]]]:
    """Join unit edges into polylines by matching endpoints."""
    # endpoint -> list of (edge_id, end_index)
    adj: Dict[Tuple[int, int], List[Tuple[int, Tuple[int, int]]]] = {}
    for i, (a, b) in enumerate(edges):
        adj.setdefault(a, []).append((i, b))
        adj.setdefault(b, []).append((i, a))
    used = [False] * len(edges)
    lines: List[List[Tuple[int, int]]] = []
    for i, (a, b) in enumerate(edges):
        if used[i]:
            continue
        used[i] = True
        line = [a, b]
        # extend forward from b
        cur = b
        while True:
            nxt = None
            for j, other in adj.get(cur, []):
                if not used[j]:
                    nxt = (j, other)
                    break
            if nxt is None:
                break
            j, other = nxt
            used[j] = True
            line.append(other)
            cur = other
        # extend backward from a
        cur = a
        while True:
            nxt = None
            for j, other in adj.get(cur, []):
                if not used[j]:
                    nxt = (j, other)
                    break
            if nxt is None:
                break
            j, other = nxt
            used[j] = True
            line.insert(0, other)
            cur = other
        lines.append(line)
    return lines


def _to_meters(
    line: Sequence[Tuple[int, int]],
    pixel_size_m: float,
    origin_x: float,
    origin_y: float,
) -> np.ndarray:
    """Pixel grid coords -> map meters (north-up: y decreases with row)."""
    pts = np.asarray(line, dtype=float)
    out = np.empty_like(pts)
    out[:, 0] = origin_x + pts[:, 0] * pixel_size_m
    out[:, 1] = origin_y - pts[:, 1] * pixel_size_m
    return out


def extract_shoreline(
    water: np.ndarray,
    valid: np.ndarray | None = None,
    pixel_size_m: float = 10.0,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    min_length_m: float = 0.0,
) -> List[np.ndarray]:
    """Extract shoreline polylines from a water mask.

    ``valid`` marks usable pixels (not cloud / not NaN); when omitted every
    pixel counts as valid. Returns a list of (N, 2) arrays in map meters.
    ``min_length_m`` drops short fragments (pond edges, speckle remnants).
    """
    water = np.asarray(water, dtype=bool)
    if valid is None:
        valid = np.ones_like(water, dtype=bool)
    edges = _boundary_edges(water, np.asarray(valid, dtype=bool))
    lines = []
    for line in _chain_edges(edges):
        pts = _to_meters(line, pixel_size_m, origin_x, origin_y)
        if len(pts) < 2:
            continue
        length = float(np.abs(np.diff(pts[:, 0])).sum() + np.abs(np.diff(pts[:, 1])).sum())
        # Manhattan length is exact for axis-aligned unit edges.
        if length >= min_length_m:
            lines.append(pts)
    return lines


def shoreline_length_m(lines: Sequence[np.ndarray]) -> float:
    """Total length of shoreline polylines in meters (Euclidean)."""
    total = 0.0
    for pts in lines:
        if len(pts) >= 2:
            total += float(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])).sum())
    return total
