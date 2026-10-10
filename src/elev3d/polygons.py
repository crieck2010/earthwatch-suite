"""Volumetric change polygons: vectorize significant-change areas to GeoJSON.

Pipeline: threshold the difference raster (|Δh| >= threshold) → label
8-connected components (pure numpy, no scipy) → drop components smaller
than ``min_pixels`` → trace each component's boundary by chaining directed
pixel edges (interior on the left) → emit GeoJSON Polygons with per-feature
cut/fill volumes and dominant change class.

Holes are preserved: rings from one component are grouped, the largest
ring is the exterior, the rest are interior holes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class ChangePolygon:
    """One vectorized change area."""

    exterior: List[Tuple[float, float]]  # map coords, closed ring
    holes: List[List[Tuple[float, float]]]
    cut_m3: float
    fill_m3: float
    net_m3: float
    dominant: str  # "cut" | "fill"
    area_m2: float
    pixel_count: int

    def to_feature(self) -> Dict:
        rings = [self.exterior] + self.holes
        return {
            "type": "Feature",
            "properties": {
                "cut_m3": self.cut_m3,
                "fill_m3": self.fill_m3,
                "net_m3": self.net_m3,
                "dominant": self.dominant,
                "area_m2": self.area_m2,
                "pixel_count": self.pixel_count,
            },
            "geometry": {"type": "Polygon", "coordinates": rings},
        }


def label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """8-connected component labelling, pure numpy. Returns (labels, n)."""
    m = np.asarray(mask, dtype=bool)
    labels = np.zeros(m.shape, dtype=np.int32)
    current = 0
    rows, cols = m.shape
    for r in range(rows):
        for c in range(cols):
            if not m[r, c] or labels[r, c]:
                continue
            current += 1
            queue = deque([(r, c)])
            labels[r, c] = current
            while queue:
                pr, pc = queue.popleft()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = pr + dr, pc + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and m[nr, nc]
                            and not labels[nr, nc]
                        ):
                            labels[nr, nc] = current
                            queue.append((nr, nc))
    return labels, current


def _boundary_edges(component: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Directed boundary edges of a boolean component mask, interior left.

    Pixel (r, c) spans corners (c..c+1, r..r+1). An edge is a boundary
    edge when the neighbour across it is not in the component.
    """
    rows, cols = component.shape
    edges = []
    for r in range(rows):
        for c in range(cols):
            if not component[r, c]:
                continue
            # top: neighbour (r-1, c)
            if r == 0 or not component[r - 1, c]:
                edges.append(((c + 1, r), (c, r)))
            # bottom: neighbour (r+1, c)
            if r == rows - 1 or not component[r + 1, c]:
                edges.append(((c, r + 1), (c + 1, r + 1)))
            # left: neighbour (r, c-1)
            if c == 0 or not component[r, c - 1]:
                edges.append(((c, r), (c, r + 1)))
            # right: neighbour (r, c+1)
            if c == cols - 1 or not component[r, c + 1]:
                edges.append(((c + 1, r + 1), (c + 1, r)))
    return edges


def _chain_rings(
    edges: List[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> List[List[Tuple[int, int]]]:
    """Chain directed edges into closed rings."""
    start_map: Dict[Tuple[int, int], List[int]] = {}
    for i, (s, _e) in enumerate(edges):
        start_map.setdefault(s, []).append(i)
    used = [False] * len(edges)
    rings = []
    for i, (s, e) in enumerate(edges):
        if used[i]:
            continue
        ring = [s]
        used[i] = True
        cur = e
        while cur != s:
            nxt = None
            for j in start_map.get(cur, ()):
                if not used[j]:
                    nxt = j
                    break
            if nxt is None:  # pragma: no cover - defensive; edges always chain
                break
            used[nxt] = True
            ring.append(cur)
            cur = edges[nxt][1]
        ring.append(s)
        rings.append(ring)
    return rings


def _ring_area(ring: Sequence[Tuple[float, float]]) -> float:
    """Signed shoelace area."""
    s = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        s += x1 * y2 - x2 * y1
    return s / 2.0


def _to_map(
    ring: Sequence[Tuple[int, int]],
    transform: Tuple[float, float, float, float, float, float],
) -> List[Tuple[float, float]]:
    a, _b, c, _d, e, f = transform
    return [(c + a * x, f + e * y) for x, y in ring]


def change_polygons(
    diff: np.ndarray,
    transform: Tuple[float, float, float, float, float, float],
    threshold: float = 0.10,
    min_pixels: int = 4,
) -> List[ChangePolygon]:
    """Vectorize significant-change areas of a difference raster (metres).

    ``threshold`` is the |Δh| change-detection threshold; components with
    fewer than ``min_pixels`` pixels are dropped as speckle.
    """
    arr = np.asarray(diff, dtype=float)
    a, _b, _c, _d, e, _f = transform
    px_area = abs(a * e)
    if px_area <= 0:
        raise ValueError("degenerate pixel area")

    significant = np.isfinite(arr) & (np.abs(arr) >= threshold)
    labels, n = label_components(significant)
    out: List[ChangePolygon] = []
    for lab in range(1, n + 1):
        comp = labels == lab
        count = int(np.count_nonzero(comp))
        if count < min_pixels:
            continue
        # Crop to the component bbox: edge tracing is O(bbox), not O(raster).
        rr, cc = np.nonzero(comp)
        r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
        sub = comp[r0:r1, c0:c1]
        rings = _chain_rings(_boundary_edges(sub))
        if not rings:
            continue
        # Shift pixel coords back to full-raster space.
        rings = [[(x + c0, y + r0) for x, y in ring] for ring in rings]
        # Largest ring (by |area|) is the exterior; the rest are holes.
        rings.sort(key=lambda r: abs(_ring_area(r)), reverse=True)
        exterior_px = rings[0]
        holes_px = rings[1:]

        vals = arr[comp]
        cut = float(np.sum(-vals[vals < 0]) * px_area)
        fill = float(np.sum(vals[vals > 0]) * px_area)
        area = abs(_ring_area(_to_map(exterior_px, transform)))
        for h in holes_px:
            area -= abs(_ring_area(_to_map(h, transform)))

        out.append(
            ChangePolygon(
                exterior=_to_map(exterior_px, transform),
                holes=[_to_map(h, transform) for h in holes_px],
                cut_m3=cut,
                fill_m3=fill,
                net_m3=fill - cut,
                dominant="fill" if fill >= cut else "cut",
                area_m2=area,
                pixel_count=count,
            )
        )
    return out


def polygons_to_geojson(polys: List[ChangePolygon]) -> Dict:
    """GeoJSON FeatureCollection of change polygons."""
    return {
        "type": "FeatureCollection",
        "features": [p.to_feature() for p in polys],
    }
