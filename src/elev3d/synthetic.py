"""Seeded synthetic DEMs for demos and tests — no network, no data files.

The synthetic site is a 100×100 raster at 10 m pixels (north-up), a gently
sloping base surface, and two planted earthworks with **known** volumes:

* **cut**: 50×25 px rectangle excavated 0.15 m → 18,750 m³ removed
* **fill**: 41×20 px rectangle raised 0.15 m → 12,300 m³ added

Plus low-amplitude noise (σ = 0.02 m) everywhere else, so the noise floor
has something honest to do. DEM1 is the "before", DEM2 the "after".
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

SHAPE = (100, 100)
#: 10 m north-up transform: x = 500000 + 10*col, y = 4600000 - 10*row
TRANSFORM = (10.0, 0.0, 500000.0, 0.0, -10.0, 4600000.0)

#: Planted earthworks, (row0, row1, col0, col1, depth_m). Negative = cut.
#: Depths (±0.15 m) sit well above the demo noise floor (0.08 m) so the
#: planted signal survives thresholding; the noise (σ = 0.02 m) does not.
PLANTED = {
    "cut": (10, 60, 5, 30, -0.15),  # 50×25 px × 100 m² × 0.15 m = 18,750 m³
    "fill": (65, 85, 60, 101, 0.15),  # 20×41 px × 100 m² × 0.15 m = 12,300 m³
}

CUT_M3 = 50 * 25 * 100.0 * 0.15  # 18,750
FILL_M3 = 20 * 41 * 100.0 * 0.15  # 12,300


def synthetic_dems(seed: int = 7) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    """Return (dem1, dem2, transform) — the before/after pair."""
    rng = np.random.default_rng(seed)
    rows, cols = SHAPE
    yy, xx = np.mgrid[0:rows, 0:cols]
    base = 100.0 + 0.002 * xx + 0.001 * yy  # gentle slope, metres
    dem1 = base + rng.normal(0.0, 0.02, SHAPE)
    dem2 = base + rng.normal(0.0, 0.02, SHAPE)
    for _name, (r0, r1, c0, c1, depth) in PLANTED.items():
        dem2[r0:r1, c0:c1] += depth
    return dem1, dem2, TRANSFORM


def synthetic_zones_geojson() -> Dict:
    """Three zones: the cut area, the fill area, and a stable road strip."""
    t = TRANSFORM
    a, _b, c, _d, e, f = t

    def rect(c0, c1, r0, r1):
        x0, x1 = c + a * c0, c + a * c1
        y1, y0 = f + e * r0, f + e * r1  # e < 0: r0 is the north edge
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": "cut-pit", "name": "Cut pit", "kind": "pit"},
                "geometry": {"type": "Polygon", "coordinates": [rect(5, 30, 10, 60)]},
            },
            {
                "type": "Feature",
                "properties": {
                    "id": "fill-mound",
                    "name": "Fill mound",
                    "kind": "stockpile",
                },
                "geometry": {"type": "Polygon", "coordinates": [rect(60, 101, 65, 85)]},
            },
            {
                "type": "Feature",
                "properties": {
                    "id": "stable-road",
                    "name": "Stable road",
                    "kind": "stable",
                },
                # a quiet strip with no planted change, for coreg checks
                "geometry": {"type": "Polygon", "coordinates": [rect(35, 55, 10, 60)]},
            },
        ],
    }
