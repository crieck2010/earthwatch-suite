"""Seeded synthetic data for offline demos and tests.

Builds a small coastal world: a straight-ish shoreline with a sinusoidal
wiggle, two transects, and seven passes over 2019–2024.

Planted ground truth (the demo and the tests assert these numbers):

- Transect ``shore-a`` erodes at **−2.0 m/yr** (shoreline retreats landward).
- Transect ``shore-b`` accretes at **+3.0 m/yr** (shoreline advances seaward).
- A storm on 2021-06-15 carves an extra **−8 m** out of transect A on the
  2021-07-20 pass; the beach recovers by the next pass (realistic scarp
  recovery — the long-term regression still sees ≈ −2.0 m/yr).

Deterministic for a given seed. Water is south (high row numbers); land is
north. Transects are drawn landward → seaward per the documented
convention, so erosion shows as *decreasing* position.
"""

from __future__ import annotations

import datetime as _dt
from typing import Dict, List, Tuple

import numpy as np

from .transects import Transect
from .zones import Site

#: Demo grid: 10 m pixels over a 1.2 km x 1.2 km tile.
DEMO_SHAPE: Tuple[int, int] = (120, 120)
DEMO_PIXEL_M: float = 10.0
DEMO_ORIGIN_X: float = 500_000.0
DEMO_ORIGIN_Y: float = 4_600_000.0

#: Base shoreline row (before shifts): gentle sinusoidal wiggle ±3 px.
BASE_ROW: float = 60.0
WIGGLE_AMP: float = 3.0

#: Planted rates (m/yr) and storm bite (m).
EROSION_RATE_A: float = -2.0
ACCRETION_RATE_B: float = 3.0
STORM_DATE: str = "2021-06-15"
STORM_BITE_M: float = -8.0

#: (pass_id suffix, date, is_post_storm)
PASSES: List[Tuple[str, str, bool]] = [
    ("2019-01-15", "2019-01-15", False),
    ("2020-01-15", "2020-01-15", False),
    ("2021-01-15", "2021-01-15", False),
    ("2021-07-20", "2021-07-20", True),
    ("2022-01-15", "2022-01-15", False),
    ("2023-01-15", "2023-01-15", False),
    ("2024-01-15", "2024-01-15", False),
]

TRANSECT_COLS: Dict[str, int] = {"shore-a": 40, "shore-b": 80}
TRANSECT_ROW0: int = 20   # landward end (origin)
TRANSECT_ROW1: int = 100  # seaward end


def _years_since(d: _dt.date, base: _dt.date) -> float:
    return (d - base).days / 365.25


def _shift_rows(transect_id: str, date: _dt.date, base: _dt.date, post_storm: bool) -> float:
    """Shoreline row shift in pixels (negative = landward/erosion)."""
    yrs = _years_since(date, base)
    if transect_id == "shore-a":
        shift_m = EROSION_RATE_A * yrs
        if post_storm:
            shift_m += STORM_BITE_M
    else:
        shift_m = ACCRETION_RATE_B * yrs
    return shift_m / DEMO_PIXEL_M


def _shore_row(col: int, shift_px: float) -> float:
    wiggle = WIGGLE_AMP * np.sin(2.0 * np.pi * col / DEMO_SHAPE[1])
    return BASE_ROW + wiggle + shift_px


def synthetic_passes(seed: int = 7) -> List[Dict[str, object]]:
    """One dict per pass: pass_id, date, ndwi (2D), cloud (2D bool)."""
    base = _dt.date.fromisoformat(PASSES[0][1])
    out: List[Dict[str, object]] = []
    for i, (_pid, datestr, post_storm) in enumerate(PASSES):
        date = _dt.date.fromisoformat(datestr)
        rng = np.random.default_rng(seed + i)
        # Per-transect shifts differ, so build the shoreline row per column
        # by interpolating the two transect shifts across the tile.
        cols = np.arange(DEMO_SHAPE[1])
        shift_a = _shift_rows("shore-a", date, base, post_storm)
        shift_b = _shift_rows("shore-b", date, base, post_storm)
        ca, cb = TRANSECT_COLS["shore-a"], TRANSECT_COLS["shore-b"]
        # Linear ramp between transect columns, constant outside.
        shift = np.where(
            cols <= ca, shift_a,
            np.where(cols >= cb, shift_b,
                     shift_a + (shift_b - shift_a) * (cols - ca) / (cb - ca)),
        )
        shore = np.array([_shore_row(c, s) for c, s in zip(cols, shift)])
        rows = np.arange(DEMO_SHAPE[0])[:, None]
        water = rows > shore[None, :]
        ndwi = np.where(
            water,
            0.55 + rng.normal(0, 0.05, DEMO_SHAPE),
            -0.25 + rng.normal(0, 0.08, DEMO_SHAPE),
        )
        cloud = rng.random(DEMO_SHAPE) < 0.06
        ndwi = np.where(cloud, np.nan, ndwi)
        out.append(
            {
                "pass_id": f"synthetic-{datestr}",
                "date": datestr,
                "ndwi": ndwi,
                "cloud": cloud,
                "post_storm": post_storm,
            }
        )
    return out


def _to_meters(col_px: float, row_px: float) -> List[float]:
    return [
        DEMO_ORIGIN_X + col_px * DEMO_PIXEL_M,
        DEMO_ORIGIN_Y - row_px * DEMO_PIXEL_M,
    ]


def synthetic_transects() -> List[Transect]:
    """Two vertical transects, landward (row 20) → seaward (row 100)."""
    transects = []
    for tid, col in TRANSECT_COLS.items():
        coords = np.array(
            [_to_meters(col, TRANSECT_ROW0), _to_meters(col, TRANSECT_ROW1)]
        )
        transects.append(Transect(id=tid, name=tid.replace("-", " ").title(), coords=coords))
    return transects


def synthetic_sites() -> List[Site]:
    """One site polygon covering the demo tile."""
    x0, y0 = _to_meters(0, 0)
    x1, y1 = _to_meters(DEMO_SHAPE[1], DEMO_SHAPE[0])
    return [
        Site(
            id="demo-cove",
            name="Demo Cove",
            geometry={
                "type": "Polygon",
                "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
            },
        )
    ]


def transects_geojson_dict() -> Dict[str, object]:
    """Transects as a GeoJSON FeatureCollection dict (for the demo/CLI)."""
    feats = []
    for t in synthetic_transects():
        feats.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [list(map(float, pt)) for pt in t.coords],
                },
                "properties": {"id": t.id, "name": t.name},
            }
        )
    return {"type": "FeatureCollection", "features": feats}
