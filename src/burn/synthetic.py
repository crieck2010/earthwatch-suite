"""Seeded synthetic data for offline demos and tests.

Builds a small world: three zones, a pre-fire NBR field (healthy
vegetation, spatially varying), and a post-fire NBR field with a planted
burn scar of *known* severity areas, plus a short NDVI/NBR pass series with
a planted clearing event for the disturbance heuristic. Deterministic for a
given seed — the demo and the tests assert the same numbers.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .zones import Zone

#: Demo grid: 10 m pixels over a 1 km x 1 km tile.
DEMO_SHAPE: Tuple[int, int] = (100, 100)
DEMO_PIXEL_M: float = 10.0
DEMO_TRANSFORM: Tuple[float, ...] = (10.0, 0.0, 500_000.0, 0.0, -10.0, 4_600_000.0)

#: Planted burn: centred square with known severity ring structure.
BURN_CENTER: Tuple[int, int] = (50, 50)


def synthetic_zones() -> List[Zone]:
    """Three demo zones as rectangles over the demo tile."""
    boxes = {
        "pine-ridge": (10, 10, 60, 60),
        "meadow-brook": (55, 55, 95, 95),
        "control-plot": (5, 70, 40, 95),
    }
    zones = []
    for zid, (x0, y0, x1, y1) in boxes.items():
        zones.append(
            Zone(
                id=zid,
                name=zid.replace("-", " ").title(),
                geometry={
                    "type": "Polygon",
                    "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
                },
            )
        )
    return zones


def _noise(shape: Tuple[int, int], seed: int, scale: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(shape).astype(np.float32) - 0.5) * 2 * scale


def synthetic_pre_nbr(shape: Tuple[int, int] = DEMO_SHAPE, seed: int = 7) -> np.ndarray:
    """Healthy-vegetation NBR field: ~0.55 mean with spatial variation."""
    rows, cols = shape
    rr, cc = np.mgrid[0:rows, 0:cols].astype(np.float32)
    base = 0.55 + 0.10 * np.sin(rr / 12.0) * np.cos(cc / 15.0)
    return (base + _noise(shape, seed, 0.04)).astype(np.float32)


def _burn_radius_map(shape: Tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    rr, cc = np.mgrid[0:rows, 0:cols].astype(np.float32)
    return np.sqrt((rr - BURN_CENTER[0]) ** 2 + (cc - BURN_CENTER[1]) ** 2)


def synthetic_post_nbr(
    shape: Tuple[int, int] = DEMO_SHAPE, seed: int = 7, burn: bool = True
) -> np.ndarray:
    """Post-fire NBR: planted circular scar with severity rings.

    Ring radii (pixels) and the dNBR they produce against the ~0.55
    pre-fire field:

    * r <= 6   → dNBR ≈ 0.85  → high severity
    * 6 < r <= 11 → dNBR ≈ 0.55 → moderate-high
    * 11 < r <= 16 → dNBR ≈ 0.35 → moderate-low
    * 16 < r <= 20 → dNBR ≈ 0.18 → low

    Everything outside r = 20 is unburned (dNBR ≈ 0).
    """
    pre = synthetic_pre_nbr(shape, seed)
    if not burn:
        return (pre + _noise(shape, seed + 99, 0.02)).astype(np.float32)
    dist = _burn_radius_map(shape)
    target_dnbr = np.zeros(shape, dtype=np.float32)
    target_dnbr[dist <= 6] = 0.85
    target_dnbr[(dist > 6) & (dist <= 11)] = 0.55
    target_dnbr[(dist > 11) & (dist <= 16)] = 0.35
    target_dnbr[(dist > 16) & (dist <= 20)] = 0.18
    post = pre - target_dnbr + _noise(shape, seed + 1, 0.015)
    return post.astype(np.float32)


def expected_burn_areas(pixel_size_m: float = DEMO_PIXEL_M) -> Dict[str, float]:
    """Planted severity areas (ha) for the synthetic scar, by class name.

    Computed from the exact pixel radii above — the demo asserts these.
    """
    dist = _burn_radius_map(DEMO_SHAPE)
    pixel_ha = (pixel_size_m ** 2) / 10_000.0
    out = {}
    rings = {
        "high_severity": dist <= 6,
        "moderate_high_severity": (dist > 6) & (dist <= 11),
        "moderate_low_severity": (dist > 11) & (dist <= 16),
        "low_severity": (dist > 16) & (dist <= 20),
    }
    for name, sel in rings.items():
        out[name + "_ha"] = float(np.count_nonzero(sel) * pixel_ha)
    out["burned_ha"] = float(sum(out.values()))
    return out


def synthetic_pass_series(
    n_passes: int = 6, shape: Tuple[int, int] = (40, 40), seed: int = 11
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    """Short NDVI/NBR pass series with a planted clearing at pass 3.

    Returns (dates, ndvi_list, nbr_list). Passes 0–2 are stable healthy
    vegetation; passes 3+ have a 12×12 cleared patch (both indices drop
    hard) — the disturbance heuristic should fire exactly once, on pass 3.
    """
    rows, cols = shape
    dates = [f"2024-06-{1 + 10 * i:02d}" for i in range(n_passes)]
    rng = np.random.default_rng(seed)
    ndvi_list, nbr_list = [], []
    for i in range(n_passes):
        ndvi = 0.72 + (rng.random(shape).astype(np.float32) - 0.5) * 0.06
        nbr = 0.55 + (rng.random(shape).astype(np.float32) - 0.5) * 0.06
        if i >= 3:
            ndvi[10:22, 10:22] = 0.18 + (rng.random((12, 12)).astype(np.float32) - 0.5) * 0.04
            nbr[10:22, 10:22] = 0.05 + (rng.random((12, 12)).astype(np.float32) - 0.5) * 0.04
        ndvi_list.append(ndvi.astype(np.float32))
        nbr_list.append(nbr.astype(np.float32))
    return dates, ndvi_list, nbr_list


#: Date of the synthetic fire (between the two demo passes).
SYNTHETIC_FIRE_DATE = "2024-07-15"
SYNTHETIC_PRE_PASS = ("syn-pre-20240615", "2024-06-15")
SYNTHETIC_POST_PASS = ("syn-post-20240801", "2024-08-01")
