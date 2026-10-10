"""Salience-aware label placement.

Why this exists: labels on a data map compete with the data. The legibility
heuristic in autopilot.compose wants labels on dark areas; this module answers
the complementary question — where is the map visually interesting, so labels
land near the features worth pointing at and away from bright clutter.

Formula (documented so it can be argued with):
    speed_n   = clip(speed / p99(speed), 0, 1)
    temp_anom = clip(|temperature - median(temperature)| / (p99 - p1), 0, 1)
    salience  = 0.6 * speed_n + 0.4 * temp_anom
NaN-safe: any NaN in the inputs maps to 0 salience. Output is float in [0, 1].

Honest limitation, stated plainly: salience != importance. This is a visual
heuristic — fast water and unusual temperature read as "interesting" to a
human eye. It is not a judgment of what matters scientifically or
operationally. Do not let a label rank stand in for domain judgment.
"""

from __future__ import annotations

import numpy as np


def salience_map(speed, temperature) -> np.ndarray:
    """Compute a 2D salience field from speed and temperature grids.

    Args:
        speed: 2D array-like (e.g. current speed in m/s).
        temperature: 2D array-like, same shape (e.g. degC).

    Returns:
        float ndarray in [0, 1], same shape. NaN inputs yield 0.
    """
    s = np.asarray(speed, dtype=np.float64)
    t = np.asarray(temperature, dtype=np.float64)

    sf = s[np.isfinite(s)]
    if sf.size == 0:
        speed_n = np.zeros_like(s)
    else:
        p99 = float(np.percentile(sf, 99))
        speed_n = np.where(p99 > 0, s / p99, 0.0)
    speed_n = np.clip(speed_n, 0.0, 1.0)

    tf = t[np.isfinite(t)]
    if tf.size == 0:
        temp_anom = np.zeros_like(t)
    else:
        med = float(np.median(tf))
        denom = float(np.percentile(tf, 99) - np.percentile(tf, 1))
        temp_anom = np.where(denom > 0, np.abs(t - med) / denom, 0.0)
    temp_anom = np.clip(temp_anom, 0.0, 1.0)

    sal = 0.6 * speed_n + 0.4 * temp_anom
    sal = np.where(np.isfinite(sal), sal, 0.0)
    return np.clip(sal, 0.0, 1.0)


def rank_label_positions(
    candidates,
    salience,
    lats,
    lons,
    brightness=None,
    w_salience=1.0,
    w_brightness=0.5,
) -> list:
    """Rank candidate label positions by salience minus brightness penalty.

    Args:
        candidates: list of {"name": str, "lon": float, "lat": float, ...}.
            Extra keys pass through untouched.
        salience: 2D array over the (lats, lons) grid.
        lats: 1D latitude coordinate array (rows of salience).
        lons: 1D longitude coordinate array (columns of salience).
        brightness: optional 2D array, same shape as salience; bright cells
            are penalized (labels should avoid bright clutter).
        w_salience: weight rewarding salience (default 1.0).
        w_brightness: weight penalizing brightness (default 0.5).

    Returns:
        Candidates sorted by score descending, each a copy with
        "salience_score" added. score = w_salience * sal - w_brightness * bri,
        sampled at the nearest grid cell. Deterministic.
    """
    sal = np.asarray(salience, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    bri = np.asarray(brightness, dtype=np.float64) if brightness is not None else None

    ranked = []
    for c in candidates:
        iy = int(np.argmin(np.abs(lats - c["lat"])))
        ix = int(np.argmin(np.abs(lons - c["lon"])))
        sal_v = float(sal[iy, ix]) if np.isfinite(sal[iy, ix]) else 0.0
        bri_v = float(bri[iy, ix]) if bri is not None else 0.0
        out = dict(c)
        out["salience_score"] = w_salience * sal_v - w_brightness * bri_v
        ranked.append(out)
    ranked.sort(key=lambda d: d["salience_score"], reverse=True)
    return ranked
