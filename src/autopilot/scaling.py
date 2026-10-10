"""Robust auto-scaling for scalar color channels.

Why this exists: raw min/max colormapping lets one bad cell (a sensor spike, a
fill value that leaked through) flatten every real feature into a single
color. Percentile limits clip the pathological tails and map the honest bulk
of the data to the full colormap.

Scope note: this generalizes the percentile mechanism. survey-viz 0.26.1
already uses p99 for the bivariate *brightness* (wind speed) channel — that
stays untouched. This module targets the *hue / scalar* channel (temperature,
SST, anomaly fields): the limits you pass to the colormap normalization.

Honest limitations:
- Percentiles are a statistical convenience, not truth. If the real signal
  lives in the clipped tails (an extreme event you care about), the caller
  must widen the limits — the clipped_*_fraction fields tell you how much you
  clipped so the choice is auditable.
- Degenerate inputs never raise and never return misleading limits; the
  "method" field always says which fallback was used. A "degenerate-constant"
  result returns vmin == vmax == c and the caller should widen before
  colormapping (a zero-width norm is a divide-by-zero in most colormaps).
"""

from __future__ import annotations

import numpy as np


def robust_limits(values, lo_pct=2.0, hi_pct=98.0) -> dict:
    """Compute robust colormap limits from percentiles, NaN-safe.

    Args:
        values: array-like of scalars; NaNs and infs are ignored.
        lo_pct: lower percentile for vmin (default 2.0).
        hi_pct: upper percentile for vmax (default 98.0).

    Returns:
        dict with:
        - "vmin", "vmax": float limits.
        - "method": "percentile" normally; "degenerate-empty" when there are
          no finite values (vmin=0.0, vmax=1.0); "degenerate-constant" when
          every finite value is identical (vmin=vmax=c — the caller should
          widen before colormapping).
        - "lo_pct", "hi_pct": the percentiles used.
        - "n_total": total element count; "n_finite": finite element count.
        - "clipped_lo_fraction": fraction of finite values strictly below vmin.
        - "clipped_hi_fraction": fraction of finite values strictly above vmax.

    Never raises on weird input; the method field tells the truth about what
    happened.
    """
    try:
        arr = np.asarray(values, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        # Not numeric at all (e.g. a plain string, ragged sequences) — there
        # is nothing finite to summarize, so report the honest empty fallback
        # instead of raising.
        arr = np.array([], dtype=np.float64)
    n_total = int(arr.size)
    finite = arr[np.isfinite(arr)]
    n_finite = int(finite.size)

    if n_finite == 0:
        return {
            "vmin": 0.0,
            "vmax": 1.0,
            "method": "degenerate-empty",
            "lo_pct": float(lo_pct),
            "hi_pct": float(hi_pct),
            "n_total": n_total,
            "n_finite": 0,
            "clipped_lo_fraction": 0.0,
            "clipped_hi_fraction": 0.0,
        }

    if np.all(finite == finite[0]):
        c = float(finite[0])
        return {
            "vmin": c,
            "vmax": c,
            "method": "degenerate-constant",
            "lo_pct": float(lo_pct),
            "hi_pct": float(hi_pct),
            "n_total": n_total,
            "n_finite": n_finite,
            "clipped_lo_fraction": 0.0,
            "clipped_hi_fraction": 0.0,
        }

    vmin = float(np.percentile(finite, lo_pct))
    vmax = float(np.percentile(finite, hi_pct))
    if vmax <= vmin:
        # Pathological tiny/granular inputs where the percentiles coincide
        # but the data is not constant — fall back to the honest data range.
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))

    return {
        "vmin": vmin,
        "vmax": vmax,
        "method": "percentile",
        "lo_pct": float(lo_pct),
        "hi_pct": float(hi_pct),
        "n_total": n_total,
        "n_finite": n_finite,
        "clipped_lo_fraction": float(np.mean(finite < vmin)),
        "clipped_hi_fraction": float(np.mean(finite > vmax)),
    }
