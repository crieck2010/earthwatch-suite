"""Fire-severity classification from dNBR.

Uses the standard USGS dNBR severity bands (Key & Benson, 2006; the table
used by USGS/NPS burn-severity mapping products such as MTBS):

=========== ================== ============ ============
class id    name               dNBR range   typical colour
=========== ================== ============ ============
0           unburned           < 0.10       dark green
1           low severity       0.10–0.269   light green
2           moderate-low       0.27–0.439   yellow
3           moderate-high      0.44–0.659   orange
4           high severity      >= 0.66      red
5           enhanced regrowth  < −0.10      blue
=========== ================== ============ ============

Notes on the bands (documented honestly, because they matter):

* The 0.10 / 0.27 / 0.44 / 0.66 break points were calibrated on
  western-US conifer forests. In grasslands, shrublands, or tropical
  forests the same dNBR can mean something different — treat the bands
  as a starting point and recalibrate per ecosystem (see
  ``docs/SEVERITY_BANDS.md``).
* ``unburned`` spans dNBR from −0.10 up to 0.10: small wiggles around
  zero are phenology and noise, not fire.
* Class ids are stable across releases — survey-change transition
  matrices and the QML palette depend on them.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

#: (class id, name, lower bound inclusive, upper bound exclusive, (r,g,b,a))
SEVERITY_CLASSES: Tuple[Tuple[int, str, float, float, Tuple[int, int, int, int]], ...] = (
    (0, "unburned", -0.10, 0.10, (56, 168, 0, 255)),
    (1, "low severity", 0.10, 0.27, (139, 209, 0, 255)),
    (2, "moderate-low severity", 0.27, 0.44, (255, 255, 0, 255)),
    (3, "moderate-high severity", 0.44, 0.66, (255, 128, 0, 255)),
    (4, "high severity", 0.66, float("inf"), (215, 25, 28, 255)),
    (5, "enhanced regrowth", float("-inf"), -0.10, (43, 131, 186, 255)),
)

#: Convenience lookups.
CLASS_BY_ID: Dict[int, Tuple[str, float, float]] = {
    cid: (name, lo, hi) for cid, name, lo, hi, _ in SEVERITY_CLASSES
}
CLASS_BY_NAME: Dict[str, int] = {name: cid for cid, name, _, _, _ in SEVERITY_CLASSES}

#: dNBR >= this counts as "burned" for burned-area statistics.
BURNED_THRESHOLD: float = 0.10

#: Class ids that count as burned (severity bands 1..4).
BURNED_CLASS_IDS: Tuple[int, ...] = (1, 2, 3, 4)


def classify(dnbr_array: np.ndarray) -> np.ndarray:
    """Classify a dNBR raster into severity class ids (uint8).

    NaN → 255 (nodata). Exact band edges belong to the *higher* class
    (0.10 is low severity, 0.27 is moderate-low, 0.44 is moderate-high,
    0.66 is high) — tested explicitly.
    """
    d = np.asarray(dnbr_array, dtype=np.float32)
    out = np.full(d.shape, 255, dtype=np.uint8)
    valid = np.isfinite(d)
    # Regrowth first (lowest band), then the burn bands in order.
    out[valid & (d < -0.10)] = 5
    out[valid & (d >= -0.10) & (d < 0.10)] = 0
    out[valid & (d >= 0.10) & (d < 0.27)] = 1
    out[valid & (d >= 0.27) & (d < 0.44)] = 2
    out[valid & (d >= 0.44) & (d < 0.66)] = 3
    out[valid & (d >= 0.66)] = 4
    return out


def burned_mask(dnbr_array: np.ndarray, threshold: float = BURNED_THRESHOLD) -> np.ndarray:
    """Bool mask of burned pixels: dNBR >= threshold, NaN → False."""
    d = np.asarray(dnbr_array, dtype=np.float32)
    return np.isfinite(d) & (d >= threshold)


def class_areas(
    severity: np.ndarray,
    valid: np.ndarray | None,
    pixel_size_m: float,
) -> Dict[str, float]:
    """Hectares per severity class over the valid pixels.

    Returns ``{class_name_snake: ha}`` for the six classes plus
    ``burned_ha`` (classes 1–4) and ``burned_fraction`` (of valid pixels).
    """
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")
    sev = np.asarray(severity, dtype=np.uint8)
    ok = np.ones(sev.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if ok.shape != sev.shape:
        raise ValueError("valid mask shape != severity shape")
    pixel_ha = (pixel_size_m ** 2) / 10_000.0
    result: Dict[str, float] = {}
    burned_px = 0
    valid_px = int(np.count_nonzero(ok))
    for cid, name, _lo, _hi, _color in SEVERITY_CLASSES:
        n = int(np.count_nonzero((sev == cid) & ok))
        result[name.replace(" ", "_").replace("-", "_") + "_ha"] = n * pixel_ha
        if cid in BURNED_CLASS_IDS:
            burned_px += n
    result["burned_ha"] = burned_px * pixel_ha
    result["burned_fraction"] = (burned_px / valid_px) if valid_px else 0.0
    result["valid_pixels"] = float(valid_px)
    return result
