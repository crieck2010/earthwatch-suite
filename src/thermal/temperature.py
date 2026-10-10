"""Land-surface-temperature (LST) conversions.

Two acquisition paths, one output (degrees Celsius):

1. **Landsat Collection 2 Level-2** ``ST_B10`` — surface temperature already
   retrieved by USGS: ``kelvin = dn * 0.00341802 + 149``. This is the
   primary path. (The older ``dn * 0.01`` scaling was a data bug — real C2
   ST products use scale 0.00341802, offset +149 K, verified against live
   STAC ``raster:bands`` metadata.)
2. **Brightness-temperature fallback** — from top-of-atmosphere thermal
   radiance (TIRS band 10): radiance → Planck inversion → brightness
   temperature → emissivity correction → LST. The fallback's assumptions
   (single thermal band, constant emissivity) are stated plainly in the
   docstrings and the README limitations.

Scaling note: ``st_dn_to_kelvin`` accepts per-asset ``scale``/``offset``
overrides and :func:`st_scale_offset_from_stac` reads them from a STAC
asset's ``raster:bands`` entry. Callers feeding real STAC items should
always prefer the asset metadata over these defaults.
"""

from __future__ import annotations

import math
from typing import Tuple

#: Landsat Collection 2 Level-2 surface-temperature scale (ST_B10),
#: verified against live STAC ``raster:bands`` metadata (Element84 Earth
#: Search ``landsat-c2-l2``, Microsoft Planetary Computer). Stored as
#: uint16; Kelvin = DN * ST_SCALE + ST_OFFSET.
ST_SCALE = 0.00341802

#: Landsat Collection 2 Level-2 surface-temperature offset, Kelvin.
#: nodata DN 0 therefore maps to exactly 149.0 K.
ST_OFFSET = 149.0

#: Kelvin → Celsius offset.
K_TO_C = 273.15

#: Effective wavelength of Landsat TIRS band 10 (micrometres).
LAMBDA_B10_UM = 10.9

#: Second radiation constant c2 = h*c/k, in micrometre·Kelvin.
C2_UM_K = 14388.0

#: Band-10 thermal constants (K1 in W/m^2/sr/um, K2 in K) from the Landsat
#: metadata (MTL) / USGS handbook. Used by the brightness-temp fallback.
BAND10_CONSTANTS = {
    # Landsat 8 TIRS
    "landsat8": {
        "k1": 774.8853,
        "k2": 1321.0789,
        "ml": 0.0003342,  # radiance multiplicative rescaling
        "al": 0.1,  # radiance additive rescaling
    },
    # Landsat 9 TIRS-2
    "landsat9": {
        "k1": 799.0284,
        "k2": 1329.2405,
        "ml": 0.0003342,
        "al": 0.1,
    },
}


def st_dn_to_kelvin(dn, scale=None, offset=None):
    """Convert Landsat C2L2 ST_B10 digital numbers to Kelvin.

    ``kelvin = dn * scale + offset``. Defaults are the verified Landsat
    Collection 2 constants (:data:`ST_SCALE`, :data:`ST_OFFSET`); pass
    per-asset values read from a STAC item's ``raster:bands`` entry (see
    :func:`st_scale_offset_from_stac`) to override them.

    Works on scalars and numpy arrays (NaN-safe: NaN in → NaN out).
    """
    scale = ST_SCALE if scale is None else scale
    offset = ST_OFFSET if offset is None else offset
    try:
        import numpy as np

        arr = np.asarray(dn, dtype=float)
        return arr * scale + offset
    except ImportError:
        return dn * scale + offset


def st_scale_offset_from_stac(asset_dict):
    """Extract (scale, offset) for a thermal asset from a STAC asset dict.

    Reads the first entry of the asset's ``raster:bands`` list when the
    asset dict comes straight from a STAC item's ``assets`` map. Falls back
    to the Landsat Collection 2 constants when the metadata is absent —
    the safe default for ``ST_B10`` assets.

    ``asset_dict`` may be a full STAC asset (``{"href": ..., "raster:bands":
    [...]}``) or just the ``raster:bands`` list itself.
    """
    bands = asset_dict
    if isinstance(asset_dict, dict):
        bands = asset_dict.get("raster:bands") or []
    scale = offset = None
    for entry in bands or []:
        if isinstance(entry, dict):
            scale = entry.get("scale")
            offset = entry.get("offset")
            break
    if scale is None:
        scale = ST_SCALE
    if offset is None:
        offset = ST_OFFSET
    return float(scale), float(offset)


def kelvin_to_celsius(kelvin):
    """Kelvin → Celsius. Scalar- or array-safe."""
    try:
        import numpy as np

        return np.asarray(kelvin, dtype=float) - K_TO_C
    except ImportError:
        return kelvin - K_TO_C


def st_dn_to_celsius(dn):
    """Landsat C2L2 ST_B10 digital numbers straight to Celsius."""
    return kelvin_to_celsius(st_dn_to_kelvin(dn))


def radiance_to_brightness_temp(
    radiance, k1: float, k2: float
):
    """Planck inversion: TOA thermal radiance → brightness temperature (K).

    ``Tb = K2 / ln(K1 / L + 1)``

    Non-positive radiances are unphysical and yield NaN. Scalar- or
    array-safe.
    """
    try:
        import numpy as np

        L = np.asarray(radiance, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            tb = k2 / np.log(k1 / L + 1.0)
        tb = np.where(L > 0, tb, np.nan)
        return tb
    except ImportError:
        if radiance is None or radiance <= 0 or not math.isfinite(radiance):
            return float("nan")
        return k2 / math.log(k1 / radiance + 1.0)


def dn_to_radiance(dn, ml: float, al: float):
    """TOA radiance from thermal-band DN: ``L = ML * Qcal + AL``."""
    try:
        import numpy as np

        return np.asarray(dn, dtype=float) * ml + al
    except ImportError:
        return dn * ml + al


def emissivity_correct(
    brightness_temp_k, emissivity: float = 0.98
):
    """Single-channel emissivity correction (Artis & Carnahan / Weng form).

    ``LST = Tb / (1 + (λ·Tb/ρ)·ln(ε))``

    with λ = 10.9 µm (TIRS band 10) and ρ = 14388 µm·K. Assumes one
    representative broadband emissivity for the whole scene — the documented
    limitation of the fallback path. Typical land emissivities: 0.96–0.99
    (vegetation ~0.98, urban ~0.95–0.97, water ~0.99).
    """
    if not 0.0 < emissivity <= 1.0:
        raise ValueError(f"emissivity must be in (0, 1], got {emissivity}")
    lam_rho = LAMBDA_B10_UM / C2_UM_K
    try:
        import numpy as np

        tb = np.asarray(brightness_temp_k, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            lst = tb / (1.0 + lam_rho * tb * math.log(emissivity))
        return np.where(np.isfinite(tb) & (tb > 0), lst, np.nan)
    except ImportError:
        tb = brightness_temp_k
        if tb is None or not math.isfinite(tb) or tb <= 0:
            return float("nan")
        return tb / (1.0 + lam_rho * tb * math.log(emissivity))


def brightness_temp_to_lst_c(
    dn,
    sensor: str = "landsat8",
    emissivity: float = 0.98,
):
    """Full fallback path: TIRS band-10 DN → LST in Celsius.

    DN → TOA radiance (ML/AL) → Planck inversion → emissivity correction →
    Celsius. ``sensor`` is ``"landsat8"`` or ``"landsat9"``.
    """
    const = BAND10_CONSTANTS.get(sensor)
    if const is None:
        raise ValueError(
            f"unknown sensor {sensor!r} (known: {sorted(BAND10_CONSTANTS)})"
        )
    radiance = dn_to_radiance(dn, const["ml"], const["al"])
    tb = radiance_to_brightness_temp(radiance, const["k1"], const["k2"])
    lst_k = emissivity_correct(tb, emissivity)
    return kelvin_to_celsius(lst_k)


def lst_anomaly_c(zone_c, reference_c) -> float:
    """Urban heat-island intensity: zone LST minus reference LST, in °C."""
    return zone_c - reference_c


def celsius_stats(values) -> Tuple[float, float, float, float, float, float, float, float]:
    """Mean, median, std, min, max, p10, p90, count of finite values.

    Pure-python fallback when numpy is unavailable; numpy path otherwise.
    Returns NaNs when there are no finite values.
    """
    try:
        import numpy as np

        arr = np.asarray(values, dtype=float).ravel()
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            nan = float("nan")
            return nan, nan, nan, nan, nan, nan, nan, 0.0
        return (
            float(np.mean(finite)),
            float(np.median(finite)),
            float(np.std(finite)),
            float(np.min(finite)),
            float(np.max(finite)),
            float(np.percentile(finite, 10)),
            float(np.percentile(finite, 90)),
            float(finite.size),
        )
    except ImportError:
        vals = sorted(v for v in values if v == v)  # drop NaN
        n = len(vals)
        if n == 0:
            nan = float("nan")
            return nan, nan, nan, nan, nan, nan, nan, 0.0
        mean = sum(vals) / n
        mid = n // 2
        median = vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2
        var = sum((v - mean) ** 2 for v in vals) / n
        def pct(p):
            k = (n - 1) * p / 100
            lo, hi = math.floor(k), math.ceil(k)
            return vals[lo] if lo == hi else vals[lo] + (vals[hi] - vals[lo]) * (k - lo)
        return mean, median, math.sqrt(var), vals[0], vals[-1], pct(10), pct(90), float(n)
