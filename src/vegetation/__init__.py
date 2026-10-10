"""Vegetation/vigor analytics for surveying and remote sensing.

Engine-first: pure-logic modules with no UI imports. Spectral-index math is
*reused* from survey-imagery (lazy import) — this package adds only the
vegetation-specific analytics on top: zonal time series, phenology-aware
anomalies, growing-season summaries, and vigor-class maps.

Public API
----------
- :mod:`vegetation.indices_ext` — vigor-index computation (NDVI/EVI/SAVI via
  survey-imagery, plus GNDVI/MSAVI defined here)
- :mod:`vegetation.zones` — zone loading from GeoJSON
- :mod:`vegetation.zonal` — pure-numpy zonal statistics
- :mod:`vegetation.timeseries` — per-zone per-pass time-series records + CSV
- :mod:`vegetation.anomalies` — phenology-aware (day-of-year) z-score anomalies
- :mod:`vegetation.season` — growing-season summaries (green-up, peak,
  senescence, season integral)
- :mod:`vegetation.vigor` — per-pixel vigor classification
- :mod:`vegetation.qml` — QGIS .qml styles for vigor-class maps
- :mod:`vegetation.synthetic` — seeded synthetic data for demos and tests
- :mod:`vegetation.acquire` — real satellite acquisition via survey-imagery
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "indices_ext",
    "zones",
    "zonal",
    "timeseries",
    "anomalies",
    "season",
    "vigor",
    "qml",
    "synthetic",
    "acquire",
]
