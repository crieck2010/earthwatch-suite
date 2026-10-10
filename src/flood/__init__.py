"""Flood/water-extent mapping for surveying and remote sensing.

Engine-first: pure-logic modules with no UI imports. Spectral-index math is
*reused* from survey-imagery (lazy import) — this package adds only the
water-specific analytics on top: water masks, water-area time series,
flood deltas vs. a dry-season baseline, and flood-extent polygons.

Public API
----------
- :mod:`flood.indices_water` — water-index computation (NDWI via
  survey-imagery, plus MNDWI defined here)
- :mod:`flood.zones` — zone loading from GeoJSON
- :mod:`flood.watermask` — threshold water masks, cloud interaction, despeckling
- :mod:`flood.zonal` — pure-numpy water-area statistics
- :mod:`flood.timeseries` — per-zone per-pass water-area records + CSV
- :mod:`flood.delta` — flood deltas vs. a dry-season baseline
- :mod:`flood.polygons` — water-mask vectorisation to GeoJSON polygons
- :mod:`flood.qml` — QGIS .qml styles for water masks
- :mod:`flood.synthetic` — seeded synthetic data for demos and tests
- :mod:`flood.acquire` — real satellite acquisition via survey-imagery
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "indices_water",
    "zones",
    "watermask",
    "zonal",
    "timeseries",
    "delta",
    "polygons",
    "qml",
    "synthetic",
    "acquire",
]
