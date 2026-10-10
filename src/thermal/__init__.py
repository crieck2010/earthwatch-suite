"""Land-surface-temperature analytics for surveying and remote sensing.

Engine-first: pure-logic modules with no UI imports. Thermal-band math is
implemented here (Landsat C2L2 ST_B10 scaling, Planck-inversion brightness
temperature, emissivity correction); acquisition reuses survey-imagery's
STAC pipeline via lazy import.

Public API
----------
- :mod:`thermal.temperature` — ST DN → Kelvin → Celsius, Planck inversion,
  emissivity correction, zonal stat helpers
- :mod:`thermal.zones` — zone loading from GeoJSON
- :mod:`thermal.zonal` — pure-numpy zonal LST statistics
- :mod:`thermal.timeseries` — per-zone per-pass LST records + CSV
- :mod:`thermal.anomalies` — day-of-year heat-anomaly z-scores
- :mod:`thermal.uhi` — urban heat-island intensity vs. a reference zone
- :mod:`thermal.heatwave` — heatwave event flagging
- :mod:`thermal.qml` — QGIS .qml styles for LST maps
- :mod:`thermal.synthetic` — seeded synthetic data for demos and tests
- :mod:`thermal.acquire` — real satellite acquisition via survey-imagery
"""

from __future__ import annotations

__version__ = "0.1.1"

__all__ = [
    "temperature",
    "zones",
    "zonal",
    "timeseries",
    "anomalies",
    "uhi",
    "heatwave",
    "qml",
    "synthetic",
    "acquire",
]
