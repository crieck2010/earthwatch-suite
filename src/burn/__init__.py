"""survey-burn: burn-scar / disturbance mapping engine.

Pure-logic core — no UI imports. Reads satellite-derived spectral indices
(NBR reused from survey-imagery's registry) and produces dNBR rasters, USGS
fire-severity classifications, per-zone burned-area time series, a
clearing/logging disturbance heuristic, and disturbance polygons.
"""

__version__ = "0.1.0"
