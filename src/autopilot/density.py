"""Density-by-area strand counts for flow-field visualization.

Why this exists: a fixed strand count looks sparse on a continental view and
cluttered on a regional zoom. Strand density should be a property of the map
area, not a magic constant copied between scripts. This module turns a
bounding box into a strand count with one documented calibration point.

Calibration: base_density=0.5 strands/deg^2 reproduces ~3000 strands for the
North America bbox (-165, 15, -55, 72): area = 110 x 57 = 6270 deg^2,
round(6270 x 0.5) = 3135 strands. That is the strand count that looked right
on the daily North-American winds reel; every other bbox scales from there.

Honest limitations:
- Area is a planar approximation: |d_lon| x |d_lat| in degrees^2. This
  overstates true area near the poles (meridians converge) and understates
  nothing at the equator. For a density heuristic this is fine; do not use it
  as a real area measurement.
- The min/max clamps (500 / 6000) are guardrails against degenerate zooms
  (a 1x1 degree box would otherwise get a handful of strands; a global view
  would get tens of thousands and melt the renderer). They are returned in
  the "clamped" flag so the caller can see when they bit.
"""

from __future__ import annotations


def strand_count_for_bbox(
    bbox, base_density=0.5, min_count=500, max_count=6000
) -> dict:
    """Compute a strand count proportional to map area.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) in degrees.
        base_density: strands per square degree (default 0.5 — calibrated so
            the North America bbox yields ~3135 strands).
        min_count: floor for tiny bboxes (default 500).
        max_count: ceiling for huge bboxes (default 6000).

    Returns:
        {"count": int, "area_deg2": float, "density_per_deg2": float,
         "method": "area", "clamped": bool}.

    Pure function, deterministic, no randomness.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    area = abs(max_lon - min_lon) * abs(max_lat - min_lat)
    raw = round(area * base_density)
    count = int(min(max(raw, min_count), max_count))
    return {
        "count": count,
        "area_deg2": float(area),
        "density_per_deg2": float(count / area) if area > 0 else 0.0,
        "method": "area",
        "clamped": count != raw,
    }
