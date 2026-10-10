"""Deterministic, keyword/rules-based format tagging for trend items.

Rule table (title + tags, lowercased; checked in order; an item may earn
several tags — this is a heuristic description, not a classification):

    particle-flow       wind/ocean rendered as moving particles or streaks
                        ("particle", "particles", "streamline", "streaks",
                        "flow trails")
    flow-field          vector-field maps without explicit particle rendering
                        ("flow field", "flowfield", "wind map", "wind flow",
                        "ocean current", "currents map", "gfs wind")
    globe-spin          rotating-globe presentation
                        ("globe", "rotating earth", "earth spin",
                        "spinning earth")
    choropleth          area-shaded statistical maps
                        ("choropleth", "by state", "by county", "by country",
                        "election map", "colored by")
    3d-extrusion        prism / extruded-block 3-D maps
                        ("3d", "extrusion", "extruded", "3-d", "prism map")
    timelapse           animation over time
                        ("timelapse", "time-lapse", "time lapse", "over time",
                        "animated map", "years of")
    before-after        change shown as a two-frame comparison
                        ("before and after", "before/after", "before-after",
                        "then and now", "then vs now", "slider compare")
    heatmap             density/intensity raster-style maps
                        ("heatmap", "heat map", "heat-map", "hotspot")
    dot-density         one-dot-per-N-observations maps
                        ("dot density", "dot-density", "one dot", "each dot",
                        "dot map")
    satellite           satellite/aerial imagery as the visual base
                        ("satellite", "landsat", "sentinel", "aerial",
                        "imagery")
    bathymetry-terrain  terrain/elevation/seafloor rendering
                        ("bathymetry", "terrain", "topography",
                        "topographic", "elevation", "relief", "seafloor",
                        "hypsometric")
    other               fallback when no rule matches

Heuristic only: titles are marketing copy, not ground truth. The table is
deliberately easy to extend — add a (tag, keywords) row to TAG_RULES and a
fixture test. Nothing here touches the network or any UI framework.
"""
from __future__ import annotations

#: (tag, keyword-tuple) rows in match order. Keep vocabulary in sync with
#: the module docstring above.
TAG_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("particle-flow", ("particle", "particles", "streamline", "streaks",
                       "flow trails")),
    ("flow-field", ("flow field", "flowfield", "wind map", "wind flow",
                    "ocean current", "currents map", "gfs wind")),
    ("globe-spin", ("globe", "rotating earth", "earth spin",
                    "spinning earth")),
    ("choropleth", ("choropleth", "by state", "by county", "by country",
                    "election map", "colored by")),
    ("3d-extrusion", ("3d", "extrusion", "extruded", "3-d", "prism map")),
    ("timelapse", ("timelapse", "time-lapse", "time lapse", "over time",
                   "animated map", "years of")),
    ("before-after", ("before and after", "before/after", "before-after",
                      "then and now", "then vs now", "slider compare")),
    ("heatmap", ("heatmap", "heat map", "heat-map", "hotspot")),
    ("dot-density", ("dot density", "dot-density", "one dot", "each dot",
                     "dot map")),
    ("satellite", ("satellite", "landsat", "sentinel", "aerial", "imagery")),
    ("bathymetry-terrain", ("bathymetry", "terrain", "topography",
                            "topographic", "elevation", "relief",
                            "seafloor", "hypsometric")),
)

#: Every tag this module can emit (v1 vocabulary).
TAG_VOCABULARY: tuple[str, ...] = tuple(tag for tag, _ in TAG_RULES) + ("other",)

FALLBACK_TAG = "other"


def tag_item(item: dict) -> list[str]:
    """Tag a TrendItem dict from its title + tags. Never mutates ``item``.

    Returns one or more vocabulary tags; ``["other"]`` when nothing
    matches. Deterministic: fixed rule order, substring matching on the
    lowercased title + tags text.
    """
    haystack = " ".join(
        [str(item.get("title") or "")]
        + [str(t) for t in (item.get("tags") or []) if t]
    ).lower()
    tags = [tag for tag, keywords in TAG_RULES
            if any(kw in haystack for kw in keywords)]
    return tags or [FALLBACK_TAG]
