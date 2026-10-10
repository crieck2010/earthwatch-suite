"""Region selection and validation.

:func:`layout_for` is the main entry point: pick a platform, declare
which content blocks the render will draw, and get back the region
rects. The content set comes from the render plan — which is how the
engine stays dataset-aware without knowing any dataset:

* continuous gridded variables (SST, ERA5, fires, ...) draw
  title + map + chart + caption + footer,
* storm tracks and streamgage maps draw title + map + caption + footer
  (no time-series chart; their colors carry meaning and per-frame
  scalar stats don't apply),
* earthquake catalogs draw title + map + ranking + chart + caption +
  footer (the "quake" flavor: a largest-events ranking panel plus a
  daily-count chart).
* chart-less combinations simply omit ``"chart"``.

The full mapping of variables to content sets lives in
docs/INTEROP.md (it belongs to survey-viz's render plan, not to this
engine).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from . import geometry
from .geometry import Rect
from .platforms import QUAKE_REGION_NAMES, REGION_NAMES, Platform, get_platform

#: Content blocks a render plan may request.
CONTENT_NAMES = ("title", "map", "ranking", "chart", "caption", "footer")


def layout_for(platform: str,
               content: Optional[Iterable[str]] = None,
               flavor: str = "standard") -> Dict[str, Rect]:
    """Region rects for ``platform``, filtered to ``content`` blocks.

    ``flavor`` is ``"standard"`` or ``"quake"`` (the earthquake
    renderer draws an extra largest-events ranking panel and gets a
    dedicated six-region set). ``content`` defaults to every region of
    the flavor. Unknown flavors/platforms/content fail fast; the
    returned rects are figure fractions ``(left, bottom, width,
    height)`` ready for ``fig.add_axes``.
    """
    if flavor not in ("standard", "quake"):
        raise ValueError(
            f"Unknown flavor {flavor!r} (expected 'standard' or 'quake')")
    plat = get_platform(platform)
    names = list(REGION_NAMES) if flavor == "standard" else list(QUAKE_REGION_NAMES)
    source = plat.regions if flavor == "standard" else plat.quake_regions
    if source is None:  # pragma: no cover — every platform ships both
        raise ValueError(f"Platform {platform!r} has no {flavor!r} regions")
    if content is None:
        wanted: List[str] = names
    else:
        wanted = list(content)
        unknown = [n for n in wanted if n not in names]
        if unknown:
            raise ValueError(
                f"Unknown content block(s) {unknown!r} for flavor "
                f"{flavor!r} (expected a subset of {names})")
    regions = {n: source[n] for n in wanted}
    problems = validate_layout(plat, regions)
    if problems:  # curated data must never ship broken
        raise AssertionError(
            f"Platform {platform!r} ({flavor}) regions failed validation: "
            + "; ".join(problems))
    return regions


def validate_layout(plat: Platform,
                    regions: Dict[str, Rect]) -> List[str]:
    """Check region rects against the canvas and the unsafe zones.

    Returns a list of human-readable problems (empty == valid). Rules:

    * every region lies inside the unit square;
    * no region's interior overlaps an unsafe rect (shared edges are OK);
    * region interiors don't overlap each other (shared edges are OK) —
      except ``caption``, which is a floating chip by design (survey-viz
      draws it over the chart/map boundary) and may overlay other
      regions, but never an unsafe rect.
    """
    problems: List[str] = []
    names = list(regions)
    for n in names:
        r = regions[n]
        if not geometry.within_canvas(r):
            problems.append(
                f"region {n!r} {geometry.describe(r)} escapes the canvas")
    for n in names:
        r = regions[n]
        for i, u in enumerate(plat.unsafe):
            if geometry.overlaps(r, u):
                problems.append(
                    f"region {n!r} {geometry.describe(r)} overlaps unsafe "
                    f"zone {i} {geometry.describe(u)}")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if "caption" in (a, b):
                continue  # floating chip: may overlay, never unsafe zones
            if geometry.overlaps(regions[a], regions[b]):
                problems.append(
                    f"regions {a!r} and {b!r} overlap "
                    f"({geometry.describe(regions[a])} vs "
                    f"{geometry.describe(regions[b])})")
    return problems
