"""Duck-typed interop with the peer engines.

survey-viz and survey-animate never import this package (peers stay
optional); instead this module builds plain-dict canvases that
``viz.render_viz(..., canvas=...)`` consumes, and validates them
fail-fast so a typo'd hand-written canvas can't silently misplace a
title under TikTok's status bar.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from .geometry import Rect, within_canvas
from .platforms import REGION_NAMES, get_platform
from .regions import layout_for


def to_viz_canvas(platform: str,
                  content: Optional[Iterable[str]] = None,
                  flavor: str = "standard") -> Dict:
    """Build the ``canvas`` dict ``viz.render_viz`` accepts.

    ``flavor`` is ``"standard"`` or ``"quake"`` (earthquake renders).
    Shape::

        {"platform": key, "flavor": flavor, "width": px, "height": px,
         "regions": {name: [left, bottom, width, height], ...},
         "unsafe": [[left, bottom, width, height], ...]}
    """
    plat = get_platform(platform)
    regions = layout_for(platform, content, flavor=flavor)
    canvas = {
        "platform": plat.key,
        "flavor": flavor,
        "width": plat.width,
        "height": plat.height,
        "regions": {n: list(r) for n, r in regions.items()},
        "unsafe": [list(u) for u in plat.unsafe],
    }
    validate_canvas(canvas)
    return canvas


def validate_canvas(canvas: Dict) -> Dict:
    """Fail fast on a malformed canvas dict; returns it unchanged."""
    if not isinstance(canvas, dict):
        raise ValueError(f"canvas must be a dict, got {type(canvas).__name__}")
    for key in ("width", "height", "regions"):
        if key not in canvas:
            raise ValueError(f"canvas is missing {key!r}")
    w, h = canvas["width"], canvas["height"]
    if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
        raise ValueError(f"canvas width/height must be positive ints, "
                         f"got {(w, h)!r}")
    regions = canvas["regions"]
    if not isinstance(regions, dict) or not regions:
        raise ValueError("canvas['regions'] must be a non-empty dict")
    allowed = set(REGION_NAMES) | {"ranking"}
    for name, rect in regions.items():
        if name not in allowed:
            raise ValueError(
                f"unknown region {name!r} "
                f"(expected one of {sorted(allowed)})")
        if (not isinstance(rect, (list, tuple)) or len(rect) != 4
                or not all(isinstance(v, (int, float)) for v in rect)):
            raise ValueError(f"region {name!r} must be [l, b, w, h], "
                             f"got {rect!r}")
        if not within_canvas(tuple(rect)):
            raise ValueError(f"region {name!r} {rect!r} escapes the canvas")
    return canvas


def canvas_from_dict(d: Dict) -> Dict:
    """Coerce a user-supplied canvas mapping (e.g. from JSON) and validate."""
    return validate_canvas(dict(d))
