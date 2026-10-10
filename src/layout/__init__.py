"""survey-layout: platform-aware canvas layouts for vertical data-viz reels.

The engine answers one question: *given a target platform, where on the
canvas may content go?*  It ships curated platform presets (canvas size +
the screen regions each platform's own chrome covers), region templates
for the standard reel content blocks (title, map, chart, caption,
footer) that clear those regions, and frame-fitting helpers for
adapting existing frames to a canvas.

Coordinates are figure fractions throughout: ``(left, bottom, width,
height)`` with the origin at the bottom-left, exactly what
``matplotlib.figure.Figure.add_axes`` takes — so the ``regions`` mapping
plugs directly into survey-viz's renderers with no translation layer.

The platform chrome measurements are community-measured approximations
(documented per platform in docs/PLATFORMS.md). Platforms change their
chrome; the presets are versioned data, not ground truth, and every
preset carries its source notes.
"""

from .platforms import (PLATFORMS, QUAKE_REGION_NAMES, REGION_NAMES,
                        Platform, get_platform, list_platforms)
from .regions import layout_for, validate_layout
from .interop import to_viz_canvas, validate_canvas

try:  # optional: needs the "fit" extra (numpy + Pillow)
    from .fit import fit_frame as _fit_frame
except ImportError:  # pragma: no cover
    _fit_frame = None


def fit_frame(*args, **kwargs):
    """Fit a frame to a canvas (needs ``pip install "survey-layout[fit]"``)."""
    if _fit_frame is None:
        raise ImportError(
            "fit_frame needs numpy and Pillow: "
            'pip install "survey-layout[fit]"')
    return _fit_frame(*args, **kwargs)

__version__ = "0.1.0"

__all__ = [
    "PLATFORMS",
    "QUAKE_REGION_NAMES",
    "REGION_NAMES",
    "Platform",
    "fit_frame",
    "get_platform",
    "layout_for",
    "list_platforms",
    "to_viz_canvas",
    "validate_canvas",
    "__version__",
]
