"""Reusable visual styles for the earthwatch-suite render stack.

A :class:`Style` is a named, versioned bundle of the aesthetic choices
survey-viz supports — base style (``"reel-dark"`` / ``"light"``),
per-variable colormaps, title/footer templates, and the basemap-underlay
toggle — so a creator can apply one named look across many reels instead
of re-picking widgets every time.

The core is stdlib-only and never imports survey-viz: styles are plain
data, and :func:`apply_style` consumes/produces plain spec dicts (the
``VizSpec`` fields ``style`` / ``title`` / ``caption`` / ``underlay``)
plus a ``render_viz`` kwargs dict (``cmap``). Anything that understands
a VizSpec dict — survey-viz itself, reel-studio, scripts — can use it.
"""

from __future__ import annotations

from .styles import (
    KNOWN_BASE_STYLES,
    STYLE_SCHEMA_VERSION,
    Style,
    StyleError,
    get_style,
    list_styles,
    register_style,
    validate_style,
)
from .templates import (
    TEMPLATE_FIELDS,
    render_template,
)
from .labels import (
    VARIABLE_LABELS,
    region_label,
    variable_label,
)
from .apply import (
    apply_style,
    render_kwargs_for,
    suggest_style,
)
from .presets import (
    BUILTIN_STYLES,
    load_styles_from_dir,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "KNOWN_BASE_STYLES",
    "STYLE_SCHEMA_VERSION",
    "BUILTIN_STYLES",
    "Style",
    "StyleError",
    "TEMPLATE_FIELDS",
    "VARIABLE_LABELS",
    "apply_style",
    "get_style",
    "list_styles",
    "load_styles_from_dir",
    "register_style",
    "region_label",
    "render_kwargs_for",
    "render_template",
    "suggest_style",
    "validate_style",
    "variable_label",
]
