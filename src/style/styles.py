"""The :class:`Style` dataclass, validation, and the style registry.

A style is data: it can live in a Python module (the built-ins in
:mod:`style.presets`), in a JSON file (user styles via
:func:`style.presets.load_styles_from_dir`), or be constructed
programmatically. :func:`validate_style` is the single gate — every
entry point runs through it, so a bad style fails fast with a message
that names the field and the fix.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Tuple

from .templates import TEMPLATE_FIELDS, StyleError, check_template  # noqa: F401

#: Version of the style *schema* (the field set below). Bumped when
#: fields are added/removed/renamed; each Style records the schema it
#: was written against so loaders can migrate or refuse cleanly.
STYLE_SCHEMA_VERSION = "1.0"

#: Base styles survey-viz understands (``VizSpec.style``). survey-style
#: never imports viz; this tuple mirrors ``viz.spec.KNOWN_STYLES`` and
#: is re-verified against the installed survey-viz in the test suite.
KNOWN_BASE_STYLES = ("reel-dark", "light")

_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class Style:
    """One reusable visual style.

    Fields:
        name: slug, e.g. ``"midnight-ocean"`` (lowercase, hyphens).
        version: version of this style *definition* (``"1.0"``).
        label: human-readable label shown in pickers.
        description: one or two sentences shown under the label.
        base_style: survey-viz base style — ``"reel-dark"`` or
            ``"light"`` (mirrors ``viz.spec.KNOWN_STYLES``).
        colormaps: variable slug -> matplotlib colormap name, e.g.
            ``{"sst": "inferno"}``. Only names matter here;
            survey-viz validates them against its registry at render.
        default_cmap: colormap for variables not in ``colormaps``.
        title_template: template for the frame title (see
            :mod:`style.templates`); ``"{title}"`` keeps the spec title.
        footer_template: template for the footer caption; ``""``
            means no caption.
        underlay: basemap underlay on/off (``VizSpec.underlay``).
        tags: free-form tags for grouping in UIs, e.g.
            ``("dark", "ocean")``.
    """

    name: str
    version: str = "1.0"
    label: str = ""
    description: str = ""
    base_style: str = "reel-dark"
    colormaps: Mapping[str, str] = field(default_factory=dict)
    default_cmap: str = "turbo"
    title_template: str = "{title}"
    footer_template: str = ""
    underlay: bool = True
    tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        errors = _validate(self)
        if errors:
            raise StyleError(
                f"style {self.name!r} is invalid:\n  - "
                + "\n  - ".join(errors))


def _validate(style: Style) -> List[str]:
    errors: List[str] = []
    if not isinstance(style.name, str) or not _NAME_RE.match(style.name):
        errors.append(
            f"name: {style.name!r} must be a lowercase hyphenated slug "
            f"(e.g. 'midnight-ocean')")
    if not isinstance(style.version, str) or not style.version.strip():
        errors.append("version: must be a non-empty string like '1.0'")
    if style.base_style not in KNOWN_BASE_STYLES:
        errors.append(
            f"base_style: {style.base_style!r} not in {KNOWN_BASE_STYLES} "
            f"(survey-viz VizSpec.style values)")
    for var, cmap in dict(style.colormaps).items():
        if not isinstance(cmap, str) or not cmap.strip():
            errors.append(
                f"colormaps[{var!r}]: colormap name must be a non-empty "
                f"string, got {cmap!r}")
    if not isinstance(style.default_cmap, str) or not style.default_cmap.strip():
        errors.append("default_cmap: must be a non-empty colormap name")
    for field_name in ("title_template", "footer_template"):
        template = getattr(style, field_name)
        if not isinstance(template, str):
            errors.append(f"{field_name}: must be a string")
            continue
        errors.extend(
            f"{field_name}: {msg}"
            for msg in check_template(template))
    if not isinstance(style.underlay, bool):
        errors.append(
            f"underlay: expected bool, got {style.underlay!r}")
    return errors


def validate_style(style: Style) -> List[str]:
    """Return a list of problems with ``style`` (empty = valid).

    Unlike the :class:`Style` constructor — which raises — this is the
    non-raising form for UIs and loaders that want to report every
    problem at once.
    """
    return _validate(style)


# --- registry ---------------------------------------------------------------

_REGISTRY: Dict[str, Style] = {}


def register_style(style: Style, replace: bool = False) -> Style:
    """Add ``style`` to the registry.

    Built-ins are registered at import; call this for programmatic or
    loaded user styles. Raises :class:`StyleError` on a name collision
    unless ``replace=True``.
    """
    if not isinstance(style, Style):
        raise StyleError(
            f"register_style: expected a Style, got {type(style).__name__}")
    if style.name in _REGISTRY and not replace:
        raise StyleError(
            f"a style named {style.name!r} is already registered "
            f"(pass replace=True to overwrite it)")
    _REGISTRY[style.name] = style
    return style


def list_styles() -> List[str]:
    """Sorted names of all registered styles (built-ins + user)."""
    return sorted(_REGISTRY)


def get_style(name: str) -> Style:
    """Return the style ``name``, or raise with a did-you-mean hint."""
    try:
        return _REGISTRY[name]
    except KeyError:
        hint = difflib.get_close_matches(name, _REGISTRY, n=3, cutoff=0.6)
        msg = f"unknown style {name!r}"
        if hint:
            msg += f" — did you mean: {', '.join(hint)}?"
        else:
            msg += f" (available: {', '.join(sorted(_REGISTRY))})"
        raise StyleError(msg) from None
