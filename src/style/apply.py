"""Applying styles to VizSpec-shaped dicts.

This is the interop seam: :func:`apply_style` takes whatever the
parser or the UI produced — a :class:`~viz.spec.VizSpec`, anything
with a ``to_dict()`` method, or a plain dict — and returns an updated
spec dict plus the ``render_viz`` kwargs the style implies.

It never imports survey-viz. The spec dict fields it writes
(``style``, ``title``, ``caption``, ``underlay``) are exactly the
``VizSpec`` fields of the same names, and ``render_kwargs_for``
returns ``{"cmap": ...}`` for ``render_viz``'s ``cmap`` parameter.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from .labels import region_label, variable_label
from .presets import SUGGESTIONS
from .styles import Style, StyleError, get_style
from .templates import TEMPLATE_FIELDS, render_template


def _spec_dict(spec: Any) -> Dict[str, Any]:
    if hasattr(spec, "to_dict"):
        d = spec.to_dict()
    elif isinstance(spec, Mapping):
        d = dict(spec)
    else:
        raise StyleError(
            "apply_style: spec must be a VizSpec, an object with "
            f"to_dict(), or a dict — got {type(spec).__name__}")
    if "variable" not in d:
        raise StyleError(
            "apply_style: spec dict has no 'variable' key — is this a "
            "VizSpec dict?")
    return d


def _template_context(spec: Dict[str, Any],
                      channel: Optional[str]) -> Dict[str, str]:
    variable = str(spec.get("variable", ""))
    region = str(spec.get("region_key", "") or "")
    start = str(spec.get("start", "") or "")
    end = str(spec.get("end", "") or "")
    ctx = {
        "title": str(spec.get("title", "") or ""),
        "variable": variable,
        "variable_label": variable_label(variable),
        "region": region,
        "region_label": region_label(region),
        "start": start,
        "end": end,
        "date": end,
        "year": end[:4] if len(end) >= 4 else "",
    }
    if channel is not None:
        ctx["channel"] = channel
    return ctx


def render_kwargs_for(style: Style, variable: str) -> Dict[str, str]:
    """The ``render_viz`` kwargs a style implies for ``variable``.

    Currently just ``{"cmap": ...}``: the style's per-variable colormap
    or its ``default_cmap``. Categorical renderers (storm tracks,
    earthquakes, streamgages) ignore ``cmap``; the title/footer parts of
    the style still apply to them.
    """
    return {"cmap": style.colormaps.get(variable, style.default_cmap)}


def suggest_style(variable: str) -> str:
    """Name of the built-in style suggested for ``variable``.

    Unknown variables fall back to ``"reel-dark"`` — the neutral
    default. A suggestion, not a rule: pass any style name you like.
    """
    return SUGGESTIONS.get(variable, "reel-dark")


def apply_style(spec: Any,
                style_name: str,
                *,
                overrides: Optional[Mapping[str, Any]] = None,
                channel: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Apply the named style to ``spec``.

    Args:
        spec: a VizSpec, an object with ``to_dict()``, or a spec dict.
        style_name: registered style name (see :func:`list_styles`).
        overrides: optional per-call tweaks — any :class:`Style`
            field (``colormaps``, ``footer_template``, ``underlay``,
            ...) replaces the style's value for this call only.
        channel: your channel name, used when a template references
            ``{channel}`` (e.g. the ``"creator"`` style).

    Returns:
        ``(spec_dict, render_kwargs)`` — the updated spec dict (with
        ``style`` / ``title`` / ``caption`` / ``underlay`` set) and
        ``{"cmap": ...}`` for ``render_viz``.

    The input is never mutated; overrides never touch the registry.
    """
    style = get_style(style_name)
    if overrides:
        unknown = sorted(set(overrides) - {
            "version", "label", "description", "base_style", "colormaps",
            "default_cmap", "title_template", "footer_template",
            "underlay", "tags"})
        if unknown:
            raise StyleError(
                f"apply_style: unknown override field(s): "
                f"{', '.join(unknown)}")
        merged = dict(
            name=style.name, version=style.version, label=style.label,
            description=style.description, base_style=style.base_style,
            colormaps=dict(style.colormaps),
            default_cmap=style.default_cmap,
            title_template=style.title_template,
            footer_template=style.footer_template,
            underlay=style.underlay, tags=style.tags)
        merged.update(overrides)
        if isinstance(merged.get("colormaps"), Mapping):
            merged["colormaps"] = dict(merged["colormaps"])
        if isinstance(merged.get("tags"), (list, tuple)):
            merged["tags"] = tuple(merged["tags"])
        style = Style(**merged)  # re-validated by the constructor

    spec_dict = _spec_dict(spec)
    ctx = _template_context(spec_dict, channel)

    title = render_template(style.title_template, ctx)
    footer = render_template(style.footer_template, ctx) \
        if style.footer_template else ""

    out = dict(spec_dict)
    out["style"] = style.base_style
    out["title"] = title
    out["caption"] = footer or None
    out["underlay"] = style.underlay

    render_kwargs = render_kwargs_for(
        style, str(spec_dict.get("variable", "")))
    return out, render_kwargs
