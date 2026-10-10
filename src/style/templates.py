"""Title/footer template rendering.

Templates are plain ``str.format``-style strings over a fixed field
set (:data:`TEMPLATE_FIELDS`). The fixed set is deliberate: a style
written today renders the same way in every consumer, and a typo'd
field fails at style-definition time — not halfway through a render.

Fields:
    title: the spec's title (what the parser or the user set).
    variable: the variable slug, e.g. ``"sst"``.
    variable_label: human label, e.g. ``"Sea Surface Temperature"``.
    region: the region slug, e.g. ``"lake-michigan"``.
    region_label: human label, e.g. ``"Lake Michigan"``.
    start / end: ISO dates from the spec.
    date: convenience — ``end`` (the most recent frame's date).
    year: convenience — ``end``'s year.
    channel: the creator's channel name (caller-supplied; a template
        that uses ``{channel}`` without the caller providing one
        raises a helpful error).
"""

from __future__ import annotations

import string
from typing import Dict, List, Mapping

#: The only fields a style template may reference.
TEMPLATE_FIELDS = (
    "title",
    "variable",
    "variable_label",
    "region",
    "region_label",
    "start",
    "end",
    "date",
    "year",
    "channel",
)

_FORMATTER = string.Formatter()


class StyleError(ValueError):
    """A template references an unknown field or misses a value."""


def check_template(template: str) -> List[str]:
    """Problems with ``template``'s field references (empty = OK)."""
    problems = []
    for _, field_name, _, _ in _FORMATTER.parse(template):
        if field_name is None:
            continue
        # "{channel!r}"-style conversions are not supported; keep the
        # field name before any "!" or ":".
        base = field_name.split("!")[0].split(":")[0]
        if base not in TEMPLATE_FIELDS:
            problems.append(
                f"unknown field {{{field_name}}} — allowed: "
                f"{', '.join('{' + f + '}' for f in TEMPLATE_FIELDS)}")
    return problems


def render_template(template: str, context: Mapping[str, str]) -> str:
    """Render ``template`` against ``context``.

    Raises :class:`StyleError` naming the unknown field or the missing
    value (with a hint for ``channel``, the usual suspect).
    """
    problems = check_template(template)
    if problems:
        raise StyleError("; ".join(problems))
    # str.format raises KeyError on missing values — translate it.
    needed = {base.split("!")[0].split(":")[0]
              for _, base, _, _ in _FORMATTER.parse(template)
              if base is not None}
    missing = [f for f in needed if f not in context or context[f] is None]
    if missing:
        hints = []
        for f in missing:
            if f == "channel":
                hints.append(
                    "'channel' — pass channel=... (your channel name) "
                    "when applying the style")
            else:
                hints.append(f"'{f}' — not present in the spec context")
        raise StyleError("template needs missing " + "; ".join(hints))
    values: Dict[str, str] = {k: str(context[k]) for k in needed}
    return template.format(**values)
