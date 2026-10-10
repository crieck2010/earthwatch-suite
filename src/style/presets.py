"""Built-in style presets, shipped with the package.

Presets are plain :class:`~style.styles.Style` instances defined as
data — no logic, no I/O. User styles live outside the package (JSON
files loaded with :func:`load_styles_from_dir`); the built-ins are the
starting catalog and the worked examples for writing your own.

Colormap choices below are drawn from survey-viz's curated set
(``viz.render.CURATED_CMAPS``); survey-viz re-validates every name
against its matplotlib registry at render time.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

from .styles import STYLE_SCHEMA_VERSION, Style, StyleError, register_style

# --- shared per-variable tables ---------------------------------------------

_DARK_DEFAULTS = {
    "sst": "turbo",
    "t2m": "turbo",
    "wind": "viridis",
    "currents": "viridis",
    "ocean-color": "YlGnBu",
    "chlorophyll": "YlGnBu",
    "tp": "Blues",
    "msl": "RdBu_r",
    "sea-ice": "Blues",
    "night-lights": "inferno",
    "topography": "terrain",
    "water-storage": "BrBG",
    "fires": "hot",
}

_LIGHT_DEFAULTS = {
    "sst": "YlOrRd",
    "t2m": "YlOrRd",
    "wind": "viridis",
    "currents": "viridis",
    "ocean-color": "YlGnBu",
    "chlorophyll": "YlGnBu",
    "tp": "Blues",
    "msl": "RdBu_r",
    "sea-ice": "Blues",
    "night-lights": "magma",
    "topography": "gist_earth",
    "water-storage": "BrBG",
    "fires": "YlOrRd",
}

#: Variable -> suggested built-in style (used by ``suggest_style``).
SUGGESTIONS: Dict[str, str] = {
    "sst": "midnight-ocean",
    "currents": "midnight-ocean",
    "ocean-color": "midnight-ocean",
    "chlorophyll": "midnight-ocean",
    "night-lights": "midnight-ocean",
    "wind": "storm-chaser",
    "msl": "storm-chaser",
    "tp": "storm-chaser",
    "fires": "storm-chaser",
    "storm-tracks": "storm-chaser",
    "topography": "field-notes",
    "water-storage": "field-notes",
    "t2m": "reel-dark",
    "sea-ice": "reel-dark",
    "earthquakes": "reel-dark",
    "streamflow": "reel-dark",
}


def _builtin_styles() -> List[Style]:
    v = STYLE_SCHEMA_VERSION
    return [
        Style(
            name="reel-dark",
            version="1.0",
            label="Reel Dark",
            description="The default survey-viz look: dark background, "
                        "per-variable curated colormaps, no footer. The "
                        "safe starting point for any variable.",
            base_style="reel-dark",
            colormaps=dict(_DARK_DEFAULTS),
            default_cmap="turbo",
            title_template="{title}",
            footer_template="",
            underlay=True,
            tags=("dark", "default"),
        ),
        Style(
            name="reel-light",
            version="1.0",
            label="Reel Light",
            description="Light-background variant of the default look for "
                        "daytime viewing and print-friendly frames.",
            base_style="light",
            colormaps=dict(_LIGHT_DEFAULTS),
            default_cmap="viridis",
            title_template="{title}",
            footer_template="",
            underlay=True,
            tags=("light", "default"),
        ),
        Style(
            name="midnight-ocean",
            version="1.0",
            label="Midnight Ocean",
            description="Cinematic dark grade for ocean variables — "
                        "inferno/plasma heat on SST and currents, quiet "
                        "footer with the variable and region.",
            base_style="reel-dark",
            colormaps={
                "sst": "inferno",
                "t2m": "inferno",
                "currents": "plasma",
                "ocean-color": "viridis",
                "chlorophyll": "viridis",
                "wind": "magma",
                "night-lights": "inferno",
                "sea-ice": "cool",
                "tp": "Blues",
                "msl": "RdBu_r",
                "topography": "terrain",
                "water-storage": "BrBG",
                "fires": "hot",
            },
            default_cmap="inferno",
            title_template="{title}",
            footer_template="{variable_label} · {region_label}",
            underlay=True,
            tags=("dark", "ocean", "cinematic"),
        ),
        Style(
            name="field-notes",
            version="1.0",
            label="Field Notes",
            description="Light, print-like style for terrain and land "
                        "variables — earthy colormaps with a dated "
                        "'Field notes' footer.",
            base_style="light",
            colormaps={
                "topography": "gist_earth",
                "sst": "YlOrRd",
                "t2m": "YlOrRd",
                "tp": "Blues",
                "msl": "RdBu_r",
                "ocean-color": "YlGnBu",
                "chlorophyll": "YlGnBu",
                "wind": "Greens",
                "currents": "ocean",
                "sea-ice": "Blues",
                "water-storage": "BrBG",
                "fires": "YlOrRd",
                "night-lights": "magma",
            },
            default_cmap="terrain",
            title_template="{title}",
            footer_template="Field notes · {date}",
            underlay=True,
            tags=("light", "land", "print"),
        ),
        Style(
            name="storm-chaser",
            version="1.0",
            label="Storm Chaser",
            description="High-contrast dark style for atmosphere and "
                        "hazards — diverging pressure, hot fires, footer "
                        "with the date range.",
            base_style="reel-dark",
            colormaps={
                "wind": "turbo",
                "msl": "seismic",
                "tp": "Blues",
                "fires": "hot",
                "sst": "coolwarm",
                "t2m": "coolwarm",
                "currents": "viridis",
                "ocean-color": "YlGnBu",
                "chlorophyll": "YlGnBu",
                "sea-ice": "Blues",
                "night-lights": "inferno",
                "topography": "terrain",
                "water-storage": "BrBG",
            },
            default_cmap="turbo",
            title_template="{title}",
            footer_template="{variable_label} · {start} → {end}",
            underlay=True,
            tags=("dark", "atmosphere", "hazards"),
        ),
        Style(
            name="creator",
            version="1.0",
            label="Creator Brand Kit",
            description="Channel brand kit: the default dark look plus a "
                        "copyright footer. Requires your channel name "
                        "(pass channel=... when applying).",
            base_style="reel-dark",
            colormaps=dict(_DARK_DEFAULTS),
            default_cmap="turbo",
            title_template="{title}",
            footer_template="© {channel}",
            underlay=True,
            tags=("dark", "brand"),
        ),
    ]


#: The built-in catalog (registered at import, in definition order).
BUILTIN_STYLES: List[Style] = _builtin_styles()
for _s in BUILTIN_STYLES:
    register_style(_s)


# --- user styles (JSON) -----------------------------------------------------

def style_from_dict(data: dict, source: str = "<dict>") -> Style:
    """Build a :class:`Style` from a plain dict (e.g. parsed JSON).

    Unknown keys raise :class:`StyleError` naming the file they came
    from; every value runs through the same validation as built-ins.
    """
    if not isinstance(data, dict):
        raise StyleError(f"{source}: style must be a JSON object")
    known = {"name", "version", "label", "description", "base_style",
             "colormaps", "default_cmap", "title_template",
             "footer_template", "underlay", "tags"}
    unknown = sorted(set(data) - known)
    if unknown:
        raise StyleError(
            f"{source}: unknown style field(s): {', '.join(unknown)} "
            f"(known: {', '.join(sorted(known))})")
    try:
        return Style(
            name=data.get("name", ""),
            version=str(data.get("version", "1.0")),
            label=data.get("label", ""),
            description=data.get("description", ""),
            base_style=data.get("base_style", "reel-dark"),
            colormaps=dict(data.get("colormaps", {}) or {}),
            default_cmap=data.get("default_cmap", "turbo"),
            title_template=data.get("title_template", "{title}"),
            footer_template=data.get("footer_template", ""),
            underlay=data.get("underlay", True),
            tags=tuple(data.get("tags", ()) or ()),
        )
    except StyleError as exc:
        raise StyleError(f"{source}: {exc}") from None


def load_styles_from_dir(path: str) -> List[Style]:
    """Load ``*.json`` user styles from ``path`` into the registry.

    Each file holds one style object (see ``docs/STYLES.md`` for the
    schema). Files are loaded in sorted order; an invalid file raises
    :class:`StyleError` naming the file — nothing is half-registered.
    """
    if not os.path.isdir(path):
        raise StyleError(f"style directory not found: {path}")
    loaded: List[Style] = []
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(path, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            raise StyleError(f"{fpath}: cannot parse JSON: {exc}") from None
        style = style_from_dict(data, source=fpath)
        register_style(style)
        loaded.append(style)
    return loaded
