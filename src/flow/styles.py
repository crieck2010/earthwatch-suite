"""Style presets for frame rendering.

Palettes are colorblind-safe (cividis/viridis families). Each preset is a
frozen :class:`Style` so render runs are reproducible and the manifest can
record exactly what was used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Style:
    """All visual decisions for one rendered frame."""

    name: str
    description: str
    # Scalar (temperature) base map
    scalar_cmap: str = "cividis"     # matplotlib colormap name
    scalar_vmin: str | float = "auto"  # "auto" or a fixed value
    scalar_vmax: str | float = "auto"
    scalar_label: str = "Water temperature (°C)"
    # Canvas
    background: str = "#0b1526"      # ocean / no-data color
    land_color: str = "#2a2f3a"
    # Particle streaks
    particle_color: str = "#ffffff"
    particle_linewidth: float = 1.2
    trail_alpha_head: float = 0.9    # alpha at the particle head
    trail_alpha_tail: float = 0.05   # alpha at the trail tail
    head_dot: bool = True            # draw a dot at each particle head
    head_size: float = 6.0
    # Decorations
    text_color: str = "#ffffff"
    colorbar: bool = True
    scalebar: bool = True
    timestamp: bool = True
    timestamp_format: str = "{timestamp}"  # format string over frame metadata
    title: str = ""


STYLES: Dict[str, Style] = {
    "reel-dark": Style(
        name="reel-dark",
        description="Dark-navy base for vertical social reels (mapped.earth look).",
        scalar_cmap="cividis",
        background="#0b1526",
        land_color="#232b3d",
        particle_color="#ffffff",
        particle_linewidth=1.3,
        text_color="#ffffff",
    ),
    "reel-light": Style(
        name="reel-light",
        description="Light base for vertical social reels; dark particles.",
        scalar_cmap="viridis",
        background="#eef1f4",
        land_color="#c9cfd6",
        particle_color="#0e1f38",
        particle_linewidth=1.3,
        text_color="#0e1f38",
    ),
    "print": Style(
        name="print",
        description="High-contrast print/report style; thin black streaks.",
        scalar_cmap="cividis",
        background="#ffffff",
        land_color="#d9d9d9",
        particle_color="#000000",
        particle_linewidth=0.8,
        trail_alpha_head=0.75,
        text_color="#000000",
    ),
}


def get_style(name: str) -> Style:
    """Return the :class:`Style` preset ``name`` or raise KeyError."""
    try:
        return STYLES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown style {name!r}; known styles: {sorted(STYLES)}") from exc


def list_styles() -> List[Style]:
    """All style presets, sorted by name."""
    return [STYLES[k] for k in sorted(STYLES)]
