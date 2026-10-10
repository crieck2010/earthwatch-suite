"""survey-flow: flow-field rendering engine.

Advect particles through surface-current vector fields, render them over a
scalar (water temperature) colormap, and emit a stable frame manifest that
survey-animate turns into video. The math core (fields + advection) is
stdlib+numpy; matplotlib is only needed for rendering.
"""

from flow.advect import AdvectionConfig, ParticleSet
from flow.fields import TimeVaryingField, VectorField
from flow.frames import FrameEntry, FrameManifest, MANIFEST_SCHEMA
from flow.styles import Style, get_style, list_styles

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AdvectionConfig",
    "ParticleSet",
    "TimeVaryingField",
    "VectorField",
    "FrameEntry",
    "FrameManifest",
    "MANIFEST_SCHEMA",
    "Style",
    "get_style",
    "list_styles",
]
