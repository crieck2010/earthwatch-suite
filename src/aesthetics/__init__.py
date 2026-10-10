"""survey-aesthetics: aesthetic rendering engine for remote-sensing reels.

Deterministic, UI-free render layers that give survey-viz output the
mapped.earth / warming.watch look: LIC flow-field streaks, advected
particle strands, additive glow with bloom, 3D prism extrusion, rotated
auto-fit framing, editorial typography, collision-aware furniture
layout, custom legends, styled basemaps, watermark furniture, and the
``teal_pink`` density-surface compositing primitive.

Every public function is deterministic: same inputs + same seed ->
byte-identical output, so survey-cache fingerprints stay stable across
frames of a reel.

Intended consumer: survey-viz (see docs/API.md for the call pattern).
"""

import matplotlib

matplotlib.use("Agg")  # headless render backend; no UI imports anywhere

from aesthetics.backgrounds import BLACK, PAPER, draw_coastlines, fig_to_rgba, new_canvas
from aesthetics.colormaps import get_colormap, list_colormaps
from aesthetics.basemap import (
    NO_BASEMAP,
    SUBTLE_LAND,
    VOID_BLACK,
    BasemapStyle,
    draw_basemap,
    get_basemap,
    list_basemaps,
)
from aesthetics.framing import (
    fit_extent_to_canvas,
    optimal_rotation,
    rotate_frame,
    rotate_frame_fill,
    rotated_render_size,
    tight_bbox,
)
from aesthetics.glow import accumulate_glow, glow_from_grid, light_beam
from aesthetics.layout import (
    FurnitureSpec,
    PlacedItem,
    box_spec,
    fit_font_size,
    place_furniture,
    text_extent_frac,
    text_spec,
)
from aesthetics.legends import (
    counter,
    date_dial,
    encoding_statement,
    gradient_bar,
    timeline,
    vertical_scale_bar,
)
from aesthetics.lic import lic_texture
from aesthetics.presets import (
    DARK_FLOW,
    DARK_GLOW,
    DARK_STRANDS,
    PAPER_PRISM,
    Preset,
    get_preset,
    list_presets,
    preset_figure,
)
from aesthetics.prism import prism_frame
from aesthetics.strands import render_strands
from aesthetics.surface import render_surface
from aesthetics.typography import (
    draw_readout,
    draw_subtitle,
    draw_title,
    place_labels,
)
from aesthetics.watermark import (
    draw_copyright,
    draw_data_source,
    draw_watermark,
    frame_furniture,
)

__version__ = "0.4.0"

__all__ = [
    "__version__",
    # backgrounds
    "BLACK",
    "PAPER",
    "new_canvas",
    "draw_coastlines",
    "fig_to_rgba",
    # basemap
    "BasemapStyle",
    "VOID_BLACK",
    "NO_BASEMAP",
    "SUBTLE_LAND",
    "list_basemaps",
    "get_basemap",
    "draw_basemap",
    # lic
    "lic_texture",
    # strands
    "render_strands",
    # surface
    "render_surface",
    # colormaps
    "get_colormap",
    "list_colormaps",
    # glow
    "accumulate_glow",
    "glow_from_grid",
    "light_beam",
    # prism
    "prism_frame",
    # framing
    "rotate_frame",
    "rotate_frame_fill",
    "optimal_rotation",
    "rotated_render_size",
    "tight_bbox",
    "fit_extent_to_canvas",
    # typography
    "draw_title",
    "draw_subtitle",
    "draw_readout",
    "place_labels",
    # layout
    "FurnitureSpec",
    "PlacedItem",
    "place_furniture",
    "text_spec",
    "box_spec",
    "text_extent_frac",
    "fit_font_size",
    # legends
    "gradient_bar",
    "vertical_scale_bar",
    "date_dial",
    "timeline",
    "counter",
    "encoding_statement",
    # watermark
    "draw_watermark",
    "draw_copyright",
    "draw_data_source",
    "frame_furniture",
    # presets
    "Preset",
    "DARK_FLOW",
    "DARK_GLOW",
    "DARK_STRANDS",
    "PAPER_PRISM",
    "get_preset",
    "list_presets",
    "preset_figure",
]
