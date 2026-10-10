"""QGIS .qml styles for elevation-difference rasters.

Δh is signed, so the style is a diverging single-band pseudocolor ramp:
red = cut (ground removed), blue = fill (ground added), near-white at
zero. Breaks are in metres and the no-change band matches the engine's
default noise floor (±0.10 m).
"""

from __future__ import annotations

from typing import Sequence, Tuple

#: (value m, label, r, g, b) stops of the diverging Δh ramp.
DEFAULT_DH_STOPS: Sequence[Tuple[float, str, int, int, int]] = (
    (-2.0, "−2 m (cut)", 165, 0, 38),
    (-0.5, "−0.5 m", 244, 109, 67),
    (-0.1, "−0.1 m", 254, 224, 139),
    (0.0, "no change", 255, 255, 255),
    (0.1, "+0.1 m", 224, 243, 248),
    (0.5, "+0.5 m", 116, 173, 209),
    (2.0, "+2 m (fill)", 49, 54, 149),
)


def dh_qml(
    stops: Sequence[Tuple[float, str, int, int, int]] = DEFAULT_DH_STOPS,
    vmin: float = -2.0,
    vmax: float = 2.0,
) -> str:
    """Return QML text for a single-band pseudocolor Δh raster (metres)."""
    items = []
    for value, label, r, g, b in stops:
        items.append(
            f'        <item value="{value}" label="{label}" '
            f'color="{r},{g},{b},255" alpha="255"/>'
        )
    items_xml = "\n".join(items)
    return f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="Symbology">
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1" alphaBand="-1" classificationMin="{vmin}" classificationMax="{vmax}" nodataColor="">
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" clip="0" classificationMode="1">
{items_xml}
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
  </pipe>
</qgis>
"""


def write_qml(path: str, **kwargs) -> str:
    """Write the Δh QML style to ``path``. Returns the path written."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dh_qml(**kwargs))
    return path
