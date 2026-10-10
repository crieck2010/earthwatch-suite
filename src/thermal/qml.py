"""QGIS .qml styles for LST maps.

LST is continuous, so the style uses a single-band pseudocolor renderer
with a diverging blue→red ramp (cold → hot), matching the palette used in
the demo report. Breaks are in °C.
"""

from __future__ import annotations

import os
from typing import Sequence, Tuple

#: (value °C, label, r, g, b) stops of the diverging ramp.
DEFAULT_LST_STOPS: Sequence[Tuple[float, str, int, int, int]] = (
    (10.0, "10 °C", 49, 54, 149),
    (20.0, "20 °C", 69, 117, 180),
    (30.0, "30 °C", 254, 224, 144),
    (40.0, "40 °C", 215, 48, 39),
    (50.0, "50 °C", 165, 0, 38),
)


def lst_qml(
    stops: Sequence[Tuple[float, str, int, int, int]] = DEFAULT_LST_STOPS,
    vmin: float = 10.0,
    vmax: float = 50.0,
) -> str:
    """Return QML text for a single-band pseudocolor LST raster (°C)."""
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
        <colorrampshader minimumValue="{vmin}" maximumValue="{vmax}" colorRampType="INTERPOLATED" clip="0">
{items_xml}
        </colorrampshader>
      </rastershader>
      <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
      <huesaturation colorizeOn="0"/>
      <rasterresampler maxOversampling="2"/>
    </rasterrenderer>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""


def write_lst_qml(path: str, **kwargs) -> str:
    """Write an LST .qml sidecar. Returns the path."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(lst_qml(**kwargs))
    return path
