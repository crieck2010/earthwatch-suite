"""QGIS .qml styles for water masks.

Water masks are binary, so the style uses a paletted renderer with two
entries — dry land transparent, water a clear blue — following the same
hand-written QML conventions as the sibling engines.
"""

from __future__ import annotations

import os

#: (value, label, (r, g, b, a))
WATER_PALETTE = (
    (0, "dry land", (0, 0, 0, 0)),
    (1, "water", (31, 120, 180, 255)),
)


def water_qml() -> str:
    """Return QML text for a paletted water-mask raster."""
    entries = "\n".join(
        f'        <paletteEntry value="{v}" label="{label}" '
        f'color="{r},{g},{b},{a}" alpha="{a}"/>'
        for v, label, (r, g, b, a) in WATER_PALETTE
    )
    return f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="Symbology">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" alphaBand="-1" nodataColor="">
      <colorPalette>
{entries}
      </colorPalette>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation colorizeOn="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""


def write_water_qml(path: str) -> str:
    """Write the water-mask .qml sidecar next to a mask raster."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(water_qml())
    return path
