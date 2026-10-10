"""QGIS .qml styles for vigor-class maps.

Class rasters are categorical, so the style uses a paletted renderer with one
entry per vigor band — matching the colours in :data:`vegetation.vigor.DEFAULT_VIGOR_BANDS`.
"""

from __future__ import annotations

import os
from typing import Sequence, Tuple

from .vigor import DEFAULT_VIGOR_BANDS


def vigor_qml(
    bands: Sequence[Tuple[int, str, float, float, Tuple[int, int, int, int]]] = DEFAULT_VIGOR_BANDS,
) -> str:
    """Return QML text for a paletted vigor-class raster."""
    entries = []
    for class_id, label, _lo, _hi, (r, g, b, _a) in bands:
        entries.append(
            f'        <paletteEntry value="{class_id}" label="{label}" '
            f'color="{r},{g},{b},255" alpha="255"/>'
        )
    entries_xml = "\n".join(entries)
    return f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="Symbology">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" alphaBand="-1" nodataColor="">
      <colorPalette>
{entries_xml}
      </colorPalette>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation colorizeOn="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""


def write_vigor_qml(path: str) -> str:
    """Write the vigor-class .qml sidecar next to a class raster."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(vigor_qml())
    return path
