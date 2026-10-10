"""QGIS .qml styles for coast rasters.

Water masks get a paletted style (water = coastal blue, land = transparent
so a basemap shows through, invalid = hatched grey). Hand-written QML,
same conventions as the sibling engines.
"""

from __future__ import annotations

_QGIS_DOCTYPE = "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>"

#: (value, label, r, g, b, alpha)
_WATER_PALETTE = [
    (0, "Land (not water)", 0, 0, 0, 0),
    (1, "Water", 31, 119, 180, 255),
    (2, "Invalid (cloud / nodata)", 160, 160, 160, 180),
]


def watermask_qml() -> str:
    """Return QML text for a paletted water-mask raster (0/1/2)."""
    entries = "\n".join(
        f'        <paletteEntry value="{v}" label="{label}" '
        f'color="{r},{g},{b},{a}" alpha="{a}"/>'
        for v, label, r, g, b, a in _WATER_PALETTE
    )
    return f"""{_QGIS_DOCTYPE}
<qgis version="3.28" styleCategories="Symbology">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" alphaBand="-1" nodataColor="">
      <colorPalette>
{entries}
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
"""


def write_qml(path: str, kind: str = "watermask") -> str:
    """Write a QML style file. Returns the path."""
    import os

    if kind != "watermask":
        raise ValueError(f"unknown QML kind {kind!r}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(watermask_qml())
    return path
