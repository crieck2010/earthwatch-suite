"""QGIS .qml styles for burn rasters.

dNBR rasters get a continuous-ish singleband pseudocolor style; severity
rasters get a paletted style with the six USGS classes (+ nodata
transparent). Both follow the same hand-written QML conventions as the
sibling engines.
"""

from __future__ import annotations

import os

from .severity import SEVERITY_CLASSES

_QGIS_DOCTYPE = "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>"


def severity_qml() -> str:
    """Return QML text for a paletted severity-class raster."""
    entries = "\n".join(
        f'        <paletteEntry value="{cid}" label="{name}" '
        f'color="{r},{g},{b},{a}" alpha="{a}"/>'
        for cid, name, _lo, _hi, (r, g, b, a) in SEVERITY_CLASSES
    )
    return f"""{_QGIS_DOCTYPE}
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


def dnbr_qml() -> str:
    """Return QML text for a dNBR raster (singleband pseudocolor)."""
    items = "\n".join(
        f'        <item value="{v}" label="{label}" color="{color}" alpha="255"/>'
        for v, label, color in (
            (-0.5, "strong regrowth", "43,131,186"),
            (-0.1, "unburned", "56,168,0"),
            (0.1, "low", "139,209,0"),
            (0.27, "moderate-low", "255,255,0"),
            (0.44, "moderate-high", "255,128,0"),
            (0.66, "high", "215,25,28"),
        )
    )
    return f"""{_QGIS_DOCTYPE}
<qgis version="3.28" styleCategories="Symbology">
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1" alphaBand="-1" nodataColor="">
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" clip="0" classificationMode="1">
{items}
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation colorizeOn="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""


def write_qml(path: str, kind: str = "severity") -> str:
    """Write a .qml sidecar next to a burn raster. ``kind``: severity|dnbr."""
    text = severity_qml() if kind == "severity" else dnbr_qml()
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path
