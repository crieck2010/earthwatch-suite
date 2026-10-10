"""Real satellite acquisition via survey-imagery.

For shoreline work the caller needs a per-pass water-index raster (NDWI or
MNDWI). This module builds a survey-imagery ``MonitorConfig`` per site (AOI
from the site polygon, the water index from imagery's registry) and runs
``imagery.monitor.run_monitor`` — reusing its STAC search, cloud masking,
index computation, and per-scene GeoTIFF output.

Cloud handling: survey-imagery's pipeline cloud-masks its index rasters, so
clouded pixels arrive as NaN — and NaN never counts as water
(:func:`coast.watermask.water_mask` treats them as invalid). The scene's
``cloud_cover`` percent travels on every record so downstream consumers can
gate on it.

Tide caveat (see README limitations): satellite passes are *not*
tide-synchronised. For erosion-rate work, prefer passes near the same tidal
stage or apply a tide correction downstream; this module passes through
whatever the catalog returns.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .zones import Site

#: ``<name>_<index>_<YYYY-MM-DD>.tif`` — the file layout run_monitor writes.
_RASTER_RE = re.compile(r"^(?P<name>.+)_(?P<index>[a-z0-9]+)_(?P<date>\d{4}-\d{2}-\d{2})\.tif$")


@dataclass
class PassInfo:
    """One acquired pass: index raster on disk + metadata."""

    pass_id: str
    date: str  # ISO YYYY-MM-DD
    index: str
    raster_path: str
    cloud_cover: float = 0.0


def _imagery():
    try:
        import imagery.monitor as _mon
        import imagery.aoi as _aoi
    except ImportError as exc:
        raise ImportError(
            "real acquisition needs the 'survey-imagery' package "
            "(pip install 'survey-coast[acquire]')"
        ) from exc
    return _mon, _aoi


def acquire_index_passes(
    site: Site,
    start: str,
    end: str,
    work_dir: str,
    index: str = "ndwi",
    collections: Sequence[str] | None = None,
    max_cloud_cover: float = 40.0,
) -> List[PassInfo]:
    """Acquire per-pass water-index scenes for a site (needs network).

    Runs survey-imagery's monitor for ``index`` and returns one
    :class:`PassInfo` per scene found, with ``raster_path`` pointing at the
    scene's GeoTIFF. Reading the rasters needs rasterio.
    """
    mon, aoi_mod = _imagery()
    try:
        import rasterio  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "reading acquired rasters needs rasterio "
            "(pip install 'survey-coast[raster]')"
        ) from exc

    os.makedirs(work_dir, exist_ok=True)
    aoi = aoi_mod.AOI.from_geojson(
        {"type": "Feature", "geometry": site.geometry, "properties": {}}
    )
    config = mon.MonitorConfig(
        name=site.id,
        aoi=aoi,
        indices=[index],
        start=start,
        end=end,
        collections=list(collections) if collections else None,
        max_cloud_cover=max_cloud_cover,
        out_dir=work_dir,
    )
    result = mon.run_monitor(config)
    passes: List[PassInfo] = []
    for scene in result.scenes:
        raster = getattr(scene, f"{index}_path", None) or getattr(scene, "raster_path", "")
        m = _RASTER_RE.match(os.path.basename(raster))
        date = m.group("date") if m else getattr(scene, "date", "")
        passes.append(
            PassInfo(
                pass_id=f"{site.id}-{date}",
                date=date,
                index=index,
                raster_path=raster,
                cloud_cover=float(getattr(scene, "cloud_cover", 0.0)),
            )
        )
    return sorted(passes, key=lambda p: p.date)
