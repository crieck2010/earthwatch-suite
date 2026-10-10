"""Real satellite acquisition via survey-imagery.

For dNBR work the caller needs two scenes: a pre-fire NBR and a post-fire
NBR. This module builds a survey-imagery ``MonitorConfig`` per fire event
(AOI from the zone polygon, ``nbr`` from imagery's registry) and runs
``imagery.monitor.run_monitor`` — reusing its STAC search, cloud masking,
index computation, and per-scene GeoTIFF output. The caller then pairs the
pre/post NBR rasters with :func:`burn.dnbr.select_pre_post` and feeds them
to :func:`burn.dnbr.dnbr`.

Cloud handling: survey-imagery's pipeline cloud-masks its index rasters, so
clouded pixels arrive as NaN — and NaN never counts as burned
(:func:`burn.severity.burned_mask`). The scene's ``cloud_cover`` percent
travels on every record so downstream consumers can gate on it.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .dnbr import PassInfo
from .zones import Zone

#: ``<name>_<index>_<YYYY-MM-DD>.tif`` — the file layout run_monitor writes.
_RASTER_RE = re.compile(r"^(?P<name>.+)_(?P<index>[a-z0-9]+)_(?P<date>\d{4}-\d{2}-\d{2})\.tif$")


def _imagery():
    try:
        import imagery.monitor as _mon
        import imagery.aoi as _aoi
    except ImportError as exc:
        raise ImportError(
            "real acquisition needs the 'survey-imagery' package "
            "(pip install 'survey-burn[acquire]')"
        ) from exc
    return _mon, _aoi


def acquire_nbr_passes(
    zone: Zone,
    start: str,
    end: str,
    work_dir: str,
    collections: Sequence[str] | None = None,
    max_cloud_cover: float = 40.0,
) -> List[PassInfo]:
    """Acquire NBR scenes for a zone from STAC catalogs (needs network).

    Runs survey-imagery's monitor for ``nbr`` and returns one
    :class:`PassInfo` per scene found, with ``nbr_path`` pointing at the
    scene's GeoTIFF. Reading the rasters needs rasterio.
    """
    mon, aoi_mod = _imagery()
    try:
        import rasterio  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "reading index rasters needs 'rasterio' "
            "(pip install 'survey-burn[raster]')"
        ) from exc
    zone_dir = os.path.join(work_dir, zone.id)
    os.makedirs(zone_dir, exist_ok=True)
    config = mon.MonitorConfig(
        name=zone.id,
        aoi=aoi_mod.AOI.from_geojson(zone.geometry),
        indices=["nbr"],
        start=start,
        end=end,
        collections=list(collections) if collections else ["sentinel-2-l2a"],
        max_cloud_cover=max_cloud_cover,
        work_dir=zone_dir,
    )
    result = mon.run_monitor(config)
    passes: List[PassInfo] = []
    for scene in result.scenes:
        m = _RASTER_RE.match(os.path.basename(scene.path))
        if not m:
            continue
        passes.append(
            PassInfo(
                pass_id=f"{zone.id}-{m.group('date')}",
                date=m.group("date"),
                cloud_cover=float(getattr(scene, "cloud_cover", 0.0) or 0.0),
                nbr_path=scene.path,
            )
        )
    passes.sort(key=lambda p: p.date)
    return passes


def read_raster(path: str) -> Tuple[np.ndarray, tuple]:
    """Read a single-band raster → (float32 array, affine 6-tuple)."""
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "reading rasters needs 'rasterio' (pip install 'survey-burn[raster]')"
        ) from exc
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        t = src.transform
        transform = (t.a, t.b, t.c, t.d, t.e, t.f)
    return arr, transform
