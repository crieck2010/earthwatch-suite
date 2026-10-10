"""Real satellite acquisition via survey-imagery.

For each zone, this builds a survey-imagery ``MonitorConfig`` (AOI from the
zone polygon, the requested water indices that imagery's registry carries)
and runs ``imagery.monitor.run_monitor`` — reusing its STAC search, cloud
masking, index computation, and per-scene GeoTIFF output. Each scene's index
raster (``<zone>_<index>_<date>.tif`` under the zone work dir) is
thresholded into a water mask by :mod:`flood.watermask` and reduced to
water-area statistics, which become :class:`flood.timeseries.ZonePassRecord`
rows with the zone id attached.

Cloud handling: survey-imagery's pipeline cloud-masks its index rasters, so
clouded pixels arrive as NaN — and NaN is never water
(:func:`flood.watermask.water_mask`). The scene's ``cloud_cover`` percent
from the monitor's own statistics travels on the record so downstream
consumers can gate on it.

Indices defined only in survey-flood (mndwi) cannot be acquired this way
yet — the caller is told to compute them from downloaded bands.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Sequence

import numpy as np

from .timeseries import ZonePassRecord
from .watermask import build_water_mask
from .zonal import water_stats
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
            "(pip install 'survey-flood[acquire]')"
        ) from exc
    return _mon, _aoi


def acquire_timeseries(
    zones: Sequence[Zone],
    start: str,
    end: str,
    indices: Sequence[str],
    work_dir: str,
    threshold: float = 0.0,
    min_pixels: int = 4,
    collections: Sequence[str] | None = None,
    max_cloud_cover: float = 40.0,
) -> List[ZonePassRecord]:
    """Acquire per-zone water time series from STAC catalogs (needs network)."""
    mon, aoi_mod = _imagery()
    from .indices_water import LOCAL_INDICES

    registry_indices = [i for i in indices if i not in LOCAL_INDICES]
    if not registry_indices:
        raise ValueError(
            "none of the requested indices are in survey-imagery's registry; "
            "mndwi needs local band computation (see docs)"
        )
    local_only = [i for i in indices if i in LOCAL_INDICES]
    records: List[ZonePassRecord] = []
    collections = list(collections) if collections else ["sentinel-2-l2a"]
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "reading index rasters needs 'rasterio' "
            "(pip install 'survey-flood[raster]')"
        ) from exc
    for zone in zones:
        aoi = aoi_mod.from_geojson(
            {"type": "Feature", "properties": {}, "geometry": zone.geometry},
            name=zone.id,
        )
        zone_dir = os.path.join(work_dir, zone.id)
        config = mon.MonitorConfig(
            name=zone.id,
            aoi=aoi,
            collections=collections,
            indices=registry_indices,
            output_dir=zone_dir,
            start=start,
            end=end,
            max_cloud_cover=max_cloud_cover,
        )
        result = mon.run_monitor(config)
        # SceneStats keyed by (date, index) for cloud-cover lookup.
        stats_by_key: Dict[tuple, Any] = {}
        for stat in result.records:
            stats_by_key[(stat.datetime[:10], stat.index)] = stat
        for path in result.rasters:
            match = _RASTER_RE.match(os.path.basename(path))
            if not match or not os.path.exists(path):
                continue
            index = match.group("index")
            date = match.group("date")
            with rasterio.open(path) as src:
                index_array = src.read(1).astype(np.float32)
                pixel_size = abs(src.transform.a)
            water, valid = build_water_mask(
                index_array, threshold=threshold, min_pixels=min_pixels
            )
            stats = water_stats(water, valid, pixel_size)
            stat = stats_by_key.get((date, index))
            cloud_cover = getattr(stat, "cloud_cover", None) if stat else None
            scene_id = getattr(stat, "scene_id", os.path.basename(path))
            records.append(
                ZonePassRecord.from_water_stats(
                    zone.id, zone.name, scene_id, date, index, stats,
                    cloud_cover=cloud_cover,
                )
            )
    if local_only:
        print(
            f"note: {', '.join(local_only)} not acquired (flood-local "
            "indices); compute from downloaded bands via flood.indices_water"
        )
    return sorted(records, key=lambda r: (r.zone_id, r.date))
