"""Real satellite acquisition via survey-imagery.

For each zone, this builds a survey-imagery ``MonitorConfig`` (AOI from the
zone polygon, the requested vigor indices that imagery's registry carries)
and runs ``imagery.monitor.run_monitor`` — reusing its STAC search,
cloud masking, index computation, and per-scene statistics. The resulting
SceneStats rows are converted to :class:`vegetation.timeseries.ZonePassRecord`
with the zone id attached.

Indices defined only in survey-vegetation (gndvi/msavi) cannot be acquired
this way yet — the caller is told to compute them from downloaded bands.
"""

from __future__ import annotations

import datetime as _dt
import os
from typing import List, Sequence

from .timeseries import ZonePassRecord
from .zones import Zone


def _imagery():
    try:
        import imagery.monitor as _mon
        import imagery.aoi as _aoi
    except ImportError as exc:
        raise ImportError(
            "real acquisition needs the 'survey-imagery' package "
            "(pip install 'survey-vegetation[acquire]')"
        ) from exc
    return _mon, _aoi


def acquire_timeseries(
    zones: Sequence[Zone],
    start: str,
    end: str,
    indices: Sequence[str],
    work_dir: str,
    collections: Sequence[str] | None = None,
    max_cloud_cover: float = 40.0,
) -> List[ZonePassRecord]:
    """Acquire per-zone vigor time series from STAC catalogs (needs network)."""
    mon, aoi_mod = _imagery()
    from .indices_ext import LOCAL_INDICES

    registry_indices = [i for i in indices if i not in LOCAL_INDICES]
    if not registry_indices:
        raise ValueError(
            "none of the requested indices are in survey-imagery's registry; "
            "gndvi/msavi need local band computation (see docs)"
        )
    local_only = [i for i in indices if i in LOCAL_INDICES]
    records: List[ZonePassRecord] = []
    collections = list(collections) if collections else ["sentinel-2-l2a"]
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
        for stat in result.records:
            day = _dt.date.fromisoformat(stat.datetime[:10])
            records.append(
                ZonePassRecord(
                    zone_id=zone.id,
                    zone_name=zone.name,
                    pass_id=stat.scene_id,
                    date=day.isoformat(),
                    doy=int(day.strftime("%j")),
                    year=day.year,
                    index=stat.index,
                    mean=stat.mean,
                    median=stat.median,
                    std=stat.std,
                    minimum=stat.minimum,
                    maximum=stat.maximum,
                    p10=stat.p10,
                    p90=stat.p90,
                    valid_pixels=stat.valid_pixels,
                    total_pixels=stat.total_pixels,
                    cloud_cover=stat.cloud_cover,
                )
            )
    if local_only:
        print(
            f"note: {', '.join(local_only)} not acquired (vegetation-local "
            "indices); compute from downloaded bands via vegetation.indices_ext"
        )
    return sorted(records, key=lambda r: (r.zone_id, r.date))
