"""Real satellite acquisition via survey-imagery (needs network).

Thermal path: Landsat Collection 2 Level-2 ``ST_B10`` (surface temperature,
Kelvin = DN × 0.00341802 + 149) through survey-imagery's STAC search, then
:func:`thermal.temperature.st_dn_to_celsius`. Falls back to top-of-atmosphere
brightness temperature (TIRS band 10) with the Planck inversion + emissivity
correction when only L1 data is available — see :mod:`thermal.temperature`.

Cloud handling: survey-imagery's pipeline cloud-masks its rasters, so
clouded pixels arrive as NaN — and NaN never counts as land temperature.
The scene's ``cloud_cover`` percent travels on every record so downstream
consumers can gate on it.

Overpass-time caveat (see README limitations): Landsat overpasses are
~10:00 local solar time. Diurnal heating means LST from different sensors
or overpass times is not directly comparable — compare like with like.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence

from .temperature import st_dn_to_celsius
from .zones import Zone


@dataclass
class PassInfo:
    """One acquired pass: ST_B10 raster on disk + metadata."""

    pass_id: str
    date: str  # ISO YYYY-MM-DD
    raster_path: str
    cloud_cover: float = 0.0


def _imagery():
    try:
        import imagery.stac as _stac
        import imagery.aoi as _aoi
        import imagery.acquisition as _acq
        import imagery.bands as _bands
    except ImportError as exc:
        raise ImportError(
            "real acquisition needs the 'survey-imagery' package "
            "(pip install 'survey-thermal[acquire]')"
        ) from exc
    return _stac, _aoi, _acq, _bands


def _resolve_thermal_asset(scene, bands) -> str:
    """Find the ST_B10 (surface temperature) asset href on a STAC scene."""
    assets = getattr(scene, "assets", {}) or {}
    alias = "tirs1"
    asset_key = None
    try:
        asset_key = bands.canonical_asset(alias, getattr(scene, "collection", ""))
    except Exception:
        asset_key = None
    for key in (asset_key, "ST_B10", "tirs1", "B10"):
        if key and key in assets:
            asset = assets[key]
            href = asset.get("href") if isinstance(asset, dict) else getattr(asset, "href", None)
            if href:
                return href
    raise RuntimeError(
        f"no thermal (ST_B10) asset found on scene {getattr(scene, 'id', '?')}"
    )


def acquire_lst_passes(
    zone: Zone,
    start: str,
    end: str,
    work_dir: str,
    collections: Sequence[str] | None = None,
    max_cloud_cover: float = 40.0,
) -> List[PassInfo]:
    """Acquire per-pass ST_B10 scenes for a zone and convert to LST GeoTIFFs.

    Returns one :class:`PassInfo` per scene; the written rasters are in
    degrees Celsius. Reading/writing rasters needs rasterio.
    """
    stac, aoi_mod, acq, bands = _imagery()
    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError as exc:
        raise ImportError(
            "reading acquired rasters needs rasterio "
            "(pip install 'survey-thermal[raster]')"
        ) from exc

    os.makedirs(work_dir, exist_ok=True)
    aoi = aoi_mod.AOI.from_geojson(
        {"type": "Feature", "geometry": zone.geometry, "properties": {}}
    )
    scenes = stac.search_scenes(
        aoi=aoi,
        start=start,
        end=end,
        collections=list(collections) if collections else None,
    )
    scenes = stac.filter_max_cloud(stac.latest_per_date(scenes), max_cloud_cover)

    passes: List[PassInfo] = []
    for scene in scenes:
        href = _resolve_thermal_asset(scene, bands)
        signer = getattr(scene, "signer", None)
        band = acq.read_band(href, aoi=aoi, signer=signer)
        lst_c = st_dn_to_celsius(band.data)
        date = str(getattr(scene, "datetime", ""))[:10]
        out_path = os.path.join(work_dir, f"{zone.id}_lst_{date}.tif")
        a, b, c, d, e, f = band.transform
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=lst_c.shape[0],
            width=lst_c.shape[1],
            count=1,
            dtype="float32",
            crs=band.crs,
            transform=Affine(a, b, c, d, e, f),
            nodata=float("nan"),
            compress="deflate",
        ) as dst:
            dst.write(lst_c.astype("float32"), 1)
        passes.append(
            PassInfo(
                pass_id=f"{zone.id}-{date}",
                date=date,
                raster_path=out_path,
                cloud_cover=float(getattr(scene, "cloud_cover", 0.0)),
            )
        )
    return passes
