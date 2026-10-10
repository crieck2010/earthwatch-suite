"""Command-line interface for survey-burn.

Subcommands: dnbr | severity | timeseries | disturbance | polygons | demo | indices
Every pipeline step also runs in --synthetic mode: no network, no data
files, deterministic seeded output — the fastest way to learn the tool.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__


def _load_npy(path: str):
    import numpy as np

    if path.endswith(".npy"):
        return np.load(path)
    raise SystemExit(f"raster input must be .npy in this build (got {path!r}); "
                     "use rasterio via the acquire path for GeoTIFFs")


def cmd_dnbr(args: argparse.Namespace) -> int:
    import numpy as np

    from . import synthetic as syn
    from .dnbr import dnbr, rdnbr

    if args.synthetic:
        pre = syn.synthetic_pre_nbr()
        post = syn.synthetic_post_nbr()
    else:
        if not args.pre or not args.post:
            raise SystemExit("--pre and --post .npy rasters are required without --synthetic")
        pre = _load_npy(args.pre)
        post = _load_npy(args.post)
    out = rdnbr(pre, post) if args.relative else dnbr(pre, post)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, out)
    kind = "RdNBR" if args.relative else "dNBR"
    finite = out[np.isfinite(out)]
    print(f"{kind}: wrote {args.out} "
          f"(n={finite.size}, mean={finite.mean():.3f}, max={finite.max():.3f})")
    return 0


def cmd_severity(args: argparse.Namespace) -> int:
    import numpy as np

    from . import qml
    from .severity import BURNED_THRESHOLD, classify

    arr = _load_npy(args.dnbr)
    sev = classify(arr)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, sev)
    burned = int(np.count_nonzero((sev >= 1) & (sev <= 4)))  # severity classes 1..4
    total = int(np.count_nonzero(sev != 255))
    print(f"severity: wrote {args.out} "
          f"(burned pixels={burned}/{total}, threshold dNBR>={BURNED_THRESHOLD})")
    if args.qml:
        qml.write_qml(args.qml, kind="severity")
        print(f"severity: wrote QML style {args.qml}")
    return 0


def cmd_timeseries(args: argparse.Namespace) -> int:
    import numpy as np

    from . import synthetic as syn
    from .dnbr import dnbr as _dnbr
    from .severity import class_areas, classify
    from .timeseries import ZonePassRecord, append_records
    from .zonal import rasterize_zone_rect
    from .zones import load_zones_geojson

    zones = load_zones_geojson(args.zones) if not args.synthetic else syn.synthetic_zones()
    shape = syn.DEMO_SHAPE
    bbox = (0, 0, shape[1], shape[0])  # pixel coords for the demo grid
    records = []
    pre = syn.synthetic_pre_nbr(shape)
    post = syn.synthetic_post_nbr(shape)
    d = _dnbr(pre, post)
    sev = classify(d)
    for zone in zones:
        ring = zone.geometry["coordinates"][0]
        zx0, zy0 = float(ring[0][0]), float(ring[0][1])
        zx1, zy1 = float(ring[2][0]), float(ring[2][1])
        zmask = rasterize_zone_rect(shape, bbox, (zx0, zy0, zx1, zy1))
        zone_sev = np.where(zmask, sev, 255).astype(np.uint8)
        areas = class_areas(zone_sev, zmask, syn.DEMO_PIXEL_M)
        records.append(
            ZonePassRecord.from_class_areas(
                zone_id=zone.id,
                zone_name=zone.name,
                pass_id=syn.SYNTHETIC_POST_PASS[0],
                date=syn.SYNTHETIC_POST_PASS[1],
                index="dnbr",
                areas=areas,
                total_pixels=int(np.count_nonzero(zmask)),
                pixel_size_m=syn.DEMO_PIXEL_M,
                cloud_cover=5.0,
            )
        )
    n = append_records(args.out, records)
    print(f"timeseries: appended {n} new rows to {args.out} ({len(records)} zones)")
    return 0


def cmd_disturbance(args: argparse.Namespace) -> int:
    from . import synthetic as syn
    from .disturbance import DisturbanceParams, detect_disturbance

    dates, ndvi_list, nbr_list = syn.synthetic_pass_series()
    params = DisturbanceParams(
        ndvi_drop=args.ndvi_drop, nbr_drop=args.nbr_drop, min_pixels=args.min_pixels
    )
    events = []
    for i in range(1, len(dates)):
        found = detect_disturbance(
            ndvi_list[i - 1], ndvi_list[i], nbr_list[i - 1], nbr_list[i],
            params=params, pixel_size_m=10.0,
            zone_id="demo-block", pass_id=f"pass-{i}", date=dates[i],
        )
        events.extend(found)
    print(f"disturbance: {len(events)} event(s) across {len(dates)} passes")
    for e in events:
        print(f"  {e.date}: {e.pixels} px = {e.area_ha:.2f} ha "
              f"(dNDVI={e.mean_ndvi_drop:.2f}, dNBR={e.mean_nbr_drop:.2f}, "
              f"confidence={e.confidence:.2f})")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump([e.__dict__ for e in events], fh, indent=2)
        print(f"disturbance: wrote {args.out}")
    return 0


def cmd_polygons(args: argparse.Namespace) -> int:
    import numpy as np

    from . import synthetic as syn
    from .dnbr import dnbr as _dnbr
    from .polygons import PIXEL_TRANSFORM, north_up_transform, vectorize, write_geojson
    from .severity import burned_mask, classify

    if args.synthetic:
        pre = syn.synthetic_pre_nbr()
        post = syn.synthetic_post_nbr()
        d = _dnbr(pre, post)
        sev = classify(d)
        mask = burned_mask(d)
        transform = syn.DEMO_TRANSFORM  # valid north-up 6-tuple
        features = vectorize(mask, transform=transform, severity=sev,
                             min_area_ha=args.min_area_ha,
                             pixel_size_m=syn.DEMO_PIXEL_M)
    else:
        if not args.mask:
            raise SystemExit("--mask .npy is required without --synthetic")
        mask = _load_npy(args.mask).astype(bool)
        sev = _load_npy(args.severity).astype("uint8") if args.severity else None
        features = vectorize(mask, min_area_ha=args.min_area_ha,
                             pixel_size_m=args.pixel_size)
    path = write_geojson(features, args.out)
    total_ha = sum(f["properties"]["area_ha"] for f in features)
    print(f"polygons: wrote {len(features)} feature(s), {total_ha:.2f} ha -> {path}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    import numpy as np

    from . import qml, synthetic as syn
    from .dnbr import dnbr as _dnbr
    from .polygons import vectorize, write_geojson
    from .severity import burned_mask, class_areas, classify

    out = os.path.abspath(args.out_dir)
    os.makedirs(out, exist_ok=True)
    zones = syn.synthetic_zones()
    pre = syn.synthetic_pre_nbr()
    post = syn.synthetic_post_nbr()
    d = _dnbr(pre, post)
    sev = classify(d)
    np.save(os.path.join(out, "dnbr.npy"), d)
    np.save(os.path.join(out, "severity.npy"), sev)
    qml.write_qml(os.path.join(out, "severity.qml"), kind="severity")
    qml.write_qml(os.path.join(out, "dnbr.qml"), kind="dnbr")
    areas = class_areas(sev, None, syn.DEMO_PIXEL_M)
    expected = syn.expected_burn_areas()
    print("demo: planted burn vs measured (whole tile)")
    for key in ("high_severity_ha", "moderate_high_severity_ha",
                "moderate_low_severity_ha", "low_severity_ha", "burned_ha"):
        print(f"  {key:26s} planted={expected[key]:7.2f} ha  measured={areas[key]:7.2f} ha")
    features = vectorize(burned_mask(d), transform=syn.DEMO_TRANSFORM,
                         severity=sev, pixel_size_m=syn.DEMO_PIXEL_M)
    gpath = write_geojson(features, os.path.join(out, "disturbance_polygons.geojson"))
    total_ha = sum(f["properties"]["area_ha"] for f in features)
    print(f"demo: {len(features)} polygon(s), {total_ha:.2f} ha -> {gpath}")
    print(f"demo: outputs in {out}")
    return 0


def cmd_indices(args: argparse.Namespace) -> int:
    from .indices_fire import BURN_INDICES

    print("burn indices: " + ", ".join(BURN_INDICES))
    print("nbr resolves through survey-imagery's registry (lazy import);")
    print("dnbr/rdnbr are two-scene indices — see `survey-burn dnbr`.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="survey-burn", description="Burn-scar / disturbance mapping engine")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("dnbr", help="compute dNBR (or RdNBR) from pre/post NBR rasters")
    s.add_argument("--pre", help=".npy pre-fire NBR raster")
    s.add_argument("--post", help=".npy post-fire NBR raster")
    s.add_argument("--out", required=True, help="output .npy path")
    s.add_argument("--relative", action="store_true", help="compute RdNBR instead")
    s.add_argument("--synthetic", action="store_true", help="use seeded demo rasters")
    s.set_defaults(func=cmd_dnbr)

    s = sub.add_parser("severity", help="classify a dNBR raster into USGS severity bands")
    s.add_argument("--dnbr", required=True, help="input .npy dNBR raster")
    s.add_argument("--out", required=True, help="output .npy severity raster")
    s.add_argument("--qml", help="write a QGIS .qml style sidecar here")
    s.set_defaults(func=cmd_severity)

    s = sub.add_parser("timeseries", help="per-zone burned-area time series (CSV)")
    s.add_argument("--zones", help="GeoJSON FeatureCollection of zones")
    s.add_argument("--out", required=True, help="output CSV path (incremental appends)")
    s.add_argument("--synthetic", action="store_true", help="use seeded demo zones/rasters")
    s.set_defaults(func=cmd_timeseries)

    s = sub.add_parser("disturbance", help="run the clearing/logging disturbance heuristic")
    s.add_argument("--ndvi-drop", type=float, default=0.15)
    s.add_argument("--nbr-drop", type=float, default=0.10)
    s.add_argument("--min-pixels", type=int, default=9)
    s.add_argument("--out", help="write events as JSON here")
    s.add_argument("--synthetic", action="store_true", default=True,
                   help="use the seeded demo pass series (default)")
    s.set_defaults(func=cmd_disturbance)

    s = sub.add_parser("polygons", help="vectorise a burn mask to GeoJSON polygons")
    s.add_argument("--mask", help=".npy bool burn mask")
    s.add_argument("--severity", help=".npy uint8 severity raster (for dominant class)")
    s.add_argument("--out", required=True, help="output .geojson path")
    s.add_argument("--min-area-ha", type=float, default=0.0)
    s.add_argument("--pixel-size", type=float, default=10.0)
    s.add_argument("--synthetic", action="store_true", help="use the seeded demo scar")
    s.set_defaults(func=cmd_polygons)

    s = sub.add_parser("demo", help="run the full pipeline on seeded synthetic data")
    s.add_argument("--out-dir", default="./burn-demo")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("indices", help="list the burn indices this engine works with")
    s.set_defaults(func=cmd_indices)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
