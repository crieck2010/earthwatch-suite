"""Command-line interface for survey-vegetation.

Subcommands: timeseries | anomalies | season | vigor-map | demo | indices
Every pipeline step also runs in --synthetic mode: no network, no data
files, deterministic seeded output — the fastest way to learn the tool.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__


def _parse_indices(text: str):
    from .indices_ext import VIGOR_INDICES

    indices = [t.strip().lower() for t in text.split(",") if t.strip()]
    unknown = [i for i in indices if i not in VIGOR_INDICES]
    if unknown:
        raise SystemExit(
            f"unknown vigor indices: {', '.join(unknown)} "
            f"(known: {', '.join(VIGOR_INDICES)})"
        )
    return indices


def cmd_timeseries(args: argparse.Namespace) -> int:
    from . import synthetic as syn
    from . import timeseries as ts
    from .zones import load_zones_geojson

    indices = _parse_indices(args.indices)
    if args.synthetic:
        zones = syn.synthetic_zones()
        records = syn.synthetic_timeseries(
            zones, start=args.start, end=args.end, indices=indices,
            seed=args.seed,
        )
    else:
        from .acquire import acquire_timeseries

        zones = load_zones_geojson(args.zones)
        records = acquire_timeseries(
            zones, args.start, args.end, indices, args.work_dir,
            collections=args.collections.split(",") if args.collections else None,
            max_cloud_cover=args.max_cloud_cover,
        )
    path, n = ts.append_csv(records, args.out)
    print(f"wrote {n} new records -> {path}")
    return 0


def cmd_anomalies(args: argparse.Namespace) -> int:
    from . import anomalies as an
    from . import timeseries as ts

    records = ts.read_csv(args.timeseries)
    out = []
    for index in _parse_indices(args.indices):
        out.extend(
            an.detect_anomalies(
                records, index,
                window_days=args.window_days,
                z_threshold=args.z_threshold,
                min_years=args.min_years,
            )
        )
    path = an.write_anomalies_csv(out, args.out)
    stressed = [r for r in out if r.stressed]
    print(f"wrote {len(out)} anomaly records -> {path}")
    print(f"stressed zone-passes: {len(stressed)}")
    for r in stressed[:20]:
        print(f"  {r.zone_id} {r.index} {r.date}: z={r.z_score:.2f}")
    return 0


def cmd_season(args: argparse.Namespace) -> int:
    from . import season as sn
    from . import timeseries as ts

    records = ts.read_csv(args.timeseries)
    out = []
    for index in _parse_indices(args.indices):
        out.extend(sn.summarize_all_years(records, index))
    path = sn.write_season_json(out, args.out)
    found = [s for s in out if s.season_found]
    print(f"wrote {len(out)} season summaries ({len(found)} with season) -> {path}")
    for s in found:
        print(
            f"  {s.zone_id} {s.year}: green-up {s.greenup_date}, "
            f"peak {s.peak_value:.2f} on {s.peak_date}, "
            f"senescence {s.senescence_date}, "
            f"integral {s.season_integral:.1f} index-days"
        )
    return 0


def cmd_vigor_map(args: argparse.Namespace) -> int:
    import numpy as np

    from . import qml, vigor
    from . import synthetic as syn

    if args.synthetic:
        index_array = syn.synthetic_index_raster(
            shape=tuple(args.shape), seed=args.seed
        )
        transform = None
        crs = None
    else:
        if not args.index_raster:
            raise SystemExit("--index-raster is required without --synthetic")
        try:
            import rasterio
        except ImportError:
            raise SystemExit("reading rasters needs 'rasterio' (pip install 'survey-vegetation[raster]')")
        with rasterio.open(args.index_raster) as src:
            index_array = src.read(1).astype(np.float32)
            transform = tuple(src.transform)[:6]
            crs = str(src.crs) if src.crs else None
    classes = vigor.classify(index_array)
    if transform is not None:
        tif = vigor.write_class_geotiff(classes, args.out, transform, crs)
        print(f"wrote class GeoTIFF -> {tif}")
    else:
        npy = vigor.write_class_npy(classes, os.path.splitext(args.out)[0] + ".npy")
        print(f"wrote class array -> {npy} (no georeferencing in synthetic mode)")
    qml_path = qml.write_vigor_qml(os.path.splitext(args.out)[0] + ".qml")
    print(f"wrote QGIS style -> {qml_path}")
    for h in vigor.class_histogram(classes):
        print(f"  {h['label']}: {h['pixels']} px ({h['fraction']:.1%})")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Full synthetic pipeline: zones -> timeseries -> anomalies -> season -> vigor map."""
    from . import anomalies as an
    from . import season as sn
    from . import synthetic as syn
    from . import timeseries as ts
    from . import qml, vigor
    from .zones import zones_to_feature_collection

    outdir = args.out_dir
    os.makedirs(outdir, exist_ok=True)
    zones = syn.synthetic_zones()
    with open(os.path.join(outdir, "zones.geojson"), "w", encoding="utf-8") as fh:
        json.dump(zones_to_feature_collection(zones), fh, indent=2)
    indices = ["ndvi", "evi"]
    records = syn.synthetic_timeseries(
        zones, start="2022-03-01", end="2024-11-30",
        indices=indices, seed=args.seed,
    )
    ts_path, n = ts.append_csv(records, os.path.join(outdir, "timeseries.csv"))
    print(f"[1/5] timeseries: {n} records -> {ts_path}")

    anoms = []
    for index in indices:
        anoms.extend(an.detect_anomalies(records, index))
    an_path = an.write_anomalies_csv(anoms, os.path.join(outdir, "anomalies.csv"))
    stressed = sum(1 for r in anoms if r.stressed)
    print(f"[2/5] anomalies: {len(anoms)} records, {stressed} stressed -> {an_path}")

    summaries = []
    for index in indices:
        summaries.extend(sn.summarize_all_years(records, index))
    sn_path = sn.write_season_json(summaries, os.path.join(outdir, "season.json"))
    print(f"[3/5] season: {len(summaries)} summaries -> {sn_path}")

    import numpy as np

    raster = syn.synthetic_index_raster(seed=args.seed)
    classes = vigor.classify(raster)
    vigor.write_class_npy(classes, os.path.join(outdir, "vigor_classes.npy"))
    qml.write_vigor_qml(os.path.join(outdir, "vigor_classes.qml"))
    print("[4/5] vigor map: vigor_classes.npy + .qml")

    print(f"[5/5] demo complete in {outdir}")
    return 0


def cmd_indices(_args: argparse.Namespace) -> int:
    from .indices_ext import VIGOR_INDICES, required_bands

    for name in VIGOR_INDICES:
        try:
            bands = ", ".join(required_bands(name))
        except ImportError:
            bands = "(needs survey-imagery: pip install 'survey-vegetation[acquire]')"
        print(f"{name}: bands {bands}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="survey-vegetation",
        description="Vegetation/vigor analytics for satellite remote sensing",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("timeseries", help="build per-zone vigor time series")
    t.add_argument("--zones", default="", help="zones GeoJSON (real mode)")
    t.add_argument("--start", default="2022-03-01")
    t.add_argument("--end", default="2024-11-30")
    t.add_argument("--indices", default="ndvi,evi")
    t.add_argument("--out", default="timeseries.csv")
    t.add_argument("--synthetic", action="store_true", help="seeded demo data, no network")
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--work-dir", default="./.veg-work")
    t.add_argument("--collections", default="")
    t.add_argument("--max-cloud-cover", type=float, default=40.0)
    t.set_defaults(func=cmd_timeseries)

    a = sub.add_parser("anomalies", help="phenology-aware anomaly detection")
    a.add_argument("--timeseries", required=True)
    a.add_argument("--indices", default="ndvi")
    a.add_argument("--out", default="anomalies.csv")
    a.add_argument("--window-days", type=int, default=15)
    a.add_argument("--z-threshold", type=float, default=-2.0)
    a.add_argument("--min-years", type=int, default=1)
    a.set_defaults(func=cmd_anomalies)

    s = sub.add_parser("season", help="growing-season summaries")
    s.add_argument("--timeseries", required=True)
    s.add_argument("--indices", default="ndvi")
    s.add_argument("--out", default="season.json")
    s.set_defaults(func=cmd_season)

    v = sub.add_parser("vigor-map", help="per-pixel vigor classification map")
    v.add_argument("--index-raster", default="", help="index GeoTIFF (real mode)")
    v.add_argument("--out", default="vigor_classes.tif")
    v.add_argument("--synthetic", action="store_true")
    v.add_argument("--seed", type=int, default=11)
    v.add_argument("--shape", type=int, nargs=2, default=[60, 60], metavar=("ROWS", "COLS"))
    v.set_defaults(func=cmd_vigor_map)

    d = sub.add_parser("demo", help="full synthetic pipeline end-to-end")
    d.add_argument("--out-dir", default="./veg-demo")
    d.add_argument("--seed", type=int, default=7)
    d.set_defaults(func=cmd_demo)

    i = sub.add_parser("indices", help="list vigor indices and required bands")
    i.set_defaults(func=cmd_indices)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
