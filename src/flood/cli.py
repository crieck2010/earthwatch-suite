"""Command-line interface for survey-flood.

Subcommands: watermask | timeseries | delta | polygons | demo | indices
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
    from .indices_water import WATER_INDICES

    indices = [t.strip().lower() for t in text.split(",") if t.strip()]
    unknown = [i for i in indices if i not in WATER_INDICES]
    if unknown:
        raise SystemExit(
            f"unknown water indices: {', '.join(unknown)} "
            f"(known: {', '.join(WATER_INDICES)})"
        )
    return indices


def _parse_months(text: str):
    months = [int(t.strip()) for t in text.split(",") if t.strip()]
    bad = [m for m in months if not 1 <= m <= 12]
    if bad:
        raise SystemExit(f"invalid months: {bad} (use 1-12)")
    return months


def cmd_watermask(args: argparse.Namespace) -> int:
    import numpy as np

    from . import qml, synthetic as syn
    from .watermask import build_water_mask

    if args.synthetic:
        zones = syn.synthetic_zones()
        zone = next((z for z in zones if z.id == args.zone), zones[0])
        level = syn.flood_level(zone.id, args.date)
        index_array = syn.synthetic_index_raster(
            tuple(args.shape), zone.id, level, seed=args.seed, index=args.index
        )
        clouds = (
            syn.synthetic_cloud_mask(tuple(args.shape), args.cloud_fraction, args.seed)
            if args.cloud_fraction > 0
            else None
        )
    else:
        if not args.index_raster:
            raise SystemExit("--index-raster is required without --synthetic")
        try:
            import rasterio
        except ImportError:
            raise SystemExit("reading rasters needs 'rasterio' (pip install 'survey-flood[raster]')")
        with rasterio.open(args.index_raster) as src:
            index_array = src.read(1).astype(np.float32)
        clouds = None
        if args.cloud_raster:
            with rasterio.open(args.cloud_raster) as src:
                clouds = src.read(1).astype(bool)
    water, valid = build_water_mask(
        index_array, threshold=args.threshold,
        cloud_mask=clouds, min_pixels=args.min_pixels,
    )
    base = os.path.splitext(args.out)[0]
    np.save(base + ".npy", water.astype(np.uint8))
    print(f"wrote water mask -> {base}.npy")
    qml_path = qml.write_water_qml(base + ".qml")
    print(f"wrote QGIS style -> {qml_path}")
    n_water = int(water.sum())
    n_valid = int(valid.sum())
    print(f"water pixels: {n_water} / {n_valid} valid ({n_water / n_valid:.1%})" if n_valid else "no valid pixels")
    return 0


def cmd_timeseries(args: argparse.Namespace) -> int:
    from . import synthetic as syn
    from . import timeseries as ts
    from .zones import load_zones_geojson

    indices = _parse_indices(args.indices)
    if args.synthetic:
        zones = syn.synthetic_zones()
        records = syn.synthetic_timeseries(
            zones, start=args.start, end=args.end, indices=indices, seed=args.seed,
        )
    else:
        from .acquire import acquire_timeseries

        zones = load_zones_geojson(args.zones)
        records = acquire_timeseries(
            zones, args.start, args.end, indices, args.work_dir,
            threshold=args.threshold, min_pixels=args.min_pixels,
            collections=args.collections.split(",") if args.collections else None,
            max_cloud_cover=args.max_cloud_cover,
        )
    path, n = ts.append_csv(records, args.out)
    print(f"wrote {n} new records -> {path}")
    return 0


def cmd_delta(args: argparse.Namespace) -> int:
    from . import delta as dl
    from . import timeseries as ts

    records = ts.read_csv(args.timeseries)
    months = _parse_months(args.baseline_months)
    out = []
    for index in _parse_indices(args.indices):
        out.extend(
            dl.compute_all_deltas(
                records, index, baseline_months=months,
                baseline_years=args.baseline_years,
                flood_threshold_ha=args.flood_threshold_ha,
                max_cloud_cover=args.max_cloud_cover,
            )
        )
    csv_path = dl.write_deltas_csv(out, args.out)
    dl.write_deltas_json(out, os.path.splitext(args.out)[0] + ".json")
    flooded = [d for d in out if d.flooded]
    print(f"wrote {len(out)} delta records -> {csv_path} (+ .json)")
    print(f"flooded zone-passes: {len(flooded)}")
    for d in flooded[:20]:
        print(f"  {d.zone_id} {d.date}: +{d.delta_ha:.1f} ha (z={d.z_score})")
    return 0


def cmd_polygons(args: argparse.Namespace) -> int:
    import numpy as np

    from . import polygons as pg
    from . import synthetic as syn
    from .watermask import build_water_mask

    if args.synthetic:
        zones = syn.synthetic_zones()
        zone = next((z for z in zones if z.id == args.zone), zones[0])
        level = syn.flood_level(zone.id, args.date)
        index_array = syn.synthetic_index_raster(
            tuple(args.shape), zone.id, level, seed=args.seed, index=args.index
        )
        water, _valid = build_water_mask(
            index_array, threshold=args.threshold, min_pixels=args.min_pixels
        )
        transform = pg.PIXEL_TRANSFORM
    else:
        if not args.mask_npy:
            raise SystemExit("--mask-npy is required without --synthetic")
        water = np.load(args.mask_npy).astype(bool)
        transform = pg.north_up_transform(args.origin_x, args.origin_y, args.pixel_size_m)
    features = pg.vectorize(water, transform=transform, min_pixels=args.min_pixels)
    extra = {"zone_id": args.zone, "date": args.date, "index": args.index}
    path = pg.write_polygons_geojson(features, args.out, extra_properties=extra)
    total_ha = sum(f["properties"]["area_ha"] for f in features)
    print(f"wrote {len(features)} flood polygons ({total_ha:.2f} ha) -> {path}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Full synthetic pipeline: zones -> watermask -> timeseries -> delta -> polygons."""
    import numpy as np

    from . import delta as dl
    from . import polygons as pg
    from . import qml, synthetic as syn
    from . import timeseries as ts
    from .watermask import build_water_mask
    from .zones import zones_to_feature_collection

    outdir = args.out_dir
    os.makedirs(outdir, exist_ok=True)
    zones = syn.synthetic_zones()
    with open(os.path.join(outdir, "zones.geojson"), "w", encoding="utf-8") as fh:
        json.dump(zones_to_feature_collection(zones), fh, indent=2)

    # [1/5] water mask of the planted flood
    date = "2024-04-15"
    zone = zones[0]
    level = syn.flood_level(zone.id, date)
    raster = syn.synthetic_index_raster((40, 60), zone.id, level, seed=args.seed)
    water, valid = build_water_mask(raster)
    np.save(os.path.join(outdir, "watermask.npy"), water.astype(np.uint8))
    qml.write_water_qml(os.path.join(outdir, "watermask.qml"))
    print(f"[1/5] watermask: {int(water.sum())} water px on {date} (flood level {level})")

    # [2/5] time series
    indices = ["ndwi"]
    records = syn.synthetic_timeseries(zones, indices=indices, seed=args.seed)
    ts_path, n = ts.append_csv(records, os.path.join(outdir, "timeseries.csv"))
    print(f"[2/5] timeseries: {n} records -> {ts_path}")

    # [3/5] deltas — the planted flood must be detected
    deltas = dl.compute_all_deltas(records, "ndwi", flood_threshold_ha=1.0)
    dl_path = dl.write_deltas_csv(deltas, os.path.join(outdir, "delta.csv"))
    flooded = [d for d in deltas if d.flooded]
    print(f"[3/5] delta: {len(deltas)} records, {len(flooded)} flooded -> {dl_path}")
    planted = [d for d in flooded if d.zone_id == "river-reach" and d.date == date]
    if not planted:
        raise SystemExit("demo FAILED: planted flood not detected")
    print(f"       planted flood detected: +{planted[0].delta_ha:.1f} ha on {date}")

    # [4/5] flood-extent polygons (10 m grid, honest hectares)
    features = pg.vectorize(
        water, transform=pg.north_up_transform(0.0, 0.0, 10.0)
    )
    pg_path = pg.write_polygons_geojson(
        features, os.path.join(outdir, "flood_polygons.geojson"),
        extra_properties={"zone_id": zone.id, "date": date, "index": "ndwi"},
    )
    total_ha = sum(f["properties"]["area_ha"] for f in features)
    print(f"[4/5] polygons: {len(features)} features, {total_ha:.2f} ha -> {pg_path}")

    print(f"[5/5] demo complete in {outdir}")
    return 0


def cmd_indices(_args: argparse.Namespace) -> int:
    from .indices_water import WATER_INDICES, required_bands

    for name in WATER_INDICES:
        try:
            bands = ", ".join(required_bands(name))
        except ImportError:
            bands = "(needs survey-imagery: pip install 'survey-flood[acquire]')"
        print(f"{name}: bands {bands}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="survey-flood",
        description="Flood/water-extent mapping for satellite remote sensing",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    w = sub.add_parser("watermask", help="threshold an index raster to a water mask")
    w.add_argument("--index-raster", default="", help="water-index GeoTIFF (real mode)")
    w.add_argument("--cloud-raster", default="", help="boolean cloud mask GeoTIFF")
    w.add_argument("--out", default="watermask.npy")
    w.add_argument("--index", default="ndwi")
    w.add_argument("--threshold", type=float, default=0.0)
    w.add_argument("--min-pixels", type=int, default=4)
    w.add_argument("--synthetic", action="store_true")
    w.add_argument("--zone", default="river-reach")
    w.add_argument("--date", default="2024-04-15")
    w.add_argument("--shape", type=int, nargs=2, default=[40, 60], metavar=("ROWS", "COLS"))
    w.add_argument("--cloud-fraction", type=float, default=0.0)
    w.add_argument("--seed", type=int, default=7)
    w.set_defaults(func=cmd_watermask)

    t = sub.add_parser("timeseries", help="build per-zone water-area time series")
    t.add_argument("--zones", default="", help="zones GeoJSON (real mode)")
    t.add_argument("--start", default="2022-01-15")
    t.add_argument("--end", default="2024-12-15")
    t.add_argument("--indices", default="ndwi")
    t.add_argument("--out", default="timeseries.csv")
    t.add_argument("--threshold", type=float, default=0.0)
    t.add_argument("--min-pixels", type=int, default=4)
    t.add_argument("--synthetic", action="store_true", help="seeded demo data, no network")
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--work-dir", default="./.flood-work")
    t.add_argument("--collections", default="")
    t.add_argument("--max-cloud-cover", type=float, default=40.0)
    t.set_defaults(func=cmd_timeseries)

    d = sub.add_parser("delta", help="flood deltas vs. dry-season baseline")
    d.add_argument("--timeseries", required=True)
    d.add_argument("--indices", default="ndwi")
    d.add_argument("--out", default="delta.csv")
    d.add_argument("--baseline-months", default="6,7,8")
    d.add_argument("--baseline-years", type=int, default=1)
    d.add_argument("--flood-threshold-ha", type=float, default=1.0)
    d.add_argument("--max-cloud-cover", type=float, default=40.0)
    d.set_defaults(func=cmd_delta)

    g = sub.add_parser("polygons", help="vectorise a water mask to GeoJSON polygons")
    g.add_argument("--mask-npy", default="", help="water mask .npy (real mode)")
    g.add_argument("--out", default="flood_polygons.geojson")
    g.add_argument("--index", default="ndwi")
    g.add_argument("--min-pixels", type=int, default=4)
    g.add_argument("--synthetic", action="store_true")
    g.add_argument("--zone", default="river-reach")
    g.add_argument("--date", default="2024-04-15")
    g.add_argument("--shape", type=int, nargs=2, default=[40, 60], metavar=("ROWS", "COLS"))
    g.add_argument("--threshold", type=float, default=0.0)
    g.add_argument("--origin-x", type=float, default=0.0)
    g.add_argument("--origin-y", type=float, default=0.0)
    g.add_argument("--pixel-size-m", type=float, default=10.0)
    g.add_argument("--seed", type=int, default=7)
    g.set_defaults(func=cmd_polygons)

    m = sub.add_parser("demo", help="full synthetic pipeline end-to-end")
    m.add_argument("--out-dir", default="./flood-demo")
    m.add_argument("--seed", type=int, default=7)
    m.set_defaults(func=cmd_demo)

    i = sub.add_parser("indices", help="list water indices and required bands")
    i.set_defaults(func=cmd_indices)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
