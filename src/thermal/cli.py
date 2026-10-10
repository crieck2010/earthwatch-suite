"""Command-line interface for survey-thermal.

Subcommands: lst | timeseries | uhi | heatwave | anomalies | demo
Every pipeline step also runs in --synthetic mode: no network, no data
files, deterministic seeded output — the fastest way to learn the tool.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__


def cmd_lst(args: argparse.Namespace) -> int:
    from . import qml as qml_mod
    from . import synthetic as syn
    from .temperature import brightness_temp_to_lst_c, st_dn_to_celsius

    if args.synthetic:
        import numpy as np

        arr, transform = syn.synthetic_lst_raster(seed=args.seed)
        source = "synthetic"
        # The synthetic raster is already land surface temperature in °C —
        # no DN conversion applies.
        lst_c = np.asarray(arr, dtype=float)
    else:
        try:
            import rasterio
        except ImportError:
            raise SystemExit(
                "reading scene rasters needs rasterio "
                "(pip install 'survey-thermal[raster]')"
            )
        with rasterio.open(args.scene) as src:
            arr = src.read(1).astype(float)
            transform = src.transform
        source = args.scene
        if args.source == "st":
            lst_c = st_dn_to_celsius(arr)
        else:
            lst_c = brightness_temp_to_lst_c(
                arr, sensor=args.sensor, emissivity=args.emissivity
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    import numpy as np

    finite = np.asarray(lst_c, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        print(f"LST stats (°C): mean {finite.mean():.2f}, "
              f"min {finite.min():.2f}, max {finite.max():.2f}")
    try:
        import rasterio
        from rasterio.transform import Affine

        a, b, c, d, e, f = (
            transform if isinstance(transform, tuple)
            else (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
        )
        with rasterio.open(
            args.out, "w", driver="GTiff",
            height=lst_c.shape[0], width=lst_c.shape[1], count=1,
            dtype="float32", transform=Affine(a, b, c, d, e, f),
            nodata=float("nan"), compress="deflate",
        ) as dst:
            dst.write(np.asarray(lst_c, dtype="float32"), 1)
        qml_path = os.path.splitext(args.out)[0] + ".qml"
        qml_mod.write_lst_qml(qml_path)
        print(f"wrote LST (°C) -> {args.out}  (+ {qml_path})")
    except ImportError:
        np.save(args.out + ".npy", np.asarray(lst_c))
        print(f"rasterio not installed; wrote numpy array -> {args.out}.npy")
    print(f"source: {source}  (path: {args.source}, sensor: {args.sensor})")
    return 0


def cmd_timeseries(args: argparse.Namespace) -> int:
    from . import synthetic as syn
    from . import timeseries as ts
    from .zonal import zonal_lst_stats
    from .zones import load_zones_geojson

    if args.synthetic:
        zones = syn.synthetic_zones()
        records = syn.synthetic_timeseries(
            zones, start=args.start, end=args.end, seed=args.seed
        )
    else:
        from .acquire import acquire_lst_passes

        zones = load_zones_geojson(args.zones)
        records = []
        for zone in zones:
            for info in acquire_lst_passes(
                zone, args.start, args.end, args.work_dir,
                collections=args.collections.split(",") if args.collections else None,
                max_cloud_cover=args.max_cloud_cover,
            ):
                import rasterio

                with rasterio.open(info.raster_path) as src:
                    arr = src.read(1).astype(float)
                    t = src.transform
                    transform = (t.a, t.b, t.c, t.d, t.e, t.f)
                stats = zonal_lst_stats(arr, zone, transform)
                records.append(
                    ts.make_record(zone.id, zone.name, info.pass_id,
                                   info.date, stats, info.cloud_cover)
                )
    path, n = ts.append_csv(records, args.out)
    print(f"wrote {n} new records -> {path}")
    return 0


def cmd_uhi(args: argparse.Namespace) -> int:
    from . import timeseries as ts
    from . import uhi

    records = ts.read_csv(args.timeseries)
    if args.reference:
        ref_id = args.reference
    else:
        ref_id = uhi.auto_reference(records)
        print(f"auto-selected reference zone: {ref_id}")
    uhi_records = uhi.compute_uhi(records, reference_id=ref_id)
    path = uhi.write_uhi_csv(uhi_records, args.out)
    if uhi_records:
        peak = max(uhi_records, key=lambda r: r.uhi_c)
        print(f"wrote {len(uhi_records)} UHI records -> {path}")
        print(f"peak UHI: {peak.zone_id} {peak.date}: +{peak.uhi_c:.2f} °C vs {ref_id}")
    else:
        print(f"wrote 0 UHI records -> {path}")
    return 0


def cmd_heatwave(args: argparse.Namespace) -> int:
    from . import heatwave as hw
    from . import timeseries as ts

    records = ts.read_csv(args.timeseries)
    events = hw.detect_heatwaves(
        records, threshold_c=args.threshold, consecutive=args.consecutive
    )
    path = hw.write_heatwaves_csv(events, args.out)
    print(f"wrote {len(events)} heatwave events -> {path}")
    for e in events[:20]:
        print(f"  {e.zone_id}: {e.start_date} → {e.end_date} "
              f"({e.duration_passes} passes, peak {e.peak_max_c:.1f} °C)")
    return 0


def cmd_anomalies(args: argparse.Namespace) -> int:
    from . import anomalies as an
    from . import timeseries as ts

    records = ts.read_csv(args.timeseries)
    out = an.detect_anomalies(
        records,
        field=args.field,
        window_days=args.window_days,
        z_threshold=args.z_threshold,
        min_years=args.min_years,
    )
    path = an.write_anomalies_csv(out, args.out)
    hot = [r for r in out if r.hot]
    print(f"wrote {len(out)} anomaly records -> {path}")
    print(f"hot zone-passes: {len(hot)}")
    for r in hot[:20]:
        print(f"  {r.zone_id} {r.date}: z={r.z_score:.2f}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from . import synthetic as syn
    from . import timeseries as ts
    from . import uhi

    report = syn.demo_report(args.out_dir, seed=args.seed)
    records = ts.read_csv(report["timeseries"])
    downtown = [r for r in records if r.zone_id == "downtown"]
    uhi_recs = uhi.compute_uhi(records, reference_id="meadow")
    mean_uhi = sum(r.uhi_c for r in uhi_recs) / len(uhi_recs)
    print("=== survey-thermal demo (synthetic, 2022–2024) ===")
    print(f"timeseries rows : {report['n_timeseries']} -> {report['timeseries']}")
    print(f"UHI records     : {report['n_uhi']} (mean downtown UHI: +{mean_uhi:.2f} °C)")
    print(f"hot anomalies   : {report['n_hot']} -> {report['anomalies']}")
    print(f"heatwave events : {report['n_heatwaves']} -> {report['heatwaves']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="survey-thermal",
        description="Land-surface-temperature analytics for earthwatch-suite.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("lst", help="convert a thermal scene to an LST map (°C)")
    s.add_argument("--scene", help="input raster (ST_B10 DN or TIRS B10 DN)")
    s.add_argument("--synthetic", action="store_true", help="use seeded synthetic raster")
    s.add_argument("--source", choices=["st", "toa"], default="st",
                   help="'st': Landsat C2L2 ST_B10 surface temp; 'toa': brightness-temp fallback")
    s.add_argument("--sensor", choices=["landsat8", "landsat9"], default="landsat8")
    s.add_argument("--emissivity", type=float, default=0.98)
    s.add_argument("--out", required=True, help="output LST GeoTIFF (°C)")
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_lst)

    s = sub.add_parser("timeseries", help="per-zone per-pass LST time series")
    s.add_argument("--zones", help="zones GeoJSON (not needed with --synthetic)")
    s.add_argument("--synthetic", action="store_true")
    s.add_argument("--start", default="2022-01-15")
    s.add_argument("--end", default="2024-12-15")
    s.add_argument("--collections", default="")
    s.add_argument("--max-cloud-cover", type=float, default=40.0)
    s.add_argument("--work-dir", default="./thermal-work")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_timeseries)

    s = sub.add_parser("uhi", help="urban heat-island intensity vs. a reference zone")
    s.add_argument("--timeseries", required=True)
    s.add_argument("--reference", default="", help="reference zone id (default: auto-select coolest)")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_uhi)

    s = sub.add_parser("heatwave", help="flag heatwave events (N consecutive hot passes)")
    s.add_argument("--timeseries", required=True)
    s.add_argument("--threshold", type=float, default=38.0, help="max-LST threshold °C")
    s.add_argument("--consecutive", type=int, default=3)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_heatwave)

    s = sub.add_parser("anomalies", help="day-of-year heat-anomaly z-scores")
    s.add_argument("--timeseries", required=True)
    s.add_argument("--field", choices=["mean", "maximum"], default="mean")
    s.add_argument("--window-days", type=int, default=15)
    s.add_argument("--z-threshold", type=float, default=2.0)
    s.add_argument("--min-years", type=int, default=1)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_anomalies)

    s = sub.add_parser("demo", help="run the full synthetic demo pipeline")
    s.add_argument("--out-dir", default="./thermal-demo")
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_demo)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
