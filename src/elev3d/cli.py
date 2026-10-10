"""Command-line interface for survey-3d.

Subcommands: diff | volumes | polygons | coreg | demo
``diff`` and ``demo`` run in --synthetic mode: no network, no data files,
deterministic seeded output — the fastest way to learn the tool.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__


def _parse_date(s: str) -> str:
    from datetime import date

    return date.fromisoformat(s).isoformat()


def cmd_diff(args: argparse.Namespace) -> int:
    from . import synthetic as syn
    from .diff import check_grids, difference

    if args.synthetic:
        dem1, dem2, transform = syn.synthetic_dems(seed=args.seed)
        crs_wkt = None
        label1, label2 = "synthetic-before", "synthetic-after"
    else:
        from . import acquire

        if not args.dem1 or not args.dem2:
            print("error: --dem1 and --dem2 are required (or use --synthetic)",
                  file=sys.stderr)
            return 2
        acquire.check_crs_match(args.dem1, args.dem2)
        dem1, transform1, crs1, _ = acquire.read_dem(args.dem1)
        dem2, transform2, _crs2, _ = acquire.read_dem(args.dem2)
        check_grids(dem1.shape, dem2.shape, transform1, transform2)
        transform, crs_wkt = transform1, crs1
        label1, label2 = args.dem1, args.dem2

    result = difference(dem1, dem2, noise_floor=args.noise_floor)
    summary = result.summary()
    summary["dem1"], summary["dem2"] = label1, label2
    print(json.dumps(summary, indent=2))

    if args.out:
        from . import acquire, qml

        acquire.write_geotiff(args.out, result.diff_thresholded, transform, crs_wkt)
        qpath = os.path.splitext(args.out)[0] + ".qml"
        qml.write_qml(qpath)
        print(f"wrote {args.out} and {qpath}")
    return 0


def cmd_volumes(args: argparse.Namespace) -> int:
    from . import synthetic as syn
    from .diff import difference
    from .timeseries import append_records, make_record
    from .volumes import zone_volumes
    from .zonal import zone_mask
    from .zones import load_zones_geojson

    if args.synthetic:
        dem1, dem2, transform = syn.synthetic_dems(seed=args.seed)
        diff = difference(dem1, dem2, noise_floor=args.noise_floor).diff_thresholded
        # load zones from the synthetic GeoJSON via a temp file
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".geojson", delete=False
        ) as fh:
            json.dump(syn.synthetic_zones_geojson(), fh)
            zones_path = fh.name
        zones = load_zones_geojson(zones_path)
        os.unlink(zones_path)
        pass_id, date = "synthetic-before_vs_synthetic-after", "2025-06-15"
    else:
        from . import acquire

        if not args.diff or not args.zones:
            print("error: --diff and --zones are required (or use --synthetic)",
                  file=sys.stderr)
            return 2
        diff, transform, _crs, _u = acquire.read_dem(args.diff)
        zones = load_zones_geojson(args.zones)
        pass_id = args.pass_id or "dem1_vs_dem2"
        date = _parse_date(args.date) if args.date else "2025-06-15"

    import numpy as np

    records = []
    for zone in zones:
        vols = zone_volumes(
            diff,
            zone,
            transform,
            noise_floor=args.noise_floor,
            rmse1=args.rmse1,
            rmse2=args.rmse2,
        )
        z = diff[zone_mask(zone, diff.shape, transform)]
        z = z[np.isfinite(z)]
        zonal = {
            "mean": float(np.mean(z)) if z.size else 0.0,
            "median": float(np.median(z)) if z.size else 0.0,
            "std": float(np.std(z)) if z.size else 0.0,
            "minimum": float(np.min(z)) if z.size else 0.0,
            "maximum": float(np.max(z)) if z.size else 0.0,
            "p10": float(np.percentile(z, 10)) if z.size else 0.0,
            "p90": float(np.percentile(z, 90)) if z.size else 0.0,
            "valid_pixels": float(z.size),
            "total_pixels": float(z.size),
            "cloud_cover": 0.0,
        }
        records.append(
            make_record(zone.id, zone.name, pass_id, date, zonal, vols.to_dict())
        )

    out = args.out or "volumes.csv"
    n = append_records(out, records)
    print(f"appended {n} records to {out} (dedupe on zone_id+pass_id+index)")
    for r in records:
        d = r.to_dict()
        print(
            f"  {d['zone_id']}: cut={d['cut_m3']:.0f} m³ "
            f"fill={d['fill_m3']:.0f} m³ net={d['net_m3']:+.0f} m³ "
            f"σ={d['volume_sigma_m3']:.0f} m³ [{d['confidence']}]"
        )
    return 0


def cmd_polygons(args: argparse.Namespace) -> int:
    from . import synthetic as syn
    from .diff import difference
    from .polygons import change_polygons, polygons_to_geojson

    if args.synthetic:
        dem1, dem2, transform = syn.synthetic_dems(seed=args.seed)
        diff = difference(dem1, dem2, noise_floor=0.0).diff
    else:
        from . import acquire

        if not args.diff:
            print("error: --diff is required (or use --synthetic)", file=sys.stderr)
            return 2
        diff, transform, _crs, _u = acquire.read_dem(args.diff)

    polys = change_polygons(
        diff, transform, threshold=args.threshold, min_pixels=args.min_pixels
    )
    out = args.out or "change_polygons.geojson"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(polygons_to_geojson(polys), fh, indent=2)
    print(f"wrote {len(polys)} change polygons to {out}")
    for i, p in enumerate(polys):
        print(
            f"  polygon {i}: {p.dominant} cut={p.cut_m3:.0f} m³ "
            f"fill={p.fill_m3:.0f} m³ area={p.area_m2:.0f} m²"
        )
    return 0


def cmd_coreg(args: argparse.Namespace) -> int:
    from . import synthetic as syn
    from .coreg import coregistration_check
    from .diff import difference
    from .zones import load_zones_geojson

    if args.synthetic:
        dem1, dem2, transform = syn.synthetic_dems(seed=args.seed)
        diff = difference(dem1, dem2, noise_floor=0.0).diff
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".geojson", delete=False
        ) as fh:
            json.dump(syn.synthetic_zones_geojson(), fh)
            zones_path = fh.name
        zones = load_zones_geojson(zones_path)
        os.unlink(zones_path)
        stable = next(z for z in zones if z.kind == "stable")
    else:
        from . import acquire

        if not args.dem1 or not args.dem2 or not args.stable:
            print(
                "error: --dem1, --dem2 and --stable are required "
                "(or use --synthetic)",
                file=sys.stderr,
            )
            return 2
        acquire.check_crs_match(args.dem1, args.dem2)
        dem1, t1, _, _ = acquire.read_dem(args.dem1)
        dem2, t2, _, _ = acquire.read_dem(args.dem2)
        diff = difference(dem1, dem2, noise_floor=0.0).diff
        transform = t1
        zones = load_zones_geojson(args.stable)
        stable = zones[0]

    report = coregistration_check(
        diff, stable, transform, threshold_m=args.threshold
    )
    print(json.dumps(report.to_dict(), indent=2))
    if not report.passed:
        print(
            "WARNING: stable ground moved — DEMs may not be coregistered. "
            "Do not trust volumes until the pair is aligned.",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    print("=== survey-3d demo: synthetic site, known cut 18,750 m³ / fill 12,300 m³ ===")
    for step, sub in (
        ("coreg check", ["coreg", "--synthetic", "--seed", str(args.seed)]),
        ("differencing", ["diff", "--synthetic", "--seed", str(args.seed),
                          "--noise-floor", "0.08"]),
        ("volumetrics", ["volumes", "--synthetic", "--seed", str(args.seed),
                         "--noise-floor", "0.08", "--rmse1", "0.03",
                         "--rmse2", "0.03", "--out", "demo_volumes.csv"]),
        ("polygons", ["polygons", "--synthetic", "--seed", str(args.seed),
                      "--threshold", "0.08", "--out", "demo_polygons.geojson"]),
    ):
        print(f"\n--- {step} ---")
        rc = main(sub)
        if rc not in (0, 1):
            return rc
    print("\nDemo complete: demo_volumes.csv, demo_polygons.geojson")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="survey-3d",
        description="DSM/DEM differencing and cut/fill volumetrics (v%s)" % __version__,
    )
    p.add_argument("--version", action="version", version="%(prog)s " + __version__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("diff", help="difference two DEMs (dem2 - dem1)")
    d.add_argument("--dem1", help="before DEM (GeoTIFF)")
    d.add_argument("--dem2", help="after DEM (GeoTIFF)")
    d.add_argument("--synthetic", action="store_true")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--noise-floor", type=float, default=0.0,
                   help="minimum |Δh| in metres to count as change")
    d.add_argument("--out", help="write thresholded difference GeoTIFF (+ .qml)")
    d.set_defaults(func=cmd_diff)

    v = sub.add_parser("volumes", help="cut/fill volumetrics per zone")
    v.add_argument("--diff", help="difference raster (GeoTIFF, metres)")
    v.add_argument("--zones", help="zones GeoJSON")
    v.add_argument("--synthetic", action="store_true")
    v.add_argument("--seed", type=int, default=7)
    v.add_argument("--noise-floor", type=float, default=0.05)
    v.add_argument("--rmse1", type=float, default=None, help="vertical RMSE of DEM1 (m)")
    v.add_argument("--rmse2", type=float, default=None, help="vertical RMSE of DEM2 (m)")
    v.add_argument("--pass-id", default=None)
    v.add_argument("--date", default=None, help="date of the later DEM (YYYY-MM-DD)")
    v.add_argument("--out", default=None, help="output CSV (default volumes.csv)")
    v.set_defaults(func=cmd_volumes)

    pg = sub.add_parser("polygons", help="vectorize change areas to GeoJSON")
    pg.add_argument("--diff", help="difference raster (GeoTIFF, metres)")
    pg.add_argument("--synthetic", action="store_true")
    pg.add_argument("--seed", type=int, default=7)
    pg.add_argument("--threshold", type=float, default=0.10,
                    help="|Δh| threshold in metres")
    pg.add_argument("--min-pixels", type=int, default=4)
    pg.add_argument("--out", default=None)
    pg.set_defaults(func=cmd_polygons)

    c = sub.add_parser("coreg", help="coregistration bias check over a stable area")
    c.add_argument("--dem1", help="before DEM (GeoTIFF)")
    c.add_argument("--dem2", help="after DEM (GeoTIFF)")
    c.add_argument("--stable", help="stable-area zones GeoJSON (first zone used)")
    c.add_argument("--synthetic", action="store_true")
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--threshold", type=float, default=0.10,
                   help="|bias| pass threshold in metres")
    c.set_defaults(func=cmd_coreg)

    dm = sub.add_parser("demo", help="run the full synthetic demo pipeline")
    dm.add_argument("--seed", type=int, default=7)
    dm.set_defaults(func=cmd_demo)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
