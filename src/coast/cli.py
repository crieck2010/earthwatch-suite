"""Command-line interface for survey-coast.

Subcommands: shoreline | transects | rates | storm | demo | indices
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
    raise SystemExit(
        f"raster input must be .npy in this build (got {path!r}); "
        "use rasterio via the acquire path for GeoTIFFs"
    )


def _shoreline_geojson_path(out_dir: str, pass_id: str) -> str:
    return os.path.join(out_dir, f"shoreline_{pass_id}.geojson")


def _write_shoreline_geojson(path, pass_id, date, index_name, lines, length_m):
    from .geojson import linestring_feature, write_geojson

    feats = [
        linestring_feature(
            [[float(x), float(y)] for x, y in line],
            {
                "pass_id": pass_id,
                "date": date,
                "index": index_name,
                "length_m": round(length_m, 2),
                "n_vertices": len(line),
            },
        )
        for line in lines
    ]
    return write_geojson(
        path, feats, pass_id=pass_id, date=date, index=index_name,
        crs_hint="map meters; pair with your site CRS in QGIS",
    )


def cmd_shoreline(args: argparse.Namespace) -> int:
    import numpy as np

    from . import shoreline as _sl
    from . import watermask as _wm

    jobs = []  # (pass_id, date, ndwi, cloud)
    if args.synthetic:
        from . import synthetic as syn

        for p in syn.synthetic_passes(seed=args.seed):
            jobs.append((p["pass_id"], p["date"], p["ndwi"], p["cloud"]))
        index_name = "ndwi"
        transform = (syn.DEMO_PIXEL_M, syn.DEMO_ORIGIN_X, syn.DEMO_ORIGIN_Y)
    else:
        if not args.ndwi or not args.date:
            raise SystemExit("--ndwi and --date are required without --synthetic")
        ndwi = _load_npy(args.ndwi)
        cloud = _load_npy(args.cloud) if args.cloud else None
        pass_id = args.pass_id or os.path.splitext(os.path.basename(args.ndwi))[0]
        jobs.append((pass_id, args.date, ndwi, cloud))
        index_name = args.index
        transform = (args.pixel_size, args.origin_x, args.origin_y)

    os.makedirs(args.out_dir, exist_ok=True)
    for pass_id, date, ndwi, cloud in jobs:
        mask = _wm.water_mask(
            np.asarray(ndwi, dtype=float),
            threshold=args.threshold,
            cloud=None if cloud is None else np.asarray(cloud, dtype=bool),
            min_pixels=args.min_pixels,
            index_name=index_name,
        )
        lines = _sl.extract_shoreline(
            mask.water,
            mask.valid,
            pixel_size_m=transform[0],
            origin_x=transform[1],
            origin_y=transform[2],
            min_length_m=args.min_length_m,
        )
        length = _sl.shoreline_length_m(lines)
        path = _shoreline_geojson_path(args.out_dir, pass_id)
        _write_shoreline_geojson(path, pass_id, date, index_name, lines, length)
        print(
            f"shoreline: {pass_id} date={date} lines={len(lines)} "
            f"length={length:.1f} m water_frac={mask.water_fraction:.3f} "
            f"cloud={mask.cloud_cover:.1f}% -> {path}"
        )
    return 0


def _read_shorelines_dir(shorelines_dir: str):
    """Yield (pass_id, date, [polylines in meters]) per shoreline file."""
    import glob

    import numpy as np

    from .geojson import read_geojson

    files = sorted(glob.glob(os.path.join(shorelines_dir, "shoreline_*.geojson")))
    if not files:
        raise SystemExit(f"no shoreline_*.geojson files in {shorelines_dir}")
    for path in files:
        obj = read_geojson(path)
        feats = obj["features"]
        pass_id = obj.get("pass_id") or (feats[0]["properties"].get("pass_id") if feats else None)
        date = obj.get("date") or (feats[0]["properties"].get("date") if feats else None)
        if not pass_id or not date:
            # fall back to the filename stem
            stem = os.path.splitext(os.path.basename(path))[0]
            pass_id = pass_id or stem.replace("shoreline_", "")
            date = date or ""
        lines = [
            np.asarray(f["geometry"]["coordinates"], dtype=float)
            for f in feats
            if (f.get("geometry") or {}).get("type") == "LineString"
        ]
        yield pass_id, date, lines


def cmd_transects(args: argparse.Namespace) -> int:
    import datetime as dt

    from . import shoreline as _sl
    from .positions import TransectPassRecord, append_records
    from .transects import load_transects_geojson, transect_position

    transects = load_transects_geojson(args.transects)
    records = []
    for pass_id, date, lines in _read_shorelines_dir(args.shorelines):
        d = dt.date.fromisoformat(date) if date else None
        for t in transects:
            pos = transect_position(t, lines)
            records.append(
                TransectPassRecord(
                    transect_id=t.id,
                    transect_name=t.name,
                    site_id=args.site_id,
                    site_name=args.site_name,
                    pass_id=pass_id,
                    date=date,
                    doy=d.timetuple().tm_yday if d else 0,
                    year=d.year if d else 0,
                    index=args.index,
                    position_m=pos,
                    shoreline_length_m=_sl.shoreline_length_m(lines),
                    valid=pos == pos,
                    cloud_cover=args.cloud_cover,
                    pixel_size_m=args.pixel_size,
                )
            )
    added, skipped = append_records(args.out, records)
    print(f"transects: wrote {args.out} (added={added}, skipped={skipped})")
    return 0


def cmd_rates(args: argparse.Namespace) -> int:
    from .positions import read_positions
    from .rates import compute_rates, write_rates_csv

    records = read_positions(args.positions)
    if not records:
        raise SystemExit(f"no records in {args.positions}")
    rates = compute_rates(records)
    write_rates_csv(args.out, rates)
    for r in rates:
        flag = "EROSION" if r.rate_m_per_yr < 0 else "ACCRETION"
        print(
            f"rates: {r.transect_id} n={r.n_passes} "
            f"rate={r.rate_m_per_yr:+.2f}±{r.rate_se_m_per_yr:.2f} m/yr "
            f"net={r.net_change_m:+.1f} m [{flag}]"
        )
    print(f"rates: wrote {args.out}")
    return 0


def cmd_storm(args: argparse.Namespace) -> int:
    from .positions import read_positions
    from .storm import storm_deltas, write_storm_csv

    records = read_positions(args.positions)
    if not records:
        raise SystemExit(f"no records in {args.positions}")
    deltas = storm_deltas(records, args.event)
    write_storm_csv(args.out, deltas)
    for d in deltas:
        print(
            f"storm: {d.transect_id} {d.before_date} -> {d.after_date} "
            f"displacement={d.displacement_m:+.1f} m"
        )
    print(f"storm: wrote {args.out}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    import numpy as np

    from . import shoreline as _sl
    from . import synthetic as syn
    from . import watermask as _wm
    from .geojson import write_geojson
    from .positions import TransectPassRecord, append_records, read_positions
    from .rates import compute_rates, write_rates_csv
    from .storm import storm_deltas, write_storm_csv
    from .transects import transect_position

    out = args.out_dir
    shore_dir = os.path.join(out, "shorelines")
    os.makedirs(shore_dir, exist_ok=True)
    transects = syn.synthetic_transects()
    site = syn.synthetic_sites()[0]
    write_geojson(
        os.path.join(out, "transects.geojson"),
        [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [list(map(float, pt)) for pt in t.coords],
                },
                "properties": {"id": t.id, "name": t.name},
            }
            for t in transects
        ],
    )

    import datetime as dt

    records = []
    for p in syn.synthetic_passes(seed=args.seed):
        mask = _wm.water_mask(
            np.asarray(p["ndwi"], dtype=float),
            threshold=0.0,
            cloud=np.asarray(p["cloud"], dtype=bool),
            min_pixels=4,
            index_name="ndwi",
        )
        lines = _sl.extract_shoreline(
            mask.water,
            mask.valid,
            pixel_size_m=syn.DEMO_PIXEL_M,
            origin_x=syn.DEMO_ORIGIN_X,
            origin_y=syn.DEMO_ORIGIN_Y,
            min_length_m=50.0,
        )
        length = _sl.shoreline_length_m(lines)
        _write_shoreline_geojson(
            _shoreline_geojson_path(shore_dir, str(p["pass_id"])),
            str(p["pass_id"]), str(p["date"]), "ndwi", lines, length,
        )
        d = dt.date.fromisoformat(str(p["date"]))
        for t in transects:
            pos = transect_position(t, lines)
            records.append(
                TransectPassRecord(
                    transect_id=t.id, transect_name=t.name,
                    site_id=site.id, site_name=site.name,
                    pass_id=str(p["pass_id"]), date=str(p["date"]),
                    doy=d.timetuple().tm_yday, year=d.year, index="ndwi",
                    position_m=pos, shoreline_length_m=length,
                    valid=pos == pos, cloud_cover=mask.cloud_cover,
                    pixel_size_m=syn.DEMO_PIXEL_M,
                )
            )
    pos_csv = os.path.join(out, "positions.csv")
    if os.path.exists(pos_csv):
        os.remove(pos_csv)
    added, _ = append_records(pos_csv, records)
    rates = compute_rates(read_positions(pos_csv))
    write_rates_csv(os.path.join(out, "rates.csv"), rates)
    deltas = storm_deltas(read_positions(pos_csv), syn.STORM_DATE)
    write_storm_csv(os.path.join(out, "storm.csv"), deltas)

    print(f"demo: {len(records)} position records ({added} new) -> {pos_csv}")
    print("demo: planted vs recovered rates (m/yr):")
    for r in rates:
        planted = syn.EROSION_RATE_A if r.transect_id == "shore-a" else syn.ACCRETION_RATE_B
        print(
            f"  {r.transect_id}: planted={planted:+.1f} "
            f"recovered={r.rate_m_per_yr:+.2f}±{r.rate_se_m_per_yr:.2f} "
            f"net={r.net_change_m:+.1f} m"
        )
    for d in deltas:
        print(
            f"demo: storm {syn.STORM_DATE} {d.transect_id} "
            f"displacement={d.displacement_m:+.1f} m "
            f"(planted bite {syn.STORM_BITE_M:+.0f} m on shore-a)"
        )
    print(f"demo: outputs in {out}")
    return 0


def cmd_indices(_args: argparse.Namespace) -> int:
    from .indices_water import describe_indices

    for name, info in describe_indices().items():
        print(f"{name}: bands={info['bands']} formula={info['formula']} ({info['reference']})")
        print(f"    use: {info['use']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="survey-coast", description="Shoreline / coastal-change engine"
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("shoreline", help="extract per-pass shorelines from water-index rasters")
    s.add_argument("--synthetic", action="store_true", help="use the seeded demo world")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--ndwi", help=".npy water-index raster (non-synthetic)")
    s.add_argument("--cloud", help=".npy bool cloud mask (non-synthetic)")
    s.add_argument("--date", help="pass date ISO (non-synthetic)")
    s.add_argument("--pass-id", help="pass id (non-synthetic)")
    s.add_argument("--index", default="ndwi")
    s.add_argument("--threshold", type=float, default=0.0)
    s.add_argument("--min-pixels", type=int, default=4)
    s.add_argument("--min-length-m", type=float, default=0.0)
    s.add_argument("--pixel-size", type=float, default=10.0)
    s.add_argument("--origin-x", type=float, default=0.0)
    s.add_argument("--origin-y", type=float, default=0.0)
    s.add_argument("--out-dir", default="shorelines")
    s.set_defaults(func=cmd_shoreline)

    t = sub.add_parser("transects", help="compute shoreline positions along transects")
    t.add_argument("--shorelines", required=True, help="dir of shoreline_*.geojson")
    t.add_argument("--transects", required=True, help="transect LineStrings GeoJSON")
    t.add_argument("--out", required=True, help="positions CSV (appended)")
    t.add_argument("--site-id", default="")
    t.add_argument("--site-name", default="")
    t.add_argument("--index", default="ndwi")
    t.add_argument("--cloud-cover", type=float, default=0.0)
    t.add_argument("--pixel-size", type=float, default=10.0)
    t.set_defaults(func=cmd_transects)

    r = sub.add_parser("rates", help="erosion/accretion rates per transect")
    r.add_argument("--positions", required=True)
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_rates)

    e = sub.add_parser("storm", help="shoreline displacement around an event date")
    e.add_argument("--positions", required=True)
    e.add_argument("--event", required=True, help="event date ISO YYYY-MM-DD")
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_storm)

    d = sub.add_parser("demo", help="full synthetic pipeline end to end")
    d.add_argument("--out-dir", default="demo_out")
    d.add_argument("--seed", type=int, default=7)
    d.set_defaults(func=cmd_demo)

    i = sub.add_parser("indices", help="list supported water indices")
    i.set_defaults(func=cmd_indices)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
