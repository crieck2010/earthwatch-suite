"""Thin CLI for survey-flow. All heavy lifting lives in the engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
import types

import numpy as np

from flow import __version__
from flow.fields import TimeVaryingField
from flow.frames import FrameManifest
from flow.pipeline import render_animation
from flow.render import SIZES
from flow.styles import list_styles


def _fail(msg: str) -> int:
    print(f"survey-flow: error: {msg}", file=sys.stderr)
    return 2


def _dict_to_field_namespace(data: dict) -> types.SimpleNamespace:
    """Adapt a CurrentField JSON dict (survey-currents) to a namespace."""
    ns = types.SimpleNamespace()
    ns.u = np.asarray(data["u"], dtype=float)
    ns.v = np.asarray(data["v"], dtype=float)
    ns.temperature = (None if data.get("temperature") is None
                      else np.asarray(data["temperature"], dtype=float))
    ns.times = list(data["times"])
    ns.lats = np.asarray(data["lats"], dtype=float)
    ns.lons = np.asarray(data["lons"], dtype=float)
    ns.crs = data.get("crs", "EPSG:4326")
    ns.source = data.get("source", "")
    return ns


def _demo_namespace(nt: int = 6, ny: int = 36, nx: int = 54) -> types.SimpleNamespace:
    """Self-contained synthetic field (rotating gyre + warm core), no deps."""
    import datetime as dt
    rng = np.random.default_rng(7)
    lats = np.linspace(41.5, 46.5, ny)
    lons = np.linspace(-92.5, -84.5, nx)
    yy, xx = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx),
                         indexing="ij")
    # Lake Michigan-ish shape: mask the corners as land.
    land = (xx ** 2 + yy ** 2) > 0.86
    u = np.empty((nt, ny, nx))
    v = np.empty((nt, ny, nx))
    temp = np.empty((nt, ny, nx))
    t0 = dt.datetime(2026, 9, 16, tzinfo=dt.timezone.utc)
    times = []
    for k in range(nt):
        uk = -yy * 0.35 + rng.normal(0, 0.02, (ny, nx))
        vk = xx * 0.35 + rng.normal(0, 0.02, (ny, nx))
        tk = 18.0 - 4.0 * (xx ** 2 + yy ** 2) + 0.2 * k
        uk = np.where(land, np.nan, uk)
        vk = np.where(land, np.nan, vk)
        tk = np.where(land, np.nan, tk)
        u[k], v[k], temp[k] = uk, vk, tk
        times.append((t0 + dt.timedelta(hours=k)).isoformat())
    ns = types.SimpleNamespace()
    ns.u, ns.v, ns.temperature = u, v, temp
    ns.times, ns.lats, ns.lons = times, lats, lons
    ns.crs, ns.source = "EPSG:4326", "survey-flow/demo"
    return ns


def _load_tvf(args) -> TimeVaryingField:
    scalar = args.scalar
    if args.cog:
        if not args.times or len(args.times) != len(args.cog):
            return _raise("--cog needs one --times value per file")
        return TimeVaryingField.from_cogs(args.cog, args.times, scalar=scalar)
    if not args.field:
        return _raise("provide --field <currentfield.json> or --cog files")
    with open(args.field, encoding="utf-8") as fh:
        data = json.load(fh)
    return TimeVaryingField.from_current_field(
        _dict_to_field_namespace(data), scalar=scalar)


def _raise(msg: str):
    raise SystemExit(f"survey-flow: error: {msg}")


def cmd_render(args) -> int:
    try:
        tvf = _load_tvf(args)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2
    except (ValueError, KeyError, OSError) as exc:
        return _fail(str(exc))
    size = tuple(int(x) for x in args.size.split("x")) if "x" in args.size else args.size
    try:
        manifest = render_animation(
            tvf, args.out, style=args.style, n_frames=args.frames,
            n_particles=args.particles, size=size, dpi=args.dpi, fps=args.fps,
            dt_seconds=args.dt_seconds, seed=args.seed,
            vmin=args.vmin, vmax=args.vmax, title=args.title,
            progress=lambda i, n: print(f"  frame {i}/{n}", end="\r",
                                        file=sys.stderr))
    except ImportError as exc:
        return _fail(str(exc))
    print(f"\nwrote {len(manifest)} frames + manifest to {args.out}",
          file=sys.stderr)
    print(os.path.join(args.out, "manifest.json"))
    return 0


def cmd_demo(args) -> int:
    ns = _demo_namespace()
    tvf = TimeVaryingField.from_current_field(ns, scalar="temperature")
    try:
        manifest = render_animation(
            tvf, args.out, style=args.style, n_frames=args.frames,
            n_particles=args.particles, size="square", dpi=50,
            seed=args.seed,
            progress=lambda i, n: print(f"  frame {i}/{n}", end="\r",
                                        file=sys.stderr))
    except ImportError as exc:
        return _fail(str(exc))
    print(f"\ndemo: {len(manifest)} frames + manifest in {args.out}",
          file=sys.stderr)
    return 0


def cmd_styles(_args) -> int:
    for s in list_styles():
        print(f"{s.name:12} {s.description}")
    return 0


def cmd_sizes(_args) -> int:
    for name, (w, h) in SIZES.items():
        print(f"{name:8} {w}x{h}")
    return 0


def cmd_manifest(args) -> int:
    """Rebuild a manifest from a directory of frame_*.png files."""
    import glob
    pngs = sorted(glob.glob(os.path.join(args.dir, "frame_*.png")))
    if not pngs:
        return _fail(f"no frame_*.png files in {args.dir}")
    from flow.frames import FrameEntry
    manifest = FrameManifest(fps=args.fps, field_source=args.source or "")
    for i, png in enumerate(pngs):
        manifest.add(FrameEntry(frame_index=i, timestamp_iso="", png_path=png,
                                style=args.style))
    out = os.path.join(args.dir, "manifest.json")
    manifest.write(out)
    print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="survey-flow",
        description="Render flow-field map frames from surface-current fields.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("render", help="render an animation from a field")
    r.add_argument("--field", help="CurrentField JSON (survey-currents to_json)")
    r.add_argument("--cog", action="append",
                   help="4-band COG input (repeatable; needs --times)")
    r.add_argument("--times", action="append", help="ISO timestamp per --cog")
    r.add_argument("--scalar", default="temperature",
                   choices=["temperature", "speed"])
    r.add_argument("--style", default="reel-dark")
    r.add_argument("--frames", type=int, default=48)
    r.add_argument("--particles", type=int, default=1500)
    r.add_argument("--size", default="reel",
                   help="reel|wide|square or WxH, e.g. 1080x1920")
    r.add_argument("--dpi", type=int, default=100)
    r.add_argument("--fps", type=float, default=30.0)
    r.add_argument("--dt-seconds", type=float, default=None)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--vmin", type=float, default=None)
    r.add_argument("--vmax", type=float, default=None)
    r.add_argument("--title", default=None)
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_render)

    d = sub.add_parser("demo", help="offline synthetic demo (no downloads)")
    d.add_argument("--out", required=True)
    d.add_argument("--style", default="reel-dark")
    d.add_argument("--frames", type=int, default=12)
    d.add_argument("--particles", type=int, default=400)
    d.add_argument("--seed", type=int, default=7)
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("styles", help="list style presets")
    s.set_defaults(func=cmd_styles)
    z = sub.add_parser("sizes", help="list named output sizes")
    z.set_defaults(func=cmd_sizes)

    m = sub.add_parser("manifest", help="rebuild manifest.json from frames/")
    m.add_argument("--dir", required=True)
    m.add_argument("--fps", type=float, default=30.0)
    m.add_argument("--style", default="reel-dark")
    m.add_argument("--source", default="")
    m.set_defaults(func=cmd_manifest)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
