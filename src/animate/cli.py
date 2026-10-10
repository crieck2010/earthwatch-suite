"""Thin CLI for survey-animate. All heavy lifting lives in the engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np

from animate import __version__
from animate.encode import EncoderUnavailable, format_for_path, resolve_ffmpeg
from animate.manifest import Manifest, ManifestFrame, frames_from_dir, read_manifest
from animate.pipeline import render_video
from animate.presets import list_presets


def _fail(msg: str) -> int:
    print(f"survey-animate: error: {msg}", file=sys.stderr)
    return 2


def _add_source_args(sp: argparse.ArgumentParser) -> None:
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", help="survey-flow frame manifest JSON")
    g.add_argument("--frames-dir",
                   help="directory of PNGs (alphabetical order)")
    sp.add_argument("--preset", default="reel",
                    help="output preset: reel, wide, square, preview "
                         "(default: reel)")
    sp.add_argument("--out", required=True, help="output video path")
    sp.add_argument("--format", choices=["mp4", "gif", "webm"], default=None,
                    help="container (default: inferred from --out suffix)")
    sp.add_argument("--fps", type=float, default=None,
                    help="frames per second (default: preset default)")
    sp.add_argument("--crf", type=int, default=None,
                    help="H.264/VP9 quality 0-51, lower is better "
                         "(default: preset default)")
    sp.add_argument("--no-timestamps", action="store_true",
                    help="do not burn per-frame timestamps")
    sp.add_argument("--timestamp-position", default="lower-left",
                    choices=["lower-left", "lower-right",
                             "upper-left", "upper-right"])
    sp.add_argument("--title", default="",
                    help="optional title-card text (needs matplotlib)")
    sp.add_argument("--subtitle", default="")
    sp.add_argument("--title-seconds", type=float, default=2.0)


def _cmd_render(a: argparse.Namespace) -> int:
    source = a.manifest or a.frames_dir
    fmt = a.format
    if fmt is None:
        try:
            fmt = format_for_path(a.out)
        except ValueError as exc:
            return _fail(str(exc))
    total = [0]

    def _progress(n: int) -> None:
        total[0] = n
        print(f"\r  encoded {n} frames...", end="", flush=True)

    try:
        result = render_video(
            source, a.out, preset=a.preset, fmt=fmt, fps=a.fps, crf=a.crf,
            burn_timestamps_=not a.no_timestamps,
            timestamp_position=a.timestamp_position,
            title=a.title, subtitle=a.subtitle,
            title_seconds=a.title_seconds, progress=_progress)
    except EncoderUnavailable as exc:
        return _fail(str(exc))
    except (ValueError, FileNotFoundError) as exc:
        return _fail(str(exc))
    print(f"\r  wrote {result.video_path} "
          f"({result.n_frames} frames, {result.width}x{result.height} "
          f"@{result.fps:g}fps, {result.fmt})")
    print(f"  sidecar: {result.sidecar_path}")
    return 0


def _demo_frames(tmp: str, n: int = 24, w: int = 360, h: int = 640):
    """Offline synthetic frames: drifting warm blob on a cool field."""
    from animate import _png
    paths = []
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    for i in range(n):
        cx = w * (0.25 + 0.5 * i / max(1, n - 1))
        cy = h * 0.5 + 40 * np.sin(2 * np.pi * i / n)
        blob = np.exp(-(((xx - cx) / 90) ** 2 + ((yy - cy) / 120) ** 2))
        base = np.linspace(0.1, 0.5, w)[None, :].repeat(h, axis=0)
        temp = np.clip(base + 0.5 * blob, 0, 1)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = (temp * 255).astype(np.uint8)          # warm -> red
        rgb[..., 2] = ((1 - temp) * 255).astype(np.uint8)    # cool -> blue
        rgb[..., 1] = (np.abs(temp - 0.5) * 120).astype(np.uint8)
        p = os.path.join(tmp, f"demo_{i:04d}.png")
        _png.write_png(p, rgb)
        paths.append(p)
    return paths


def _cmd_demo(a: argparse.Namespace) -> int:
    import datetime as _dt
    with tempfile.TemporaryDirectory(prefix="animate_demo_") as tmp:
        paths = _demo_frames(tmp)
        t0 = _dt.datetime(2026, 9, 16, tzinfo=_dt.timezone.utc)
        frames = [ManifestFrame(
            index=i, png_path=p,
            timestamp_iso=(t0 + _dt.timedelta(hours=i)).isoformat())
            for i, p in enumerate(paths)]
        manifest = Manifest(frames=frames, fps=12.0,
                            field_source="synthetic/demo",
                            generator="survey-animate demo")
        try:
            result = render_video(
                manifest, a.out, preset=a.preset,
                fmt=a.format or format_for_path(a.out),
                title="survey-animate demo",
                subtitle="synthetic frames — fully offline")
        except EncoderUnavailable as exc:
            return _fail(str(exc))
    print(f"wrote {result.video_path} "
          f"({result.n_frames} frames @ {result.fps:g} fps)")
    print(f"sidecar: {result.sidecar_path}")
    return 0


def _cmd_presets(_a: argparse.Namespace) -> int:
    for p in list_presets():
        print(f"{p.name:8s} {p.width}x{p.height}  "
              f"{p.fps:g} fps  crf {p.crf}  — {p.description}")
    return 0


def _cmd_manifest_info(a: argparse.Namespace) -> int:
    try:
        if os.path.isdir(a.source):
            m = frames_from_dir(a.source)
        else:
            m = read_manifest(a.source)
    except (ValueError, FileNotFoundError) as exc:
        return _fail(str(exc))
    info = {"n_frames": len(m), "fps": m.fps, "size_px": list(m.size_px),
            "field_source": m.field_source, "generator": m.generator,
            "source_path": m.source_path,
            "first_timestamp": m.frames[0].timestamp_iso if m.frames else "",
            "last_timestamp": m.frames[-1].timestamp_iso if m.frames else ""}
    print(json.dumps(info, indent=2))
    return 0


def _cmd_check(_a: argparse.Namespace) -> int:
    try:
        exe = resolve_ffmpeg()
    except EncoderUnavailable as exc:
        return _fail(str(exc))
    print(f"ffmpeg: {exe}")
    try:
        import imageio_ffmpeg  # noqa: F401
        print("imageio-ffmpeg: installed (bundled static binary preferred)")
    except ImportError:
        print("imageio-ffmpeg: not installed (using system ffmpeg)")
    try:
        import matplotlib  # noqa: F401
        print(f"matplotlib: installed (text overlays available)")
    except ImportError:
        print("matplotlib: not installed (text overlays disabled)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="survey-animate",
        description="Encode PNG frame sequences into MP4/GIF/WebM videos.")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("render", help="encode a manifest or PNG directory")
    _add_source_args(p)
    p.set_defaults(func=_cmd_render)

    p = sub.add_parser("demo",
                       help="offline synthetic demo render (no network)")
    p.add_argument("--out", default="demo-reel.mp4")
    p.add_argument("--preset", default="reel")
    p.add_argument("--format", choices=["mp4", "gif", "webm"], default=None)
    p.set_defaults(func=_cmd_demo)

    p = sub.add_parser("presets", help="list output presets")
    p.set_defaults(func=_cmd_presets)

    p = sub.add_parser("manifest-info",
                       help="validate and summarize a manifest or PNG dir")
    p.add_argument("source")
    p.set_defaults(func=_cmd_manifest_info)

    p = sub.add_parser("check", help="report encoder backend availability")
    p.set_defaults(func=_cmd_check)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
