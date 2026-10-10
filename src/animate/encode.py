"""Video encoding via an ffmpeg binary.

Backend resolution order:
1. ``imageio-ffmpeg`` wheel's bundled static binary (no system install
   needed on user machines),
2. a system ``ffmpeg`` on PATH,
3. otherwise :class:`EncoderUnavailable` with install guidance.

Frames are piped to ffmpeg as raw RGB24 on stdin — no intermediate files,
no per-frame process spawns. MP4 uses H.264 (``libx264`` when the binary
provides it, else ``mpeg4``); GIF uses a two-pass palette for quality;
WebM uses VP9.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np

from animate.presets import even_dims


class EncoderUnavailable(RuntimeError):
    """No ffmpeg backend found (neither imageio-ffmpeg nor system ffmpeg)."""


def resolve_ffmpeg() -> str:
    """Return an ffmpeg executable path, or raise EncoderUnavailable."""
    try:
        import imageio_ffmpeg  # lazy: optional dependency
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise EncoderUnavailable(
        "no ffmpeg backend found: install it with "
        "`pip install imageio-ffmpeg` (bundles a static binary) or install "
        "a system ffmpeg and put it on PATH")


def _encoders(ffmpeg_exe: str) -> str:
    try:
        out = subprocess.run([ffmpeg_exe, "-hide_banner", "-encoders"],
                             capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    return out


def resolve_h264_codec(ffmpeg_exe: str) -> str:
    """``libx264`` when available, else the ubiquitous ``mpeg4``."""
    return "libx264" if "libx264" in _encoders(ffmpeg_exe) else "mpeg4"


def _check_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.uint8)
    if arr.shape != (height, width, 3):
        raise ValueError(
            f"encoder needs (H={height}, W={width}, 3) uint8 frames, "
            f"got shape {arr.shape} dtype {arr.dtype}")
    return np.ascontiguousarray(arr)


def _run(ffmpeg_exe: str, args: List[str], data_iter=None) -> None:
    """Run ffmpeg; stream raw frames on stdin when ``data_iter`` is given."""
    cmd = [ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-y"] + args
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
    except OSError as exc:
        raise EncoderUnavailable(f"could not launch ffmpeg at {ffmpeg_exe}: {exc}") from exc
    try:
        if data_iter is not None:
            assert proc.stdin is not None
            try:
                for chunk in data_iter:
                    proc.stdin.write(chunk)
            except BrokenPipeError:
                pass
            proc.stdin.close()
        rc = proc.wait(timeout=600)
        assert proc.stderr is not None
        stderr = proc.stderr.read().decode(errors="replace")
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("ffmpeg timed out while encoding")
    if rc != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {rc}): {stderr[-2000:]}")


def _frame_bytes(frames: Iterable[np.ndarray],
                 width: int, height: int,
                 progress: Optional[Callable[[int], None]] = None):
    n = 0
    for f in frames:
        yield _check_frame(f, width, height).tobytes()
        n += 1
        if progress is not None:
            progress(n)


def encode_mp4(frames: Iterable[np.ndarray],
               out_path: str,
               fps: float = 30.0,
               crf: int = 18,
               pix_fmt: str = "yuv420p",
               width: Optional[int] = None,
               height: Optional[int] = None,
               video_filters: Iterable[str] = (),
               audio_path: Optional[str] = None,
               ffmpeg_exe: Optional[str] = None,
               progress: Optional[Callable[[int], None]] = None) -> str:
    """Encode RGB frames to MP4 (H.264). ``width``/``height`` default to the
    first frame's shape. Dimensions are rounded up to even for yuv420p.

    ``video_filters`` are extra ``-vf`` filters (e.g. a zoompan motion
    filter) applied after the scale. ``audio_path`` muxes an audio file
    (MP3/WAV/…) as AAC, cut to the video length with ``-shortest``.
    """
    ffmpeg_exe = ffmpeg_exe or resolve_ffmpeg()
    frames = iter(frames)
    try:
        first = next(frames)
    except StopIteration:
        raise ValueError("encode_mp4 needs at least one frame")
    first = np.asarray(first, dtype=np.uint8)
    fh, fw = first.shape[:2]
    width, height = even_dims(width or fw, height or fh)

    def _all():
        yield first
        yield from frames

    codec = resolve_h264_codec(ffmpeg_exe)
    args = ["-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{fw}x{fh}", "-r", str(float(fps)), "-i", "-"]
    if audio_path:
        args += ["-i", audio_path]
    vf = []
    if (fw, fh) != (width, height):
        vf.append(f"scale={width}:{height}")
    vf.extend(video_filters)
    if pix_fmt:
        args += ["-pix_fmt", pix_fmt]
    if vf:
        args += ["-vf", ",".join(vf)]
    args += ["-c:v", codec]
    if codec == "libx264":
        args += ["-crf", str(int(crf)), "-preset", "medium"]
    if audio_path:
        args += ["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "160k",
                 "-shortest"]
    args += ["-movflags", "+faststart", out_path]
    _run(ffmpeg_exe, args, _frame_bytes(_all(), fw, fh, progress))
    return out_path


def encode_gif(frames: Iterable[np.ndarray],
               out_path: str,
               fps: float = 10.0,
               width: Optional[int] = None,
               video_filters: Iterable[str] = (),
               ffmpeg_exe: Optional[str] = None,
               progress: Optional[Callable[[int], None]] = None) -> str:
    """Encode RGB frames to GIF with a two-pass global palette (quality).

    ``video_filters`` (e.g. a zoompan motion filter) are applied before
    the palette pass so the motion is baked into the palette."""
    ffmpeg_exe = ffmpeg_exe or resolve_ffmpeg()
    materialized = [np.asarray(f, dtype=np.uint8) for f in frames]
    if not materialized:
        raise ValueError("encode_gif needs at least one frame")
    fh, fw = materialized[0].shape[:2]
    width = width or fw
    height = int(round(fh * width / fw))
    width, height = even_dims(width, height)
    with tempfile.TemporaryDirectory(prefix="animate_gif_") as tmp:
        palette = os.path.join(tmp, "palette.png")
        motion_vf = ",".join(video_filters)
        base_vf = f"scale={width}:{height}"
        if motion_vf:
            base_vf += f",{motion_vf}"
        _run(ffmpeg_exe,
             ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{fw}x{fh}",
              "-r", str(float(fps)), "-i", "-",
              "-vf", f"{base_vf},palettegen", palette],
             _frame_bytes(iter(materialized), fw, fh, None))
        _run(ffmpeg_exe,
             ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{fw}x{fh}",
              "-r", str(float(fps)), "-i", "-",
              "-i", palette, "-lavfi",
              f"{base_vf} [x]; [x][1:v] paletteuse",
              "-r", str(float(fps)), out_path],
             _frame_bytes(iter(materialized), fw, fh, progress))
    return out_path


def encode_webm(frames: Iterable[np.ndarray],
                out_path: str,
                fps: float = 30.0,
                crf: int = 18,
                width: Optional[int] = None,
                height: Optional[int] = None,
                video_filters: Iterable[str] = (),
                audio_path: Optional[str] = None,
                ffmpeg_exe: Optional[str] = None,
                progress: Optional[Callable[[int], None]] = None) -> str:
    """Encode RGB frames to WebM (VP9). ``video_filters`` and
    ``audio_path`` behave as in :func:`encode_mp4` (Opus audio)."""
    ffmpeg_exe = ffmpeg_exe or resolve_ffmpeg()
    frames = iter(frames)
    try:
        first = next(frames)
    except StopIteration:
        raise ValueError("encode_webm needs at least one frame")
    first = np.asarray(first, dtype=np.uint8)
    fh, fw = first.shape[:2]
    width, height = even_dims(width or fw, height or fh)

    def _all():
        yield first
        yield from frames

    args = ["-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{fw}x{fh}", "-r", str(float(fps)), "-i", "-"]
    if audio_path:
        args += ["-i", audio_path]
    vf = []
    if (fw, fh) != (width, height):
        vf.append(f"scale={width}:{height}")
    vf.extend(video_filters)
    if vf:
        args += ["-vf", ",".join(vf)]
    args += ["-c:v", "libvpx-vp9", "-crf", str(int(crf)), "-b:v", "0",
             "-pix_fmt", "yuv420p"]
    if audio_path:
        args += ["-map", "0:v", "-map", "1:a", "-c:a", "libopus", "-b:a",
                 "128k", "-shortest"]
    args += [out_path]
    _run(ffmpeg_exe, args, _frame_bytes(_all(), fw, fh, progress))
    return out_path


def encode(frames: Iterable[np.ndarray],
           out_path: str,
           fmt: str,
           **kwargs) -> str:
    """Dispatch to :func:`encode_mp4` / :func:`encode_gif` / :func:`encode_webm`."""
    fmt = fmt.lower().lstrip(".")
    if fmt == "mp4":
        return encode_mp4(frames, out_path, **kwargs)
    if fmt == "gif":
        return encode_gif(frames, out_path, **kwargs)
    if fmt == "webm":
        return encode_webm(frames, out_path, **kwargs)
    raise ValueError(f"unsupported video format {fmt!r} (mp4, gif, webm)")


def format_for_path(out_path: str) -> str:
    """Infer ``mp4``/``gif``/``webm`` from the output suffix."""
    ext = os.path.splitext(out_path)[1].lower().lstrip(".")
    if ext in ("mp4", "m4v"):
        return "mp4"
    if ext in ("gif",):
        return "gif"
    if ext in ("webm",):
        return "webm"
    raise ValueError(
        f"cannot infer video format from {out_path!r}; use --format mp4|gif|webm")
