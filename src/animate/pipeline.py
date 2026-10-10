"""End-to-end pipeline: manifest/PNGs -> finished video + provenance sidecar.

:func:`render_video` is the one call most users need:

1. load frames (survey-flow manifest, or a plain PNG directory),
2. aspect-fit every frame into the preset canvas,
3. optionally prepend title-card frames and burn per-frame timestamps,
4. encode (MP4/GIF/WebM),
5. write the ``.provenance.json`` sidecar.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from animate import _png
from animate.encode import (EncoderUnavailable, encode, format_for_path,
                            resolve_ffmpeg)
from animate.manifest import Manifest, frames_from_dir, read_manifest
from animate.motion import (MotionSpec, blend_frames, blend_stamps,
                            coerce_motion, motion_filter)
from animate.overlays import TextOverlayUnavailable, burn_timestamps, make_title_card
from animate.presets import Preset, get_preset, resize_pad
from animate.provenance import sha256_file, write_sidecar


@dataclass
class VideoResult:
    """Everything produced by one :func:`render_video` call."""

    video_path: str
    sidecar_path: str
    fmt: str
    preset: str
    fps: float
    n_frames: int
    width: int
    height: int
    encoder: str
    params: Dict = field(default_factory=dict)


def _load_frame(path: str) -> np.ndarray:
    """Decode a PNG to ``(H, W, 3)`` uint8, preferring imageio when present."""
    try:
        import imageio.v2 as imageio  # lazy: optional
        img = np.asarray(imageio.imread(path))
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img[:, :, :3].astype(np.uint8)
    except Exception:
        pass
    img = _png.read_png(path)
    return img[:, :, :3].copy()


def _resolve_manifest(source) -> Manifest:
    if isinstance(source, Manifest):
        return source
    if not isinstance(source, str):
        raise TypeError(
            f"source must be a manifest path, directory, or Manifest; "
            f"got {type(source).__name__}")
    if os.path.isdir(source):
        return frames_from_dir(source)
    return read_manifest(source)


def render_video(source,
                 out_path: str,
                 *,
                 preset: str = "reel",
                 fmt: Optional[str] = None,
                 fps: Optional[float] = None,
                 crf: Optional[int] = None,
                 burn_timestamps_: bool = True,
                 timestamp_position: str = "lower-left",
                 title: str = "",
                 subtitle: str = "",
                 title_seconds: float = 2.0,
                 min_duration_s: float = 0.0,
                 pad_color: tuple = (0, 0, 0),
                 motion=None,
                 audio_path: Optional[str] = None,
                 progress: Optional[Callable[[int], None]] = None,
                 ffmpeg_exe: Optional[str] = None) -> VideoResult:
    """Render ``source`` (manifest path, PNG directory, or Manifest) to video.

    Returns a :class:`VideoResult`. Raises :class:`EncoderUnavailable`
    when no ffmpeg backend exists, and the manifest errors from
    :mod:`animate.manifest` on bad input.

    ``motion`` is ``None``, a :class:`animate.motion.MotionSpec`, or a
    dict of its fields: Ken Burns zoom/pan drift (applied at encode via
    ffmpeg's ``zoompan``) plus optional crossfade blending between data
    frames (``smooth``/``smooth_steps``, computed in numpy). If the
    ffmpeg binary lacks ``zoompan``, the clip encodes without drift and
    the sidecar records why.

    ``audio_path`` muxes an audio file (MP3/WAV/…) as a second input,
    encoded to AAC (MP4) / Opus (WebM) and cut to the video length with
    ``-shortest``. GIF has no audio track: the file is ignored and the
    sidecar notes it. A missing path raises ``FileNotFoundError``.

    ``min_duration_s`` holds each data frame (by repetition) so the
    *data* portion of the reel lasts at least that many seconds: with
    ``n`` data frames at ``fps``, each frame repeats
    ``k = ceil(ceil(min_duration_s * fps) / n)`` times. Applied after
    blending and before the title-card prepend, so the title card (when
    enabled) keeps its fixed length on top of the held data. The
    per-frame timestamp list is repeated to match, so burned timestamps
    stay correct. ``0`` (default) disables the hold — behavior
    unchanged. Negative values raise ``ValueError``.
    """
    pre = get_preset(preset)
    fmt = fmt or format_for_path(out_path)
    fps = float(fps if fps is not None else pre.fps)
    crf = int(crf if crf is not None else pre.crf)
    ffmpeg_exe = ffmpeg_exe or resolve_ffmpeg()

    manifest = _resolve_manifest(source)
    if not manifest.frames:
        raise ValueError("no frames to encode")

    frames: List[np.ndarray] = []
    for mf in manifest.frames:
        frames.append(resize_pad(_load_frame(mf.png_path),
                                 pre.width, pre.height, pad_color=pad_color))
    stamps: List[str] = [mf.timestamp_iso for mf in manifest.frames]

    params: Dict = {"crf": crf, "pad_color": list(pad_color),
                    "timestamp_position": timestamp_position}

    # Cinematic motion: validate + normalize the spec up front so bad
    # values fail fast, before any encoding work.
    motion_spec: Optional[MotionSpec] = coerce_motion(motion)
    if motion_spec is not None:
        params["motion"] = motion_spec.to_dict()

    # Smooth crossfade blending between data frames (numpy, deterministic).
    if motion_spec is not None and motion_spec.blending:
        frames = blend_frames(frames, motion_spec.smooth_steps)
        stamps = blend_stamps(stamps, motion_spec.smooth_steps)

    # Minimum data duration: hold each data frame by repeating it k
    # times so short reels still play at least min_duration_s seconds
    # of *data*. Applied after blending and BEFORE the title-card
    # prepend, so the card (when enabled) keeps its fixed length on
    # top of the held data. Stamps are repeated to match, keeping
    # burned timestamps correct.
    if min_duration_s < 0:
        raise ValueError(
            f"min_duration_s must be >= 0, got {min_duration_s!r}")
    if min_duration_s > 0:
        need = math.ceil(float(min_duration_s) * fps)
        if len(frames) < need:
            k = math.ceil(need / len(frames))
            frames = [f for f in frames for _ in range(k)]
            stamps = [s for s in stamps for _ in range(k)]
            params["min_duration_s"] = float(min_duration_s)
            params["frame_hold"] = k

    # Audio: fail fast on a missing file; GIF cannot carry audio.
    if audio_path is not None and not os.path.isfile(audio_path):
        raise FileNotFoundError(f"audio file not found: {audio_path}")
    if audio_path is not None:
        params["audio"] = os.path.basename(audio_path)
        if fmt == "gif":
            params["audio_skipped"] = "gif has no audio track"
            audio_path = None

    # Title card first (so it plays before the animation).
    if title:
        try:
            cards = make_title_card(title, subtitle, pre.width, pre.height,
                                    n_frames=max(1, int(round(title_seconds * fps))))
            frames = cards + frames
            params["title"] = title
            params["subtitle"] = subtitle
        except TextOverlayUnavailable:
            params["title_skipped"] = "matplotlib not installed"

    # Timestamps (skip title-card frames — they have no model time).
    if burn_timestamps_:
        n_cards = len(frames) - len(stamps)
        card_stamps = [""] * n_cards + stamps
        try:
            frames = burn_timestamps(frames, card_stamps,
                                     position=timestamp_position)
        except TextOverlayUnavailable:
            params["timestamps_skipped"] = "matplotlib not installed"

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    video_filters: List[str] = []
    if motion_spec is not None and motion_spec.animated:
        mfilt = motion_filter(motion_spec, pre.width, pre.height,
                              len(frames))
        if mfilt:
            video_filters.append(mfilt)
    encode_kwargs: Dict = {"fps": fps, "ffmpeg_exe": ffmpeg_exe,
                           "progress": progress,
                           "video_filters": video_filters}
    if fmt in ("mp4", "webm"):
        encode_kwargs["crf"] = crf
        encode_kwargs["width"] = pre.width
        encode_kwargs["height"] = pre.height
    if audio_path is not None:
        encode_kwargs["audio_path"] = audio_path
    try:
        encode(frames, out_path, fmt, **encode_kwargs)
    except RuntimeError as exc:
        # Ancient/minimal ffmpeg builds may lack zoompan; degrade to a
        # static clip rather than failing the whole render.
        if video_filters and "No such filter" in str(exc):
            params["motion_skipped"] = ("ffmpeg lacks the zoompan filter; "
                                        "encoded without drift")
            encode_kwargs["video_filters"] = []
            encode(frames, out_path, fmt, **encode_kwargs)
        else:
            raise

    try:
        source_hash = sha256_file(manifest.source_path) \
            if manifest.source_path and os.path.isfile(manifest.source_path) else ""
    except OSError:
        source_hash = ""
    import animate as _animate_pkg
    sidecar = write_sidecar(
        out_path, source=manifest.source_path or "(in-memory manifest)",
        source_sha256=source_hash, preset=pre.name, fmt=fmt, fps=fps,
        n_frames=len(frames), params=params, encoder=ffmpeg_exe,
        generator=f"survey-animate {_animate_pkg.__version__}")

    return VideoResult(video_path=os.path.abspath(out_path),
                       sidecar_path=sidecar, fmt=fmt, preset=pre.name,
                       fps=fps, n_frames=len(frames),
                       width=pre.width, height=pre.height,
                       encoder=ffmpeg_exe, params=params)
