"""End-to-end animation pipeline: field -> particles -> frames -> manifest.

:func:`render_animation` is the one-call path the CLI and future
orchestrators (survey-reels) use. It is deliberately boring: sample the
time-varying field at ``n_frames`` fractional times, advect one shared
particle set through them, render each frame, and record the manifest.
"""

from __future__ import annotations

import datetime as _dt
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from flow import __version__
from flow.advect import AdvectionConfig, ParticleSet
from flow.fields import TimeVaryingField, VectorField
from flow.frames import FrameEntry, FrameManifest, interpolate_times
from flow.render import SIZES, render_frame
from flow.styles import Style, get_style


def _model_span_seconds(times_iso: Sequence[str]) -> Optional[float]:
    try:
        parsed = [_dt.datetime.fromisoformat(t) for t in times_iso]
    except ValueError:
        return None
    if len(parsed) < 2:
        return None
    return max((parsed[-1] - parsed[0]).total_seconds(), 0.0)


def render_animation(
    time_varying: TimeVaryingField,
    out_dir: str,
    *,
    style: Style | str = "reel-dark",
    n_frames: int = 48,
    n_particles: int = 1500,
    size: str | Tuple[int, int] = "reel",
    dpi: int = 100,
    fps: float = 30.0,
    dt_seconds: Optional[float] = None,
    seed: int = 7,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> FrameManifest:
    """Render ``n_frames`` PNGs + manifest into ``out_dir``.

    Returns the :class:`FrameManifest`. Frame ``i`` samples the field at
    fractional time ``i * (nt-1) / (n_frames-1)``. ``dt_seconds`` defaults
    to the model span divided by ``n_frames`` so particle motion matches
    the animation clock.
    """
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    style_obj = get_style(style) if isinstance(style, str) else style
    if isinstance(size, str) and size not in SIZES:
        raise ValueError(f"unknown size {size!r}; want one of {sorted(SIZES)}")
    size_px = SIZES[size] if isinstance(size, str) else (int(size[0]), int(size[1]))

    span = _model_span_seconds(time_varying.times)
    if dt_seconds is None:
        dt_seconds = (span / n_frames) if span else 3600.0
    config = AdvectionConfig(dt_seconds=float(dt_seconds), seed=seed)

    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    particles = ParticleSet(n_particles, time_varying.at(0.0), config=config)
    labels = interpolate_times(time_varying.times, n_frames)
    manifest = FrameManifest(fps=fps, size_px=size_px,
                             field_source=time_varying.fields[0].source,
                             generator=f"survey-flow {__version__}")

    # Fixed scalar range across frames so the colormap does not flicker.
    gmin, gmax = np.inf, -np.inf
    for f in time_varying.fields:
        if f.scalar is not None:
            lo, hi = f.scalar_range()
            gmin, gmax = min(gmin, lo), max(gmax, hi)
    if vmin is None:
        vmin = float(gmin) if np.isfinite(gmin) else 0.0
    if vmax is None:
        vmax = float(gmax) if np.isfinite(gmax) else 1.0

    last_stats: Dict[str, float] = {}
    for i in range(n_frames):
        t = i * (time_varying.nt - 1) / max(n_frames - 1, 1)
        vf = time_varying.at(t)
        last_stats = particles.step(vf)
        png = os.path.join(frames_dir, f"frame_{i:04d}.png")
        render_frame(vf, particles, style_obj, png, timestamp=labels[i],
                     vmin=vmin, vmax=vmax, size=size_px, dpi=dpi, title=title)
        manifest.add(FrameEntry(
            frame_index=i, timestamp_iso=labels[i], png_path=png,
            scalar=time_varying.fields[0].scalar_name,
            scalar_range=(vmin, vmax), colormap=style_obj.scalar_cmap,
            style=style_obj.name,
            render_params={"dpi": dpi, "size_px": list(size_px),
                           "n_particles": n_particles,
                           "dt_seconds": float(dt_seconds),
                           "seed": seed, "title": title or ""}))
        if progress:
            progress(i + 1, n_frames)

    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest.write(manifest_path)
    manifest.run_stats = last_stats  # informational, not serialized
    return manifest
