"""Cinematic motion for encoded videos.

Two orthogonal effects, both optional and composable:

* **zoom/pan** — a Ken Burns drift applied at encode time with ffmpeg's
  ``zoompan`` filter. The viewport eases across the whole clip: zoom
  ramps 1.0 -> ``1 + 0.6*zoom_speed`` (in) or the reverse (out), and the
  pan drifts the window center along one of eight compass directions.
  Pure numpy is deliberately *not* used here: ``zoompan`` does smooth
  sub-pixel resampling in the encoder with no per-frame Python cost,
  and the same filter string works for MP4, WebM, and GIF.
* **smooth** — crossfade blending between consecutive data frames,
  computed in numpy *before* encode (deterministic, no ffmpeg filter
  quirks). ``smooth_steps`` blended frames are inserted per gap, so a
  monthly time step melts into the next instead of jumping.

Everything here is pure data + pure functions except the final
``-vf`` string, which :mod:`animate.encode` passes straight through to
ffmpeg. If the ffmpeg binary lacks ``zoompan`` (essentially never —
it is a core libavfilter filter), the pipeline logs the miss in the
sidecar and encodes without motion rather than failing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

ZOOM_MODES = ("off", "in", "out")
PAN_MODES = ("off", "up", "down", "left", "right",
             "up-left", "up-right", "down-left", "down-right")

# compass direction -> (x, y) unit vector (y down, matching image coords)
_PAN_VECTORS = {
    "up": (0.0, -1.0), "down": (0.0, 1.0),
    "left": (-1.0, 0.0), "right": (1.0, 0.0),
    "up-left": (-0.7071, -0.7071), "up-right": (0.7071, -0.7071),
    "down-left": (-0.7071, 0.7071), "down-right": (0.7071, 0.7071),
}

#: zoom range: zoom_speed 0..1 maps to a total zoom factor of 1.0..1.6
ZOOM_RANGE = 0.6
#: pan headroom zoom applied when panning without an explicit zoom
PAN_HEADROOM = 0.15
#: pan_speed 0..1 maps to a viewport travel of 0..30% of the frame
PAN_TRAVEL = 0.30


@dataclass(frozen=True)
class MotionSpec:
    """Validated cinematic-motion parameters.

    ``zoom``: ``"off"`` | ``"in"`` | ``"out"``.
    ``zoom_speed`` / ``pan_speed``: 0.0..1.0 (UI sliders).
    ``pan``: compass direction or ``"off"``.
    ``smooth``: insert crossfade-blended frames between data frames;
    ``smooth_steps``: how many per gap (0 disables even if ``smooth``).
    """

    zoom: str = "off"
    zoom_speed: float = 0.25
    pan: str = "off"
    pan_speed: float = 0.25
    smooth: bool = False
    smooth_steps: int = 3

    def __post_init__(self) -> None:
        if self.zoom not in ZOOM_MODES:
            raise ValueError(f"zoom must be one of {ZOOM_MODES}, "
                             f"got {self.zoom!r}")
        if self.pan not in PAN_MODES:
            raise ValueError(f"pan must be one of {PAN_MODES}, "
                             f"got {self.pan!r}")
        for name in ("zoom_speed", "pan_speed"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or \
                    not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in 0.0..1.0, got {value!r}")
        if not isinstance(self.smooth_steps, int) or self.smooth_steps < 0:
            raise ValueError("smooth_steps must be a non-negative int, "
                             f"got {self.smooth_steps!r}")

    @property
    def animated(self) -> bool:
        """True when any zoom/pan drift applies (ignores ``smooth``)."""
        return self.zoom != "off" or self.pan != "off"

    @property
    def blending(self) -> bool:
        """True when crossfade blending applies."""
        return self.smooth and self.smooth_steps > 0

    def to_dict(self) -> Dict:
        return {"zoom": self.zoom, "zoom_speed": self.zoom_speed,
                "pan": self.pan, "pan_speed": self.pan_speed,
                "smooth": self.smooth, "smooth_steps": self.smooth_steps}

    @classmethod
    def from_dict(cls, d: Dict) -> "MotionSpec":
        unknown = set(d) - {"zoom", "zoom_speed", "pan", "pan_speed",
                            "smooth", "smooth_steps"}
        if unknown:
            raise ValueError(f"unknown motion keys: {sorted(unknown)}")
        return cls(**d)


def coerce_motion(motion) -> Optional[MotionSpec]:
    """Accept ``None`` | ``MotionSpec`` | ``dict``; return a MotionSpec or None."""
    if motion is None:
        return None
    if isinstance(motion, MotionSpec):
        return motion
    if isinstance(motion, dict):
        return MotionSpec.from_dict(motion)
    raise TypeError(f"motion must be None, a MotionSpec, or a dict, "
                    f"got {type(motion).__name__}")


def motion_filter(spec: MotionSpec, width: int, height: int,
                  n_frames: int) -> Optional[str]:
    """Build the ffmpeg ``zoompan`` filter for ``spec``.

    The animation is expressed in ``on`` (output frame count, 0-based),
    so with ``d=1`` it drifts continuously across the whole clip.
    Returns None when no zoom/pan applies.
    """
    if not spec.animated:
        return None
    n = max(1, int(n_frames))
    # progress 0->1 across the clip; constant 0 for a single frame
    t = f"(on/{n - 1})" if n > 1 else "0"

    if spec.zoom == "in":
        z0, z1 = 1.0, 1.0 + ZOOM_RANGE * spec.zoom_speed
    elif spec.zoom == "out":
        z0, z1 = 1.0 + ZOOM_RANGE * spec.zoom_speed, 1.0
    else:  # pan-only: hold a slight zoom so the window has room to drift
        z0 = z1 = 1.0 + PAN_HEADROOM * max(spec.pan_speed, 0.15)
    z_expr = f"{z0:.4f}+({z1:.4f}-{z0:.4f})*{t}"

    dx, dy = _PAN_VECTORS.get(spec.pan, (0.0, 0.0))
    # viewport center drifts from 0.5 -/+ dir*travel/2; travel scales with speed
    tx = dx * PAN_TRAVEL * spec.pan_speed
    ty = dy * PAN_TRAVEL * spec.pan_speed
    x_expr = f"(iw-iw/zoom)*(0.5+({tx:.4f})*({t}-0.5))"
    y_expr = f"(ih-ih/zoom)*(0.5+({ty:.4f})*({t}-0.5))"

    return (f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':"
            f"d=1:s={width}x{height}")


def blend_frames(frames: List[np.ndarray], steps: int) -> List[np.ndarray]:
    """Insert ``steps`` linearly crossfaded frames per consecutive pair.

    Endpoints are returned unchanged (no copies); blended frames are new
    uint8 arrays. Pure numpy — deterministic and ffmpeg-independent.
    """
    frames = list(frames)
    if steps <= 0 or len(frames) < 2:
        return frames
    out: List[np.ndarray] = []
    for a, b in zip(frames, frames[1:]):
        a_arr = np.asarray(a, dtype=np.float32)
        b_arr = np.asarray(b, dtype=np.float32)
        out.append(a)
        for s in range(1, steps + 1):
            alpha = s / (steps + 1)
            out.append(((1.0 - alpha) * a_arr + alpha * b_arr)
                       .astype(np.uint8))
    out.append(frames[-1])
    return out


def blend_stamps(stamps: List[str], steps: int) -> List[str]:
    """Extend a per-frame stamp list to match :func:`blend_frames` output.

    Blended frames inherit the *previous* original frame's stamp, so the
    burned-in time never jumps ahead of the data being shown.
    """
    stamps = list(stamps)
    if steps <= 0 or len(stamps) < 2:
        return stamps
    out: List[str] = []
    for a, b in zip(stamps, stamps[1:]):
        out.append(a)
        out.extend([a] * steps)
    out.append(stamps[-1])
    return out
