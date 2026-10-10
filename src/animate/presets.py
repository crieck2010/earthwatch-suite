"""Output presets and frame geometry.

Presets fix the delivered video's dimensions, default frame rate, and
quality. Input frames are aspect-fit into the preset canvas (bilinear
rescale + padding) so nothing is ever stretched or cropped silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Preset:
    """One named output geometry + quality bundle."""

    name: str
    width: int
    height: int
    fps: float
    crf: int          # H.264 quality (lower = better, 18 ~ visually lossless)
    pix_fmt: str      # pixel format handed to the encoder
    description: str


PRESETS: Dict[str, Preset] = {
    "reel": Preset(
        name="reel", width=1080, height=1920, fps=30.0, crf=18,
        pix_fmt="yuv420p",
        description="Vertical 9:16 reel (1080x1920) — Instagram/TikTok/Shorts"),
    "wide": Preset(
        name="wide", width=1920, height=1080, fps=30.0, crf=18,
        pix_fmt="yuv420p",
        description="Widescreen 16:9 (1920x1080) — YouTube/presentations"),
    "square": Preset(
        name="square", width=1080, height=1080, fps=30.0, crf=18,
        pix_fmt="yuv420p",
        description="Square 1:1 (1080x1080) — feed posts"),
    "preview": Preset(
        name="preview", width=540, height=960, fps=15.0, crf=23,
        pix_fmt="yuv420p",
        description="Small 9:16 draft (540x960) — fast iteration"),
}


def get_preset(name: str) -> Preset:
    """Return the preset, or raise with the valid names listed."""
    try:
        return PRESETS[name]
    except KeyError:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown preset {name!r}; valid presets: {valid}") from None


def list_presets() -> List[Preset]:
    return [PRESETS[k] for k in sorted(PRESETS)]


def _bilinear_resize(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Bilinear rescale of a (H, W, C) uint8/float array in pure numpy."""
    h, w = img.shape[:2]
    if (h, w) == (new_h, new_w):
        return img
    ys = np.linspace(0, h - 1, new_h)
    xs = np.linspace(0, w - 1, new_w)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    top = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
    bot = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
    out = top * (1 - wy) + bot * wy
    if np.issubdtype(img.dtype, np.integer):
        out = np.clip(np.rint(out), 0, 255).astype(img.dtype)
    return out


def resize_pad(frame: np.ndarray,
               width: int,
               height: int,
               pad_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Aspect-fit ``frame`` into a ``(height, width, 3)`` canvas.

    The frame is bilinearly rescaled to the largest size that fits inside
    the canvas, then centered on a solid ``pad_color`` background. Never
    stretches or crops.
    """
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(f"resize_pad needs (H, W, 3|4), got {frame.shape}")
    rgb = frame[:, :, :3]
    h, w = rgb.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError(f"degenerate frame shape {frame.shape}")
    scale = min(width / w, height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    small = _bilinear_resize(rgb, new_h, new_w)
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:, :] = pad_color
    y = (height - new_h) // 2
    x = (width - new_w) // 2
    canvas[y:y + new_h, x:x + new_w] = small.astype(np.uint8)
    return canvas


def even_dims(width: int, height: int) -> Tuple[int, int]:
    """Round dimensions up to even (H.264 yuv420p requires even sizes)."""
    return (width + (width & 1), height + (height & 1))
