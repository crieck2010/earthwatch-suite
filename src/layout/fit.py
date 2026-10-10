"""Fit existing frames to a platform canvas.

Three strategies, all deterministic:

* ``contain`` — scale to fit inside the canvas, letterbox the rest with
  a solid background color.
* ``cover`` — scale to cover the canvas, center-crop the overflow.
* ``blur-fill`` — the classic vertical-reel treatment: a blurred,
  darkened ``cover`` copy fills the canvas and the sharp ``contain``
  image sits on top. 16:9 masters become 9:16 reels this way.

Pillow does the resampling (LANCZOS) and the blur; numpy carries the
pixels. ``fit_frame`` accepts and returns numpy uint8 arrays
(H, W, 3|4).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from PIL import Image, ImageFilter
except ImportError:  # pragma: no cover
    Image = None  # type: ignore
    ImageFilter = None  # type: ignore


def _require_pil() -> None:
    if Image is None:
        raise ImportError(
            "fit_frame needs Pillow: pip install 'survey-layout[fit]' "
            "(or pip install pillow)")


def _as_image(arr: np.ndarray) -> "Image.Image":
    _require_pil()
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        raise ValueError(f"fit_frame needs uint8 frames, got {a.dtype}")
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError(f"fit_frame needs (H, W, 3|4), got {a.shape}")
    mode = "RGB" if a.shape[2] == 3 else "RGBA"
    return Image.fromarray(a, mode=mode)


def _to_array(img: "Image.Image", channels: int) -> np.ndarray:
    a = np.asarray(img.convert("RGB" if channels == 3 else "RGBA"))
    return np.ascontiguousarray(a)


def _scaled(img: "Image.Image", size: Tuple[int, int]) -> "Image.Image":
    return img.resize(size, Image.LANCZOS)


def fit_frame(frame: np.ndarray, width: int, height: int,
              mode: str = "blur-fill",
              bg_color: Tuple[int, int, int] = (10, 10, 14),
              blur_radius: float = 25.0,
              darken: float = 0.55) -> np.ndarray:
    """Fit ``frame`` to ``(width, height)`` with ``mode``.

    ``bg_color`` is the letterbox color for ``contain``; ``blur_radius``
    / ``darken`` tune the ``blur-fill`` background (``darken`` multiplies
    the background brightness, 0–1).
    """
    if mode not in ("contain", "cover", "blur-fill"):
        raise ValueError(
            f"Unknown fit mode {mode!r} "
            "(expected 'contain', 'cover', or 'blur-fill')")
    if width <= 0 or height <= 0:
        raise ValueError(f"Canvas must be positive, got {(width, height)}")
    img = _as_image(frame)
    channels = 3 if img.mode == "RGB" else 4
    iw, ih = img.size

    def _cover() -> "Image.Image":
        scale = max(width / iw, height / ih)
        big = _scaled(img, (max(1, int(round(iw * scale))),
                            max(1, int(round(ih * scale)))))
        x = (big.size[0] - width) // 2
        y = (big.size[1] - height) // 2
        return big.crop((x, y, x + width, y + height))

    if mode == "cover":
        return _to_array(_cover(), channels)

    # contain geometry, shared by contain and blur-fill
    scale = min(width / iw, height / ih)
    small = _scaled(img, (max(1, int(round(iw * scale))),
                          max(1, int(round(ih * scale)))))
    ox, oy = (width - small.size[0]) // 2, (height - small.size[1]) // 2

    if mode == "contain":
        canvas = Image.new(img.mode, (width, height),
                           bg_color + ((255,) if channels == 4 else ()))
        canvas.paste(small, (ox, oy))
        return _to_array(canvas, channels)

    # blur-fill
    bg = _cover().filter(ImageFilter.GaussianBlur(blur_radius))
    bg_arr = np.asarray(bg).astype(np.float32) * float(darken)
    bg = Image.fromarray(np.clip(bg_arr, 0, 255).astype(np.uint8), mode=bg.mode)
    if channels == 4:
        bg = bg.convert("RGBA")
    else:
        bg = bg.convert("RGB")
    bg.paste(small.convert(bg.mode), (ox, oy))
    return _to_array(bg, channels)
