"""Burned-in overlays: timestamps and title cards.

Text rendering needs matplotlib (Agg backend, lazy import). The text is
rendered once into a small RGBA strip and alpha-composited onto frames
with numpy, so per-frame cost stays tiny. Without matplotlib,
:func:`burn_timestamps` passes frames through unchanged (documented) and
:func:`make_title_card` raises :class:`TextOverlayUnavailable`.

survey-flow already burns a colorbar legend into its frames — this module
deliberately does not draw another one.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


class TextOverlayUnavailable(RuntimeError):
    """matplotlib is required for text overlays."""


def _require_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt  # noqa: F401
        return plt
    except Exception as exc:
        raise TextOverlayUnavailable(
            "text overlays need matplotlib: `pip install matplotlib` "
            f"({exc})") from exc


def render_text_rgba(text: str,
                     fontsize_px: int = 42,
                     color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                     bg: Tuple[float, float, float, float] = (0, 0, 0, 0.45),
                     pad_px: int = 14) -> np.ndarray:
    """Render ``text`` to an ``(H, W, 4)`` uint8 RGBA strip."""
    plt = _require_mpl()
    fig = plt.figure(figsize=(8, 1.2), dpi=100)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.5, text, fontsize=fontsize_px, color=color,
            va="center", ha="left",
            bbox=dict(boxstyle=f"round,pad={pad_px / fontsize_px:.2f}",
                      facecolor=bg[:3], alpha=bg[3], edgecolor="none"))
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    # Tight-crop transparent margins.
    alpha = buf[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any():
        return buf[:1, :1].copy()
    return buf[rows][:, cols].copy()


def _composite(base: np.ndarray, overlay: np.ndarray,
               x: int, y: int) -> np.ndarray:
    """Alpha-composite RGBA ``overlay`` onto RGB ``base`` at (x, y)."""
    out = base.copy()
    h, w = overlay.shape[:2]
    H, W = base.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return out
    ov = overlay[y0 - y:y1 - y, x0 - x:x1 - x].astype(float)
    a = (ov[:, :, 3:4] / 255.0)
    out[y0:y1, x0:x1] = (ov[:, :, :3] * a + out[y0:y1, x0:x1] * (1 - a)
                         ).astype(np.uint8)
    return out


def pretty_timestamp(timestamp_iso: str) -> str:
    """``2026-09-16T00:00:00+00:00`` -> ``16 Sep 2026 00:00 UTC``."""
    import datetime as _dt
    s = (timestamp_iso or "").strip()
    if not s:
        return ""
    try:
        dt = _dt.datetime.fromisoformat(s)
    except ValueError:
        return s
    label = dt.strftime("%d %b %Y %H:%M")
    if dt.tzinfo is not None:
        try:
            label += " " + dt.tzname()
        except Exception:
            pass
    return label


def burn_timestamps(frames: Sequence[np.ndarray],
                    timestamps: Sequence[str],
                    position: str = "lower-left",
                    fontsize_px: int = 42,
                    margin_px: int = 36) -> List[np.ndarray]:
    """Burn one timestamp label per frame.

    ``position``: ``lower-left`` | ``lower-right`` | ``upper-left`` |
    ``upper-right``. Frames whose timestamp is blank are left untouched.
    """
    frames = list(frames)
    labels = [pretty_timestamp(t) for t in timestamps]
    if not any(labels):
        return [np.asarray(f) for f in frames]
    _require_mpl()  # fail fast with a clear error before compositing
    strips = {label: render_text_rgba(label, fontsize_px=fontsize_px)
              for label in dict.fromkeys(l for l in labels if l)}
    out = []
    for frame, label in zip(frames, labels):
        base = np.asarray(frame, dtype=np.uint8)
        if not label:
            out.append(base)
            continue
        strip = strips[label]
        sh, sw = strip.shape[:2]
        H, W = base.shape[:2]
        if position == "lower-right":
            x, y = W - sw - margin_px, H - sh - margin_px
        elif position == "upper-left":
            x, y = margin_px, margin_px
        elif position == "upper-right":
            x, y = W - sw - margin_px, margin_px
        else:  # lower-left
            x, y = margin_px, H - sh - margin_px
        out.append(_composite(base, strip, x, y))
    return out


def make_title_card(title: str,
                    subtitle: str = "",
                    width: int = 1080,
                    height: int = 1920,
                    bg_color: Tuple[int, int, int] = (10, 14, 20),
                    n_frames: int = 60) -> List[np.ndarray]:
    """Solid-background title frames with centered title/subtitle text."""
    plt = _require_mpl()
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor(tuple(c / 255 for c in bg_color))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.55, title, fontsize=64, color="white",
            ha="center", va="center", weight="bold", wrap=True)
    if subtitle:
        ax.text(0.5, 0.44, subtitle, fontsize=36, color=(0.75, 0.78, 0.82),
                ha="center", va="center", wrap=True)
    fig.canvas.draw()
    card = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    if (card.shape[1], card.shape[0]) != (width, height):
        from animate.presets import resize_pad
        card = resize_pad(card, width, height, pad_color=bg_color)
    return [card.copy() for _ in range(max(1, n_frames))]
