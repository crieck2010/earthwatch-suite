"""Collision-aware furniture placement for the preset title blocks.

The v0.1.0 presets put every furniture item at a fixed axes-fraction
position; on real reels those fixed positions collide (observed: the
date dial over the subtitle and over place labels, the title overflowing
the canvas width on long titles). This module places the furniture set
— title, subtitle, date dial, gradient bar, timeline, counter, encoding
line, watermark block, north arrow — by priority, trying each item's
preferred rect then its fallback alternatives, with optional font
shrinking for text. Fixed preset positions remain as the preferred
rects (and the last-resort fallback), not the law.

Algorithm (deterministic, cheap — pure rect arithmetic, no iteration):

1. Sort specs by descending priority (stable: ties keep input order).
2. For each spec, try the preferred rect, then each alternative, then
   each shrink scale applied to the preferred rect (text only).
3. Accept the first candidate that lies inside [0, 1]^2 and does not
   overlap any already-placed rect (expanded by ``pad``).
4. If nothing fits: optional items are dropped (``placed=False``);
   required items fall back to their preferred rect so the reel never
   renders empty-handed.

Text measurement uses the real Agg renderer
(:func:`text_extent_frac`), so shrink-to-fit reflects the actual
rasterized glyphs for the pinned matplotlib version — the same
measurement the final draw uses, which is what keeps survey-cache
fingerprints stable.

Limits (documented honestly):

- Overlap is tested on bounding rects, not glyph shapes; tight but
  non-overlapping rects can still look close.
- The placer knows nothing about the *map* furniture (place labels on
  the data). Pass placed furniture rects to
  :func:`aesthetics.typography.place_labels` as ``obstacles`` so city
  labels avoid the dial/legends too.
- Shrinking only scales w/h about the rect's top-left corner; it does
  not reflow or re-anchor text.
- Measurement needs a drawn canvas (``fig.canvas.draw()``); each
  :func:`text_spec` call costs one draw (~10 ms at 1080x1920).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

Rect = Tuple[float, float, float, float]  # (x, y, w, h), axes fraction


@dataclass(frozen=True)
class FurnitureSpec:
    """One furniture item's placement request."""

    kind: str
    rect: Rect                       # preferred rect (x, y, w, h)
    priority: int = 0                # higher is placed first
    alternatives: Tuple[Rect, ...] = ()   # fallback rects, tried in order
    optional: bool = False           # drop (placed=False) if nothing fits
    shrink_steps: Tuple[float, ...] = ()  # font scales to try (text only)


@dataclass
class PlacedItem:
    """Placement result for one spec."""

    kind: str
    rect: Rect
    placed: bool
    scale: float = 1.0               # < 1 when a shrink step was used


def _overlaps(a: Rect, b: Rect, pad: float) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (ax - pad < bx + bw + pad and ax + aw + pad > bx - pad
            and ay - pad < by + bh + pad and ay + ah + pad > by - pad)


def _inside(r: Rect) -> bool:
    x, y, w, h = r
    return x >= 0.0 and y >= 0.0 and x + w <= 1.0 and y + h <= 1.0 and w > 0 and h > 0


def _scaled(rect: Rect, scale: float) -> Rect:
    x, y, w, h = rect
    return (x, y, w * scale, h * scale)


def place_furniture(
    specs: Sequence[FurnitureSpec],
    *,
    pad: float = 0.012,
) -> List[PlacedItem]:
    """Place furniture specs by priority; returns one PlacedItem per spec
    in the *input* order."""
    order = sorted(range(len(specs)),
                   key=lambda i: (-specs[i].priority, i))
    placed_rects: List[Rect] = []
    results: List[Optional[PlacedItem]] = [None] * len(specs)
    for i in order:
        spec = specs[i]
        chosen: Optional[Tuple[Rect, float]] = None
        candidates = [spec.rect, *spec.alternatives]
        for rect in candidates:
            if _inside(rect) and not any(
                    _overlaps(rect, q, pad) for q in placed_rects):
                chosen = (rect, 1.0)
                break
        if chosen is None:
            for scale in spec.shrink_steps:
                rect = _scaled(spec.rect, scale)
                if _inside(rect) and not any(
                        _overlaps(rect, q, pad) for q in placed_rects):
                    chosen = (rect, scale)
                    break
        if chosen is None:
            if spec.optional:
                results[i] = PlacedItem(spec.kind, spec.rect, False, 1.0)
                continue
            chosen = (spec.rect, 1.0)  # last resort: the fixed position
        rect, scale = chosen
        placed_rects.append(rect)
        results[i] = PlacedItem(spec.kind, rect, True, scale)
    return [r for r in results if r is not None]


# --- text measurement / shrink-to-fit ---------------------------------------


def text_extent_frac(
    ax,
    text: str,
    *,
    family: str = "DejaVu Sans",
    size: float = 20,
    weight: str = "normal",
) -> Tuple[float, float]:
    """Measured (width, height) of ``text`` in axes fraction, using the
    real Agg renderer. Deterministic for a pinned matplotlib version."""
    fig = ax.figure
    artist = fig.text(0, 0, text, family=family, fontsize=size, weight=weight)
    fig.canvas.draw()
    bb = artist.get_window_extent(fig.canvas.get_renderer())
    artist.remove()
    fw, fh = fig.get_size_inches() * fig.dpi
    return bb.width / fw, bb.height / fh


def fit_font_size(
    ax,
    text: str,
    *,
    family: str = "DejaVu Sans",
    weight: str = "normal",
    start_size: float = 64,
    max_width_frac: float = 0.88,
    min_size: float = 20,
) -> float:
    """Largest font size in ``[min_size, start_size]`` whose rendered
    width fits ``max_width_frac`` of the canvas. Binary search on the
    real renderer; deterministic."""
    if text_extent_frac(ax, text, family=family, size=start_size,
                        weight=weight)[0] <= max_width_frac:
        return start_size
    lo, hi = min_size, start_size
    for _ in range(8):
        mid = (lo + hi) / 2.0
        w, _ = text_extent_frac(ax, text, family=family, size=mid,
                                weight=weight)
        if w <= max_width_frac:
            lo = mid
        else:
            hi = mid
    return lo


def text_spec(
    ax,
    kind: str,
    text: str,
    x: float,
    y: float,
    *,
    family: str = "DejaVu Sans",
    size: float = 20,
    weight: str = "normal",
    va: str = "top",
    ha: str = "left",
    priority: int = 0,
    alternatives: Sequence[Rect] = (),
    optional: bool = False,
    shrink: bool = True,
) -> FurnitureSpec:
    """Build a FurnitureSpec for a text item, measuring its real extent.

    ``(x, y)`` is the anchor per ``ha``/``va`` (only ``left``/``top``
    supported — the preset title grammar); the rect spans the measured
    glyphs. When ``shrink`` is true, shrink steps (0.92 … 0.70) are
    offered so long titles fit instead of colliding.
    """
    if ha != "left" or va != "top":
        raise ValueError("text_spec supports ha='left', va='top' only")
    w, h = text_extent_frac(ax, text, family=family, size=size, weight=weight)
    rect = (x, y - h, w, h)
    shrink_steps = (0.92, 0.85, 0.78, 0.70) if shrink else ()
    return FurnitureSpec(
        kind=kind, rect=rect, priority=priority,
        alternatives=tuple(alternatives), optional=optional,
        shrink_steps=shrink_steps,
    )


def box_spec(
    kind: str,
    rect: Rect,
    *,
    priority: int = 0,
    alternatives: Sequence[Rect] = (),
    optional: bool = False,
) -> FurnitureSpec:
    """Build a FurnitureSpec for a non-text item (dial, bar, timeline)."""
    return FurnitureSpec(
        kind=kind, rect=rect, priority=priority,
        alternatives=tuple(alternatives), optional=optional,
    )
