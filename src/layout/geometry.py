"""Small rectangle helpers. Everything is figure fractions: (left, bottom,
width, height), origin bottom-left. Edge-touching is NOT overlap."""

from __future__ import annotations

from typing import Iterable, List, Tuple

Rect = Tuple[float, float, float, float]


def _check(rect: Rect, name: str = "rect") -> Rect:
    if len(rect) != 4:
        raise ValueError(f"{name} must be (left, bottom, width, height), "
                         f"got {rect!r}")
    l, b, w, h = (float(v) for v in rect)
    if w <= 0 or h <= 0:
        raise ValueError(f"{name} must have positive size, got {rect!r}")
    return (l, b, w, h)


def overlaps(a: Rect, b: Rect) -> bool:
    """True when the interiors intersect (shared edges don't count)."""
    al, ab, aw, ah = _check(a, "a")
    bl, bb, bw, bh = _check(b, "b")
    return (al < bl + bw and bl < al + aw and
            ab < bb + bh and bb < ab + ah)


def contains(outer: Rect, inner: Rect) -> bool:
    """True when ``inner`` lies fully inside ``outer`` (edges allowed)."""
    al, ab, aw, ah = _check(outer, "outer")
    bl, bb, bw, bh = _check(inner, "inner")
    eps = 1e-9  # fraction coords: far below a pixel; absorbs float noise
    return (al - eps <= bl and ab - eps <= bb and
            bl + bw <= al + aw + eps and bb + bh <= ab + ah + eps)


def within_canvas(rect: Rect) -> bool:
    """True when the rect lies fully inside the unit square."""
    return contains((0.0, 0.0, 1.0, 1.0), rect)


def to_pixels(rect: Rect, width: int, height: int) -> Tuple[int, int, int, int]:
    """Fraction rect -> integer pixel rect (left, top, w, h), top-origin."""
    l, b, w, h = _check(rect)
    px = int(round(l * width))
    pw = int(round(w * width))
    # bottom-origin -> top-origin
    py = int(round((1.0 - b - h) * height))
    ph = int(round(h * height))
    return (px, py, pw, ph)


def from_top_rect(left: float, top: float, width: float,
                  height: float) -> Rect:
    """Top-origin fractions (as chrome is usually measured) -> our rect."""
    return _check((left, 1.0 - top - height, width, height), "rect")


def any_overlap(rect: Rect, others: Iterable[Rect]) -> bool:
    return any(overlaps(rect, o) for o in others)


def describe(rect: Rect) -> str:
    l, b, w, h = rect
    return (f"x {l:.3f}–{l + w:.3f}, y {b:.3f}–{b + h:.3f}")
