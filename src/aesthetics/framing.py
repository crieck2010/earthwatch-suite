"""Rotated auto-fit framing and tight zoom.

The mapped.earth Lake Ontario move: rotate the map so an elongated region
fills the vertical 9:16 canvas at maximum zoom, instead of wasting the
frame on empty space above and below a north-up strip. Includes the
rotated north arrow angle so typography can draw it honestly.

All geometry here is deterministic (pure numpy/PIL, no RNG).
"""

from typing import Tuple

import numpy as np
from PIL import Image

CanvasWH = Tuple[int, int]
BBox = Tuple[float, float, float, float]  # (lon0, lon1, lat0, lat1)


def rotate_frame(
    rgba: np.ndarray,
    angle_deg: float,
    *,
    fill: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> np.ndarray:
    """Rotate a composited frame by ``angle_deg`` counter-clockwise.

    Output has the same (H, W) as the input (``expand=False``); exposed
    corners are filled with ``fill`` (default opaque black). Uses PIL
    bicubic resampling; deterministic.
    """
    arr = np.asarray(rgba, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("rgba must be an (H, W, 4) array")
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
                          mode="RGBA")
    fill255 = tuple(int(np.clip(c, 0.0, 1.0) * 255.0 + 0.5) for c in fill)
    rot = img.rotate(float(angle_deg), resample=Image.BICUBIC,
                     expand=False, fillcolor=fill255)
    return np.asarray(rot, dtype=np.float64) / 255.0


def optimal_rotation(
    lon0: float,
    lon1: float,
    lat0: float,
    lat1: float,
    *,
    canvas_wh: CanvasWH = (1080, 1920),
    ref_lat: float = None,
) -> float:
    """Return the rotation angle (degrees, counter-clockwise) that maximizes
    the region's displayed area on the canvas.

    The region's lon/lat extents are converted to kilometres at ``ref_lat``
    (defaults to the bbox centre), then the rotation in [0, 90) maximizing
    ``scale**2`` — where ``scale = min(W/w', H/h')`` fits the rotated
    bounding box into the canvas — is found on a 0.5° grid.

    For a region much wider than tall on a vertical canvas this returns
    ~90° (the Lake Ontario case); for a square-ish region it returns ~0°.
    """
    if not (lon1 > lon0 and lat1 > lat0):
        raise ValueError("bbox must have positive width and height")
    w_px, h_px = canvas_wh
    lat = ref_lat if ref_lat is not None else (lat0 + lat1) / 2.0
    w_km = (lon1 - lon0) * 111.32 * max(np.cos(np.deg2rad(lat)), 1e-6)
    h_km = (lat1 - lat0) * 110.57

    best_angle = 0.0
    best_score = -1.0
    for deg in np.linspace(0.0, 90.0, 181):
        t = np.deg2rad(deg)
        c, s = abs(np.cos(t)), abs(np.sin(t))
        wp = w_km * c + h_km * s
        hp = w_km * s + h_km * c
        scale = min(w_px / wp, h_px / hp)
        score = scale ** 2
        if score > best_score:
            best_score = score
            best_angle = float(deg)
    return best_angle


def rotated_render_size(
    angle_deg: float, canvas_wh: CanvasWH = (1080, 1920)
) -> CanvasWH:
    """Canvas size at which to render the north-up map so that rotating it
    by ``angle_deg`` (via :func:`rotate_frame`, ``expand=False``) fills a
    ``canvas_wh`` frame edge to edge.

    Pattern for survey-viz::

        rw, rh = rotated_render_size(angle, (1080, 1920))
        fig, ax = new_canvas(rw, rh, background="black")
        ... draw the north-up map ...
        rgba = rotate_frame(fig_to_rgba(fig), angle)  # now 1080x1920
        ... then draw titles, legends, furniture (unrotated) ...
    """
    w, h = canvas_wh
    t = np.deg2rad(angle_deg)
    c, s = abs(np.cos(t)), abs(np.sin(t))
    return (int(round(w * c + h * s)), int(round(w * s + h * c)))


def rotate_frame_fill(
    rgba: np.ndarray,
    angle_deg: float,
    canvas_wh: CanvasWH = (1080, 1920),
    *,
    fill: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> np.ndarray:
    """Rotate by ``angle_deg`` counter-clockwise and center-crop to ``canvas_wh``.

    Unlike :func:`rotate_frame` (same-size output), this rotates with
    ``expand=True`` and then crops the center ``canvas_wh`` window, so no
    content is squeezed: use it with :func:`rotated_render_size` for the
    full auto-fit pattern::

        rw, rh = rotated_render_size(angle, (1080, 1920))
        ... render the north-up map at (rw, rh) ...
        rgba = rotate_frame_fill(map_rgba, angle, (1080, 1920))  # 1080x1920

    Deterministic (PIL bicubic).
    """
    arr = np.asarray(rgba, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("rgba must be an (H, W, 4) array")
    w, h = int(canvas_wh[0]), int(canvas_wh[1])
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
                          mode="RGBA")
    fill255 = tuple(int(np.clip(c, 0.0, 1.0) * 255.0 + 0.5) for c in fill)
    rot = img.rotate(float(angle_deg), resample=Image.BICUBIC,
                     expand=True, fillcolor=fill255)
    ew, eh = rot.size
    if ew < w or eh < h:
        raise ValueError(
            f"rotated frame {ew}x{eh} is smaller than canvas {w}x{h}; "
            f"render at rotated_render_size({angle_deg}, {(w, h)}) first"
        )
    left = (ew - w) // 2
    top = (eh - h) // 2
    cropped = rot.crop((left, top, left + w, top + h))
    return np.asarray(cropped, dtype=np.float64) / 255.0


def tight_bbox(
    lons,
    lats,
    mask,
    *,
    pad_frac: float = 0.04,
) -> BBox:
    """Smallest lon/lat bbox containing ``mask``, padded by ``pad_frac``
    of each span on every side. ``lons``/``lats`` are 1D coordinate axes
    (or 2D arrays matching ``mask``).
    """
    mask = np.asarray(mask, dtype=bool)
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    if not np.any(mask):
        raise ValueError("mask is empty: nothing to frame")
    if lons.ndim == 1 and lats.ndim == 1:
        lon2d, lat2d = np.meshgrid(lons, lats)
    else:
        lon2d, lat2d = lons, lats
    m_lons = lon2d[mask]
    m_lats = lat2d[mask]
    x0, x1 = float(np.min(m_lons)), float(np.max(m_lons))
    y0, y1 = float(np.min(m_lats)), float(np.max(m_lats))
    dx = max(x1 - x0, 1e-9)
    dy = max(y1 - y0, 1e-9)
    return (x0 - dx * pad_frac, x1 + dx * pad_frac,
            y0 - dy * pad_frac, y1 + dy * pad_frac)


def fit_extent_to_canvas(bbox: BBox, canvas_wh: CanvasWH = (1080, 1920)) -> BBox:
    """Expand ``bbox`` (never shrink) so its aspect ratio matches the canvas.

    The region stays centred; the dimension that would otherwise leave
    empty gutters is widened. Guarantees a full-bleed map with no
    letterboxing.
    """
    lon0, lon1, lat0, lat1 = bbox
    w_px, h_px = canvas_wh
    target = w_px / h_px
    w = lon1 - lon0
    h = lat1 - lat0
    if h <= 0 or w <= 0:
        raise ValueError("bbox must have positive width and height")
    current = w / h
    if current < target:  # too tall: widen
        new_w = h * target
        cx = (lon0 + lon1) / 2.0
        return (cx - new_w / 2.0, cx + new_w / 2.0, lat0, lat1)
    if current > target:  # too wide: grow taller
        new_h = w / target
        cy = (lat0 + lat1) / 2.0
        return (lon0, lon1, cy - new_h / 2.0, cy + new_h / 2.0)
    return bbox
