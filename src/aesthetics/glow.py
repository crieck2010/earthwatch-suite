"""Additive glow layers with bloom for event/point data.

Renders scattered events (lightning flashes, quake epicenters, lighthouse
positions) as light accumulating on a dark background: values are binned
additively onto a grid, colormapped, then softened with a Gaussian core
plus a wider, dimmer bloom pass. Overlapping events sum — dense regions
burn brighter, exactly the mapped.earth "year of lightning" look.

Also provides :func:`light_beam`, an anisotropic glowing wedge for
directional sources (lighthouse beams on their true cycle).

Fully deterministic: no RNG anywhere in this module.
"""

from typing import Tuple, Union

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")

ColormapLike = Union[str, matplotlib.colors.Colormap]


def _as_cmap(cmap: ColormapLike) -> matplotlib.colors.Colormap:
    if isinstance(cmap, str):
        try:
            return matplotlib.colormaps[cmap]
        except KeyError as exc:
            raise ValueError(f"unknown colormap {cmap!r}") from exc
    return cmap


def accumulate_glow(
    xs,
    ys,
    values,
    shape: Tuple[int, int],
    *,
    vmin: float = 0.0,
    vmax: float,
    cmap: ColormapLike = "inferno",
    sigma: float = 5.0,
    bloom_sigma: float = 22.0,
    bloom_gain: float = 0.55,
    gain: float = 1.0,
) -> np.ndarray:
    """Bin points additively and render as glow + bloom.

    Parameters
    ----------
    xs, ys : array-like
        Point locations in **pixel** coordinates (floats allowed).
    values : array-like
        Per-point weight (e.g. flash count, magnitude energy). Values
        accumulate **linearly** on the grid — overlapping events sum —
        then the blurred energy is colormapped.
    shape : (H, W)
        Output pixel grid.
    vmin, vmax : float
        Fixed energy range. Use reel-wide fixed values to avoid flicker.
    cmap : colormap name or Colormap.
    sigma : float
        Gaussian core width in pixels (the tight glow around each event).
    bloom_sigma, bloom_gain : float
        Width and relative strength of the wide halo pass.
    gain : float
        Global multiplier applied before normalization.

    Returns
    -------
    (H, W, 4) float64 RGBA in [0, 1]: straight (non-premultiplied) RGB
    from the colormapped blurred energy, alpha = blurred energy.
    Composite with normal alpha blending over a dark background.
    """
    h, w = int(shape[0]), int(shape[1])
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel() * gain
    if not (xs.shape == ys.shape == values.shape):
        raise ValueError("xs, ys, values must have the same shape")
    if not vmax > vmin:
        raise ValueError("vmax must be greater than vmin")
    if h <= 0 or w <= 0:
        raise ValueError("shape must be positive")

    grid = np.zeros((h, w), dtype=np.float64)
    if xs.size:
        ix = np.clip(np.round(xs).astype(int), 0, w - 1)
        iy = np.clip(np.round(ys).astype(int), 0, h - 1)
        np.add.at(grid, (iy, ix), values)

    return glow_from_grid(
        grid, vmin=vmin, vmax=vmax, cmap=cmap, sigma=sigma,
        bloom_sigma=bloom_sigma, bloom_gain=bloom_gain, gain=gain,
    )


def glow_from_grid(
    grid,
    *,
    vmin: float = 0.0,
    vmax: float,
    cmap: ColormapLike = "inferno",
    sigma: float = 5.0,
    bloom_sigma: float = 22.0,
    bloom_gain: float = 0.55,
    gain: float = 1.0,
) -> np.ndarray:
    """Render a pre-gridded energy field as glow + bloom (no binning).

    For data that already arrives aggregated per pixel/cell (e.g. GOES
    flash counts per 5 km cell rasterized to the render grid): values
    accumulate **linearly**, are blurred (core + bloom), then colormapped.
    The halo takes the colormap's dim colors instead of a smeared hue.

    Returns (H, W, 4) float64 RGBA: straight RGB from the colormapped
    blurred energy, alpha = blurred energy. Composite with normal alpha
    blending over a dark background. Deterministic.
    """
    grid = np.asarray(grid, dtype=np.float64) * gain
    if grid.ndim != 2:
        raise ValueError("grid must be a 2D array")
    if not vmax > vmin:
        raise ValueError("vmax must be greater than vmin")
    # Linear energy first (truly additive), blur, *then* colormap.
    energy = np.clip((grid - vmin) / (vmax - vmin), 0.0, 1.0)
    core = gaussian_filter(energy, sigma=sigma)
    bloom = gaussian_filter(energy, sigma=bloom_sigma) * bloom_gain
    e = np.clip(core + bloom, 0.0, 1.0)
    rgb = _as_cmap(cmap)(e)[..., :3]
    alpha = np.clip(core + bloom, 0.0, 1.0)
    return np.dstack([rgb, alpha])


def light_beam(
    shape: Tuple[int, int],
    x: float,
    y: float,
    angle_deg: float,
    length_px: float,
    width_px: float,
    *,
    color: Tuple[float, float, float] = (1.0, 0.95, 0.85),
    intensity: float = 1.0,
) -> np.ndarray:
    """Render a single glowing directional beam (lighthouse light).

    A wedge of light starting at ``(x, y)`` pointing ``angle_deg``
    (degrees, 0 = east, counter-clockwise positive), widening and fading
    with distance. Additive: sum several beams (or add to
    :func:`accumulate_glow` output) for a coastline of lights.
    """
    h, w = int(shape[0]), int(shape[1])
    if length_px <= 0 or width_px <= 0:
        raise ValueError("length_px and width_px must be positive")
    theta = np.deg2rad(angle_deg)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx = xx - x
    dy = yy - y
    along = dx * np.cos(theta) + dy * np.sin(theta)
    across = -dx * np.sin(theta) + dy * np.cos(theta)
    in_beam = (along >= 0.0) & (along <= length_px)
    spread = width_px * (0.35 + 0.65 * np.clip(along / length_px, 0.0, 1.0))
    envelope = np.exp(-((across / np.maximum(spread, 1e-9)) ** 2))
    falloff = np.exp(-along / (length_px * 0.7))
    beam = np.where(in_beam, envelope * falloff, 0.0) * intensity
    rgb = np.asarray(color, dtype=np.float64).reshape(1, 1, 3) * beam[..., None]
    alpha = np.clip(beam, 0.0, 1.0)
    return np.dstack([np.clip(rgb, 0.0, 1.0), alpha])
