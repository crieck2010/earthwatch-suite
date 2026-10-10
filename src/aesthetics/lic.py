"""Line-integral-convolution (LIC) streak textures for vector fields.

Renders a flow field ``(u, v)`` as a fine streak texture over a scalar
field (e.g. water temperature): **hue encodes the scalar, brightness
encodes speed** — the mapped.earth "week of currents" look.

Implementation: classic LIC via streamline integration, vectorized with
``scipy.ndimage.map_coordinates``. A seeded white-noise texture is
advected forward and backward along normalized streamlines and averaged
(box kernel); the result modulates a colormapped scalar field, with
per-pixel brightness scaled by speed.

Determinism: the noise texture comes from ``numpy.random.default_rng(seed)``;
identical inputs + seed give bit-identical output.
"""

from typing import Union

import matplotlib
import numpy as np
from scipy.ndimage import map_coordinates

matplotlib.use("Agg")

ColormapLike = Union[str, matplotlib.colors.Colormap]


def _as_cmap(cmap: ColormapLike) -> matplotlib.colors.Colormap:
    if isinstance(cmap, str):
        try:
            return matplotlib.colormaps[cmap]
        except KeyError as exc:
            raise ValueError(f"unknown colormap {cmap!r}") from exc
    return cmap


def lic_texture(
    u,
    v,
    scalar,
    *,
    vmin: float,
    vmax: float,
    cmap: ColormapLike = "turbo",
    seed: int = 0,
    speed_max: float = None,
    kernel: int = 12,
    step: float = 0.6,
    brightness_floor: float = 0.12,
    streak_contrast: float = 0.65,
) -> np.ndarray:
    """Render ``(u, v, scalar)`` grids as an LIC streak RGBA texture.

    Parameters
    ----------
    u, v, scalar : 2D arrays of identical shape (Ny, Nx).
        Vector components and the scalar used for hue. NaN marks missing
        data (transparent in the output).
    vmin, vmax : float
        Fixed scalar range for the colormap. **Pass reel-wide fixed values,
        not per-frame data min/max**, or the reel will flicker.
    cmap : colormap name or Colormap, default ``"turbo"``.
    seed : int
        RNG seed for the noise texture. Same seed -> identical texture.
    speed_max : float or None
        Speed that maps to full brightness. When None, the data max is
        used; for reels, pass a fixed value to avoid brightness flicker.
    kernel : int
        Streamline half-length in integration steps (streak length).
    step : float
        Integration step in pixels.
    brightness_floor : float
        Brightness of still water (0 = pitch black, 1 = full).
    streak_contrast : float
        How strongly the LIC texture modulates the color (0 = flat).

    Returns
    -------
    (Ny, Nx, 4) float64 RGBA array in [0, 1]; alpha is 0 where any input
    is NaN and 1 elsewhere.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    s = np.asarray(scalar, dtype=np.float64)
    if not (u.shape == v.shape == s.shape and u.ndim == 2):
        raise ValueError("u, v, scalar must be 2D arrays of identical shape")
    if not vmax > vmin:
        raise ValueError("vmax must be greater than vmin")
    if kernel < 1:
        raise ValueError("kernel must be >= 1")

    ny, nx = u.shape
    valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(s)
    speed = np.hypot(u, v)
    speed[~valid] = 0.0

    if speed_max is None:
        speed_max = float(np.max(speed)) if np.any(valid) else 1.0
    speed_max = max(float(speed_max), 1e-12)

    # Unit direction field; zero where invalid or still.
    denom = np.maximum(speed, 1e-12)
    uh = np.where(valid, u / denom, 0.0)
    vh = np.where(valid, v / denom, 0.0)

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((ny, nx))

    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)

    def _advect(sign: float) -> np.ndarray:
        px = xx.copy()
        py = yy.copy()
        acc = noise.copy()
        for _ in range(kernel):
            du = map_coordinates(uh, [py, px], order=1, mode="nearest")
            dv = map_coordinates(vh, [py, px], order=1, mode="nearest")
            px += sign * du * step
            py += sign * dv * step
            acc += map_coordinates(noise, [py, px], order=1, mode="nearest")
        return acc

    acc = _advect(1.0) + _advect(-1.0) - noise  # noise counted twice
    lic = acc / (2.0 * kernel + 1.0)

    # Robust normalization so a few extreme streaks don't wash the texture out.
    if np.any(valid):
        lo, hi = np.percentile(lic[valid], [1.0, 99.0])
    else:
        lo, hi = 0.0, 1.0
    if not hi > lo:
        hi = lo + 1e-12
    tex = np.clip((lic - lo) / (hi - lo), 0.0, 1.0)
    modulation = 1.0 - streak_contrast * (1.0 - tex)

    snorm = np.clip((s - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = _as_cmap(cmap)(snorm)[..., :3]
    bright = brightness_floor + (1.0 - brightness_floor) * np.clip(
        speed / speed_max, 0.0, 1.0
    )

    out = np.zeros((ny, nx, 3))
    out[valid] = (
        rgb[valid] * modulation[valid, None] * bright[valid, None]
    )
    rgba = np.dstack([np.clip(out, 0.0, 1.0), valid.astype(np.float64)])
    return rgba
