"""Surface compositing: a colored density field over land, with shadow.

:func:`render_surface` is the mapped.earth population-density look as a
single pure function: a 2D scalar grid is resampled to an exact pixel
size, colormapped (the engine's ``teal_pink`` ramp by default), dressed
with thin data-driven contour lines, and composited over a near-black
teal background with a soft drop shadow of the land (or of the data
footprint) underneath. No-data cells are honest: inside ``land_mask``
they render as flat dark grey, outside land they are transparent and
the background/shadow show through.

Dependency rule: this module is numpy + matplotlib only. All resampling
and the NaN-aware Gaussian smoothing are hand-rolled numpy here — no
scipy — so the engine stays dependency-light.

Determinism: no RNG anywhere. Identical inputs give byte-identical
uint8 RGBA (for a pinned matplotlib version, the caveat the whole
engine carries).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from aesthetics.colormaps import get_colormap

# Fixed feather for the drop shadow, in output pixels. The public
# signature exposes shadow offset/alpha but no blur control, so the
# blur is a house constant (documented, deterministic) rather than a
# hidden default that changes between calls.
_SHADOW_BLUR_SIGMA = 3.0


def _as_rgba(value, name: str) -> np.ndarray:
    """Validate an RGB/RGBA color to a float64 array of 4 in [0, 1]."""
    try:
        arr = np.asarray(value, dtype=np.float64).ravel()
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be an RGB or RGBA tuple") from exc
    if arr.shape == (3,):
        arr = np.concatenate([arr, [1.0]])
    if arr.shape != (4,) or not np.all(np.isfinite(arr)) \
            or np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(
            f"{name} must be RGB/RGBA channels in [0, 1], got {value!r}")
    return arr


def _resample_nearest(mask: np.ndarray, height_px: int,
                      width_px: int) -> np.ndarray:
    """Nearest-neighbor resample of a 2D bool array to (H, W).

    Pixel-center mapping, the same convention as
    ``aesthetics.strands._resample_mask``: output pixel ``(j, i)``
    samples the input cell its center falls in. Row 0 is the top in
    both input and output, so no flip is needed.
    """
    my, mx = mask.shape
    jj = np.clip((((np.arange(height_px) + 0.5) / height_px) * my
                  ).astype(int), 0, my - 1)
    ii = np.clip((((np.arange(width_px) + 0.5) / width_px) * mx
                  ).astype(int), 0, mx - 1)
    return mask[jj[:, None], ii[None, :]]


def _resample_bilinear_nanaware(grid: np.ndarray, valid: np.ndarray,
                                height_px: int, width_px: int):
    """Bilinear resample to (H, W) that never invents data over gaps.

    Pixel-center mapping: output pixel ``(j, i)`` maps to input
    coordinates ``((j + 0.5) * ny / H - 0.5, (i + 0.5) * nx / W - 0.5)``.
    The four bilinear corner values are averaged with weights
    renormalized over the *valid corners only* (a NaN hole contributes
    nothing to its finite neighbors' values). But the output data mask
    itself is the **nearest-neighbor** resample of the input mask, so a
    pixel whose center falls in a no-data cell stays no-data: color
    never bleeds into a gap, and gaps keep their honest shape.

    Returns ``(values, valid_out)`` with NaN wherever ``valid_out`` is
    False.
    """
    ny, nx = grid.shape
    y = (np.arange(height_px) + 0.5) * ny / height_px - 0.5
    x = (np.arange(width_px) + 0.5) * nx / width_px - 0.5
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    wy1 = (y - y0)[:, None]
    wx1 = (x - x0)[None, :]
    num = np.zeros((height_px, width_px), dtype=np.float64)
    den = np.zeros((height_px, width_px), dtype=np.float64)
    for dy in (0, 1):
        yc = np.clip(y0 + dy, 0, ny - 1)
        wy = wy1 if dy else 1.0 - wy1
        for dx in (0, 1):
            xc = np.clip(x0 + dx, 0, nx - 1)
            wx = wx1 if dx else 1.0 - wx1
            corner_valid = valid[yc[:, None], xc[None, :]]
            corner_val = np.where(
                corner_valid, grid[yc[:, None], xc[None, :]], 0.0)
            w = wy * wx * corner_valid
            num += w * corner_val
            den += w
    values = np.where(den > 1e-12, num / np.maximum(den, 1e-12), np.nan)
    valid_out = _resample_nearest(valid, height_px, width_px)
    values = np.where(valid_out, values, np.nan)
    return values, valid_out


def _gaussian_kernel(sigma: float) -> np.ndarray:
    """1D Gaussian kernel, radius ceil(3*sigma), sum-normalized."""
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def _convolve_axis(arr: np.ndarray, kernel: np.ndarray,
                   axis: int) -> np.ndarray:
    """Convolve along one axis with zero padding (numpy only)."""
    radius = kernel.size // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(arr, pad_width, mode="constant")
    out = np.zeros_like(arr, dtype=np.float64)
    for tap, weight in enumerate(kernel):
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(tap, tap + arr.shape[axis])
        out += weight * padded[tuple(sl)]
    return out


def _gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur of a plain float array (zero padding)."""
    if sigma <= 0:
        return np.array(arr, dtype=np.float64, copy=True)
    kernel = _gaussian_kernel(sigma)
    return _convolve_axis(_convolve_axis(arr, kernel, 0), kernel, 1)


def _gaussian_smooth_nanaware(values: np.ndarray, valid: np.ndarray,
                              sigma: float) -> np.ndarray:
    """Gaussian smoothing that respects no-data cells exactly.

    Renormalized convolution: both the zero-filled values and the
    validity mask are blurred with the same separable Gaussian, and
    the result is their ratio — so a finite cell surrounded by no-data
    keeps (approximately) its own value instead of being dragged
    toward zero. The output is then re-masked with the *original*
    validity mask: no-data cells remain no-data after smoothing. Data
    never bleeds into a gap (the mapped.earth rule: gaps are gaps,
    not interpolation opportunities).
    """
    if sigma <= 0:
        return np.where(valid, values, np.nan)
    kernel = _gaussian_kernel(sigma)
    zeroed = np.where(valid, values, 0.0)
    mask_f = valid.astype(np.float64)
    num = _convolve_axis(_convolve_axis(zeroed, kernel, 0), kernel, 1)
    den = _convolve_axis(_convolve_axis(mask_f, kernel, 0), kernel, 1)
    smoothed = np.where(den > 1e-12, num / np.maximum(den, 1e-12), np.nan)
    return np.where(valid, smoothed, np.nan)


def _shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a 2D array by (dy rows down, dx cols right), zero-filled."""
    out = np.zeros_like(arr)
    h, w = arr.shape
    src_y = slice(max(0, -dy), h - max(0, dy))
    dst_y = slice(max(0, dy), h - max(0, -dy))
    src_x = slice(max(0, -dx), w - max(0, dx))
    dst_x = slice(max(0, dx), w - max(0, -dx))
    out[dst_y, dst_x] = arr[src_y, src_x]
    return out


def _over(base: np.ndarray, fg_rgb: np.ndarray,
          fg_alpha: np.ndarray) -> np.ndarray:
    """Alpha-over composite of an RGB+alpha layer onto an RGBA canvas."""
    fa = fg_alpha
    ba = base[..., 3:4]
    out_a = fa + ba * (1.0 - fa)
    safe_a = np.maximum(out_a, 1e-12)
    out_rgb = np.where(
        out_a > 0.0,
        (fg_rgb * fa + base[..., :3] * ba * (1.0 - fa)) / safe_a,
        0.0,
    )
    return np.concatenate([out_rgb, out_a], axis=2)


def _apply_shadow(canvas: np.ndarray, silhouette: np.ndarray,
                  offset_px, alpha: float) -> np.ndarray:
    """Composite the blurred, offset black silhouette under the surface."""
    if not silhouette.any():
        return canvas
    blurred = _gaussian_blur(silhouette.astype(np.float64),
                             _SHADOW_BLUR_SIGMA)
    shifted = _shift(blurred, int(offset_px[1]), int(offset_px[0]))
    shadow_alpha = np.clip(alpha * shifted, 0.0, 1.0)[..., None]
    black = np.zeros(canvas.shape[:2] + (3,), dtype=np.float64)
    return _over(canvas, black, shadow_alpha)


def _auto_contour_levels(vmin: float, vmax: float, scale: str,
                         explicit) -> np.ndarray:
    """Contour levels: explicit array, or auto data-scale levels.

    Auto levels are up to 7 interior levels between ``vmin`` and
    ``vmax`` (which are data-driven when not caller-fixed): equally
    spaced for ``linear``, log-spaced (``geomspace``) for ``log``.
    """
    if explicit is not None:
        levels = np.asarray(explicit, dtype=np.float64).ravel()
        if levels.size and (not np.all(np.isfinite(levels))
                            or np.any(np.diff(levels) <= 0)):
            raise ValueError(
                "contour_levels must be finite and strictly increasing")
        return levels
    if scale == "log":
        return np.geomspace(vmin, vmax, num=9)[1:-1]
    return np.linspace(vmin, vmax, num=9)[1:-1]


def render_surface(
    grid,                              # (ny, nx) float array; NaN = no data
    *,
    width_px: int,
    height_px: int,
    cmap: str = "teal_pink",
    vmin=None, vmax=None,              # default: finite nanmin/nanmax of grid
    scale: str = "linear",             # "linear" | "log"
    smoothing: float = 0.0,            # Gaussian sigma in OUTPUT pixels; 0 = off
    contours: bool = True,
    contour_levels=None,               # None = auto data-driven (max ~8); or explicit array
    contour_color=(1.0, 1.0, 1.0, 0.35),
    contour_linewidth: float = 0.6,
    shadow: bool = True,
    land_mask=None,                    # (ny, nx) bool, same shape as grid
    shadow_offset_px=(6, 10),
    shadow_alpha: float = 0.45,
    background=(0.012, 0.043, 0.055, 1.0),   # near-black teal
    nodata=(0.16, 0.16, 0.16, 1.0),          # flat dark grey
) -> np.ndarray:  # (height_px, width_px, 4) uint8 RGBA
    """Render a scalar grid as a colored surface over shadowed land.

    Parameters
    ----------
    grid : (ny, nx) array-like of float
        Data grid, row 0 = top. NaN (and, for ``scale="log"``,
        non-positive values; also +/-inf) means *no data* — never
        invented, never interpolated over.
    width_px, height_px : int
        Exact output size. The output is ``(height_px, width_px, 4)``
        uint8 RGBA.
    cmap : str or Colormap, default ``"teal_pink"``
        Any matplotlib colormap name (plus the engine's registered
        ``teal_pink``), or a Colormap object passed through.
    vmin, vmax : float or None
        Color range. ``None`` (default) uses the finite min/max of the
        valid data (for ``log``, of the positive data). Like every
        engine layer, pass **reel-wide fixed values** for animation or
        the reel will flicker. ``vmax > vmin`` is required.
    scale : "linear" | "log"
        ``"log"`` applies a log normalization — for density-like data
        spanning orders of magnitude. Non-positive values are treated
        as no-data (documented, by design), and auto contour levels are
        log-spaced. ``vmin`` must be positive for ``log``. A constant
        valid field (auto ``vmin == vmax``) is expanded to a unit
        (linear) or decade (log) range instead of raising; an
        explicitly passed ``vmax <= vmin`` does raise.
    smoothing : float
        Gaussian sigma in **output pixels**; ``0`` = off. NaN-aware
        renormalized convolution (see module notes): no-data cells
        stay no-data, and data does not bleed into them.
    contours : bool
        Draw thin contour lines at data-driven levels over the colored
        field, suppressed over no-data cells.
    contour_levels : array-like or None
        ``None`` = auto (up to 7 interior levels between vmin/vmax,
        log-spaced for ``log``). Or an explicit, finite, strictly
        increasing array of levels in data units.
    contour_color, contour_linewidth
        Line color (RGBA in [0, 1]) and width in points.
    shadow : bool
        Composite a soft drop shadow under the surface: the silhouette
        of ``land_mask`` (or, when ``land_mask`` is None, of the
        finite-cell footprint), blurred with a fixed 3 px Gaussian,
        offset by ``shadow_offset_px``, at ``shadow_alpha`` strength.
    land_mask : (ny, nx) bool array or None
        Same shape as ``grid``; True = land. NaN cells on land render
        as the flat ``nodata`` grey; NaN cells off land (or everywhere
        when ``land_mask`` is None) are transparent, so the
        background/shadow show through. Finite cells are colormapped
        regardless of the mask.
    shadow_offset_px : (dx, dy)
        Shadow offset in output pixels: right and **down** (image
        coordinates, so the shadow falls below-right of the land).
    shadow_alpha : float in [0, 1]
        Peak shadow opacity.
    background, nodata : RGBA tuples in [0, 1]
        Canvas color (a near-black teal by default) and the flat grey
        for no-data-on-land.

    Returns
    -------
    (height_px, width_px, 4) uint8 RGBA. Compositing order is
    background -> shadow -> surface (contours merged into the surface
    layer). With the default opaque background the output is opaque.

    Raises
    ------
    ValueError
        For a non-2D or empty ``grid``, a ``land_mask`` of the wrong
        shape, non-positive output size, ``scale`` not in
        ``{"linear", "log"}``, ``vmax <= vmin``, a non-positive
        ``vmin`` for ``log``, negative ``smoothing``, an unknown
        colormap, or malformed colors/levels.

    Notes
    -----
    Resampling is bilinear with renormalized weights over valid
    corners, but the output data mask is nearest-neighbor resampled —
    see :func:`_resample_bilinear_nanaware`. Contours are drawn with
    matplotlib on an offscreen figure at the exact pixel size (the
    same canvas pattern as ``render_strands``), then masked to valid
    cells before compositing, so no contour ink ever lands on a
    no-data pixel.

    An **all-NaN grid is not an error**: with ``land_mask`` it renders
    grey land over background + shadow; without, it renders the plain
    background. (An *empty* grid — zero cells — does raise.)

    Determinism: identical inputs give byte-identical output. No RNG.
    """
    # --- validate the scalar arguments --------------------------------
    if isinstance(width_px, bool) or isinstance(height_px, bool) \
            or not isinstance(width_px, (int, np.integer)) \
            or not isinstance(height_px, (int, np.integer)) \
            or int(width_px) <= 0 or int(height_px) <= 0:
        raise ValueError(
            f"width_px and height_px must be positive integers, "
            f"got {width_px!r} x {height_px!r}")
    width_px, height_px = int(width_px), int(height_px)
    if scale not in ("linear", "log"):
        raise ValueError(
            f'scale must be "linear" or "log", got {scale!r}')
    try:
        smoothing = float(smoothing)
    except (TypeError, ValueError) as exc:
        raise ValueError("smoothing must be a number") from exc
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError(f"smoothing must be finite and >= 0, got {smoothing}")
    bg_rgba = _as_rgba(background, "background")
    nodata_rgba = _as_rgba(nodata, "nodata")
    contour_rgba = _as_rgba(contour_color, "contour_color")
    if not np.isfinite(float(contour_linewidth)) \
            or float(contour_linewidth) < 0:
        raise ValueError(
            f"contour_linewidth must be finite and >= 0, "
            f"got {contour_linewidth!r}")
    try:
        offset = tuple(shadow_offset_px)
    except TypeError as exc:
        raise ValueError("shadow_offset_px must be a (dx, dy) pair") from exc
    if len(offset) != 2 or not all(
            isinstance(v, (int, np.integer)) and not isinstance(v, bool)
            for v in offset):
        raise ValueError(
            f"shadow_offset_px must be a (dx, dy) pair of ints, "
            f"got {shadow_offset_px!r}")
    if not (0.0 <= float(shadow_alpha) <= 1.0):
        raise ValueError(
            f"shadow_alpha must be in [0, 1], got {shadow_alpha!r}")
    cmap_obj = get_colormap(cmap)

    # --- validate the grid and mask ------------------------------------
    try:
        data = np.asarray(grid, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("grid must be a 2D float array") from exc
    if data.ndim != 2:
        raise ValueError(f"grid must be 2D, got shape {data.shape}")
    if data.size == 0:
        raise ValueError("grid must be non-empty (got zero cells)")
    land = None
    if land_mask is not None:
        land = np.asarray(land_mask, dtype=bool)
        if land.shape != data.shape:
            raise ValueError(
                f"land_mask must have the same shape as grid "
                f"{data.shape}, got {land.shape}")

    # Valid = finite, and (for log) strictly positive.
    valid = np.isfinite(data)
    if scale == "log":
        valid &= data > 0.0

    bg_tile = np.broadcast_to(
        bg_rgba, (height_px, width_px, 4)).astype(np.float64).copy()
    land_px = _resample_nearest(land, height_px, width_px) \
        if land is not None else None

    def _finish(surface: np.ndarray, silhouette: np.ndarray) -> np.ndarray:
        canvas = bg_tile
        if shadow:
            canvas = _apply_shadow(canvas, silhouette, offset,
                                   float(shadow_alpha))
        canvas = _over(canvas, surface[..., :3], surface[..., 3:4])
        return np.rint(np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)

    # --- all-NaN (or all-non-positive for log): graceful, not an error --
    if not valid.any():
        silhouette = land_px if land_px is not None else _resample_nearest(
            valid, height_px, width_px)
        surface = np.zeros((height_px, width_px, 4), dtype=np.float64)
        if land_px is not None:
            surface[land_px] = nodata_rgba
        return _finish(surface, silhouette)

    # --- color range ----------------------------------------------------
    finite_vals = data[valid]
    lo = float(np.min(finite_vals)) if vmin is None else float(vmin)
    hi = float(np.max(finite_vals)) if vmax is None else float(vmax)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"vmin/vmax must be finite, got {lo}..{hi}")
    if scale == "log" and lo <= 0.0:
        raise ValueError(
            f'scale="log" requires a positive vmin, got {lo}')
    if hi < lo:
        raise ValueError(f"need vmax > vmin, got vmin={lo}, vmax={hi}")
    if hi == lo:
        # A constant valid field has no range to normalize by. Raising
        # would make a single finite cell (or a flat density field)
        # unrenderable, so the auto side is expanded instead; only an
        # explicitly caller-fixed degenerate range is an error.
        if vmin is not None and vmax is not None:
            raise ValueError(f"need vmax > vmin, got vmin={lo}, vmax={hi}")
        if scale == "log":
            if vmin is not None:
                hi = lo * 10.0
            elif vmax is not None:
                lo = hi / 10.0
            else:
                lo, hi = lo / np.sqrt(10.0), hi * np.sqrt(10.0)
        else:
            if vmin is not None:
                hi = lo + 1.0
            elif vmax is not None:
                lo = hi - 1.0
            else:
                lo, hi = lo - 0.5, hi + 0.5

    # --- resample + smooth ----------------------------------------------
    values_px, valid_px = _resample_bilinear_nanaware(
        data, valid, height_px, width_px)
    if smoothing > 0:
        values_px = _gaussian_smooth_nanaware(values_px, valid_px, smoothing)

    # --- colormap the surface -------------------------------------------
    if scale == "log":
        t = (np.log(np.where(valid_px, values_px, 1.0)) - np.log(lo)) \
            / (np.log(hi) - np.log(lo))
    else:
        t = (np.where(valid_px, values_px, lo) - lo) / (hi - lo)
    t = np.clip(t, 0.0, 1.0)

    surface = np.zeros((height_px, width_px, 4), dtype=np.float64)
    if valid_px.any():
        colored = cmap_obj(t[valid_px])  # (n, 4)
        colored[:, 3] = 1.0  # finite cells are opaque, per the contract
        surface[valid_px] = colored
    if land_px is not None:
        grey = (~valid_px) & land_px
        surface[grey] = nodata_rgba

    # --- contours over the colored field, never over no-data -------------
    levels = _auto_contour_levels(lo, hi, scale, contour_levels)
    if contours and levels.size and float(contour_linewidth) > 0:
        dpi = 100
        fig = plt.figure(
            figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
        fig.patch.set_facecolor((0.0, 0.0, 0.0, 0.0))
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        ax.set_facecolor((0.0, 0.0, 0.0, 0.0))
        ax.set_xlim(0.0, width_px)
        ax.set_ylim(0.0, height_px)
        # Flip vertically: buffer row 0 is the top, contour y grows up.
        z_plot = np.ma.masked_where(~valid_px[::-1], values_px[::-1])
        xs = np.arange(width_px) + 0.5
        ys = np.arange(height_px) + 0.5
        ax.contour(xs, ys, z_plot, levels=levels,
                   colors=[tuple(contour_rgba)],
                   linewidths=float(contour_linewidth))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba(),
                         dtype=np.float64) / 255.0
        plt.close(fig)
        ink_a = np.where(valid_px[..., None], buf[..., 3:4], 0.0)
        surface[..., :3] = buf[..., :3] * ink_a \
            + surface[..., :3] * (1.0 - ink_a)
        surface[..., 3:4] = ink_a + surface[..., 3:4] * (1.0 - ink_a)

    silhouette = land_px if land_px is not None else valid_px
    return _finish(surface, silhouette)
