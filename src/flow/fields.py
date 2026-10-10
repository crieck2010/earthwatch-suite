"""Vector-field sampling: the spatial side of the flow engine.

A :class:`VectorField` is one timestep of surface currents on a regular
lat/lon grid: eastward/northward velocity (m/s) plus an optional scalar
grid (water temperature, degC by default). Land is represented as NaN
velocity — particles die on land and the renderer paints it as coastline.

A :class:`TimeVaryingField` blends consecutive timesteps so animation
frames can sample the flow at any fractional time ``t`` in ``[0, nt-1]``.

Inputs:
  * any object with the survey-currents ``CurrentField`` attributes
    (``u, v, temperature, times, lats, lons, crs, source``) — duck-typed,
    so survey-currents is not a hard dependency;
  * survey-currents 4-band per-timestep COGs (``speed, u, v, temperature``)
    via :meth:`VectorField.from_cog` (needs rasterio, lazy import).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

# Bands written by survey-currents' COG exporter (docs/INTEROP.md).
COG_BAND_NAMES = ("speed", "u", "v", "temperature")


def _bilinear(
    grid: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    qlons: np.ndarray,
    qlats: np.ndarray,
) -> np.ndarray:
    """NaN-aware bilinear interpolation of ``grid`` (ny, nx) at query points.

    Points outside the grid return NaN. Where some of the four surrounding
    corners are NaN (ragged coastlines), the valid corners are renormalized;
    where all four are NaN the result is NaN (treated as land).
    """
    grid = np.asarray(grid, dtype=float)
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    qlons = np.asarray(qlons, dtype=float).ravel()
    qlats = np.asarray(qlats, dtype=float).ravel()
    ny, nx = grid.shape
    out = np.full(qlons.shape, np.nan)

    in_bounds = (
        (qlons >= lons[0]) & (qlons <= lons[-1])
        & (qlats >= lats[0]) & (qlats <= lats[-1])
        & (nx >= 2) & (ny >= 2)
    )
    if not np.any(in_bounds):
        return out
    xl = qlons[in_bounds]
    yl = qlats[in_bounds]

    ix = np.clip(np.searchsorted(lons, xl, side="right") - 1, 0, nx - 2)
    iy = np.clip(np.searchsorted(lats, yl, side="right") - 1, 0, ny - 2)
    dx = lons[ix + 1] - lons[ix]
    dy = lats[iy + 1] - lats[iy]
    dx = np.where(dx == 0.0, 1.0, dx)
    dy = np.where(dy == 0.0, 1.0, dy)
    fx = (xl - lons[ix]) / dx
    fy = (yl - lats[iy]) / dy

    g00 = grid[iy, ix]
    g10 = grid[iy, ix + 1]
    g01 = grid[iy + 1, ix]
    g11 = grid[iy + 1, ix + 1]
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    acc = np.zeros_like(xl)
    wsum = np.zeros_like(xl)
    for g, w in ((g00, w00), (g10, w10), (g01, w01), (g11, w11)):
        valid = ~np.isnan(g)
        acc += np.where(valid, g * w, 0.0)
        wsum += np.where(valid, w, 0.0)
    vals = np.where(wsum > 0.0, acc / np.where(wsum == 0.0, 1.0, wsum), np.nan)
    out[in_bounds] = vals
    return out


class VectorField:
    """One timestep of surface currents on a regular lat/lon grid."""

    def __init__(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        scalar: Optional[np.ndarray] = None,
        scalar_name: str = "temperature",
        crs: str = "EPSG:4326",
        source: str = "",
        timestamp: str = "",
    ) -> None:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        if u.shape != v.shape or u.shape != (lats.size, lons.size):
            raise ValueError(
                f"u/v shape {u.shape}/{v.shape} must equal "
                f"(ny, nx)={((lats.size, lons.size))}"
            )
        if scalar is not None:
            scalar = np.asarray(scalar, dtype=float)
            if scalar.shape != u.shape:
                raise ValueError(
                    f"scalar shape {scalar.shape} must match u/v shape {u.shape}"
                )
        if not (np.all(np.diff(lats) > 0) and np.all(np.diff(lons) > 0)):
            raise ValueError("lats and lons must be strictly increasing")
        self.u = u
        self.v = v
        self.lats = lats
        self.lons = lons
        self.scalar = scalar
        self.scalar_name = scalar_name
        self.crs = crs
        self.source = source
        self.timestamp = timestamp
        # Land where velocity is missing.
        self.landmask = np.isnan(u) | np.isnan(v)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_current_field(cls, cf: Any, index: int = 0,
                           scalar: str = "temperature") -> "VectorField":
        """Build from a survey-currents ``CurrentField`` (duck-typed).

        ``scalar`` selects the overlay grid: ``"temperature"`` (default) or
        ``"speed"``. A field without temperature falls back to speed.
        """
        nt = len(cf.times)
        if not 0 <= index < nt:
            raise IndexError(f"index {index} out of range for {nt} timesteps")
        if scalar == "temperature" and getattr(cf, "temperature", None) is None:
            scalar = "speed"
        grid: Optional[np.ndarray] = None
        if scalar == "temperature":
            grid = np.asarray(cf.temperature[index], dtype=float)
        elif scalar == "speed":
            with np.errstate(invalid="ignore"):
                grid = np.sqrt(
                    np.asarray(cf.u[index], dtype=float) ** 2
                    + np.asarray(cf.v[index], dtype=float) ** 2
                )
        else:
            raise ValueError(f"unknown scalar {scalar!r}; want 'temperature' or 'speed'")
        return cls(
            u=np.asarray(cf.u[index], dtype=float),
            v=np.asarray(cf.v[index], dtype=float),
            lats=np.asarray(cf.lats, dtype=float),
            lons=np.asarray(cf.lons, dtype=float),
            scalar=grid,
            scalar_name=scalar,
            crs=getattr(cf, "crs", "EPSG:4326"),
            source=getattr(cf, "source", ""),
            timestamp=cf.times[index],
        )

    @classmethod
    def from_cog(cls, path: str, scalar: str = "temperature") -> "VectorField":
        """Build from a survey-currents 4-band COG (needs rasterio).

        Bands are ``speed, u, v, temperature``. Lats/lons are derived from
        the GeoTransform; the file must be a north-up regular grid.
        """
        try:
            import rasterio
        except ImportError as exc:
            raise ImportError(
                "from_cog needs rasterio (pip install 'survey-flow[raster]')"
            ) from exc
        if scalar not in COG_BAND_NAMES:
            raise ValueError(f"unknown scalar {scalar!r}; bands are {COG_BAND_NAMES}")
        with rasterio.open(path) as dst:
            if dst.count < 4:
                raise ValueError(
                    f"{path}: expected >= 4 bands (speed, u, v, temperature), "
                    f"got {dst.count}"
                )
            arr = dst.read()  # (4, ny, nx)
            transform = dst.transform
            crs = str(dst.crs) if dst.crs else "EPSG:4326"
            tags = dst.tags()
        band = {name: arr[i] for i, name in enumerate(COG_BAND_NAMES)}
        ny, nx = band["u"].shape
        # North-up regular grid: derive cell-center coordinates.
        xs = transform.c + (np.arange(nx) + 0.5) * transform.a
        ys = transform.f + (np.arange(ny) + 0.5) * transform.e
        if transform.e > 0:  # south-up; flip to increasing latitude
            ys = ys[::-1]
            band = {k: v[::-1, :] for k, v in band.items()}
        return cls(
            u=band["u"], v=band["v"], lats=ys, lons=xs,
            scalar=band[scalar], scalar_name=scalar,
            crs=crs, source=f"cog:{path}",
            timestamp=tags.get("timestamp", ""),
        )

    # -- sampling ------------------------------------------------------------

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat)."""
        return (float(self.lons[0]), float(self.lats[0]),
                float(self.lons[-1]), float(self.lats[-1]))

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.lats.size, self.lons.size)

    def sample(self, qlons: np.ndarray, qlats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Bilinear (u, v) at query points; NaN off-grid or on land."""
        return (
            _bilinear(self.u, self.lats, self.lons, qlons, qlats),
            _bilinear(self.v, self.lats, self.lons, qlons, qlats),
        )

    def sample_scalar(self, qlons: np.ndarray, qlats: np.ndarray) -> np.ndarray:
        """Bilinear scalar (temperature/speed) at query points."""
        if self.scalar is None:
            raise ValueError("field has no scalar grid")
        return _bilinear(self.scalar, self.lats, self.lons, qlons, qlats)

    def is_land(self, qlons: np.ndarray, qlats: np.ndarray) -> np.ndarray:
        """True where the query point is on land (or off-grid)."""
        qlons = np.asarray(qlons, dtype=float).ravel()
        qlats = np.asarray(qlats, dtype=float).ravel()
        out = np.ones(qlons.shape, dtype=bool)
        in_bounds = (
            (qlons >= self.lons[0]) & (qlons <= self.lons[-1])
            & (qlats >= self.lats[0]) & (qlats <= self.lats[-1])
        )
        if np.any(in_bounds):
            ix = np.clip(
                np.searchsorted(self.lons, qlons[in_bounds], side="right") - 1,
                0, self.lons.size - 1)
            iy = np.clip(
                np.searchsorted(self.lats, qlats[in_bounds], side="right") - 1,
                0, self.lats.size - 1)
            out[in_bounds] = self.landmask[iy, ix]
        return out

    def in_bounds(self, qlons: np.ndarray, qlats: np.ndarray) -> np.ndarray:
        qlons = np.asarray(qlons, dtype=float).ravel()
        qlats = np.asarray(qlats, dtype=float).ravel()
        return (
            (qlons >= self.lons[0]) & (qlons <= self.lons[-1])
            & (qlats >= self.lats[0]) & (qlats <= self.lats[-1])
        )

    def scalar_range(self) -> Tuple[float, float]:
        """(nanmin, nanmax) of the scalar grid."""
        if self.scalar is None:
            raise ValueError("field has no scalar grid")
        with np.errstate(invalid="ignore"):
            return (float(np.nanmin(self.scalar)), float(np.nanmax(self.scalar)))

    def mean_speed(self) -> float:
        with np.errstate(invalid="ignore"):
            return float(np.nanmean(np.sqrt(self.u ** 2 + self.v ** 2)))


class TimeVaryingField:
    """A time-ordered stack of :class:`VectorField` with fractional sampling.

    ``at(t)`` blends the two bracketing timesteps so animation frames sample
    a smooth flow even when model output is hourly.
    """

    def __init__(self, fields: Sequence[VectorField], times: Sequence[str]) -> None:
        fields = list(fields)
        times = list(times)
        if len(fields) != len(times) or len(fields) == 0:
            raise ValueError("fields and times must be non-empty and the same length")
        ref = fields[0]
        for f in fields[1:]:
            if f.shape != ref.shape or f.scalar_name != ref.scalar_name:
                raise ValueError("all timesteps must share grid shape and scalar")
        self.fields = fields
        self.times = times

    @classmethod
    def from_current_field(cls, cf: Any, scalar: str = "temperature") -> "TimeVaryingField":
        fields = [VectorField.from_current_field(cf, i, scalar=scalar)
                  for i in range(len(cf.times))]
        return cls(fields, list(cf.times))

    @classmethod
    def from_cogs(cls, paths: Sequence[str], times: Sequence[str],
                  scalar: str = "temperature") -> "TimeVaryingField":
        fields = [VectorField.from_cog(p, scalar=scalar) for p in paths]
        return cls(fields, list(times))

    @property
    def nt(self) -> int:
        return len(self.fields)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return self.fields[0].bounds

    def at(self, t: float) -> VectorField:
        """Blend to fractional time ``t`` in ``[0, nt-1]`` (clamped)."""
        t = float(np.clip(t, 0.0, self.nt - 1))
        k0 = int(np.floor(t))
        k1 = min(k0 + 1, self.nt - 1)
        alpha = t - k0
        if k0 == k1 or alpha == 0.0:
            return self.fields[k0]
        a, b = self.fields[k0], self.fields[k1]
        with np.errstate(invalid="ignore"):
            u = a.u * (1.0 - alpha) + b.u * alpha
            v = a.v * (1.0 - alpha) + b.v * alpha
            scalar = None
            if a.scalar is not None and b.scalar is not None:
                scalar = a.scalar * (1.0 - alpha) + b.scalar * alpha
        # Blend timestamps by picking the nearer step's label.
        stamp = a.timestamp if alpha < 0.5 else b.timestamp
        return VectorField(u=u, v=v, lats=a.lats, lons=a.lons,
                           scalar=scalar, scalar_name=a.scalar_name,
                           crs=a.crs, source=a.source, timestamp=stamp)

    def seed_bounds(self) -> Tuple[float, float, float, float]:
        return self.bounds
