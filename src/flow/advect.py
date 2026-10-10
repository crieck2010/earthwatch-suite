"""Particle advection: move tracers through the vector field over time.

Positions are longitude/latitude degrees; velocities are m/s, converted
with a spherical-earth approximation. Integration is vectorized over all
particles. Trails are ring buffers so the renderer can draw fading streaks.

Everything here is stdlib+numpy — no rendering dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from flow.fields import TimeVaryingField, VectorField

EARTH_RADIUS_M = 6_371_000.0


def velocity_to_deg(u: np.ndarray, v: np.ndarray, lat_deg: np.ndarray,
                    dt_seconds: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (u, v) m/s to (dlon, dlat) degrees over ``dt_seconds``."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    lat_deg = np.asarray(lat_deg, dtype=float)
    dlat = np.degrees(v * dt_seconds / EARTH_RADIUS_M)
    coslat = np.cos(np.radians(np.clip(lat_deg, -89.9, 89.9)))
    coslat = np.where(coslat < 1e-6, 1e-6, coslat)
    dlon = np.degrees(u * dt_seconds / (EARTH_RADIUS_M * coslat))
    return dlon, dlat


@dataclass
class AdvectionConfig:
    """Tuning knobs for the particle system."""

    dt_seconds: float = 3600.0   # model-time advanced per animation frame
    method: str = "rk2"          # "rk2" (midpoint, default) or "rk4"
    trail_length: int = 12       # positions kept per particle for streaks
    max_age_frames: int = 240    # respawn after this many frames
    respawn: bool = True         # reseed dead particles (keeps density constant)
    seed: int = 7                # deterministic seeding

    def __post_init__(self) -> None:
        if self.method not in ("rk2", "rk4"):
            raise ValueError(f"method must be 'rk2' or 'rk4', got {self.method!r}")
        if self.trail_length < 2:
            raise ValueError("trail_length must be >= 2")
        if self.dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive")


def sample_water_positions(n: int, field: VectorField,
                           rng: np.random.Generator,
                           bounds: Optional[Tuple[float, float, float, float]] = None,
                           max_tries: int = 50) -> np.ndarray:
    """Uniform random positions over water inside ``bounds``; shape (n, 2).

    Rejection-samples against the field landmask. Raises ValueError if the
    field has no water inside the bounds.
    """
    minx, miny, maxx, maxy = bounds or field.bounds
    if maxx <= minx or maxy <= miny:
        raise ValueError(f"degenerate bounds {(minx, miny, maxx, maxy)}")
    got = np.zeros((0, 2))
    tries = 0
    while got.shape[0] < n and tries < max_tries:
        tries += 1
        batch = max(n * 4, 64)
        cand = np.column_stack([
            rng.uniform(minx, maxx, batch),
            rng.uniform(miny, maxy, batch),
        ])
        water = ~field.is_land(cand[:, 0], cand[:, 1])
        got = np.vstack([got, cand[water]]) if got.size else cand[water]
    if got.shape[0] < n:
        raise ValueError(
            f"could not seed {n} water particles in {(minx, miny, maxx, maxy)} "
            f"after {max_tries} tries (field may be all land)"
        )
    return got[:n]


class ParticleSet:
    """A set of tracer particles with trail history and respawn rules."""

    def __init__(self, n_particles: int, field: VectorField,
                 config: Optional[AdvectionConfig] = None,
                 seed: Optional[int] = None) -> None:
        if n_particles <= 0:
            raise ValueError("n_particles must be positive")
        self.config = config or AdvectionConfig()
        self.rng = np.random.default_rng(
            self.config.seed if seed is None else seed)
        self.positions = sample_water_positions(
            n_particles, field, self.rng)                       # (n, 2) lon/lat
        self.trails = np.repeat(self.positions[:, None, :],
                                self.config.trail_length, axis=1).copy()
        self.ages = np.zeros(n_particles, dtype=int)
        self.n_respawned_total = 0

    @property
    def n(self) -> int:
        return self.positions.shape[0]

    def _step_rk2(self, field: VectorField, dt: float) -> None:
        lon, lat = self.positions[:, 0], self.positions[:, 1]
        u1, v1 = field.sample(lon, lat)
        dlon1, dlat1 = velocity_to_deg(u1, v1, lat, dt * 0.5)
        mlon, mlat = lon + dlon1, lat + dlat1
        u2, v2 = field.sample(mlon, mlat)
        dlon, dlat = velocity_to_deg(u2, v2, mlat, dt)
        self.positions[:, 0] = lon + np.where(np.isnan(dlon), 0.0, dlon)
        self.positions[:, 1] = lat + np.where(np.isnan(dlat), 0.0, dlat)
        # Particles whose midpoint or endpoint velocity was NaN die.
        dead_velocity = np.isnan(u1) | np.isnan(v1) | np.isnan(u2) | np.isnan(v2)
        self._respawn(dead_velocity, field)

    def _rate(self, field: VectorField, lon: np.ndarray,
                lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray]:
        """(dlon, dlat) degrees per second plus raw (u, v) for NaN checks."""
        u, v = field.sample(lon, lat)
        dlon, dlat = velocity_to_deg(u, v, lat, 1.0)
        return dlon, dlat, u, v

    def _step_rk4(self, field: VectorField, dt: float) -> None:
        lon, lat = self.positions[:, 0], self.positions[:, 1]
        r1lon, r1lat, u1, v1 = self._rate(field, lon, lat)
        r2lon, r2lat, u2, v2 = self._rate(
            field, lon + r1lon * dt / 2.0, lat + r1lat * dt / 2.0)
        r3lon, r3lat, u3, v3 = self._rate(
            field, lon + r2lon * dt / 2.0, lat + r2lat * dt / 2.0)
        r4lon, r4lat, u4, v4 = self._rate(
            field, lon + r3lon * dt, lat + r3lat * dt)
        dlon = (r1lon + 2.0 * r2lon + 2.0 * r3lon + r4lon) * dt / 6.0
        dlat = (r1lat + 2.0 * r2lat + 2.0 * r3lat + r4lat) * dt / 6.0
        dead_velocity = (np.isnan(u1) | np.isnan(v1) | np.isnan(u2)
                         | np.isnan(v2) | np.isnan(u3) | np.isnan(v3)
                         | np.isnan(u4) | np.isnan(v4))
        self.positions[:, 0] = lon + np.where(np.isnan(dlon), 0.0, dlon)
        self.positions[:, 1] = lat + np.where(np.isnan(dlat), 0.0, dlat)
        self._respawn(dead_velocity, field)

    def _respawn(self, dead_velocity: np.ndarray, field: VectorField) -> int:
        """Respawn dead particles; return the count respawned this step."""
        cfg = self.config
        dead = (dead_velocity
                | ~field.in_bounds(self.positions[:, 0], self.positions[:, 1])
                | field.is_land(self.positions[:, 0], self.positions[:, 1])
                | (self.ages >= cfg.max_age_frames))
        n_dead = int(np.sum(dead))
        if n_dead and cfg.respawn:
            fresh = sample_water_positions(n_dead, field, self.rng)
            self.positions[dead] = fresh
            self.trails[dead] = np.repeat(
                fresh[:, None, :], cfg.trail_length, axis=1)
            self.ages[dead] = 0
            self.n_respawned_total += n_dead
        elif n_dead and not cfg.respawn:
            # Without respawn, park dead particles off-trail (they fade out).
            self.positions[dead] = np.nan
        return n_dead

    def step(self, field: VectorField) -> Dict[str, float]:
        """Advance one frame; return step statistics."""
        cfg = self.config
        if cfg.method == "rk2":
            self._step_rk2(field, cfg.dt_seconds)
        else:
            self._step_rk4(field, cfg.dt_seconds)
        # Roll trail history (oldest dropped) and record the new head.
        self.trails[:, :-1, :] = self.trails[:, 1:, :]
        self.trails[:, -1, :] = self.positions
        self.ages += 1
        with np.errstate(invalid="ignore"):
            u, v = field.sample(self.positions[:, 0], self.positions[:, 1])
            mean_speed = float(np.nanmean(np.sqrt(u ** 2 + v ** 2)))
        return {"mean_speed_m_s": mean_speed,
                "total_respawned": float(self.n_respawned_total)}

    def trail_segments(self) -> Tuple[np.ndarray, np.ndarray]:
        """Line segments for streak rendering.

        Returns ``(segments, valid)`` where segments is
        ``(n, trail_length-1, 2, 2)`` and valid marks segments whose
        endpoints are both finite.
        """
        tr = self.trails
        segs = np.stack([tr[:, :-1, :], tr[:, 1:, :]], axis=2)
        valid = np.isfinite(segs).all(axis=(2, 3))
        return segs, valid

    def state_dict(self) -> Dict[str, object]:
        return {
            "positions": self.positions.tolist(),
            "ages": self.ages.tolist(),
            "config": {
                "dt_seconds": self.config.dt_seconds,
                "method": self.config.method,
                "trail_length": self.config.trail_length,
                "max_age_frames": self.config.max_age_frames,
                "respawn": self.config.respawn,
                "seed": self.config.seed,
            },
        }
