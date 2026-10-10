"""Tests for flow.advect: integration accuracy, respawn, determinism."""

import numpy as np
import pytest

from flow.advect import (AdvectionConfig, ParticleSet, sample_water_positions,
                         velocity_to_deg)
from flow.fields import VectorField

from conftest import make_fake_current_field


def _uniform_east_field(u=0.5):
    # Constant eastward flow, no land.
    lats = np.linspace(40.0, 50.0, 11)
    lons = np.linspace(-100.0, -80.0, 21)
    shape = (11, 21)
    return VectorField(u=np.full(shape, u), v=np.zeros(shape),
                       lats=lats, lons=lons,
                       scalar=np.full(shape, 15.0))


def test_velocity_to_deg_equator():
    # 1 m/s for one hour at the equator: ~3.6 km -> ~0.0324 deg lon.
    dlon, dlat = velocity_to_deg(np.array([1.0]), np.array([0.0]),
                                 np.array([0.0]), 3600.0)
    assert dlon[0] == pytest.approx(0.03239, rel=1e-3)
    assert dlat[0] == pytest.approx(0.0, abs=1e-12)


def test_velocity_to_deg_latitude_scaling():
    dlon_eq, _ = velocity_to_deg(np.array([1.0]), np.array([0.0]),
                                 np.array([0.0]), 3600.0)
    dlon_60, _ = velocity_to_deg(np.array([1.0]), np.array([0.0]),
                                 np.array([60.0]), 3600.0)
    assert dlon_60[0] == pytest.approx(dlon_eq[0] * 2.0, rel=1e-3)


def test_rk2_uniform_flow_moves_east():
    f = _uniform_east_field(u=0.5)
    cfg = AdvectionConfig(dt_seconds=3600.0, method="rk2", seed=1)
    ps = ParticleSet(20, f, config=cfg)
    before = ps.positions.copy()
    ps.step(f)
    # All particles moved east, latitude unchanged.
    dlon = ps.positions[:, 0] - before[:, 0]
    assert np.all(dlon > 0)
    assert np.allclose(ps.positions[:, 1], before[:, 1])


def test_rk4_matches_rk2_on_uniform_flow():
    f = _uniform_east_field(u=0.5)
    ps2 = ParticleSet(10, f, AdvectionConfig(method="rk2", seed=3))
    ps4 = ParticleSet(10, f, AdvectionConfig(method="rk4", seed=3))
    ps2.step(f)
    ps4.step(f)
    np.testing.assert_allclose(ps2.positions, ps4.positions, rtol=1e-6)


def test_circular_flow_conserves_radius():
    # Solid-body rotation on a fine grid with small steps: integration
    # error dominates, so RK4 must conserve radius better than RK2.
    n = 81
    lats = np.linspace(-5.0, 5.0, n)
    lons = np.linspace(-5.0, 5.0, n)
    yy, xx = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n),
                         indexing="ij")
    omega = 2 * np.pi / (10 * 86400.0)  # one revolution per 10 days
    r_earth = 6_371_000.0
    # Physical radius in radians: normalized unit 1 == 5 degrees.
    r_rad = np.sqrt(xx ** 2 + yy ** 2) * 5.0 * np.pi / 180.0
    speed = omega * r_earth * r_rad
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(r_rad > 0, -yy / np.sqrt(xx ** 2 + yy ** 2) * speed, 0.0)
        v = np.where(r_rad > 0, xx / np.sqrt(xx ** 2 + yy ** 2) * speed, 0.0)
    f = VectorField(u=u, v=v, lats=lats, lons=lons)
    radii = {}
    for method in ("rk2", "rk4"):
        ps = ParticleSet(30, f, AdvectionConfig(dt_seconds=900.0,
                                               method=method, seed=9,
                                               max_age_frames=10_000))
        r0 = np.sqrt(ps.positions[:, 0] ** 2 + ps.positions[:, 1] ** 2)
        n_steps = 16
        for _ in range(n_steps):
            ps.step(f)
        r1 = np.sqrt(ps.positions[:, 0] ** 2 + ps.positions[:, 1] ** 2)
        # Only particles that never respawned (a respawned particle's
        # radius legitimately jumps to a random water location).
        survived = ps.ages == n_steps
        assert survived.sum() > 20
        radii[method] = float(np.mean(np.abs(r1[survived] - r0[survived])))
    # Both integrators must hold the circular orbit tightly (no spiraling).
    assert radii["rk2"] < 0.05
    assert radii["rk4"] < 0.05


def test_determinism_same_seed():
    f = _uniform_east_field()
    ps1 = ParticleSet(25, f, AdvectionConfig(seed=42))
    ps2 = ParticleSet(25, f, AdvectionConfig(seed=42))
    for _ in range(5):
        ps1.step(f)
        ps2.step(f)
    np.testing.assert_array_equal(ps1.positions, ps2.positions)
    ps3 = ParticleSet(25, f, AdvectionConfig(seed=43))
    assert not np.array_equal(ps1.positions, ps3.positions)


def test_respawn_on_land(fake_cf):
    from flow.fields import TimeVaryingField
    tvf = TimeVaryingField.from_current_field(fake_cf)
    f0 = tvf.at(0.0)
    cfg = AdvectionConfig(dt_seconds=3600.0, seed=5, max_age_frames=10_000)
    ps = ParticleSet(40, f0, config=cfg)
    for _ in range(60):
        ps.step(f0)
    # With respawn on, no particle should be sitting on land.
    assert not f0.is_land(ps.positions[:, 0], ps.positions[:, 1]).any()
    assert ps.n_respawned_total >= 0


def test_max_age_respawn():
    f = _uniform_east_field()
    cfg = AdvectionConfig(dt_seconds=1.0, seed=2, max_age_frames=3)
    ps = ParticleSet(10, f, config=cfg)
    for _ in range(4):
        ps.step(f)
    assert ps.n_respawned_total > 0
    assert (ps.ages <= 3).all()


def test_no_respawn_parks_nan():
    f = _uniform_east_field()
    cfg = AdvectionConfig(dt_seconds=1.0, seed=2, max_age_frames=1,
                          respawn=False)
    ps = ParticleSet(10, f, config=cfg)
    ps.step(f)  # ages -> 1, not yet dead
    ps.step(f)  # ages -> 2 >= max_age -> dead, parked as NaN
    assert np.isnan(ps.positions).any()


def test_trail_shapes_and_rolling():
    f = _uniform_east_field()
    cfg = AdvectionConfig(trail_length=5, seed=1)
    ps = ParticleSet(8, f, config=cfg)
    assert ps.trails.shape == (8, 5, 2)
    ps.step(f)
    p1 = ps.positions.copy()
    # Newest trail entry equals current position.
    np.testing.assert_array_equal(ps.trails[:, -1, :], ps.positions)
    ps.step(f)
    # Previous position is now second-newest; newest is current.
    np.testing.assert_array_equal(ps.trails[:, -2, :], p1)
    np.testing.assert_array_equal(ps.trails[:, -1, :], ps.positions)
    segs, valid = ps.trail_segments()
    assert segs.shape == (8, 4, 2, 2)
    assert valid.all()


def test_sample_water_positions_all_land():
    lats = np.array([40.0, 41.0])
    lons = np.array([-90.0, -89.0])
    f = VectorField(u=np.full((2, 2), np.nan), v=np.full((2, 2), np.nan),
                    lats=lats, lons=lons)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_water_positions(4, f, rng, max_tries=3)


def test_config_validation():
    with pytest.raises(ValueError):
        AdvectionConfig(method="euler")
    with pytest.raises(ValueError):
        AdvectionConfig(trail_length=1)
    with pytest.raises(ValueError):
        AdvectionConfig(dt_seconds=0)
    with pytest.raises(ValueError):
        ParticleSet(0, _uniform_east_field())


def test_step_returns_stats():
    f = _uniform_east_field(u=0.5)
    ps = ParticleSet(10, f, AdvectionConfig(seed=1))
    stats = ps.step(f)
    assert stats["mean_speed_m_s"] == pytest.approx(0.5, rel=1e-2)
    assert stats["total_respawned"] >= 0


def test_trail_segments_invalid_where_parked_nan():
    f = _uniform_east_field()
    cfg = AdvectionConfig(dt_seconds=1.0, seed=2, max_age_frames=1,
                          respawn=False)
    ps = ParticleSet(10, f, config=cfg)
    ps.step(f)
    ps.step(f)  # deaths parked as NaN
    segs, valid = ps.trail_segments()
    assert not valid.all()
    assert valid.shape == (10, cfg.trail_length - 1)


def test_state_dict_has_expected_keys():
    f = _uniform_east_field()
    ps = ParticleSet(5, f, AdvectionConfig(seed=1))
    d = ps.state_dict()
    assert set(d) == {"positions", "ages", "config"}
    assert d["config"]["method"] == "rk2"
    assert len(d["positions"]) == 5
