"""Tests for flow.fields: sampling, landmask, time blending, COG input."""

import numpy as np
import pytest

from flow.fields import TimeVaryingField, VectorField, _bilinear

from conftest import make_fake_current_field


def _plane_field():
    # u = lon, v = lat exactly -> bilinear must reproduce interior values.
    lats = np.array([40.0, 41.0, 42.0])
    lons = np.array([-90.0, -89.0, -88.0])
    yy, xx = np.meshgrid(lats, lons, indexing="ij")
    return VectorField(u=xx, v=yy, lats=lats, lons=lons,
                       scalar=np.full((3, 3), 20.0))


def test_bilinear_reproduces_plane():
    f = _plane_field()
    u, v = f.sample(np.array([-89.4]), np.array([41.3]))
    assert u[0] == pytest.approx(-89.4)
    assert v[0] == pytest.approx(41.3)


def test_bilinear_out_of_bounds_is_nan():
    f = _plane_field()
    u, v = f.sample(np.array([-100.0, -89.0]), np.array([41.0, 99.0]))
    assert np.isnan(u).all() and np.isnan(v).all()


def test_bilinear_all_nan_neighborhood():
    lats = np.array([40.0, 41.0])
    lons = np.array([-90.0, -89.0])
    f = VectorField(u=np.full((2, 2), np.nan), v=np.full((2, 2), np.nan),
                    lats=lats, lons=lons)
    u, _ = f.sample(np.array([-89.5]), np.array([40.5]))
    assert np.isnan(u[0])


def test_bilinear_partial_nan_renormalizes():
    lats = np.array([40.0, 41.0])
    lons = np.array([-90.0, -89.0])
    u = np.array([[1.0, np.nan], [np.nan, np.nan]])
    f = VectorField(u=u, v=np.zeros((2, 2)), lats=lats, lons=lons)
    # Query exactly at the valid corner -> its value.
    uu, _ = f.sample(np.array([-90.0]), np.array([40.0]))
    assert uu[0] == pytest.approx(1.0)


def test_sample_vectorized_shape():
    f = _plane_field()
    qlons = np.array([-89.9, -89.1])
    qlats = np.array([40.2, 41.8])
    u, v = f.sample(qlons, qlats)
    assert u.shape == (2,) and v.shape == (2,)


def test_constructor_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        VectorField(u=np.zeros((3, 3)), v=np.zeros((3, 4)),
                    lats=np.arange(3.0), lons=np.arange(3.0))
    with pytest.raises(ValueError):
        VectorField(u=np.zeros((3, 3)), v=np.zeros((3, 3)),
                    lats=np.array([42.0, 41.0, 40.0]),  # decreasing
                    lons=np.arange(3.0))


def test_is_land_and_in_bounds(fake_cf):
    f = VectorField.from_current_field(fake_cf, 0)
    # Corners are land in the fake field; center is water.
    assert f.is_land(np.array([-92.4]), np.array([46.4]))[0]
    assert not f.is_land(np.array([-88.5]), np.array([44.0]))[0]
    assert f.is_land(np.array([-100.0]), np.array([44.0]))[0]  # off-grid
    assert not f.in_bounds(np.array([-100.0]), np.array([44.0]))[0]
    assert f.in_bounds(np.array([-88.5]), np.array([44.0]))[0]


def test_from_current_field_duck_typed(fake_cf):
    f = VectorField.from_current_field(fake_cf, 1)
    assert f.shape == (12, 16)
    assert f.scalar_name == "temperature"
    assert f.timestamp == fake_cf.times[1]
    assert f.source == "test/fake"
    lo, hi = f.scalar_range()
    assert hi > lo
    assert f.mean_speed() > 0


def test_from_current_field_scalar_speed_fallback():
    cf = make_fake_current_field(with_temperature=False)
    f = VectorField.from_current_field(cf, 0)  # temperature -> speed fallback
    assert f.scalar_name == "speed"
    with pytest.raises(ValueError):
        VectorField.from_current_field(cf, 0, scalar="bogus")
    with pytest.raises(IndexError):
        VectorField.from_current_field(cf, 99)


def test_from_cog_without_rasterio(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "rasterio", None)
    with pytest.raises(ImportError):
        VectorField.from_cog("whatever.tif")


def test_time_varying_blend(fake_cf):
    tvf = TimeVaryingField.from_current_field(fake_cf)
    assert tvf.nt == 3
    a, b = tvf.at(0.0), tvf.at(2.0)
    mid = tvf.at(1.0)
    # t=1.0 is exactly timestep 1 (no blending); t=0.5 blends 0 and 1.
    np.testing.assert_allclose(mid.u, tvf.fields[1].u, equal_nan=True)
    half = tvf.at(0.5)
    expected = (tvf.fields[0].u + tvf.fields[1].u) / 2.0
    np.testing.assert_allclose(half.u, expected, equal_nan=True)
    # Clamping.
    assert tvf.at(-5.0) is tvf.fields[0]
    assert tvf.at(99.0) is tvf.fields[-1]


def test_time_varying_rejects_mismatched_grids(fake_cf):
    tvf = TimeVaryingField.from_current_field(fake_cf)
    other = VectorField(u=np.zeros((4, 4)), v=np.zeros((4, 4)),
                        lats=np.arange(4.0), lons=np.arange(4.0))
    with pytest.raises(ValueError):
        TimeVaryingField([tvf.fields[0], other], ["a", "b"])
    with pytest.raises(ValueError):
        TimeVaryingField([], [])


def test_bounds_and_shape(fake_cf):
    tvf = TimeVaryingField.from_current_field(fake_cf)
    assert tvf.bounds == (-92.5, 41.5, -84.5, 46.5)
    assert tvf.fields[0].shape == (12, 16)


def test_sample_scalar_requires_grid(fake_cf):
    f = VectorField(u=np.zeros((3, 3)), v=np.zeros((3, 3)),
                    lats=np.arange(40.0, 43.0), lons=np.arange(-90.0, -87.0))
    with pytest.raises(ValueError):
        f.sample_scalar(np.array([-89.0]), np.array([41.0]))
    with pytest.raises(ValueError):
        f.scalar_range()


def test_time_varying_from_current_field_scalar_speed(fake_cf):
    tvf = TimeVaryingField.from_current_field(fake_cf, scalar="speed")
    assert tvf.fields[0].scalar_name == "speed"
    assert tvf.fields[0].scalar is not None


def test_bilinear_degenerate_grid_returns_nan():
    f = VectorField(u=np.array([[1.0]]), v=np.array([[0.0]]),
                    lats=np.array([40.0]), lons=np.array([-90.0]))
    u, _ = f.sample(np.array([-90.0]), np.array([40.0]))
    assert np.isnan(u[0])


def test_from_current_field_keeps_metadata(fake_cf):
    f = VectorField.from_current_field(fake_cf, 2)
    assert f.crs == "EPSG:4326"
    assert f.timestamp == fake_cf.times[2]
    assert f.bounds == (-92.5, 41.5, -84.5, 46.5)
