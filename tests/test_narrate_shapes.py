"""Shape detection / normalization tests (synthetic fields only, no network)."""

import numpy as np
import pytest

from narrate import UnrecognizedFieldError, normalize_field


def _wind_field(nt=2, ny=3, nx=4, temperature_unit="°F"):
    rng = np.random.default_rng(7)
    u10 = rng.normal(5.0, 1.0, (nt, ny, nx))
    v10 = rng.normal(-2.0, 1.0, (nt, ny, nx))
    return {
        "grids": {"u10": u10, "v10": v10, "t2m": np.full((nt, ny, nx), 12.0)},
        "air_temperature": np.full((nt, ny, nx), 53.6),  # °F
        "temperature_unit": temperature_unit,
        "times": [f"2026-09-{28 + i:02d}T00:00:00Z" for i in range(nt)],
        "lats": np.linspace(47.0, 48.0, ny),
        "lons": np.linspace(-124.0, -122.0, nx),
    }


def _current_field(nt=2, ny=3, nx=4, with_temperature=True):
    rng = np.random.default_rng(11)
    u = rng.normal(0.5, 0.2, (nt, ny, nx))
    v = rng.normal(0.1, 0.2, (nt, ny, nx))
    return {
        "u": u,
        "v": v,
        "temperature": np.full((nt, ny, nx), 8.5) if with_temperature else None,
        "times": [f"2026-09-{28 + i:02d}T06:00:00Z" for i in range(nt)],
        "lats": np.linspace(47.0, 48.0, ny),
        "lons": np.linspace(-124.0, -122.0, nx),
    }


class _WindObj:
    """Object-shaped twin of the GFS-wind dict (duck-typing check)."""

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def test_detects_gfs_wind_dict():
    nf = normalize_field(_wind_field())
    assert nf.kind == "wind"
    assert nf.u.shape == (2, 3, 4)
    assert nf.temperature is not None
    assert nf.temperature_unit == "°F"


def test_detects_gfs_wind_object():
    nf = normalize_field(_WindObj(_wind_field()))
    assert nf.kind == "wind"
    assert nf.times[0] == "2026-09-28T00:00:00Z"


def test_detects_current_dict():
    nf = normalize_field(_current_field())
    assert nf.kind == "current"
    assert nf.temperature is not None
    assert nf.temperature_unit == "°C"  # CurrentField convention


def test_detects_current_object():
    class C:
        pass

    d = _current_field()
    obj = C()
    for k, v in d.items():
        setattr(obj, k, v)
    nf = normalize_field(obj)
    assert nf.kind == "current"
    assert nf.u.shape == (2, 3, 4)


def test_gfs_t2m_fallback_when_no_air_temperature():
    f = _wind_field()
    del f["air_temperature"]
    nf = normalize_field(f)
    assert nf.temperature is not None
    assert nf.temperature_unit == "°C"  # GFS t2m store convention
    assert float(np.mean(nf.temperature)) == pytest.approx(12.0)


def test_temperature_unit_hint_overrides_declared():
    nf = normalize_field(_wind_field(), temperature_unit="°C")
    assert nf.temperature_unit == "°C"


def test_bad_temperature_unit_hint_raises():
    with pytest.raises(ValueError, match="temperature_unit"):
        normalize_field(_wind_field(), temperature_unit="rankine")


def test_unrecognized_shape_raises_for_plain_dict():
    with pytest.raises(UnrecognizedFieldError, match="unrecognized field shape"):
        normalize_field({"foo": 1})


def test_unrecognized_shape_raises_for_none_and_int():
    with pytest.raises(UnrecognizedFieldError):
        normalize_field(None)
    with pytest.raises(UnrecognizedFieldError):
        normalize_field(42)


def test_grids_without_u10_v10_raises_not_guesses():
    f = _wind_field()
    f["grids"] = {"t2m": f["grids"]["t2m"]}
    with pytest.raises(UnrecognizedFieldError, match="u10"):
        normalize_field(f)


def test_coordinate_mismatch_raises():
    f = _current_field()
    f["lats"] = np.linspace(47.0, 48.0, 5)
    with pytest.raises(ValueError, match="lats/lons"):
        normalize_field(f)


def test_times_length_mismatch_raises():
    f = _current_field()
    f["times"] = ["2026-09-28T00:00:00Z"]
    with pytest.raises(ValueError, match="len\\(times\\)"):
        normalize_field(f)


def test_current_without_temperature_is_fine():
    nf = normalize_field(_current_field(with_temperature=False))
    assert nf.temperature is None
    assert nf.temperature_unit is None
