"""story_facts tests (synthetic fields only, no network)."""

import json

import numpy as np
import pytest

from narrate import MS_TO_KNOTS, NoValidDataError, story_facts


def _hotspot_field():
    """Known hotspot: 10 m/s (pure eastward) at the far corner, t=1."""
    nt, ny, nx = 2, 3, 4
    u10 = np.full((nt, ny, nx), 3.0)
    v10 = np.full((nt, ny, nx), 0.0)
    u10[1, 2, 3] = 10.0  # hotspot: (t=1, y=2, x=3)
    return {
        "grids": {"u10": u10, "v10": v10, "t2m": np.full((nt, ny, nx), 10.0)},
        "air_temperature": np.full((nt, ny, nx), 50.0),
        "temperature_unit": "°F",
        "times": ["2026-09-28T00:00:00Z", "2026-09-29T00:00:00Z"],
        "lats": np.linspace(40.0, 42.0, ny),
        "lons": np.linspace(-10.0, -7.0, nx),
    }


def test_peak_speed_value_location_timestamp():
    facts = story_facts(_hotspot_field())
    peak = facts["peak"]
    assert peak["value_m_s"] == pytest.approx(10.0)
    assert peak["lat"] == pytest.approx(42.0)
    assert peak["lon"] == pytest.approx(-7.0)
    assert peak["time"] == "2026-09-29T00:00:00Z"


def test_knots_conversion_exactness():
    facts = story_facts(_hotspot_field())
    assert facts["peak"]["value_knots"] == pytest.approx(10.0 * 1.9438444924)
    assert MS_TO_KNOTS == pytest.approx(1.9438444924)
    one = story_facts(
        {
            "u": np.ones((1, 1, 1)),
            "v": np.zeros((1, 1, 1)),
            "times": ["2026-09-28T00:00:00Z"],
            "lats": np.array([0.0]),
            "lons": np.array([0.0]),
        }
    )
    assert one["peak"]["value_knots"] == pytest.approx(1.9438444924)


def test_mean_and_p95_speeds():
    facts = story_facts(_hotspot_field())
    n = 2 * 3 * 4
    expected_mean = (10.0 + 3.0 * (n - 1)) / n
    assert facts["mean_speed_m_s"] == pytest.approx(expected_mean)
    assert facts["mean_speed_knots"] == pytest.approx(expected_mean * MS_TO_KNOTS)
    assert facts["p95_speed_m_s"] == pytest.approx(3.0)
    assert facts["p95_speed_knots"] == pytest.approx(3.0 * MS_TO_KNOTS)


def test_uniform_eastward_direction_convention():
    f = _hotspot_field()
    f["grids"]["u10"] = np.full((2, 3, 4), 5.0)
    f["grids"]["v10"] = np.zeros((2, 3, 4))
    facts = story_facts(f)
    assert facts["direction"]["bearing_deg"] == pytest.approx(90.0)
    assert facts["direction"]["label"] == "easterly"


def test_direction_bearings_cardinal():
    cases = [
        ((0.0, 1.0), 0.0, "northerly"),     # toward north
        ((1.0, 0.0), 90.0, "easterly"),     # toward east
        ((0.0, -1.0), 180.0, "southerly"),  # toward south
        ((-1.0, 0.0), 270.0, "westerly"),   # toward west
        ((1.0, 1.0), 45.0, "northeasterly"),
    ]
    for (u_val, v_val), bearing, label in cases:
        f = {
            "u": np.full((1, 2, 2), u_val),
            "v": np.full((1, 2, 2), v_val),
            "times": ["2026-09-28T00:00:00Z"],
            "lats": np.array([0.0, 1.0]),
            "lons": np.array([0.0, 1.0]),
        }
        facts = story_facts(f)
        assert facts["direction"]["bearing_deg"] == pytest.approx(bearing), (u_val, v_val)
        assert facts["direction"]["label"] == label


def test_direction_note_documents_flow_not_meteo_convention():
    facts = story_facts(_hotspot_field())
    assert "NOT the meteorological" in facts["direction"]["note"]


def test_zero_vector_mean_has_no_dominant_direction():
    f = {
        "u": np.array([[[1.0, -1.0]]]),
        "v": np.array([[[0.0, 0.0]]]),
        "times": ["2026-09-28T00:00:00Z"],
        "lats": np.array([0.0]),
        "lons": np.array([0.0, 1.0]),
    }
    facts = story_facts(f)
    assert facts["direction"]["bearing_deg"] is None
    assert facts["direction"]["label"] is None


def test_nan_block_is_excluded_not_filled():
    nt, ny, nx = 1, 4, 4
    u = np.full((nt, ny, nx), 4.0)
    v = np.zeros((nt, ny, nx))
    u[0, 0:2, :] = np.nan  # top half is a gap
    v[0, 0:2, :] = np.nan
    f = {
        "u": u,
        "v": v,
        "times": ["2026-09-28T00:00:00Z"],
        "lats": np.linspace(40.0, 43.0, ny),
        "lons": np.linspace(-10.0, -7.0, nx),
    }
    facts = story_facts(f)
    assert facts["peak"]["value_m_s"] == pytest.approx(4.0)
    assert facts["mean_speed_m_s"] == pytest.approx(4.0)
    assert facts["n_valid_cells"] == 8
    assert facts["n_total_cells"] == 16
    assert facts["coverage_fraction"] == pytest.approx(0.5)
    assert facts["direction"]["label"] == "easterly"


def test_all_nan_field_raises():
    f = {
        "u": np.full((1, 2, 2), np.nan),
        "v": np.full((1, 2, 2), np.nan),
        "times": ["2026-09-28T00:00:00Z"],
        "lats": np.array([0.0, 1.0]),
        "lons": np.array([0.0, 1.0]),
    }
    with pytest.raises(NoValidDataError, match="no valid"):
        story_facts(f)


def test_temperature_fahrenheit_declared():
    facts = story_facts(_hotspot_field())
    assert facts["temperature"] == {"min": 50.0, "max": 50.0, "unit": "°F"}


def test_temperature_celsius_via_t2m_fallback():
    f = _hotspot_field()
    del f["air_temperature"]
    f["grids"]["t2m"] = np.arange(24, dtype=float).reshape(2, 3, 4)
    facts = story_facts(f)
    assert facts["temperature"]["unit"] == "°C"
    assert facts["temperature"]["min"] == pytest.approx(0.0)
    assert facts["temperature"]["max"] == pytest.approx(23.0)


def test_temperature_unit_hint_overrides():
    facts = story_facts(_hotspot_field(), temperature_unit="°C")
    assert facts["temperature"]["unit"] == "°C"


def test_no_temperature_gives_none():
    f = {
        "u": np.ones((1, 1, 1)),
        "v": np.zeros((1, 1, 1)),
        "temperature": None,
        "times": ["2026-09-28T00:00:00Z"],
        "lats": np.array([0.0]),
        "lons": np.array([0.0]),
    }
    assert story_facts(f)["temperature"] is None


def test_time_span_and_daily_cadence():
    facts = story_facts(_hotspot_field())
    span = facts["time_span"]
    assert span["start"] == "2026-09-28T00:00:00Z"
    assert span["end"] == "2026-09-29T00:00:00Z"
    assert span["n_timesteps"] == 2
    assert span["cadence"] == "daily"
    assert span["median_step_s"] == pytest.approx(86400.0)


def test_hourly_cadence_guess():
    f = _hotspot_field()
    f["times"] = ["2026-09-28T00:00:00Z", "2026-09-28T01:00:00Z",
                  "2026-09-28T02:00:00Z"]
    f["grids"]["u10"] = np.full((3, 3, 4), 5.0)
    f["grids"]["v10"] = np.zeros((3, 3, 4))
    f["grids"]["t2m"] = np.full((3, 3, 4), 10.0)
    f["air_temperature"] = np.full((3, 3, 4), 50.0)
    facts = story_facts(f)
    assert facts["time_span"]["cadence"] == "hourly"


def test_irregular_cadence_guess():
    f = _hotspot_field()
    f["times"] = ["2026-09-28T00:00:00Z", "2026-09-29T07:13:00Z"]
    f["grids"]["u10"] = np.full((2, 3, 4), 5.0)
    f["grids"]["v10"] = np.zeros((2, 3, 4))
    facts = story_facts(f)
    assert facts["time_span"]["cadence"] == "irregular"


def test_region_name_recorded_verbatim():
    facts = story_facts(_hotspot_field(), region_name="the North Sound")
    assert facts["region_name"] == "the North Sound"
    facts2 = story_facts(_hotspot_field())
    assert facts2["region_name"] == ""


def test_facts_json_round_trip():
    facts = story_facts(_hotspot_field(), region_name="the North Sound")
    dumped = json.dumps(facts)
    back = json.loads(dumped)
    assert back["peak"]["value_knots"] == pytest.approx(facts["peak"]["value_knots"])
    assert back["temperature"]["unit"] == "°F"
    assert back["direction"]["label"] == "easterly"


def test_current_shape_peak():
    f = {
        "u": np.zeros((1, 2, 2)),
        "v": np.zeros((1, 2, 2)),
        "temperature": np.full((1, 2, 2), 8.0),
        "times": ["2026-09-30T12:00:00Z"],
        "lats": np.array([48.0, 48.5]),
        "lons": np.array([-123.0, -122.5]),
    }
    f["u"][0, 1, 0] = -2.0  # 2 m/s westward hotspot
    facts = story_facts(f, region_name="the Strait")
    assert facts["field_kind"] == "current"
    assert facts["peak"]["value_m_s"] == pytest.approx(2.0)
    assert facts["peak"]["lat"] == pytest.approx(48.5)
    assert facts["peak"]["lon"] == pytest.approx(-123.0)
    assert facts["direction"]["label"] == "westerly"
