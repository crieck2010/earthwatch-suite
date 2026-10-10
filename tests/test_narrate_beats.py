"""headline_beats tests (synthetic fields only, no network)."""

import numpy as np
import pytest

from narrate import headline_beats, story_facts


def _hotspot_field(region_name="North America"):
    """Known hotspot: 10 m/s (pure eastward) at the far corner, t=1."""
    nt, ny, nx = 2, 3, 4
    u10 = np.full((nt, ny, nx), 3.0)
    v10 = np.full((nt, ny, nx), 0.0)
    u10[1, 2, 3] = 10.0  # hotspot: (t=1, y=2, x=3)
    field = {
        "grids": {"u10": u10, "v10": v10, "t2m": np.full((nt, ny, nx), 10.0)},
        "air_temperature": np.full((nt, ny, nx), 50.0),
        "temperature_unit": "°F",
        "times": ["2026-09-28T00:00:00Z", "2026-09-29T00:00:00Z"],
        "lats": np.linspace(40.0, 42.0, ny),
        "lons": np.linspace(-10.0, -7.0, nx),
    }
    return story_facts(field, region_name=region_name)


def _facts_with_temp_range():
    """Wind field with a meaningful temperature range (40–60 °F)."""
    nt, ny, nx = 3, 2, 2
    u10 = np.full((nt, ny, nx), 5.0)
    v10 = np.zeros((nt, ny, nx))
    u10[1, 1, 1] = 12.0
    temp = np.linspace(40.0, 60.0, nt * ny * nx).reshape(nt, ny, nx)
    field = {
        "grids": {"u10": u10, "v10": v10, "t2m": np.full((nt, ny, nx), 10.0)},
        "air_temperature": temp,
        "temperature_unit": "°F",
        "times": [
            "2026-09-28T00:00:00Z",
            "2026-09-28T12:00:00Z",
            "2026-09-29T00:00:00Z",
        ],
        "lats": np.array([40.0, 41.0]),
        "lons": np.array([-10.0, -9.0]),
    }
    return story_facts(field, region_name="North America")


def test_beat_count_sorted_and_in_range():
    facts = _hotspot_field()
    beats = headline_beats(facts)
    assert 1 <= len(beats) <= 4
    fractions = [f for f, _ in beats]
    assert fractions == sorted(fractions)
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert beats[0][0] == 0.0


def test_opening_uses_region_and_subject():
    facts = _hotspot_field(region_name="North America")
    beats = headline_beats(facts)
    assert beats[0][1] == "North America — the wind in motion"


def test_peak_fraction_matches_peak_time_position():
    facts = _hotspot_field()
    # Peak is at the second of two daily timesteps -> fraction 1.0.
    assert facts["peak"]["time"] == "2026-09-29T00:00:00Z"
    beats = headline_beats(facts)
    peak_beats = [(f, t) for f, t in beats if t.startswith("Peak:")]
    assert len(peak_beats) == 1
    assert peak_beats[0][0] == pytest.approx(1.0)


def test_peak_fraction_mid_span():
    facts = _facts_with_temp_range()
    # Peak at t=1 of 0..2 spanning 24h -> fraction 0.5.
    beats = headline_beats(facts)
    peak_beats = [(f, t) for f, t in beats if t.startswith("Peak:")]
    assert peak_beats[0][0] == pytest.approx(0.5)


def test_peak_text_contains_knot_value_and_location():
    facts = _hotspot_field()
    beats = headline_beats(facts)
    peak_text = [t for _, t in beats if t.startswith("Peak:")][0]
    peak_kt = facts["peak"]["value_knots"]  # 10 m/s -> ~19 knots
    assert f"{peak_kt:.0f}" in peak_text
    assert "knots" in peak_text
    assert "42.00°N" in peak_text
    assert "7.00°W" in peak_text


def test_regionless_facts_still_produce_opening():
    facts = _hotspot_field(region_name="")
    beats = headline_beats(facts)
    assert len(beats) >= 1
    assert beats[0][0] == 0.0
    assert beats[0][1] == "The wind in motion"


def test_single_timestep_does_not_crash():
    field = {
        "u": np.full((1, 2, 2), 4.0),
        "v": np.zeros((1, 2, 2)),
        "times": ["2026-09-28T00:00:00Z"],
        "lats": np.array([40.0, 41.0]),
        "lons": np.array([-10.0, -9.0]),
    }
    facts = story_facts(field, region_name="North America")
    assert facts["time_span"]["n_timesteps"] == 1
    beats = headline_beats(facts)
    assert len(beats) >= 1
    assert beats[0][0] == 0.0
    fractions = [f for f, _ in beats]
    assert fractions == sorted(fractions)
    assert all(0.0 <= f <= 1.0 for f in fractions)
    # Single timestep: peak position cannot be derived -> fallback 0.5.
    peak_beats = [(f, t) for f, t in beats if t.startswith("Peak:")]
    assert peak_beats[0][0] == pytest.approx(0.5)


def test_max_beats_one_returns_just_opening():
    facts = _hotspot_field()
    beats = headline_beats(facts, max_beats=1)
    assert len(beats) == 1
    assert beats[0][0] == 0.0
    assert "North America" in beats[0][1]


def test_max_beats_respected():
    facts = _facts_with_temp_range()
    for max_beats in (1, 2, 3, 4, 6):
        beats = headline_beats(facts, max_beats=max_beats)
        assert len(beats) <= max_beats


def test_max_beats_two_and_three_counts():
    facts = _hotspot_field()
    assert len(headline_beats(facts, max_beats=2)) == 2
    assert len(headline_beats(facts, max_beats=3)) == 3


def test_closing_beat_at_point_nine_restates_mean_and_direction():
    facts = _hotspot_field()
    beats = headline_beats(facts)
    closing = [t for f, t in beats if f == pytest.approx(0.9)]
    assert len(closing) == 1
    assert "Average:" in closing[0]
    assert "knots" in closing[0]
    assert "easterly" in closing[0]


def test_temperature_mid_beat_only_when_meaningful_and_max_allows():
    facts = _facts_with_temp_range()
    assert facts["temperature"]["max"] - facts["temperature"]["min"] >= 1.0
    beats4 = headline_beats(facts, max_beats=4)
    mid = [(f, t) for f, t in beats4 if f == pytest.approx(0.65)]
    assert len(mid) == 1
    assert "temperatures ranged" in mid[0][1].lower()
    # With max_beats=3 the mid beat is not added even though temp is meaningful.
    beats3 = headline_beats(facts, max_beats=3)
    assert all(f != pytest.approx(0.65) for f, _ in beats3)
    assert len(beats3) == 3


def test_no_temperature_range_no_mid_beat():
    facts = _hotspot_field()
    # Constant 50°F -> range 0, not meaningful.
    assert facts["temperature"]["max"] - facts["temperature"]["min"] == 0.0
    beats = headline_beats(facts, max_beats=4)
    assert all(f != pytest.approx(0.65) for f, _ in beats)
    assert len(beats) == 3  # opening + peak + closing, no filler


def test_no_temperature_at_all_still_valid():
    field = {
        "u": np.ones((2, 1, 1)),
        "v": np.zeros((2, 1, 1)),
        "times": ["2026-09-28T00:00:00Z", "2026-09-29T00:00:00Z"],
        "lats": np.array([0.0]),
        "lons": np.array([0.0]),
    }
    facts = story_facts(field)
    assert facts["temperature"] is None
    beats = headline_beats(facts)
    assert len(beats) >= 1
    assert beats[0][0] == 0.0


def test_current_field_beats_use_knots_and_surface_water():
    field = {
        "u": np.zeros((2, 2, 2)),
        "v": np.zeros((2, 2, 2)),
        "temperature": np.full((2, 2, 2), 8.0),
        "times": ["2026-09-28T00:00:00Z", "2026-09-29T00:00:00Z"],
        "lats": np.array([48.0, 48.5]),
        "lons": np.array([-123.0, -122.5]),
    }
    field["u"][1, 1, 0] = -2.0
    facts = story_facts(field, region_name="the Strait")
    assert facts["field_kind"] == "current"
    beats = headline_beats(facts)
    assert beats[0][1] == "the Strait — the surface water in motion"
    peak_text = [t for _, t in beats if t.startswith("Peak:")][0]
    assert "knots" in peak_text


def test_invalid_facts_raise_value_error():
    with pytest.raises(ValueError, match="facts"):
        headline_beats({})
    with pytest.raises(ValueError):
        headline_beats(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="peak"):
        headline_beats({"field_kind": "wind"})
    facts = _hotspot_field()
    broken = {k: v for k, v in facts.items() if k != "peak"}
    with pytest.raises(ValueError, match="peak"):
        headline_beats(broken)
    broken2 = {k: v for k, v in facts.items() if k != "time_span"}
    with pytest.raises(ValueError, match="time_span"):
        headline_beats(broken2)


def test_invalid_max_beats_raises():
    facts = _hotspot_field()
    with pytest.raises(ValueError, match="max_beats"):
        headline_beats(facts, max_beats=0)
    with pytest.raises(ValueError, match="max_beats"):
        headline_beats(facts, max_beats=-1)


def test_beats_are_tuples_of_float_and_str():
    facts = _hotspot_field()
    for frac, text in headline_beats(facts):
        assert isinstance(frac, float)
        assert isinstance(text, str)
        assert text


def test_unparsable_peak_time_falls_back_to_half():
    facts = _hotspot_field()
    facts = dict(facts)
    facts["peak"] = dict(facts["peak"], time="not-a-timestamp")
    beats = headline_beats(facts)
    peak_beats = [(f, t) for f, t in beats if t.startswith("Peak:")]
    assert peak_beats[0][0] == pytest.approx(0.5)
