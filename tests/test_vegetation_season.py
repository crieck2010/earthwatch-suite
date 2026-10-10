"""Tests for growing-season summaries."""

import datetime as dt

import pytest

from vegetation import season as sn
from vegetation.synthetic import phenology_value, synthetic_timeseries
from vegetation.timeseries import ZonePassRecord


def _series(zone, year, doys, values, index="ndvi"):
    recs = []
    for doy, val in zip(doys, values):
        date = dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)
        recs.append(
            ZonePassRecord.from_zonal_stats(
                zone, zone, f"p-{doy}", date.isoformat(), index,
                {"mean": val, "median": val, "std": 0.02,
                 "minimum": val - 0.05, "maximum": val + 0.05,
                 "p10": val - 0.03, "p90": val + 0.03,
                 "valid_pixels": 100, "total_pixels": 100},
            )
        )
    return recs


def test_known_phenology_recovered():
    # Synthetic curve: onset 110, peak 200, senescence onset 290.
    # With the default 0.5 amplitude fraction, green-up/senescence are the
    # mid-rise / mid-fall crossings: 155 and 245.
    doys = list(range(60, 340, 8))
    values = [phenology_value(d) for d in doys]
    recs = _series("z1", 2024, doys, values)
    s = sn.season_summary(recs, "z1", "ndvi", 2024)
    assert s.season_found
    assert abs(dt.date.fromisoformat(s.greenup_date).timetuple().tm_yday - 155) <= 8
    assert abs(dt.date.fromisoformat(s.peak_date).timetuple().tm_yday - 200) <= 8
    assert abs(dt.date.fromisoformat(s.senescence_date).timetuple().tm_yday - 245) <= 8
    assert s.peak_value == pytest.approx(0.85, abs=0.05)  # 8-day sampling misses the exact peak
    assert s.season_days and 60 <= s.season_days <= 130
    assert s.season_integral and s.season_integral > 0


def test_amplitude_fraction_shifts_edges():
    doys = list(range(60, 340, 8))
    values = [phenology_value(d) for d in doys]
    recs = _series("z1", 2024, doys, values)
    early = sn.season_summary(recs, "z1", "ndvi", 2024, amplitude_fraction=0.2)
    late = sn.season_summary(recs, "z1", "ndvi", 2024, amplitude_fraction=0.5)
    assert early.season_found and late.season_found
    assert early.greenup_date < late.greenup_date  # 0.2 crosses earlier on the rise
    assert early.senescence_date > late.senescence_date  # ...and later on the fall


def test_trapz_hand_computed():
    # Triangle 0->1 over 10 days: area = 5.
    area = sn._trapz([0.0, 10.0], [0.0, 1.0], 0.0, 10.0)
    assert area == pytest.approx(5.0)
    # Partial window with interpolated edges: 2->8 on the same triangle.
    area2 = sn._trapz([0.0, 10.0], [0.0, 1.0], 2.0, 8.0)
    assert area2 == pytest.approx(3.0)  # (0.2+0.8)/2 * 6


def test_flat_series_no_season():
    recs = _series("z1", 2024, [100, 150, 200, 250], [0.5, 0.51, 0.5, 0.49])
    s = sn.season_summary(recs, "z1", "ndvi", 2024)
    assert s.season_found is False
    assert "amplitude" in s.note


def test_too_few_passes():
    recs = _series("z1", 2024, [100, 150], [0.3, 0.6])
    s = sn.season_summary(recs, "z1", "ndvi", 2024)
    assert s.season_found is False


def test_integral_z_against_prior_years():
    doys = list(range(60, 340, 16))
    recs = []
    for year, peak in ((2022, 0.85), (2023, 0.83), (2024, 0.55)):
        recs.extend(
            _series("z1", year, doys, [phenology_value(d, peak=peak) for d in doys])
        )
    out = sn.summarize_all_years(recs, "ndvi")
    by_year = {s.year: s for s in out}
    assert by_year[2024].season_found
    assert by_year[2024].integral_z_score is not None
    assert by_year[2024].integral_z_score < -1.0  # weak 2024 vs priors
    assert by_year[2022].integral_z_score is None  # no priors yet


def test_synthetic_demo_seasons():
    recs = synthetic_timeseries()
    out = sn.summarize_all_years(recs, "ndvi")
    found = [s for s in out if s.season_found]
    assert len(found) == 9  # 3 zones x 3 years
    for s in found:
        assert 60 <= s.season_days <= 220
