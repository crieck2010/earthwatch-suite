"""Tests for erosion/accretion rate fitting."""

import datetime as dt

import pytest

from coast.positions import TransectPassRecord
from coast.rates import (
    MONITOR_CONTRACT_COLUMNS,
    RATES_COLUMNS,
    compute_rates,
    fit_rate,
    write_rates_csv,
)


def _rec(tid, date, pos, site="s1"):
    d = dt.date.fromisoformat(date)
    return TransectPassRecord(
        transect_id=tid, transect_name=tid, site_id=site, site_name=site,
        pass_id=f"p-{date}", date=date, doy=d.timetuple().tm_yday,
        year=d.year, index="ndwi", position_m=pos,
        shoreline_length_m=1000.0, valid=pos == pos,
        cloud_cover=0.0, pixel_size_m=10.0,
    )


def test_fit_recovers_planted_rate():
    # Exactly -2.0 m/yr over 5 years, tiny noise-free.
    dates = [dt.date(y, 1, 15) for y in (2019, 2020, 2021, 2022, 2023, 2024)]
    positions = [500.0 - 2.0 * (d - dates[0]).days / 365.25 for d in dates]
    fit = fit_rate(dates, positions)
    assert fit["rate"] == pytest.approx(-2.0, abs=1e-9)
    assert fit["se"] == pytest.approx(0.0, abs=1e-9)
    assert fit["r2"] == pytest.approx(1.0)
    assert fit["net"] == pytest.approx(positions[-1] - positions[0])


def test_fit_needs_three_points():
    dates = [dt.date(2020, 1, 1), dt.date(2021, 1, 1)]
    fit = fit_rate(dates, [100.0, 90.0])
    assert fit["n"] == 2
    assert fit["rate"] != fit["rate"]  # NaN
    # ...but net change and endpoint rate are still reported
    assert fit["net"] == pytest.approx(-10.0)
    assert fit["endpoint"] == pytest.approx(-10.0, abs=0.05)


def test_fit_single_point_nan():
    fit = fit_rate([dt.date(2020, 1, 1)], [100.0])
    assert fit["n"] == 1 and fit["net"] != fit["net"]


def test_fit_skips_nan_positions():
    dates = [dt.date(y, 1, 1) for y in (2020, 2021, 2022, 2023)]
    fit = fit_rate(dates, [100.0, float("nan"), 80.0, 70.0])
    assert fit["n"] == 3
    assert fit["rate"] == pytest.approx(-10.0, abs=0.05)  # leap years skew exactness


def test_compute_rates_groups_and_sign():
    recs = [
        _rec("eroding", "2020-01-01", 500.0),
        _rec("eroding", "2021-01-01", 498.0),
        _rec("eroding", "2022-01-01", 496.0),
        _rec("growing", "2020-01-01", 300.0),
        _rec("growing", "2021-01-01", 303.0),
        _rec("growing", "2022-01-01", 306.0),
    ]
    rates = {r.transect_id: r for r in compute_rates(recs)}
    assert rates["eroding"].rate_m_per_yr == pytest.approx(-2.0, abs=0.01)
    assert rates["growing"].rate_m_per_yr == pytest.approx(3.0, abs=0.05)
    assert rates["eroding"].n_passes == 3


def test_compute_rates_skips_invalid():
    recs = [
        _rec("t", "2020-01-01", 500.0),
        _rec("t", "2021-01-01", float("nan")),
        _rec("t", "2022-01-01", 496.0),
        _rec("t", "2023-01-01", 494.0),
    ]
    (rate,) = compute_rates(recs)
    assert rate.n_passes == 3


def test_write_rates_csv(tmp_path):
    recs = [_rec("t", f"{y}-01-01", 500.0 - 2 * (y - 2020)) for y in (2020, 2021, 2022)]
    path = str(tmp_path / "rates.csv")
    write_rates_csv(path, compute_rates(recs))
    text = open(path).read()
    assert text.splitlines()[0] == ",".join(RATES_COLUMNS)
    assert "t," in text


def test_monitor_contract_columns():
    assert set(MONITOR_CONTRACT_COLUMNS) <= set(RATES_COLUMNS)
