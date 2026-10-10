"""Tests for survey-derive (the `derive` package)."""

import datetime as dt

import numpy as np
import pytest

import derive
from derive import (
    Climatology,
    anomaly,
    climatology,
    monthly_means,
    percent_of_normal,
    standardized_anomaly,
    suggest_symmetric_limits,
    trend,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _daily_dates(start: dt.date, n: int):
    return [start + dt.timedelta(days=i) for i in range(n)]


def _grid(ny=4, nx=5):
    lats = np.linspace(41.0, 44.0, ny)
    lons = np.linspace(-88.0, -84.0, nx)
    return lats, lons


def _seasonal_field(start, years, ny=4, nx=5, noise=0.0, as_dict=True, variable="sst",
                    units="degC", long_name="Sea surface temperature"):
    """values = 10 + 5*sin(2*pi*doy/365), identical every year."""
    lats, lons = _grid(ny, nx)
    times = _daily_dates(start, 365 * years)
    doys = np.array([t.timetuple().tm_yday for t in times])
    base = 10.0 + 5.0 * np.sin(2 * np.pi * doys / 365.0)
    values = np.broadcast_to(base[:, None, None], (len(times), ny, nx)).copy()
    if noise:
        rng = np.random.default_rng(0)
        values = values + rng.normal(0, noise, values.shape)
    d = {"times": times, "lats": lats, "lons": lons, "values": values,
         "variable": variable, "units": units, "long_name": long_name}
    return d if as_dict else _Stub(**d)


class _Stub:
    """Duck-typed field object: .times/.lats/.lons/.values attributes."""

    def __init__(self, times, lats, lons, values, variable="sst", units="degC",
                 long_name=None):
        self.times = times
        self.lats = lats
        self.lons = lons
        self.values = values
        self.variable = variable
        self.units = units
        self.long_name = long_name


# ---------------------------------------------------------------------------
# version / API surface
# ---------------------------------------------------------------------------

def test_version():
    assert derive.__version__ == "0.1.0"


# ---------------------------------------------------------------------------
# climatology
# ---------------------------------------------------------------------------

def test_climatology_zero_window_recovers_exact_cycle():
    # window_days=0 pools only exact doy matches; identical seasonal cycle
    # each year -> doy means equal the cycle exactly.
    f = _seasonal_field(dt.date(2017, 1, 1), 3)
    clim = climatology(f, window_days=0)
    assert isinstance(clim, Climatology)
    assert clim.doy_means.shape == (366, 4, 5)
    assert clim.doy_counts.shape == (366, 4, 5)
    assert clim.doy_counts[0, 0, 0] == 3          # doy 1: 3 yearly samples
    assert clim.doy_counts[179, 2, 3] == 3
    assert clim.doy_counts[365, 0, 0] == 0        # doy 366: no leap years
    assert np.isnan(clim.doy_means[365]).all()
    expected = 10.0 + 5.0 * np.sin(2 * np.pi * 1 / 365.0)
    assert clim.doy_means[0, 0, 0] == pytest.approx(expected)
    assert clim.baseline_start == dt.date(2017, 1, 1)
    assert clim.baseline_end == dt.date(2019, 12, 31)
    assert clim.window_days == 0
    assert clim.min_samples == 10
    assert clim.source_variable == "sst"
    assert clim.source_units == "degC"
    assert np.array_equal(clim.lats, f["lats"])
    p = clim.provenance
    assert p["method"] == "doy_window"
    assert p["baseline_start"] == "2017-01-01"
    assert p["baseline_end"] == "2019-12-31"
    assert p["n_input_times"] == 365 * 3
    assert p["derive_version"] == "0.1.0"


def test_climatology_window_counts_and_wrap():
    f = _seasonal_field(dt.date(2017, 1, 1), 3)   # no leap years in baseline
    clim = climatology(f, window_days=15)
    # 31 doys in a ±15 window x 3 years = 93 pooled samples per cell.
    assert clim.doy_counts[179, 0, 0] == 93
    # Year-boundary wrap: doy 1 pools Dec 17..31 + Jan 1..16. doy 366 has
    # no samples in a leap-free baseline, so the wrapped window holds 90.
    assert clim.doy_counts[0, 0, 0] == 90
    assert clim.doy_counts[364, 0, 0] == 90   # doy 365 wraps into January
    # doy 366's own ±15 window pools its *neighbors'* samples (90 of them);
    # only exact-doy pooling (window_days=0) isolates the leap-only slot.
    assert clim.doy_counts[365, 0, 0] == 90
    assert climatology(f, window_days=0).doy_counts[365, 0, 0] == 0
    # Windowed mean ≈ true cycle (smoothing is small for a slow cycle).
    expected = 10.0 + 5.0 * np.sin(2 * np.pi * 180 / 365.0)
    assert clim.doy_means[179, 0, 0] == pytest.approx(expected, abs=0.05)


def test_climatology_leap_day_handling():
    # Baseline 2019-01-01 + 1095 days spans 2019..2021 and includes the
    # leap year 2020.
    f = _seasonal_field(dt.date(2019, 1, 1), 3)
    assert any(t.month == 2 and t.day == 29 for t in f["times"])
    clim = climatology(f, window_days=0)
    # doy 60 exists every year (Feb 29 in leap years, Mar 1 otherwise),
    # so it pools one sample per year like any ordinary doy.
    assert clim.doy_counts[59, 0, 0] == 3
    # doy 366 (Dec 31) exists ONLY in leap years -> exactly one sample:
    # the 2020-12-31 value, recovered exactly.
    assert clim.doy_counts[365, 0, 0] == 1
    leap_val = f["values"][f["times"].index(dt.date(2020, 12, 31)), 0, 0]
    assert clim.doy_means[365, 0, 0] == pytest.approx(leap_val)
    # Ordinary doys still pool one sample per year.
    assert clim.doy_counts[0, 0, 0] == 3
    # Windowed pooling still wraps the year boundary (doy 1 sees December).
    # (The 1095-day baseline ends 2021-12-30, so the wrapped window holds
    # 90 pooled samples, not a full 93.)
    clim_w = climatology(f, window_days=15)
    assert clim_w.doy_counts[0, 0, 0] == 90
    assert clim_w.doy_counts[364, 0, 0] == 90


def test_climatology_nan_excluded_from_counts():
    f = _seasonal_field(dt.date(2017, 1, 1), 3)
    f["values"][:, 0, 0] = np.nan          # one cell always missing
    f["values"][0, 1, 1] = np.nan          # one missing sample
    clim = climatology(f, window_days=0)
    assert clim.doy_counts[0, 0, 0] == 0
    assert np.isnan(clim.doy_means[0, 0, 0])
    assert clim.doy_counts[0, 1, 1] == 2   # one of three samples missing


def test_climatology_validation():
    f = _seasonal_field(dt.date(2017, 1, 1), 1)
    with pytest.raises(ValueError):
        climatology(f, window_days=-1)
    with pytest.raises(ValueError):
        climatology(f, min_samples=0)
    bad = dict(f)
    bad["times"] = []
    with pytest.raises(ValueError):
        climatology(bad)


def test_climatology_duck_typed_object():
    stub = _seasonal_field(dt.date(2017, 1, 1), 2, as_dict=False)
    clim = climatology(stub, window_days=0)
    assert clim.doy_counts[0, 0, 0] == 2
    # ISO-string times (the survey-currents CurrentField shape) coerce too.
    stub2 = _seasonal_field(dt.date(2017, 1, 1), 2, as_dict=False)
    stub2.times = [t.isoformat() for t in stub2.times]
    clim2 = climatology(stub2, window_days=0)
    assert np.array_equal(clim2.doy_means, clim.doy_means, equal_nan=True)


# ---------------------------------------------------------------------------
# anomaly
# ---------------------------------------------------------------------------

def test_anomaly_of_baseline_period_is_zero():
    f = _seasonal_field(dt.date(2017, 1, 1), 3)
    clim = climatology(f, window_days=0)
    out = anomaly(f, clim)
    assert np.nanmax(np.abs(out["values"])) < 1e-9
    assert out["variable"] == "sst-anomaly"
    assert out["units"] == "degC"
    assert out["long_name"] == "Sea surface temperature anomaly"
    assert out["times"] == f["times"]
    p = out["provenance"]
    assert p["method"] == "anomaly"
    assert p["baseline_start"] == "2017-01-01"
    assert p["baseline_end"] == "2019-12-31"
    assert p["window_days"] == 0
    assert p["derive_version"] == "0.1.0"


def test_anomaly_detects_warm_event():
    base = _seasonal_field(dt.date(2017, 1, 1), 3)
    clim = climatology(base, window_days=0)
    f = _seasonal_field(dt.date(2020, 6, 1), 1)
    f["values"] = f["values"] + 2.0                     # uniform +2 deg warm event
    out = anomaly(f, clim)
    assert np.nanmax(np.abs(out["values"] - 2.0)) < 1e-9


def test_anomaly_long_name_from_variable():
    f = _seasonal_field(dt.date(2017, 1, 1), 3, long_name=None)
    del f["long_name"]
    clim = climatology(f, window_days=0)
    out = anomaly(f, clim)
    assert out["long_name"] == "Sst anomaly"


def test_anomaly_grid_mismatch_raises():
    f = _seasonal_field(dt.date(2017, 1, 1), 3)
    clim = climatology(f, window_days=0)
    g = dict(f)
    g["lats"] = f["lats"] + 0.5
    with pytest.raises(ValueError, match="lats"):
        anomaly(g, clim)
    g2 = dict(f)
    g2["lons"] = f["lons"][:-1]
    g2["values"] = f["values"][:, :, :-1]
    with pytest.raises(ValueError, match="lons"):
        anomaly(g2, clim)


def test_anomaly_nan_propagates():
    f = _seasonal_field(dt.date(2017, 1, 1), 3)
    f["values"][10, 2, 2] = np.nan
    clim = climatology(f, window_days=0)
    out = anomaly(f, clim)
    assert np.isnan(out["values"][10, 2, 2])
    assert np.nanmax(np.abs(out["values"])) < 1e-9


# ---------------------------------------------------------------------------
# standardized anomaly
# ---------------------------------------------------------------------------

def test_standardized_anomaly_basic():
    f = _seasonal_field(dt.date(2017, 1, 1), 3, noise=0.5)
    clim = climatology(f, window_days=15, min_samples=10)
    out = standardized_anomaly(f, clim)
    assert out["units"] == "σ"
    assert out["variable"] == "sst-anomaly"      # same variable family
    assert out["provenance"]["method"] == "standardized_anomaly"
    # Baseline-period standardized anomalies have ~unit spread per doy pool.
    assert np.isfinite(out["values"]).all()


def test_standardized_anomaly_masks_zero_std():
    # Constant field -> climatological std == 0 -> all NaN.
    lats, lons = _grid()
    times = _daily_dates(dt.date(2017, 1, 1), 365 * 3)
    values = np.full((len(times), 4, 5), 7.0)
    f = {"times": times, "lats": lats, "lons": lons, "values": values,
         "variable": "sst", "units": "degC"}
    clim = climatology(f, window_days=15)
    out = standardized_anomaly(f, clim)
    assert np.isnan(out["values"]).all()


def test_standardized_anomaly_masks_low_counts():
    f = _seasonal_field(dt.date(2017, 1, 1), 3, noise=0.5)
    clim = climatology(f, window_days=15, min_samples=10**9)  # impossible bar
    out = standardized_anomaly(f, clim)
    assert np.isnan(out["values"]).all()


# ---------------------------------------------------------------------------
# percent of normal
# ---------------------------------------------------------------------------

def test_percent_of_normal_basic():
    lats, lons = _grid()
    base_times = _daily_dates(dt.date(2017, 1, 1), 365 * 3)
    base = {"times": base_times, "lats": lats, "lons": lons,
            "values": np.full((len(base_times), 4, 5), 5.0),
            "variable": "precip", "units": "mm/day"}
    clim = climatology(base, window_days=0)
    ana_times = _daily_dates(dt.date(2020, 1, 1), 10)
    ana = {"times": ana_times, "lats": lats, "lons": lons,
           "values": np.full((len(ana_times), 4, 5), 10.0),
           "variable": "precip", "units": "mm/day"}
    out = percent_of_normal(ana, clim)
    assert np.nanmax(np.abs(out["values"] - 200.0)) < 1e-9
    assert out["units"] == "%"
    assert out["variable"] == "precip-percent-of-normal"
    assert out["provenance"]["method"] == "percent_of_normal"


def test_percent_of_normal_masks_near_zero_climatology():
    lats, lons = _grid()
    base_times = _daily_dates(dt.date(2017, 1, 1), 365 * 3)
    base = {"times": base_times, "lats": lats, "lons": lons,
            "values": np.zeros((len(base_times), 4, 5)),
            "variable": "precip", "units": "mm/day"}
    clim = climatology(base, window_days=0)
    ana = {"times": _daily_dates(dt.date(2020, 1, 1), 5), "lats": lats,
           "lons": lons, "values": np.full((5, 4, 5), 3.0),
           "variable": "precip", "units": "mm/day"}
    out = percent_of_normal(ana, clim)
    assert np.isnan(out["values"]).all()


# ---------------------------------------------------------------------------
# monthly means
# ---------------------------------------------------------------------------

def test_monthly_means():
    lats, lons = _grid()
    # January 2020: values = day-of-month -> mean 16.0; February: mean 15.0.
    times = _daily_dates(dt.date(2020, 1, 1), 60)   # Jan 1 .. Feb 29
    vals = np.array([t.day for t in times], dtype=float)
    values = np.broadcast_to(vals[:, None, None], (60, 4, 5)).copy()
    f = {"times": times, "lats": lats, "lons": lons, "values": values,
         "variable": "sst", "units": "degC"}
    out = monthly_means(f)
    assert out["times"] == [dt.date(2020, 1, 1), dt.date(2020, 2, 1)]
    assert out["values"].shape == (2, 4, 5)
    assert out["values"][0, 0, 0] == pytest.approx(16.0)
    assert out["values"][1, 0, 0] == pytest.approx(15.0)   # leap Feb: 1..29
    assert out["variable"] == "sst"                        # unchanged
    assert out["units"] == "degC"
    assert out["provenance"]["method"] == "monthly_means"
    assert out["provenance"]["n_months"] == 2


def test_monthly_means_nan_aware():
    lats, lons = _grid()
    times = _daily_dates(dt.date(2020, 3, 1), 31)
    values = np.full((31, 4, 5), 4.0)
    values[0] = np.nan
    f = {"times": times, "lats": lats, "lons": lons, "values": values,
         "variable": "sst", "units": "degC"}
    out = monthly_means(f)
    assert out["values"][0, 0, 0] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# trend
# ---------------------------------------------------------------------------

def test_trend_recovers_known_slope():
    lats, lons = _grid()
    times = _daily_dates(dt.date(2015, 1, 1), 365 * 5 + 1)  # 5 years, leap incl.
    yrs = np.array([(t - times[0]).days for t in times]) / 365.25
    values = np.broadcast_to((20.0 + 0.5 * yrs)[:, None, None],
                             (len(times), 4, 5)).copy()
    f = {"times": times, "lats": lats, "lons": lons, "values": values,
         "variable": "sst", "units": "degC"}
    out = trend(f)
    assert out["slope_per_year"].shape == (4, 5)
    assert np.nanmax(np.abs(out["slope_per_year"] - 0.5)) < 0.005  # within 1%
    assert (out["n_obs"] == len(times)).all()
    assert out["variable"] == "sst-trend"
    assert out["units"] == "degC/yr"
    assert out["time_start"] == dt.date(2015, 1, 1)
    assert out["provenance"]["method"] == "ols_trend"
    # intercept = fitted value at time_start ≈ 20.
    assert np.nanmax(np.abs(out["intercept"] - 20.0)) < 0.05


def test_trend_nan_cells():
    lats, lons = _grid()
    times = _daily_dates(dt.date(2015, 1, 1), 100)
    values = np.full((100, 4, 5), 3.0)
    values[:, 0, 0] = np.nan                       # never observed
    values[1:, 1, 1] = np.nan                      # single observation
    f = {"times": times, "lats": lats, "lons": lons, "values": values,
         "variable": "sst", "units": "degC"}
    out = trend(f)
    assert np.isnan(out["slope_per_year"][0, 0])
    assert np.isnan(out["slope_per_year"][1, 1])
    assert out["slope_per_year"][2, 2] == pytest.approx(0.0, abs=1e-12)
    assert out["n_obs"][0, 0] == 0
    assert out["n_obs"][1, 1] == 1


# ---------------------------------------------------------------------------
# symmetric limits
# ---------------------------------------------------------------------------

def test_symmetric_limits():
    rng = np.random.default_rng(1)
    vals = rng.normal(0, 2, size=(50, 6, 6))
    vmin, vmax = suggest_symmetric_limits(vals, quantile=0.99)
    assert vmin == -vmax
    assert vmax == pytest.approx(float(np.quantile(np.abs(vals), 0.99)))
    assert vmin < 0 < vmax


def test_symmetric_limits_degenerate():
    assert suggest_symmetric_limits(np.full((3, 2, 2), np.nan)) == (-1.0, 1.0)
    assert suggest_symmetric_limits(np.zeros((3, 2, 2))) == (-1.0, 1.0)


def test_symmetric_limits_bad_quantile():
    with pytest.raises(ValueError):
        suggest_symmetric_limits(np.ones(5), quantile=0.0)
    with pytest.raises(ValueError):
        suggest_symmetric_limits(np.ones(5), quantile=1.0)
    with pytest.raises(ValueError):
        suggest_symmetric_limits(np.ones(5), quantile=1.5)


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

def test_empty_times_raises():
    lats, lons = _grid()
    f = {"times": [], "lats": lats, "lons": lons,
         "values": np.zeros((0, 4, 5)), "variable": "sst", "units": "degC"}
    with pytest.raises(ValueError, match="empty"):
        climatology(f)


def test_shape_mismatch_raises():
    lats, lons = _grid()
    times = _daily_dates(dt.date(2020, 1, 1), 10)
    f = {"times": times, "lats": lats, "lons": lons,
         "values": np.zeros((9, 4, 5)), "variable": "sst", "units": "degC"}
    with pytest.raises(ValueError, match="9 time steps"):
        anomaly(f, climatology(
            {"times": times, "lats": lats, "lons": lons,
             "values": np.zeros((10, 4, 5)), "variable": "sst", "units": "degC"},
            window_days=0))


def test_missing_keys_raise():
    with pytest.raises(ValueError, match="'values'"):
        climatology({"times": [dt.date(2020, 1, 1)], "lats": [1.0], "lons": [2.0]})
    with pytest.raises(ValueError, match="missing attribute"):
        climatology(object())
