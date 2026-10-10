"""Tests for phenology-aware anomaly detection."""

import math

import pytest

from vegetation import anomalies as an
from vegetation.synthetic import synthetic_timeseries
from vegetation.timeseries import ZonePassRecord


def _stats(mean):
    return {
        "mean": mean, "median": mean, "std": 0.02,
        "minimum": mean - 0.05, "maximum": mean + 0.05,
        "p10": mean - 0.03, "p90": mean + 0.03,
        "valid_pixels": 100, "total_pixels": 100,
    }


def _rec(zone, year, doy, mean, index="ndvi"):
    import datetime as dt

    date = dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)
    return ZonePassRecord.from_zonal_stats(
        zone, zone, f"p-{year}-{doy}", date.isoformat(), index, _stats(mean)
    )


def test_zscore_hand_computed():
    # baseline: 0.70, 0.74 at doy 180 in 2022/2023
    recs = [
        _rec("z1", 2022, 180, 0.70),
        _rec("z1", 2023, 180, 0.74),
        _rec("z1", 2024, 180, 0.60),  # target
    ]
    out = an.detect_anomalies(recs, "ndvi", window_days=15, z_threshold=-2.0)
    target = [r for r in out if r.year == 2024][0]
    mean = 0.72
    std = math.sqrt(((0.70 - 0.72) ** 2 + (0.74 - 0.72) ** 2) / 1)
    assert target.baseline_mean == pytest.approx(mean)
    assert target.baseline_std == pytest.approx(std)
    assert target.z_score == pytest.approx((0.60 - mean) / std)
    assert target.stressed is True
    assert target.baseline_years == 2


def test_healthy_pass_not_stressed():
    recs = [
        _rec("z1", 2022, 180, 0.70),
        _rec("z1", 2023, 180, 0.74),
        _rec("z1", 2024, 180, 0.73),
    ]
    out = an.detect_anomalies(recs, "ndvi")
    target = [r for r in out if r.year == 2024][0]
    assert target.stressed is False
    assert abs(target.z_score) < 2.0


def test_insufficient_history_noted():
    recs = [_rec("z1", 2024, 180, 0.60)]
    out = an.detect_anomalies(recs, "ndvi", min_years=1)
    assert out[0].z_score is None
    assert "history" in out[0].note


def test_phenology_window_excludes_off_season():
    # A January baseline must not judge a June pass: window=15 keeps them apart.
    recs = [
        _rec("z1", 2022, 15, 0.20),
        _rec("z1", 2023, 15, 0.22),
        _rec("z1", 2024, 180, 0.65),
    ]
    out = an.detect_anomalies(recs, "ndvi", window_days=15)
    target = [r for r in out if r.year == 2024][0]
    assert target.baseline_n == 0
    assert target.z_score is None


def test_doy_wraparound():
    assert an._doy_distance(363, 2) == 4
    assert an._doy_distance(180, 190) == 10


def test_planted_drought_detected():
    recs = synthetic_timeseries()
    out = an.detect_anomalies(recs, "ndvi")
    stressed = [r for r in out if r.stressed]
    # The planted 2024 orchard-north drought must be caught...
    hit = [r for r in stressed if r.zone_id == "orchard-north" and r.year == 2024]
    assert len(hit) > 0
    # ...and the healthy woodlot must stay quiet in 2024.
    false = [r for r in stressed if r.zone_id == "woodlot-south" and r.year == 2024]
    assert len(false) == 0


def test_anomaly_csv_columns(tmp_path):
    recs = [_rec("z1", 2022, 180, 0.7), _rec("z1", 2023, 180, 0.72)]
    out = an.detect_anomalies(recs, "ndvi")
    path = an.write_anomalies_csv(out, str(tmp_path / "a.csv"))
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    assert header == an.ANOMALY_COLUMNS
