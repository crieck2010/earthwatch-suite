"""Tests for day-of-year heat anomalies."""

import math

import pytest

from thermal import synthetic as syn
from thermal.anomalies import ANOMALY_COLUMNS, detect_anomalies, write_anomalies_csv


@pytest.fixture
def records():
    zones = syn.synthetic_zones()
    return syn.synthetic_timeseries(zones)


def test_anomaly_flags_hot_july_2024(records):
    anom = detect_anomalies(records, field="mean")
    hot = [a for a in anom if a.hot and a.zone_id == "downtown"]
    dates = [a.date for a in hot]
    assert "2024-07-15" in dates  # the planted heatwave peak


def test_anomaly_baseline_uses_prior_years_only(records):
    anom = detect_anomalies(records, field="mean")
    july24 = next(a for a in anom
                  if a.zone_id == "downtown" and a.date == "2024-07-15")
    assert july24.baseline_years == 2  # 2022 and 2023
    assert "2024" not in str(july24.baseline_mean)


def test_anomaly_insufficient_history_flagged(records):
    anom = detect_anomalies(records, field="mean", min_years=1)
    first = [a for a in anom if a.year == 2022]
    assert all(a.z_score is None for a in first)
    assert all("insufficient" in a.note for a in first)


def test_anomaly_z_score_sign_hot_positive(records):
    anom = detect_anomalies(records, field="mean")
    hot = [a for a in anom if a.hot]
    assert hot, "expected at least one hot anomaly in the planted demo"
    assert all(a.z_score >= 2.0 for a in hot)


def test_anomaly_cold_pass_not_hot(records):
    anom = detect_anomalies(records, field="mean")
    jan = [a for a in anom if a.date == "2024-01-15" and a.zone_id == "downtown"]
    assert jan and not any(a.hot for a in jan)


def test_anomaly_csv_schema(records, tmp_path):
    anom = detect_anomalies(records)
    path = write_anomalies_csv(anom, str(tmp_path / "anom.csv"))
    import csv

    with open(path, newline="") as fh:
        header = next(csv.reader(fh))
    assert header == ANOMALY_COLUMNS
