"""Tests for time-series records: CSV round-trip, incremental dedupe, schema."""

import os

import pytest

from thermal import synthetic as syn
from thermal.timeseries import (
    TIMESERIES_COLUMNS,
    append_csv,
    make_record,
    read_csv,
    select,
)


def _stats(mean=30.0):
    return {"mean": mean, "median": mean - 0.4, "std": 2.0,
            "minimum": mean - 6.0, "maximum": mean + 5.0,
            "p10": mean - 3.5, "p90": mean + 3.5,
            "valid_pixels": 100.0, "total_pixels": 100.0}


def test_make_record_doy_and_index():
    rec = make_record("z", "Z", "p-2024-07-15", "2024-07-15", _stats())
    assert rec.doy == 197
    assert rec.year == 2024
    assert rec.index == "lst"


def test_csv_round_trip(tmp_path):
    path = str(tmp_path / "ts.csv")
    rec = make_record("z", "Z", "p-2024-07-15", "2024-07-15", _stats(31.5),
                      cloud_cover=12.5)
    _, n = append_csv([rec], path)
    assert n == 1
    back = read_csv(path)
    assert len(back) == 1
    assert back[0].mean == pytest.approx(31.5)
    assert back[0].cloud_cover == pytest.approx(12.5)
    assert back[0].index == "lst"


def test_append_is_idempotent(tmp_path):
    path = str(tmp_path / "ts.csv")
    rec = make_record("z", "Z", "p-2024-07-15", "2024-07-15", _stats())
    _, n1 = append_csv([rec], path)
    _, n2 = append_csv([rec], path)
    assert (n1, n2) == (1, 0)
    assert len(read_csv(path)) == 1


def test_append_only_adds_new_passes(tmp_path):
    path = str(tmp_path / "ts.csv")
    r1 = make_record("z", "Z", "p-2024-06-15", "2024-06-15", _stats())
    r2 = make_record("z", "Z", "p-2024-07-15", "2024-07-15", _stats())
    append_csv([r1], path)
    _, n = append_csv([r1, r2], path)
    assert n == 1
    assert len(read_csv(path)) == 2


def test_schema_columns_stable():
    assert TIMESERIES_COLUMNS[0] == "zone_id"
    assert "mean" in TIMESERIES_COLUMNS
    assert "maximum" in TIMESERIES_COLUMNS
    assert "cloud_cover" in TIMESERIES_COLUMNS
    assert TIMESERIES_COLUMNS.count("zone_id") == 1


def test_csv_header_matches_schema(tmp_path):
    import csv

    path = str(tmp_path / "ts.csv")
    append_csv([make_record("z", "Z", "p-2024-07-15", "2024-07-15", _stats())], path)
    with open(path, newline="") as fh:
        header = next(csv.reader(fh))
    assert header == TIMESERIES_COLUMNS


def test_select_filters_and_sorts():
    zones = syn.synthetic_zones()
    records = syn.synthetic_timeseries(zones, start="2024-01-15", end="2024-03-15")
    sel = select(records, zone_id="downtown")
    assert all(r.zone_id == "downtown" for r in sel)
    dates = [r.date for r in sel]
    assert dates == sorted(dates)


def test_synthetic_series_shape():
    zones = syn.synthetic_zones()
    records = syn.synthetic_timeseries(zones)
    # 2 zones x 36 monthly passes (2022-2024)
    assert len(records) == 72
