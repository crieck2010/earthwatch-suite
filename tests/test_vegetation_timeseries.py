"""Tests for time-series CSV records: schema, round-trip, incremental append."""

import pytest

from vegetation import timeseries as ts
from vegetation.synthetic import synthetic_timeseries


def _stats(mean):
    return {
        "mean": mean, "median": mean, "std": 0.02,
        "minimum": mean - 0.05, "maximum": mean + 0.05,
        "p10": mean - 0.03, "p90": mean + 0.03,
        "valid_pixels": 100, "total_pixels": 100,
    }


def _rec(zone="z1", date="2024-06-15", index="ndvi", mean=0.7):
    return ts.ZonePassRecord.from_zonal_stats(
        zone, "Zone 1", f"pass-{date}", date, index, _stats(mean)
    )


def test_record_derived_fields():
    r = _rec(date="2024-06-15")
    assert r.doy == 167
    assert r.year == 2024
    assert r.key() == ("z1", "pass-2024-06-15", "ndvi")


def test_csv_roundtrip(tmp_path):
    recs = [_rec(mean=0.7), _rec(date="2024-07-01", mean=0.75, index="evi")]
    path = str(tmp_path / "ts.csv")
    ts.write_csv(recs, path)
    back = ts.read_csv(path)
    assert len(back) == 2
    assert back[0].mean == pytest.approx(0.7)
    assert back[1].index == "evi"
    assert back[0].cloud_cover is None


def test_csv_schema_columns(tmp_path):
    path = str(tmp_path / "ts.csv")
    ts.write_csv([_rec()], path)
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    assert header == ts.TIMESERIES_COLUMNS


def test_append_dedupes(tmp_path):
    path = str(tmp_path / "ts.csv")
    _, n1 = ts.append_csv([_rec(), _rec(date="2024-07-01")], path)
    _, n2 = ts.append_csv([_rec(), _rec(date="2024-08-01")], path)  # one dup
    assert (n1, n2) == (2, 1)
    assert len(ts.read_csv(path)) == 3


def test_select_filters_and_sorts():
    recs = [
        _rec(zone="b", date="2024-07-01"),
        _rec(zone="a", date="2024-06-01"),
        _rec(zone="a", date="2024-05-01", index="evi"),
    ]
    out = ts.select(recs, zone_id="a", index="ndvi")
    assert [r.date for r in out] == ["2024-06-01"]
    out = ts.select(recs, zone_id="a")
    assert [r.date for r in out] == ["2024-05-01", "2024-06-01"]


def test_read_csv_missing_column(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("zone_id,date\nz1,2024-01-01\n")
    with pytest.raises(ValueError, match="missing columns"):
        ts.read_csv(str(path))


def test_synthetic_series_shape():
    recs = synthetic_timeseries()
    zones = {r.zone_id for r in recs}
    assert zones == {"orchard-north", "pasture-east", "woodlot-south"}
    assert {r.index for r in recs} == {"ndvi", "evi"}
    # one row per zone per pass per index
    ndvi = [r for r in recs if r.index == "ndvi"]
    assert len({(r.zone_id, r.pass_id) for r in ndvi}) == len(ndvi)


def test_synthetic_deterministic():
    a = synthetic_timeseries(seed=42)
    b = synthetic_timeseries(seed=42)
    assert [r.mean for r in a] == [r.mean for r in b]
