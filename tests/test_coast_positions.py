"""Tests for positions CSV (incremental appends, dedupe, round-trip)."""

from coast.positions import (
    POSITIONS_COLUMNS,
    TransectPassRecord,
    append_records,
    read_positions,
)


def _rec(tid="t1", pid="p1", pos=100.0, valid=True):
    return TransectPassRecord(
        transect_id=tid, transect_name="T1", site_id="s1", site_name="S1",
        pass_id=pid, date="2021-06-01", doy=152, year=2021, index="ndwi",
        position_m=pos, shoreline_length_m=500.0, valid=valid,
        cloud_cover=5.0, pixel_size_m=10.0,
    )


def test_append_and_read_round_trip(tmp_path):
    path = str(tmp_path / "positions.csv")
    added, skipped = append_records(path, [_rec(), _rec(tid="t2", pos=200.0)])
    assert (added, skipped) == (2, 0)
    recs = read_positions(path)
    assert len(recs) == 2
    assert recs[0].position_m == 100.0
    assert recs[1].transect_id == "t2"
    assert recs[0].valid is True


def test_dedupe_on_transect_and_pass(tmp_path):
    path = str(tmp_path / "positions.csv")
    append_records(path, [_rec()])
    added, skipped = append_records(path, [_rec(), _rec(pid="p2")])
    assert (added, skipped) == (1, 1)
    assert len(read_positions(path)) == 2


def test_nan_round_trip(tmp_path):
    path = str(tmp_path / "positions.csv")
    append_records(path, [_rec(pos=float("nan"), valid=False)])
    (rec,) = read_positions(path)
    assert rec.position_m != rec.position_m  # NaN
    assert rec.valid is False


def test_missing_file_reads_empty(tmp_path):
    assert read_positions(str(tmp_path / "nope.csv")) == []


def test_column_order_stable():
    assert POSITIONS_COLUMNS[:4] == [
        "transect_id", "transect_name", "site_id", "site_name",
    ]
    assert "position_m" in POSITIONS_COLUMNS
