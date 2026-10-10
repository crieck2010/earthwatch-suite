"""Tests for storm-event deltas."""

import datetime as dt

import pytest

from coast.positions import TransectPassRecord
from coast.storm import STORM_COLUMNS, storm_deltas, write_storm_csv


def _rec(tid, date, pos):
    d = dt.date.fromisoformat(date)
    return TransectPassRecord(
        transect_id=tid, transect_name=tid, site_id="s1", site_name="S1",
        pass_id=f"p-{date}", date=date, doy=d.timetuple().tm_yday,
        year=d.year, index="ndwi", position_m=pos,
        shoreline_length_m=1000.0, valid=pos == pos,
        cloud_cover=0.0, pixel_size_m=10.0,
    )


def test_storm_displacement_math():
    recs = [
        _rec("t", "2021-01-15", 430.0),
        _rec("t", "2021-07-20", 420.0),  # after the storm
        _rec("t", "2022-01-15", 428.0),
    ]
    (d,) = storm_deltas(recs, "2021-06-15")
    assert d.before_date == "2021-01-15"
    assert d.after_date == "2021-07-20"
    assert d.displacement_m == pytest.approx(-10.0)


def test_event_on_pass_date_excluded():
    # A pass exactly on the event date is neither before nor after.
    recs = [_rec("t", "2021-06-15", 425.0), _rec("t", "2021-07-20", 420.0)]
    assert storm_deltas(recs, "2021-06-15") == []


def test_missing_side_skipped_not_zero_filled():
    recs = [_rec("t", "2021-07-20", 420.0), _rec("t", "2022-01-15", 428.0)]
    assert storm_deltas(recs, "2021-06-15") == []


def test_multiple_transects():
    recs = [
        _rec("a", "2021-01-15", 430.0), _rec("a", "2021-07-20", 420.0),
        _rec("b", "2021-01-15", 380.0), _rec("b", "2021-07-20", 380.0),
    ]
    deltas = {d.transect_id: d for d in storm_deltas(recs, "2021-06-15")}
    assert deltas["a"].displacement_m == pytest.approx(-10.0)
    assert deltas["b"].displacement_m == pytest.approx(0.0)


def test_write_storm_csv(tmp_path):
    recs = [_rec("t", "2021-01-15", 430.0), _rec("t", "2021-07-20", 420.0)]
    path = str(tmp_path / "storm.csv")
    write_storm_csv(path, storm_deltas(recs, "2021-06-15"))
    lines = open(path).read().splitlines()
    assert lines[0] == ",".join(STORM_COLUMNS)
    assert "-10.000" in lines[1]
