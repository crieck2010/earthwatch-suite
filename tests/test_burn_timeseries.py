"""Tests for burn.timeseries: records, CSV appends, dedupe, schema."""

import os

import pytest

from burn.severity import class_areas
from burn.synthetic import DEMO_PIXEL_M
from burn.timeseries import (
    TIMESERIES_COLUMNS,
    ZonePassRecord,
    append_records,
    read_records,
)


def _record(**kw):
    base = dict(
        zone_id="z1", zone_name="Z1", pass_id="p1", date="2024-08-01",
        doy=214, year=2024, index="dnbr", valid_pixels=100, total_pixels=100,
        burned_pixels=25, burned_fraction=0.25, burned_ha=25.0,
        pixel_size_m=10.0, cloud_cover=5.0,
    )
    base.update(kw)
    return ZonePassRecord(**base)


def test_columns_stable_and_shared_prefix():
    # The shared prefix must mirror the sibling engines' schemas.
    assert TIMESERIES_COLUMNS[:7] == [
        "zone_id", "zone_name", "pass_id", "date", "doy", "year", "index",
    ]
    assert "burned_ha" in TIMESERIES_COLUMNS
    assert "burned_fraction" in TIMESERIES_COLUMNS
    assert "cloud_cover" in TIMESERIES_COLUMNS
    assert len(TIMESERIES_COLUMNS) == len(set(TIMESERIES_COLUMNS))


def test_append_and_read_roundtrip(tmp_path):
    path = str(tmp_path / "ts.csv")
    recs = [_record(), _record(zone_id="z2", pass_id="p2")]
    assert append_records(path, recs) == 2
    back = read_records(path)
    assert len(back) == 2
    assert back[0].zone_id == "z1"
    assert back[0].burned_ha == pytest.approx(25.0)
    assert back[0].cloud_cover == pytest.approx(5.0)


def test_dedupe_on_zone_pass_index(tmp_path):
    path = str(tmp_path / "ts.csv")
    assert append_records(path, [_record()]) == 1
    assert append_records(path, [_record()]) == 0  # no-op re-append
    assert append_records(path, [_record(index="disturbance")]) == 1  # new index
    assert len(read_records(path)) == 2


def test_incremental_appends_dont_rewrite(tmp_path):
    path = str(tmp_path / "ts.csv")
    append_records(path, [_record(pass_id="p1")])
    mtime1 = os.path.getmtime(path)
    append_records(path, [_record(pass_id="p2")])
    rows = read_records(path)
    assert [r.pass_id for r in rows] == ["p1", "p2"]


def test_from_class_areas(tmp_path):
    import numpy as np

    sev = np.full((10, 10), 0, dtype=np.uint8)
    sev[:5, :] = 4  # 50 px high severity
    areas = class_areas(sev, None, DEMO_PIXEL_M)
    rec = ZonePassRecord.from_class_areas(
        zone_id="z", zone_name="Z", pass_id="p", date="2024-08-01",
        index="dnbr", areas=areas, total_pixels=100,
        pixel_size_m=DEMO_PIXEL_M, cloud_cover=2.0,
    )
    assert rec.burned_ha == pytest.approx(0.50)
    assert rec.high_severity_ha == pytest.approx(0.50)
    assert rec.burned_pixels == 50
    assert rec.doy == 214 and rec.year == 2024


def test_missing_cloud_cover_roundtrips(tmp_path):
    path = str(tmp_path / "ts.csv")
    rec = _record(cloud_cover=None)
    append_records(path, [rec])
    back = read_records(path)[0]
    assert back.cloud_cover is None
