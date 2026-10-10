import os

import pytest

from flood.timeseries import (
    TIMESERIES_COLUMNS,
    ZonePassRecord,
    append_csv,
    read_csv,
    select,
    write_csv,
)


def _rec(zone="z1", date="2024-04-15", index="ndwi", ha=5.0, seed=0):
    return ZonePassRecord(
        zone_id=zone, zone_name=f"Zone {zone}", pass_id=f"p-{zone}-{date}-{seed}",
        date=date, doy=106, year=2024, index=index, water_pixels=500,
        valid_pixels=2400, total_pixels=2400, water_fraction=0.208,
        water_ha=ha, pixel_size_m=10.0, cloud_cover=0.05,
    )


def test_columns_include_monitor_contract():
    for col in ("water_ha", "water_fraction", "zone_id", "pass_id", "index", "date"):
        assert col in TIMESERIES_COLUMNS


def test_write_read_roundtrip(tmp_path):
    path = str(tmp_path / "ts.csv")
    write_csv([_rec(ha=5.0), _rec(date="2024-05-15", ha=6.0, seed=1)], path)
    back = read_csv(path)
    assert len(back) == 2
    assert back[0].water_ha == pytest.approx(5.0)
    assert back[1].date == "2024-05-15"
    assert back[0].cloud_cover == pytest.approx(0.05)


def test_append_dedupes(tmp_path):
    path = str(tmp_path / "ts.csv")
    recs = [_rec()]
    _, n1 = append_csv(recs, path)
    _, n2 = append_csv(recs, path)  # same records again
    assert (n1, n2) == (1, 0)
    assert len(read_csv(path)) == 1


def test_append_creates_file(tmp_path):
    path = str(tmp_path / "sub" / "ts.csv")
    _, n = append_csv([_rec()], path)
    assert n == 1 and os.path.exists(path)


def test_select_filters(tmp_path):
    recs = [_rec(zone="a"), _rec(zone="b", index="mndwi"), _rec(zone="a", date="2023-04-15", seed=2)]
    assert len(select(recs, zone_id="a")) == 2
    assert len(select(recs, index="mndwi")) == 1
    out = select(recs, zone_id="a")
    assert [r.date for r in out] == sorted(r.date for r in out)


def test_missing_columns_raise(tmp_path):
    path = str(tmp_path / "bad.csv")
    with open(path, "w") as fh:
        fh.write("zone_id,date\nz1,2024-01-01\n")
    with pytest.raises(ValueError, match="missing columns"):
        read_csv(path)


def test_key_uniqueness():
    assert _rec().key() == ("z1", "p-z1-2024-04-15-0", "ndwi")
