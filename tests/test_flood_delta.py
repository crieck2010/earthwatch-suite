import os

import pytest

from flood.delta import (
    baseline_records,
    compute_all_deltas,
    compute_delta,
    write_deltas_csv,
    write_deltas_json,
)
from flood.timeseries import ZonePassRecord


def _rec(zone, date, ha, index="ndwi", cloud=None):
    return ZonePassRecord(
        zone_id=zone, zone_name=zone, pass_id=f"p-{zone}-{date}", date=date,
        doy=int(date[5:7]) * 30, year=int(date[:4]), index=index,
        water_pixels=int(ha * 100), valid_pixels=2400, total_pixels=2400,
        water_fraction=ha / 24.0, water_ha=ha, pixel_size_m=10.0,
        cloud_cover=cloud,
    )


def _baseline():
    # Dry-season (JJA) 2023 baseline for zone z1: 10, 12, 14 ha.
    return [
        _rec("z1", "2023-06-15", 10.0),
        _rec("z1", "2023-07-15", 12.0),
        _rec("z1", "2023-08-15", 14.0),
    ]


def test_delta_hand_computed():
    records = _baseline()
    target = _rec("z1", "2024-04-15", 20.0)
    d = compute_delta(target, records)
    assert d.baseline_mean_ha == pytest.approx(12.0)
    assert d.baseline_std_ha == pytest.approx(2.0)
    assert d.delta_ha == pytest.approx(8.0)
    assert d.rel_change == pytest.approx(8.0 / 12.0)
    assert d.z_score == pytest.approx(4.0)
    assert d.baseline_passes == 3
    assert d.flooded  # default threshold 0.0


def test_no_baseline_gives_nones():
    d = compute_delta(_rec("z1", "2024-04-15", 20.0), [])
    assert d.delta_ha is None and d.z_score is None and d.rel_change is None
    assert d.baseline_passes == 0 and not d.flooded


def test_single_baseline_pass_no_z():
    records = [_rec("z1", "2023-07-15", 12.0)]
    d = compute_delta(_rec("z1", "2024-04-15", 20.0), records)
    assert d.delta_ha == pytest.approx(8.0)
    assert d.z_score is None  # std undefined with n=1
    assert d.baseline_std_ha is None


def test_zero_baseline_mean_rel_is_none():
    records = [_rec("z1", "2023-07-15", 0.0), _rec("z1", "2023-08-15", 0.0)]
    d = compute_delta(_rec("z1", "2024-04-15", 5.0), records)
    assert d.delta_ha == pytest.approx(5.0)
    assert d.rel_change is None  # normally-dry catchment, first water


def test_baseline_window_selection():
    records = _baseline() + [
        _rec("z1", "2024-04-15", 99.0),  # the target itself: excluded
        _rec("z1", "2023-04-15", 99.0),  # wrong month: excluded
        _rec("z1", "2022-07-15", 99.0),  # two years back: excluded (years=1)
        _rec("z1", "2023-07-15", 99.0, cloud=90.0),  # cloudy duplicate month
        _rec("z2", "2023-07-15", 99.0),  # other zone: excluded
    ]
    base = baseline_records(records, "z1", "ndwi", "2024-04-15")
    # The cloudy 2023-07-15 record is excluded; the clean one stays.
    assert sorted(r.date for r in base) == ["2023-06-15", "2023-07-15", "2023-08-15"]
    assert all(r.water_ha < 50 for r in base)


def test_flood_threshold_gating():
    records = _baseline()
    d = compute_delta(_rec("z1", "2024-04-15", 12.5), records, flood_threshold_ha=1.0)
    assert not d.flooded  # delta 0.5 < 1.0
    d2 = compute_delta(_rec("z1", "2024-04-15", 14.0), records, flood_threshold_ha=1.0)
    assert d2.flooded


def test_compute_all_deltas_sorts_and_covers_zones():
    records = _baseline() + [_rec("z1", "2024-04-15", 20.0), _rec("z1", "2024-05-15", 13.0)]
    out = compute_all_deltas(records, "ndwi")
    assert len(out) == 5  # baseline passes get deltas vs empty baseline
    assert [d.date for d in out] == sorted(d.date for d in out)


def test_write_csv_and_json(tmp_path):
    records = _baseline() + [_rec("z1", "2024-04-15", 20.0)]
    out = compute_all_deltas(records, "ndwi")
    csv_path = write_deltas_csv(out, str(tmp_path / "delta.csv"))
    json_path = write_deltas_json(out, str(tmp_path / "delta.json"))
    assert os.path.exists(csv_path) and os.path.exists(json_path)
    with open(csv_path) as fh:
        header = fh.readline().strip().split(",")
    assert "delta_ha" in header and "z_score" in header and "flooded" in header
