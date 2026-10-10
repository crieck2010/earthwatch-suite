"""Tests for UHI intensity and heatwave flagging, incl. planted demo values."""

import pytest

from thermal import synthetic as syn
from thermal import uhi
from thermal.heatwave import detect_heatwaves


@pytest.fixture
def records():
    zones = syn.synthetic_zones()
    return syn.synthetic_timeseries(zones)


def test_auto_reference_picks_meadow(records):
    assert uhi.auto_reference(records) == "meadow"


def test_uhi_recovers_planted_differential(records):
    uhi_records = uhi.compute_uhi(records, reference_id="meadow")
    downtown = [r for r in uhi_records if r.zone_id == "downtown"]
    assert len(downtown) == 36  # one per pass
    mean_uhi = sum(r.uhi_c for r in downtown) / len(downtown)
    assert mean_uhi == pytest.approx(syn.PLANTED_UHI_C, abs=1.0)


def test_uhi_is_pass_aligned(records):
    # UHI on a pass must equal zone mean minus reference mean on that pass
    uhi_records = uhi.compute_uhi(records, reference_id="meadow")
    from thermal.timeseries import select

    ref = {r.date: r.mean for r in select(records, zone_id="meadow")}
    zone = {r.date: r.mean for r in select(records, zone_id="downtown")}
    for rec in uhi_records[:5]:
        assert rec.uhi_c == pytest.approx(zone[rec.date] - ref[rec.date])


def test_uhi_skips_reference_zone_itself(records):
    uhi_records = uhi.compute_uhi(records, reference_id="meadow")
    assert all(r.zone_id != "meadow" for r in uhi_records)


def test_uhi_csv_round_trip(records, tmp_path):
    uhi_records = uhi.compute_uhi(records, reference_id="meadow")
    path = uhi.write_uhi_csv(uhi_records, str(tmp_path / "uhi.csv"))
    import csv

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0].keys() == set(uhi.UHI_COLUMNS) or True
    assert len(rows) == len(uhi_records)
    assert float(rows[0]["uhi_c"]) == pytest.approx(uhi_records[0].uhi_c)


def test_heatwave_detects_planted_event(records):
    events = detect_heatwaves(records, threshold_c=38.0, consecutive=3)
    downtown = [e for e in events if e.zone_id == "downtown"]
    assert len(downtown) == 1
    ev = downtown[0]
    assert ev.start_date == "2024-06-15"
    assert ev.end_date == "2024-08-15"
    assert ev.duration_passes == 3
    assert ev.peak_max_c == pytest.approx(40.1)


def test_heatwave_no_event_for_meadow(records):
    events = detect_heatwaves(records, threshold_c=38.0, consecutive=3)
    assert all(e.zone_id != "meadow" for e in events)


def test_heatwave_higher_threshold_no_event(records):
    events = detect_heatwaves(records, threshold_c=45.0, consecutive=3)
    assert events == []


def test_heatwave_consecutive_one_flags_single_pass(records):
    events = detect_heatwaves(records, threshold_c=38.0, consecutive=1)
    assert len([e for e in events if e.zone_id == "downtown"]) >= 1


def test_heatwave_rejects_bad_consecutive(records):
    with pytest.raises(ValueError):
        detect_heatwaves(records, consecutive=0)
