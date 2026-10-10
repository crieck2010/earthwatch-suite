"""Tests for autopilot.stride."""

import pytest

from autopilot import STRIDE_TABLE, recommend_stride


def test_each_phenomenon_returns_documented_values():
    tide = recommend_stride("tide")
    assert (tide["stride_days"], tide["stride_hours"], tide["forecast_hours"]) == (1, 3, None)
    assert "per_event" not in tide

    syn = recommend_stride("synoptic")
    assert (syn["stride_days"], syn["stride_hours"]) == (1, 6)
    assert syn["forecast_hours"] == (0, 6, 12, 18)

    seas = recommend_stride("seasonal")
    assert (seas["stride_days"], seas["stride_hours"], seas["forecast_hours"]) == (7, 0, None)

    clim = recommend_stride("climate")
    assert (clim["stride_days"], clim["stride_hours"], clim["forecast_hours"]) == (30, 0, None)

    ev = recommend_stride("event")
    assert ev["per_event"] is True
    assert ev["stride_days"] == 1


def test_all_table_keys_have_reasons():
    for key, entry in STRIDE_TABLE.items():
        assert entry["reason"], f"{key} missing reason"
        out = recommend_stride(key)
        assert out["phenomenon"] == key
        assert out["reason"] == entry["reason"]


def test_source_is_recorded():
    out = recommend_stride("synoptic", source="gfs-wind")
    assert out["source"] == "gfs-wind"
    assert recommend_stride("tide")["source"] is None


def test_unknown_phenomenon_raises_listing_valid_keys():
    with pytest.raises(ValueError) as exc:
        recommend_stride("earthquake")
    msg = str(exc.value)
    for key in ("tide", "synoptic", "seasonal", "climate", "event"):
        assert key in msg
