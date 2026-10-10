"""Tests for zones, zonal masks, and the volume time-series CSV."""

import csv
import json
import os
import tempfile

import numpy as np
import pytest

from elev3d import synthetic as syn
from elev3d.diff import difference
from elev3d.timeseries import (
    TIMESERIES_COLUMNS,
    append_records,
    make_record,
    read_records,
)
from elev3d.volumes import zone_volumes
from elev3d.zonal import zone_mask
from elev3d.zones import Zone, load_zones_geojson


def _write_zones_tmp():
    fh = tempfile.NamedTemporaryFile("w", suffix=".geojson", delete=False)
    json.dump(syn.synthetic_zones_geojson(), fh)
    fh.close()
    return fh.name


def test_load_zones():
    path = _write_zones_tmp()
    try:
        zones = load_zones_geojson(path)
        assert len(zones) == 3
        assert {z.id for z in zones} == {"cut-pit", "fill-mound", "stable-road"}
        assert zones[0].kind == "pit"
    finally:
        os.unlink(path)


def test_load_zones_rejects_non_polygon():
    fh = tempfile.NamedTemporaryFile("w", suffix=".geojson", delete=False)
    json.dump({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {},
         "geometry": {"type": "Point", "coordinates": [0, 0]}}]}, fh)
    fh.close()
    try:
        with pytest.raises(ValueError, match="only Polygon"):
            load_zones_geojson(fh.name)
    finally:
        os.unlink(fh.name)


def test_rectangle_zone_mask_counts():
    dem1, dem2, t = syn.synthetic_dems()
    path = _write_zones_tmp()
    try:
        zones = load_zones_geojson(path)
    finally:
        os.unlink(path)
    cut = next(z for z in zones if z.id == "cut-pit")
    mask = zone_mask(cut, dem1.shape, t)
    # cut zone: cols 5–30 × rows 10–60 → 25 × 50 = 1250 px
    assert mask.sum() == 1250


def test_polygon_zone_mask_non_rectangular():
    zone = Zone(
        id="tri", name="tri",
        geometry={"type": "Polygon", "coordinates": [
            [[500000, 4600000], [500100, 4600000],
             [500000, 4599900], [500000, 4600000]]]},
    )
    mask = zone_mask(zone, (100, 100), syn.TRANSFORM)
    assert 0 < mask.sum() < 100  # right triangle ≈ half of 10×10


def test_timeseries_append_and_dedupe():
    dem1, dem2, t = syn.synthetic_dems()
    diff = difference(dem1, dem2, noise_floor=0.08).diff_thresholded
    zone = Zone(id="z1", name="Z1", geometry={
        "type": "Polygon", "coordinates": [
            [[500000, 4599000], [501000, 4599000], [501000, 4600000],
             [500000, 4600000], [500000, 4599000]]]})
    vols = zone_volumes(diff, zone, t, noise_floor=0.08)
    z = diff[zone_mask(zone, diff.shape, t)]
    z = z[np.isfinite(z)]
    zonal = {"mean": float(np.mean(z)), "median": float(np.median(z)),
             "std": float(np.std(z)), "minimum": float(np.min(z)),
             "maximum": float(np.max(z)), "p10": float(np.percentile(z, 10)),
             "p90": float(np.percentile(z, 90)),
             "valid_pixels": float(z.size), "total_pixels": float(z.size),
             "cloud_cover": 0.0}
    rec = make_record("z1", "Z1", "a_vs_b", "2025-06-15", zonal, vols.to_dict())

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "volumes.csv")
        assert append_records(path, [rec]) == 1
        assert append_records(path, [rec]) == 0  # dedupe: no-op
        rows = read_records(path)
        assert len(rows) == 1
        assert rows[0]["index"] == "dem-diff"
        assert float(rows[0]["cut_m3"]) == pytest.approx(vols.cut_m3)
        assert float(rows[0]["net_m3"]) == pytest.approx(vols.net_m3)


def test_timeseries_column_schema_stable():
    assert TIMESERIES_COLUMNS[:17] == [
        "zone_id", "zone_name", "pass_id", "date", "doy", "year", "index",
        "mean", "median", "std", "minimum", "maximum", "p10", "p90",
        "valid_pixels", "total_pixels", "cloud_cover",
    ]
    for col in ("cut_m3", "fill_m3", "net_m3", "volume_sigma_m3", "confidence"):
        assert col in TIMESERIES_COLUMNS


def test_make_record_date_parsing():
    rec = make_record("z", "Z", "p", "2025-06-15", {}, {})
    assert rec.doy == 166
    assert rec.year == 2025
    assert rec.index == "dem-diff"


def test_csv_header_written_once():
    rec = make_record("z", "Z", "p", "2025-06-15", {}, {})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "v.csv")
        append_records(path, [rec])
        rec2 = make_record("z", "Z", "p2", "2025-06-16", {}, {})
        append_records(path, [rec2])
        with open(path, encoding="utf-8") as fh:
            header_lines = [ln for ln in fh if ln.startswith("zone_id")]
        assert len(header_lines) == 1
        assert len(read_records(path)) == 2
