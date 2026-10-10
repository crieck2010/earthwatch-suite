"""Tests for zone loading and zonal LST statistics."""

import json
import os

import numpy as np
import pytest

from thermal.zonal import zonal_lst_stats, zone_mask
from thermal.zones import Zone, load_zones_geojson


@pytest.fixture
def zones_path(tmp_path):
    path = tmp_path / "zones.geojson"
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "properties": {"id": "a", "name": "Alpha", "kind": "urban"},
             "geometry": {"type": "Polygon",
                          "coordinates": [[[0, 0], [60, 0], [60, 60], [0, 60], [0, 0]]]}},
            {"type": "Feature",
             "properties": {"id": "b"},
             "geometry": {"type": "Polygon",
                          "coordinates": [[[60, 0], [120, 0], [120, 60], [60, 60], [60, 0]]]}},
        ],
    }))
    return str(path)


def test_load_zones(zones_path):
    zones = load_zones_geojson(zones_path)
    assert [z.id for z in zones] == ["a", "b"]
    assert zones[0].name == "Alpha"
    assert zones[0].kind == "urban"
    assert zones[1].name == "b"  # falls back to id


def test_load_zones_rejects_non_polygon(tmp_path):
    path = tmp_path / "bad.geojson"
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {},
                      "geometry": {"type": "Point", "coordinates": [0, 0]}}],
    }))
    with pytest.raises(ValueError):
        load_zones_geojson(path)


def test_load_zones_empty(tmp_path):
    path = tmp_path / "empty.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    with pytest.raises(ValueError):
        load_zones_geojson(path)


def _rect_zone():
    return Zone(id="r", name="R",
                geometry={"type": "Polygon",
                          "coordinates": [[[0, 0], [60, 0], [60, 60], [0, 60], [0, 0]]]})


def test_zone_mask_rectangle_exact():
    # 30 m pixels, 4x4 raster covering 0..120 m; zone covers 0..60 m
    mask = zone_mask(_rect_zone(), (4, 4), (30.0, 0.0, 0.0, 0.0, -30.0, 120.0))
    assert mask.shape == (4, 4)
    # rows 2..3 (y 0..60), cols 0..1 (x 0..60)
    expected = np.zeros((4, 4), dtype=bool)
    expected[2:4, 0:2] = True
    assert (mask == expected).all()


def test_zonal_stats_known_values():
    arr = np.full((4, 4), 25.0)
    arr[2:4, 0:2] = 35.0  # the zone's pixels
    stats = zonal_lst_stats(arr, _rect_zone(), (30.0, 0.0, 0.0, 0.0, -30.0, 120.0))
    assert stats["mean"] == pytest.approx(35.0)
    assert stats["minimum"] == pytest.approx(35.0)
    assert stats["maximum"] == pytest.approx(35.0)
    assert stats["valid_pixels"] == 4.0
    assert stats["total_pixels"] == 4.0


def test_zonal_stats_ignores_nan():
    arr = np.full((4, 4), 25.0)
    arr[2, 0] = np.nan  # clouded pixel inside the zone
    stats = zonal_lst_stats(arr, _rect_zone(), (30.0, 0.0, 0.0, 0.0, -30.0, 120.0))
    assert stats["mean"] == pytest.approx(25.0)
    assert stats["valid_pixels"] == 3.0
    assert stats["total_pixels"] == 4.0


def test_zonal_stats_all_nan():
    arr = np.full((4, 4), np.nan)
    stats = zonal_lst_stats(arr, _rect_zone(), (30.0, 0.0, 0.0, 0.0, -30.0, 120.0))
    assert np.isnan(stats["mean"])
    assert stats["valid_pixels"] == 0.0
