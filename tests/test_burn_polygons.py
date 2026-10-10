"""Tests for burn.polygons: vectorisation, area honesty, filtering."""

import json

import numpy as np
import pytest

from burn.polygons import (
    PIXEL_TRANSFORM,
    north_up_transform,
    vectorize,
    write_geojson,
)


def test_square_polygon_area_exact():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 3:7] = True  # 4x4 = 16 px
    feats = vectorize(mask, transform=PIXEL_TRANSFORM, pixel_size_m=10.0)
    assert len(feats) == 1
    f = feats[0]
    assert f["properties"]["pixels"] == 16
    assert f["properties"]["area_ha"] == pytest.approx(16 * 100 / 10_000.0)
    assert f["geometry"]["type"] == "Polygon"
    ring = f["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]  # explicitly closed


def test_two_components_two_features():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:2, 0:2] = True
    mask[7:9, 7:9] = True
    feats = vectorize(mask, pixel_size_m=10.0)
    assert len(feats) == 2
    assert sorted(f["properties"]["pixels"] for f in feats) == [4, 4]


def test_min_area_ha_filters_speckles():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True          # 1 px = 0.01 ha at 10 m
    mask[5:8, 5:8] = True      # 9 px = 0.09 ha
    feats = vectorize(mask, min_area_ha=0.05, pixel_size_m=10.0)
    assert len(feats) == 1
    assert feats[0]["properties"]["pixels"] == 9


def test_dominant_severity_attribute():
    mask = np.zeros((6, 6), dtype=bool)
    mask[:, :] = True
    sev = np.full((6, 6), 4, dtype=np.uint8)
    sev[0:2, :] = 1  # 12 px low vs 24 px high
    feats = vectorize(mask, severity=sev, pixel_size_m=10.0)
    assert feats[0]["properties"]["dominant_severity"] == 4
    assert feats[0]["properties"]["dominant_severity_name"] == "high severity"


def test_dominant_severity_none_without_raster():
    mask = np.zeros((4, 4), dtype=bool)
    mask[:, :] = True
    feats = vectorize(mask, pixel_size_m=10.0)
    assert feats[0]["properties"]["dominant_severity"] is None


def test_north_up_transform_geometry():
    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 0] = True
    t = north_up_transform(500_000.0, 4_600_000.0, 10.0)
    feats = vectorize(mask, transform=t, pixel_size_m=10.0)
    ring = feats[0]["geometry"]["coordinates"][0]
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    assert min(xs) == pytest.approx(500_000.0)
    assert max(xs) == pytest.approx(500_010.0)
    assert max(ys) == pytest.approx(4_600_000.0)
    assert min(ys) == pytest.approx(4_599_990.0)


def test_write_geojson_valid(tmp_path):
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    feats = vectorize(mask, pixel_size_m=10.0)
    path = write_geojson(feats, str(tmp_path / "out.geojson"))
    with open(path, encoding="utf-8") as fh:
        fc = json.load(fh)
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1


def test_empty_mask_no_features():
    assert vectorize(np.zeros((4, 4), dtype=bool)) == []
