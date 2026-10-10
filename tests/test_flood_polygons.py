import json

import numpy as np
import pytest

from flood.polygons import (
    PIXEL_TRANSFORM,
    north_up_transform,
    vectorize,
    write_polygons_geojson,
)


def _ring_valid(ring):
    return len(ring) >= 4 and ring[0] == ring[-1]


def test_single_block_exact_area():
    mask = np.zeros((6, 6), dtype=bool)
    mask[2:4, 2:4] = True  # 2x2 block
    feats = vectorize(mask, transform=PIXEL_TRANSFORM)
    assert len(feats) == 1
    props = feats[0]["properties"]
    assert props["pixel_count"] == 4
    # pixel transform => 1x1 unit pixels, area exactly 4
    ring = feats[0]["geometry"]["coordinates"][0]
    assert _ring_valid(ring)
    assert props["area_ha"] == pytest.approx(4 / 10_000)


def test_l_shape_area():
    mask = np.array([[1, 0], [1, 0], [1, 1]], dtype=bool)
    feats = vectorize(mask, transform=PIXEL_TRANSFORM, min_pixels=1)
    assert len(feats) == 1
    assert feats[0]["properties"]["pixel_count"] == 4


def test_min_pixels_filters():
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, 0] = True
    mask[4:7, 4:7] = True
    feats = vectorize(mask, transform=PIXEL_TRANSFORM, min_pixels=4)
    assert len(feats) == 1
    assert feats[0]["properties"]["pixel_count"] == 9


def test_empty_mask_no_features():
    assert vectorize(np.zeros((4, 4), dtype=bool)) == []


def test_donut_has_hole():
    mask = np.ones((5, 5), dtype=bool)
    mask[1:4, 1:4] = False  # 3x3 hole
    feats = vectorize(mask, transform=PIXEL_TRANSFORM, min_pixels=1)
    assert len(feats) == 1
    rings = feats[0]["geometry"]["coordinates"]
    assert len(rings) == 2  # exterior + hole
    assert all(_ring_valid(r) for r in rings)
    assert feats[0]["properties"]["pixel_count"] == 25 - 9


def test_north_up_transform_georeferences():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    t = north_up_transform(100.0, 200.0, 10.0)
    feats = vectorize(mask, transform=t, min_pixels=1)
    ring = feats[0]["geometry"]["coordinates"][0]
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    assert min(xs) == pytest.approx(100.0) and max(xs) == pytest.approx(110.0)
    assert min(ys) == pytest.approx(190.0) and max(ys) == pytest.approx(200.0)
    assert feats[0]["properties"]["area_ha"] == pytest.approx(0.01)


def test_bad_transform_raises():
    with pytest.raises(ValueError):
        vectorize(np.ones((2, 2), bool), transform=(1.0, 0.0))


def test_write_geojson_merges_extra(tmp_path):
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    feats = vectorize(mask, transform=PIXEL_TRANSFORM, min_pixels=1)
    path = write_polygons_geojson(
        feats, str(tmp_path / "p.geojson"),
        extra_properties={"zone_id": "z1", "date": "2024-04-15"},
    )
    with open(path) as fh:
        data = json.load(fh)
    assert data["type"] == "FeatureCollection"
    props = data["features"][0]["properties"]
    assert props["zone_id"] == "z1" and props["pixel_count"] == 4
