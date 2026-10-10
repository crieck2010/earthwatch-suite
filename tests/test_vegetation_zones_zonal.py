"""Tests for zones and zonal statistics."""

import json

import numpy as np
import pytest

from vegetation import zonal
from vegetation.synthetic import synthetic_zones
from vegetation.zones import (
    Zone,
    load_zones_geojson,
    zones_from_feature_collection,
    zones_to_feature_collection,
)


def _fc():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": "a", "name": "Field A"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "Field B"},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[[[3, 3], [4, 3], [4, 4], [3, 4], [3, 3]]]],
                },
            },
        ],
    }


def test_load_zones_ids_and_names():
    zones = zones_from_feature_collection(_fc())
    assert [z.id for z in zones] == ["a", "Field B"]
    assert zones[0].name == "Field A"


def test_zone_bbox():
    zones = zones_from_feature_collection(_fc())
    assert zones[0].bbox() == (0.0, 0.0, 2.0, 2.0)


def test_reject_non_collection():
    with pytest.raises(ValueError, match="FeatureCollection"):
        zones_from_feature_collection({"type": "Point"})


def test_reject_non_polygon():
    fc = _fc()
    fc["features"][0]["geometry"] = {"type": "Point", "coordinates": [0, 0]}
    with pytest.raises(ValueError, match="Polygon/MultiPolygon"):
        zones_from_feature_collection(fc)


def test_reject_duplicate_ids():
    fc = _fc()
    fc["features"][1]["properties"] = {"id": "a"}
    with pytest.raises(ValueError, match="duplicate zone id"):
        zones_from_feature_collection(fc)


def test_roundtrip_geojson(tmp_path):
    zones = zones_from_feature_collection(_fc())
    path = tmp_path / "z.geojson"
    path.write_text(json.dumps(zones_to_feature_collection(zones)))
    back = load_zones_geojson(str(path))
    assert [z.id for z in back] == ["a", "Field B"]


def test_zonal_stats_hand_computed():
    arr = np.array([[1.0, 2.0], [3.0, np.nan]])
    mask = np.array([[True, True], [True, False]])
    s = zonal.zonal_stats(arr, mask)
    assert s["mean"] == pytest.approx(2.0)
    assert s["median"] == pytest.approx(2.0)
    assert s["minimum"] == pytest.approx(1.0)
    assert s["maximum"] == pytest.approx(3.0)
    assert s["valid_pixels"] == 3
    assert s["total_pixels"] == 3


def test_zonal_stats_empty_mask():
    s = zonal.zonal_stats(np.ones((2, 2)), np.zeros((2, 2), dtype=bool))
    assert s["valid_pixels"] == 0
    assert np.isnan(s["mean"])


def test_zonal_stats_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        zonal.zonal_stats(np.ones((2, 2)), np.ones((3, 3), dtype=bool))


def test_rect_masks_without_rasterio():
    zones = synthetic_zones()  # rectangles by construction
    masks = zonal.masks_for_zones(zones, (6, 8), grid_bbox=(0, 0, 8, 6.5))
    assert len(masks) == 3
    assert masks[0].sum() > 0
    # masks are disjoint
    assert not (masks[0] & masks[1]).any()
    assert not (masks[0] & masks[2]).any()


def test_non_rectangle_needs_rasterio():
    # transform=None forces the pure-numpy path even when rasterio exists.
    zone = Zone(
        id="tri",
        name="triangle",
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [0, 1], [0, 0]]]},
    )
    with pytest.raises(ValueError, match="axis-aligned rectangles"):
        zonal.masks_for_zones([zone], (4, 4), grid_bbox=(0, 0, 1, 1))
