import csv
import json
import os

import pytest

from flood.zones import (
    load_zones_geojson,
    zones_from_feature_collection,
    zones_to_feature_collection,
)
from flood.synthetic import synthetic_zones


def test_synthetic_zones_valid():
    zones = synthetic_zones()
    assert len(zones) == 3
    assert {z.id for z in zones} == {"river-reach", "wetland-north", "upland-control"}
    for z in zones:
        minx, miny, maxx, maxy = z.bbox()
        assert minx < maxx and miny < maxy


def test_roundtrip_geojson(tmp_path):
    zones = synthetic_zones()
    path = str(tmp_path / "zones.geojson")
    with open(path, "w") as fh:
        json.dump(zones_to_feature_collection(zones), fh)
    back = load_zones_geojson(path)
    assert [z.id for z in back] == [z.id for z in zones]


def test_not_a_feature_collection():
    with pytest.raises(ValueError, match="FeatureCollection"):
        zones_from_feature_collection({"type": "Point"})


def test_empty_features():
    with pytest.raises(ValueError, match="no features"):
        zones_from_feature_collection({"type": "FeatureCollection", "features": []})


def test_bad_geometry_type():
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"id": "a"},
             "geometry": {"type": "Point", "coordinates": [0, 0]}}
        ],
    }
    with pytest.raises(ValueError, match="Polygon/MultiPolygon"):
        zones_from_feature_collection(fc)


def test_duplicate_ids_rejected():
    def feat(i):
        return {"type": "Feature", "properties": {"id": "dup"},
                "geometry": {"type": "Polygon",
                             "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]] }}
    with pytest.raises(ValueError, match="duplicate zone id"):
        zones_from_feature_collection(
            {"type": "FeatureCollection", "features": [feat(0), feat(1)]}
        )


def test_id_falls_back_to_name_then_index():
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"name": "Reach A"},
             "geometry": {"type": "Polygon",
                          "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}},
            {"type": "Feature", "properties": {},
             "geometry": {"type": "Polygon",
                          "coordinates": [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]]}},
        ],
    }
    zones = zones_from_feature_collection(fc)
    assert zones[0].id == "Reach A"
    assert zones[1].id == "zone-2"
