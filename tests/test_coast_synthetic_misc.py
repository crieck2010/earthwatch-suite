"""Tests for the seeded synthetic world and the misc small modules."""

import json

import numpy as np
import pytest

from coast import synthetic as syn
from coast.geojson import read_geojson, write_geojson
from coast.qml import watermask_qml, write_qml
from coast.zones import load_sites_geojson


def test_synthetic_deterministic():
    a = syn.synthetic_passes(seed=7)
    b = syn.synthetic_passes(seed=7)
    np.testing.assert_array_equal(a[0]["ndwi"], b[0]["ndwi"])
    c = syn.synthetic_passes(seed=8)
    assert not np.allclose(a[0]["ndwi"], c[0]["ndwi"], equal_nan=True)


def test_synthetic_pass_structure():
    passes = syn.synthetic_passes()
    assert len(passes) == 7
    assert passes[0]["date"] == "2019-01-15"
    assert passes[0]["ndwi"].shape == syn.DEMO_SHAPE
    assert passes[0]["cloud"].dtype == bool


def test_synthetic_water_fraction_sane():
    # Roughly half the tile is water (shoreline near the middle).
    p = syn.synthetic_passes()[0]
    water = np.nan_to_num(p["ndwi"], nan=-1.0) > 0.0
    frac = water.mean()
    assert 0.35 < frac < 0.65


def test_synthetic_storm_bite_visible():
    # The post-storm pass has less water near transect A than the trend.
    from coast import transects as tr
    from coast import shoreline as sl
    from coast import watermask as wm

    passes = syn.synthetic_passes()
    t = syn.synthetic_transects()[0]  # shore-a
    positions = []
    for p in passes:
        m = wm.water_mask(
            np.asarray(p["ndwi"]), threshold=0.0,
            cloud=np.asarray(p["cloud"]), min_pixels=4,
        )
        lines = sl.extract_shoreline(
            m.water, m.valid, pixel_size_m=syn.DEMO_PIXEL_M,
            origin_x=syn.DEMO_ORIGIN_X, origin_y=syn.DEMO_ORIGIN_Y,
            min_length_m=50.0,
        )
        positions.append(tr.transect_position(t, lines))
    pre = positions[2]   # 2021-01-15
    post = positions[3]  # 2021-07-20, storm bite planted
    assert post < pre
    assert (pre - post) == pytest.approx(abs(syn.STORM_BITE_M), abs=12.0)


def test_synthetic_transects_landward_to_seaward():
    ts = syn.synthetic_transects()
    assert [t.id for t in ts] == ["shore-a", "shore-b"]
    # origin (first vertex) is landward: smaller row -> larger y in meters
    assert ts[0].coords[0, 1] > ts[0].coords[-1, 1]


def test_synthetic_sites():
    (site,) = syn.synthetic_sites()
    assert site.id == "demo-cove"
    minx, miny, maxx, maxy = site.bbox()
    assert maxx - minx == pytest.approx(120 * 10.0)


def test_transects_geojson_dict_valid():
    d = syn.transects_geojson_dict()
    assert d["type"] == "FeatureCollection"
    assert len(d["features"]) == 2
    assert d["features"][0]["geometry"]["type"] == "LineString"


def test_geojson_round_trip(tmp_path):
    path = str(tmp_path / "x.geojson")
    write_geojson(path, [], pass_id="p1")
    obj = read_geojson(path)
    assert obj["pass_id"] == "p1" and obj["features"] == []


def test_geojson_rejects_non_collection(tmp_path):
    path = tmp_path / "bad.geojson"
    path.write_text(json.dumps({"type": "Point", "coordinates": [0, 0]}))
    with pytest.raises(ValueError, match="FeatureCollection"):
        read_geojson(str(path))


def test_load_sites(tmp_path):
    path = tmp_path / "sites.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                        },
                        "properties": {"id": "cove-1", "name": "Cove One"},
                    }
                ],
            }
        )
    )
    (site,) = load_sites_geojson(str(path))
    assert site.id == "cove-1" and site.name == "Cove One"
    assert site.bbox() == (0.0, 0.0, 10.0, 10.0)


def test_load_sites_rejects_point(tmp_path):
    path = tmp_path / "bad.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [0, 0]},
                        "properties": {},
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="Polygon"):
        load_sites_geojson(str(path))


def test_load_sites_empty_rejected(tmp_path):
    path = tmp_path / "empty.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    with pytest.raises(ValueError, match="no site"):
        load_sites_geojson(str(path))


def test_qml_content():
    qml = watermask_qml()
    assert "Water" in qml and "Land (not water)" in qml
    assert 'value="1"' in qml


def test_write_qml(tmp_path):
    path = write_qml(str(tmp_path / "water.qml"))
    assert open(path).read().startswith("<!DOCTYPE qgis")
    with pytest.raises(ValueError, match="unknown QML kind"):
        write_qml(str(tmp_path / "x.qml"), kind="bogus")
