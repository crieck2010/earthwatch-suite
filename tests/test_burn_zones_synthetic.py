"""Tests for burn.zones and burn.synthetic: fixtures and determinism."""

import numpy as np
import pytest

from burn.dnbr import dnbr
from burn.severity import class_areas, classify
from burn.synthetic import (
    DEMO_PIXEL_M,
    expected_burn_areas,
    synthetic_pass_series,
    synthetic_post_nbr,
    synthetic_pre_nbr,
    synthetic_zones,
)
from burn.zones import Zone, load_zones_geojson, zones_from_feature_collection


def _fc():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": "a"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "Bee"},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]]],
                },
            },
        ],
    }


def test_zones_load_and_ids():
    zones = zones_from_feature_collection(_fc())
    assert [z.id for z in zones] == ["a", "Bee"]
    assert zones[0].bbox() == (0, 0, 1, 1)


def test_zones_reject_bad_type():
    bad = {"type": "FeatureCollection", "features": []}
    with pytest.raises(ValueError, match="no features"):
        zones_from_feature_collection(bad)


def test_zones_reject_duplicate_ids():
    fc = _fc()
    fc["features"][1]["properties"] = {"id": "a"}
    with pytest.raises(ValueError, match="duplicate zone id"):
        zones_from_feature_collection(fc)


def test_zones_reject_point_geometry():
    fc = _fc()
    fc["features"][0]["geometry"] = {"type": "Point", "coordinates": [0, 0]}
    with pytest.raises(ValueError, match="Polygon/MultiPolygon"):
        zones_from_feature_collection(fc)


def test_zones_load_from_file(tmp_path):
    import json

    p = tmp_path / "zones.geojson"
    p.write_text(json.dumps(_fc()), encoding="utf-8")
    zones = load_zones_geojson(str(p))
    assert len(zones) == 2


def test_synthetic_zones_three():
    zones = synthetic_zones()
    assert len(zones) == 3
    assert len({z.id for z in zones}) == 3


def test_planted_burn_matches_expected_areas():
    pre = synthetic_pre_nbr()
    post = synthetic_post_nbr()
    d = dnbr(pre, post)
    sev = classify(d)
    areas = class_areas(sev, None, DEMO_PIXEL_M)
    expected = expected_burn_areas()
    # Noise on the rasters can nudge a few boundary pixels across bands;
    # assert within 2% relative of the planted areas.
    for key in ("high_severity_ha", "moderate_high_severity_ha",
                "moderate_low_severity_ha", "low_severity_ha", "burned_ha"):
        assert areas[key] == pytest.approx(expected[key], rel=0.02), key


def test_unburned_synthetic_has_no_burn():
    pre = synthetic_pre_nbr()
    post = synthetic_post_nbr(burn=False)
    d = dnbr(pre, post)
    sev = classify(d)
    areas = class_areas(sev, None, DEMO_PIXEL_M)
    assert areas["burned_ha"] == pytest.approx(0.0, abs=1.0)


def test_synthetic_deterministic():
    a = synthetic_post_nbr(seed=7)
    b = synthetic_post_nbr(seed=7)
    assert np.array_equal(a, b)
    c = synthetic_post_nbr(seed=8)
    assert not np.array_equal(a, c)


def test_pass_series_shapes():
    dates, ndvi, nbr = synthetic_pass_series()
    assert len(dates) == len(ndvi) == len(nbr) == 6
    assert ndvi[0].shape == (40, 40)
