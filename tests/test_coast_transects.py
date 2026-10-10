"""Tests for transect loading and shoreline-crossing geometry."""

import json

import numpy as np
import pytest

from coast.transects import (
    Transect,
    load_transects_geojson,
    transect_position,
    transect_shoreline_crossings,
)


def _transect(x=0.0, y0=0.0, y1=100.0):
    return Transect(
        id="t1",
        name="T1",
        coords=np.array([[x, y0], [x, y1]]),
    )


def test_crossing_hand_computed():
    # Horizontal shoreline at y=40, vertical transect x=0 from y=0..100.
    t = _transect()
    shore = [np.array([[-50.0, 40.0], [50.0, 40.0]])]
    assert transect_shoreline_crossings(t, shore) == pytest.approx([40.0])
    assert transect_position(t, shore) == pytest.approx(40.0)


def test_diagonal_shoreline_crossing():
    t = _transect()
    # Shoreline from (-50, 0) to (50, 100): crosses x=0 at y=50.
    shore = [np.array([[-50.0, 0.0], [50.0, 100.0]])]
    assert transect_position(t, shore) == pytest.approx(50.0)


def test_crossing_beyond_transect_ignored():
    t = _transect(y1=100.0)
    shore = [np.array([[-50.0, 140.0], [50.0, 140.0]])]  # past the end
    assert transect_shoreline_crossings(t, shore) == []
    assert np.isnan(transect_position(t, shore))


def test_median_of_multiple_crossings():
    t = _transect()
    shores = [
        np.array([[-50.0, 30.0], [50.0, 30.0]]),
        np.array([[-50.0, 50.0], [50.0, 50.0]]),
        np.array([[-50.0, 90.0], [50.0, 90.0]]),  # spurious fragment
    ]
    assert transect_position(t, shores) == pytest.approx(50.0)


def test_no_shorelines_nan():
    assert np.isnan(transect_position(_transect(), []))


def test_parallel_shoreline_no_crossing():
    t = _transect()
    shore = [np.array([[10.0, 0.0], [10.0, 100.0]])]  # parallel, offset
    assert transect_shoreline_crossings(t, shore) == []


def test_degenerate_transect_raises():
    t = Transect(id="bad", name="bad", coords=np.array([[1.0, 1.0], [1.0, 1.0]]))
    with pytest.raises(ValueError, match="first == last"):
        _ = t.direction


def test_load_transects(tmp_path):
    path = tmp_path / "transects.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[0, 0], [0, 100]],
                        },
                        "properties": {"id": "a", "name": "Alpha"},
                    }
                ],
            }
        )
    )
    ts = load_transects_geojson(str(path))
    assert len(ts) == 1 and ts[0].id == "a" and ts[0].length_m == 100.0


def test_load_transects_rejects_polygon(tmp_path):
    path = tmp_path / "bad.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                        },
                        "properties": {},
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="LineString"):
        load_transects_geojson(str(path))


def test_load_transects_empty_rejected(tmp_path):
    path = tmp_path / "empty.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    with pytest.raises(ValueError, match="no transect"):
        load_transects_geojson(str(path))
