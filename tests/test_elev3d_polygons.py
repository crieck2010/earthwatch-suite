"""Tests for change polygons: labelling, tracing, areas, speckle filter."""

import json
import tempfile
import os

import numpy as np
import pytest

from elev3d.polygons import (
    _chain_rings,
    _boundary_edges,
    change_polygons,
    label_components,
    polygons_to_geojson,
)
from elev3d.synthetic import TRANSFORM, synthetic_dems
from elev3d.diff import difference


def test_label_components_two_blobs():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True
    mask[7:9, 7:9] = True
    labels, n = label_components(mask)
    assert n == 2
    assert labels[1, 1] != labels[7, 7]


def test_label_components_diagonal_connected():
    # 8-connectivity: diagonal neighbours are one component
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True
    _labels, n = label_components(mask)
    assert n == 1


def test_square_traces_to_closed_ring():
    comp = np.zeros((6, 6), dtype=bool)
    comp[2:4, 2:4] = True
    rings = _chain_rings(_boundary_edges(comp))
    assert len(rings) == 1
    ring = rings[0]
    assert ring[0] == ring[-1]  # closed
    assert len(ring) == 9  # 2×2 square → 8 edges + closing point


def test_hole_preserved():
    # 5×5 block with a 1-px hole in the middle
    comp = np.zeros((7, 7), dtype=bool)
    comp[1:6, 1:6] = True
    comp[3, 3] = False
    polys = change_polygons(comp.astype(float) * -1.0, TRANSFORM,
                            threshold=0.5, min_pixels=1)
    assert len(polys) == 1
    assert len(polys[0].holes) == 1


def test_speckle_filter():
    diff = np.zeros((20, 20))
    diff[2, 2] = -1.0  # single pixel
    diff[10:14, 10:14] = -1.0  # 4×4 block
    polys = change_polygons(diff, TRANSFORM, threshold=0.5, min_pixels=4)
    assert len(polys) == 1
    assert polys[0].pixel_count == 16


def test_polygon_volumes_hand_computed():
    # 2×2 px at -0.5 m, 10 m pixels → 200 m³ cut
    diff = np.zeros((6, 6))
    diff[2:4, 2:4] = -0.5
    polys = change_polygons(diff, TRANSFORM, threshold=0.1, min_pixels=1)
    assert len(polys) == 1
    assert polys[0].cut_m3 == pytest.approx(200.0)
    assert polys[0].dominant == "cut"
    assert polys[0].area_m2 == pytest.approx(400.0)


def test_polygon_dominant_fill():
    diff = np.zeros((6, 6))
    diff[2:4, 2:4] = 0.7
    polys = change_polygons(diff, TRANSFORM, threshold=0.1, min_pixels=1)
    assert polys[0].dominant == "fill"
    assert polys[0].fill_m3 == pytest.approx(280.0)


def test_synthetic_site_two_polygons():
    dem1, dem2, t = synthetic_dems()
    diff = difference(dem1, dem2, noise_floor=0.0).diff
    polys = change_polygons(diff, t, threshold=0.08, min_pixels=4)
    dominants = sorted(p.dominant for p in polys)
    assert dominants == ["cut", "fill"]
    cut = next(p for p in polys if p.dominant == "cut")
    fill = next(p for p in polys if p.dominant == "fill")
    assert cut.cut_m3 == pytest.approx(18750.0, rel=0.03)
    assert fill.fill_m3 == pytest.approx(12300.0, rel=0.05)


def test_geojson_structure():
    diff = np.zeros((6, 6))
    diff[2:4, 2:4] = -0.5
    polys = change_polygons(diff, TRANSFORM, threshold=0.1, min_pixels=1)
    gj = polygons_to_geojson(polys)
    assert gj["type"] == "FeatureCollection"
    feat = gj["features"][0]
    assert feat["geometry"]["type"] == "Polygon"
    ring = feat["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]
    # valid JSON round-trip
    with tempfile.NamedTemporaryFile("w", suffix=".geojson",
                                     delete=False) as fh:
        json.dump(gj, fh)
        path = fh.name
    with open(path) as fh:
        json.load(fh)
    os.unlink(path)
