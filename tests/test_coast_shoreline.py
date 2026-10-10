"""Tests for shoreline extraction (contour tracing on the water mask)."""

import numpy as np

from coast.shoreline import extract_shoreline, shoreline_length_m


def _straight_coast(rows=10, cols=10, shore_row=5):
    """Water south of shore_row, all pixels valid."""
    water = np.zeros((rows, cols), dtype=bool)
    water[shore_row:, :] = True
    return water, np.ones_like(water, dtype=bool)


def test_straight_shoreline_length():
    water, valid = _straight_coast()
    lines = extract_shoreline(water, valid, pixel_size_m=10.0)
    assert len(lines) == 1
    # 10 columns -> 10 unit edges -> 100 m
    assert shoreline_length_m(lines) == 100.0


def test_shoreline_meter_conversion():
    water, valid = _straight_coast(rows=4, cols=3, shore_row=2)
    lines = extract_shoreline(
        water, valid, pixel_size_m=10.0, origin_x=500_000.0, origin_y=4_600_000.0
    )
    pts = lines[0]
    # shoreline runs along row 2 -> y = 4600000 - 2*10
    assert (pts[:, 1] == 4_600_000.0 - 20.0).all()
    assert pts[:, 0].min() == 500_000.0
    assert pts[:, 0].max() == 500_030.0


def test_all_land_no_shoreline():
    water = np.zeros((6, 6), dtype=bool)
    assert extract_shoreline(water) == []


def test_all_water_no_shoreline():
    # Water everywhere incl. tile edge -> no boundary emitted at tile edge.
    water = np.ones((6, 6), dtype=bool)
    assert extract_shoreline(water) == []


def test_cloud_pixels_emit_no_shoreline():
    water, _ = _straight_coast()
    valid = np.ones_like(water, dtype=bool)
    valid[5:, 4:6] = False  # cloud blob over the water side
    lines = extract_shoreline(water, valid, pixel_size_m=10.0)
    # The shoreline still exists where valid; the cloud creates no fake ring.
    total = shoreline_length_m(lines)
    assert 0 < total < 100.0


def test_min_length_filters_fragments():
    water, valid = _straight_coast(rows=6, cols=6, shore_row=3)
    lines = extract_shoreline(water, valid, pixel_size_m=10.0, min_length_m=1000.0)
    assert lines == []


def test_lake_produces_closed_loop():
    water = np.zeros((8, 8), dtype=bool)
    water[2:6, 2:6] = True  # inland lake
    lines = extract_shoreline(water, pixel_size_m=10.0)
    assert len(lines) == 1
    pts = lines[0]
    # closed loop: first == last vertex
    assert np.allclose(pts[0], pts[-1])
    # 4x4 lake -> perimeter 16 unit edges -> 160 m
    assert shoreline_length_m(lines) == 160.0


def test_default_valid_is_all_true():
    water, _ = _straight_coast(rows=5, cols=5, shore_row=2)
    lines = extract_shoreline(water, pixel_size_m=10.0)
    assert shoreline_length_m(lines) == 50.0
