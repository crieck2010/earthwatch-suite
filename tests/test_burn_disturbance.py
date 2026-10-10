"""Tests for burn.disturbance: the clearing/logging heuristic."""

import numpy as np
import pytest

from burn.disturbance import DisturbanceParams, detect_disturbance
from burn.synthetic import synthetic_pass_series


def test_detects_planted_clearing():
    dates, ndvi, nbr = synthetic_pass_series()
    events = detect_disturbance(
        ndvi[2], ndvi[3], nbr[2], nbr[3],
        zone_id="demo-block", pass_id="pass-3", date=dates[3],
    )
    assert len(events) == 1
    e = events[0]
    assert e.pixels == 144  # 12x12 planted patch
    assert e.area_ha == pytest.approx(144 * 100 / 10_000.0)
    assert e.mean_ndvi_drop > 0.4
    assert e.mean_nbr_drop > 0.4
    assert 0.0 < e.confidence <= 1.0
    assert e.date == dates[3]


def test_no_event_on_stable_passes():
    dates, ndvi, nbr = synthetic_pass_series()
    events = detect_disturbance(ndvi[0], ndvi[1], nbr[0], nbr[1])
    assert events == []


def test_no_event_after_clearing_persists():
    # Pass 3 -> 4: already cleared, no *new* drop -> no event.
    dates, ndvi, nbr = synthetic_pass_series()
    events = detect_disturbance(ndvi[3], ndvi[4], nbr[3], nbr[4])
    assert events == []


def test_ndvi_only_drop_is_not_disturbance():
    # Phenology-like: NDVI falls but NBR holds -> heuristic stays quiet.
    shape = (10, 10)
    ndvi_prev = np.full(shape, 0.7, dtype=np.float32)
    ndvi_now = np.full(shape, 0.5, dtype=np.float32)   # -0.20 drop
    nbr_prev = np.full(shape, 0.55, dtype=np.float32)
    nbr_now = np.full(shape, 0.50, dtype=np.float32)    # -0.05, under threshold
    events = detect_disturbance(ndvi_prev, ndvi_now, nbr_prev, nbr_now)
    assert events == []


def test_min_pixels_filters_speckles():
    shape = (10, 10)
    ndvi_prev = np.full(shape, 0.7, dtype=np.float32)
    ndvi_now = np.full(shape, 0.2, dtype=np.float32)
    nbr_prev = np.full(shape, 0.55, dtype=np.float32)
    nbr_now = np.full(shape, 0.05, dtype=np.float32)
    params = DisturbanceParams(min_pixels=200)  # bigger than the 100-px field
    events = detect_disturbance(ndvi_prev, ndvi_now, nbr_prev, nbr_now, params=params)
    assert events == []


def test_nan_pixels_disqualified():
    shape = (8, 8)
    ndvi_prev = np.full(shape, 0.7, dtype=np.float32)
    ndvi_now = np.full(shape, 0.2, dtype=np.float32)
    nbr_prev = np.full(shape, 0.55, dtype=np.float32)
    nbr_now = np.full(shape, 0.05, dtype=np.float32)
    nbr_now[:, :] = np.nan  # clouds ate the whole scene
    events = detect_disturbance(ndvi_prev, ndvi_now, nbr_prev, nbr_now)
    assert events == []


def test_valid_mask_respected():
    shape = (8, 8)
    ndvi_prev = np.full(shape, 0.7, dtype=np.float32)
    ndvi_now = np.full(shape, 0.2, dtype=np.float32)
    nbr_prev = np.full(shape, 0.55, dtype=np.float32)
    nbr_now = np.full(shape, 0.05, dtype=np.float32)
    valid = np.zeros(shape, dtype=bool)
    valid[:4, :] = True
    events = detect_disturbance(
        ndvi_prev, ndvi_now, nbr_prev, nbr_now, valid=valid,
        params=DisturbanceParams(min_pixels=1),
    )
    assert len(events) == 1
    assert events[0].pixels == 32


def test_events_sorted_by_size_desc():
    shape = (20, 20)
    ndvi_prev = np.full(shape, 0.7, dtype=np.float32)
    ndvi_now = np.full(shape, 0.7, dtype=np.float32)
    nbr_prev = np.full(shape, 0.55, dtype=np.float32)
    nbr_now = np.full(shape, 0.55, dtype=np.float32)
    ndvi_now[0:3, 0:3] = 0.2
    nbr_now[0:3, 0:3] = 0.05
    ndvi_now[10:16, 10:16] = 0.2
    nbr_now[10:16, 10:16] = 0.05
    events = detect_disturbance(
        ndvi_prev, ndvi_now, nbr_prev, nbr_now,
        params=DisturbanceParams(min_pixels=1),
    )
    assert len(events) == 2
    assert events[0].pixels == 36
    assert events[1].pixels == 9
