"""Tests for autopilot.salience."""

import numpy as np

from autopilot import rank_label_positions, salience_map


def _fields():
    # Fast water + warm anomaly in the top-left quadrant.
    lats = np.linspace(50, 30, 20)  # rows: north -> south
    lons = np.linspace(-130, -110, 30)
    speed = np.full((20, 30), 0.5)
    speed[:8, :10] = 8.0
    temp = np.full((20, 30), 10.0)
    temp[:8, :10] = 25.0
    return speed, temp, lats, lons


def test_salience_map_peaks_at_fast_anomalous_region():
    speed, temp, _, _ = _fields()
    sal = salience_map(speed, temp)
    assert sal.shape == speed.shape
    assert sal.min() >= 0.0 and sal.max() <= 1.0
    # Top-left (fast + anomalous) is the global max region.
    assert sal[:8, :10].mean() > sal[8:, 10:].mean()
    assert sal[:8, :10].max() == sal.max()


def test_salience_map_nan_safe():
    speed, temp, _, _ = _fields()
    speed[0, 0] = np.nan
    temp[5, 5] = np.nan
    sal = salience_map(speed, temp)
    assert np.all(np.isfinite(sal))
    assert sal.min() >= 0.0 and sal.max() <= 1.0


def test_salience_map_all_nan_is_zeros():
    sal = salience_map(np.full((4, 4), np.nan), np.full((4, 4), np.nan))
    assert np.all(sal == 0.0)


def test_rank_label_positions_prefers_salient_region():
    speed, temp, lats, lons = _fields()
    sal = salience_map(speed, temp)
    cands = [
        {"name": "far", "lon": -112.0, "lat": 32.0},
        {"name": "near", "lon": -127.0, "lat": 48.0},
    ]
    ranked = rank_label_positions(cands, sal, lats, lons)
    assert ranked[0]["name"] == "near"
    assert "salience_score" in ranked[0]
    assert ranked[0]["salience_score"] > ranked[1]["salience_score"]


def test_rank_label_positions_brightness_penalty():
    speed, temp, lats, lons = _fields()
    sal = salience_map(speed, temp)
    # Bright clutter sits on the BORING bottom-right region, away from the
    # salient top-left region.
    brightness = np.zeros_like(sal)
    brightness[12:, 20:] = 0.95
    cands = [
        {"name": "salient-dim", "lon": -127.0, "lat": 48.0},   # top-left: salient
        {"name": "bright-boring", "lon": -112.0, "lat": 32.0},  # bottom-right: bright
        {"name": "dim-boring", "lon": -120.0, "lat": 40.0},     # middle: dim, boring
    ]
    plain = rank_label_positions(cands, sal, lats, lons)
    penalized = rank_label_positions(cands, sal, lats, lons, brightness=brightness)

    # Top-ranked is near the fast/warm-anomaly region in both cases ...
    assert plain[0]["name"] == "salient-dim"
    assert penalized[0]["name"] == "salient-dim"
    # ... and the brightness penalty only demotes the bright candidate.
    plain_scores = {d["name"]: d["salience_score"] for d in plain}
    pen_scores = {d["name"]: d["salience_score"] for d in penalized}
    assert pen_scores["bright-boring"] < plain_scores["bright-boring"]
    assert pen_scores["salient-dim"] == plain_scores["salient-dim"]
    assert pen_scores["dim-boring"] == plain_scores["dim-boring"]


def test_rank_label_positions_passthrough_and_deterministic():
    speed, temp, lats, lons = _fields()
    sal = salience_map(speed, temp)
    cands = [{"name": "a", "lon": -120.0, "lat": 40.0, "custom": 42}]
    ranked = rank_label_positions(cands, sal, lats, lons)
    assert ranked[0]["custom"] == 42
    assert "salience_score" not in cands[0]  # input not mutated
    assert rank_label_positions(cands, sal, lats, lons) == ranked
