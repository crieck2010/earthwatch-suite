"""Tests for autopilot.scaling."""

import numpy as np

from autopilot import robust_limits


def test_outlier_does_not_move_vmax():
    values = list(range(1, 101)) + [1e9]
    r = robust_limits(values)
    assert r["method"] == "percentile"
    # p98 of 1..100 + 1e9 lands at 99.0, nowhere near the spike.
    assert r["vmax"] == 99.0
    assert r["vmin"] == 3.0  # p2 -> index 2 -> value 3
    # 100 and 1e9 are above vmax=99: 2/101 clipped high.
    assert abs(r["clipped_hi_fraction"] - 2 / 101) < 1e-12
    assert r["clipped_lo_fraction"] < 0.05
    assert r["n_total"] == 101
    assert r["n_finite"] == 101


def test_nan_and_inf_are_ignored():
    values = [1.0, 2.0, 3.0, np.nan, np.inf, -np.inf]
    r = robust_limits(values)
    assert r["n_total"] == 6
    assert r["n_finite"] == 3
    assert r["method"] == "percentile"
    # p2/p98 of [1,2,3] via linear interpolation: 1.04 / 2.96
    assert 1.0 <= r["vmin"] <= r["vmax"] <= 3.0
    assert abs(r["vmin"] - 1.04) < 1e-9
    assert abs(r["vmax"] - 2.96) < 1e-9


def test_degenerate_empty():
    r = robust_limits([np.nan, np.inf])
    assert r["method"] == "degenerate-empty"
    assert r["vmin"] == 0.0 and r["vmax"] == 1.0
    assert r["n_finite"] == 0


def test_degenerate_empty_list():
    r = robust_limits([])
    assert r["method"] == "degenerate-empty"


def test_degenerate_constant():
    r = robust_limits([5.0, 5.0, 5.0, np.nan])
    assert r["method"] == "degenerate-constant"
    assert r["vmin"] == 5.0 and r["vmax"] == 5.0
    assert r["n_finite"] == 3


def test_metadata_echoes_percentiles():
    r = robust_limits(np.arange(1000.0), lo_pct=5.0, hi_pct=95.0)
    assert r["lo_pct"] == 5.0
    assert r["hi_pct"] == 95.0
    assert r["method"] == "percentile"
    assert abs(r["clipped_lo_fraction"] - 0.05) < 1e-3
    assert abs(r["clipped_hi_fraction"] - 0.05) < 1e-3


def test_never_raises_on_weird_input():
    assert robust_limits("not an array")  # string -> degenerate path
    assert robust_limits([None])["method"] == "degenerate-empty"
    r = robust_limits([[1, 2], [3, 4]])
    assert r["method"] == "percentile"
