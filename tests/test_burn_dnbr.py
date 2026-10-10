"""Tests for burn.dnbr: dNBR/RdNBR math and pre/post pass selection."""

import numpy as np
import pytest

from burn.dnbr import PassInfo, dnbr, rdnbr, select_pre_post


def test_dnbr_basic_math():
    pre = np.array([[0.6, 0.5]], dtype=np.float32)
    post = np.array([[0.1, 0.4]], dtype=np.float32)
    out = dnbr(pre, post)
    assert out.shape == (1, 2)
    assert out[0, 0] == pytest.approx(0.5)
    assert out[0, 1] == pytest.approx(0.1)


def test_dnbr_nan_honest():
    pre = np.array([[0.6, np.nan]], dtype=np.float32)
    post = np.array([[np.nan, 0.4]], dtype=np.float32)
    out = dnbr(pre, post)
    assert np.isnan(out[0, 0]) and np.isnan(out[0, 1])


def test_dnbr_unburned_near_zero():
    pre = np.full((4, 4), 0.55, dtype=np.float32)
    post = pre + 0.01
    out = dnbr(pre, post)
    assert np.all(np.abs(out) < 0.02)


def test_rdnbr_normalises_sparse_vegetation():
    # Same absolute drop counts more where pre-fire cover was thin.
    pre_dense = np.array([[0.7]], dtype=np.float32)
    pre_sparse = np.array([[0.2]], dtype=np.float32)
    drop = np.array([[0.3]], dtype=np.float32)
    r_dense = rdnbr(pre_dense, pre_dense - drop)
    r_sparse = rdnbr(pre_sparse, pre_sparse - drop)
    assert r_sparse[0, 0] > r_dense[0, 0]


def test_rdnbr_nan_where_pre_zero():
    pre = np.array([[0.0]], dtype=np.float32)
    post = np.array([[-0.2]], dtype=np.float32)
    assert np.isnan(rdnbr(pre, post)[0, 0])


def _passes():
    return [
        PassInfo("p1", "2024-06-01", 5.0),
        PassInfo("p2", "2024-06-20", 60.0),  # too cloudy
        PassInfo("p3", "2024-07-10", 10.0),
        PassInfo("p4", "2024-08-05", 8.0),
        PassInfo("p5", "2024-09-01", 3.0),
    ]


def test_select_pre_post_picks_nearest_clear():
    pre, post = select_pre_post(_passes(), "2024-07-15")
    assert pre.pass_id == "p3"   # p2 disqualified on clouds
    assert post.pass_id == "p4"


def test_select_pre_post_no_pre_raises():
    with pytest.raises(ValueError, match="no pre-fire pass"):
        select_pre_post(_passes(), "2024-05-01")


def test_select_pre_post_no_post_raises():
    with pytest.raises(ValueError, match="no post-fire pass"):
        select_pre_post(_passes(), "2024-10-01")


def test_select_pre_post_bad_windows():
    with pytest.raises(ValueError):
        select_pre_post(_passes(), "2024-07-15", pre_days=0)
