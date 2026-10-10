"""Tests for burn.masking: pure-numpy connected components."""

import numpy as np

from burn.masking import label_components, remove_small_components


def test_two_components_labelled():
    m = np.zeros((5, 5), dtype=bool)
    m[0:2, 0:2] = True
    m[3:5, 3:5] = True
    labels, sizes = label_components(m, connectivity=8)
    assert len(sizes) == 2
    assert sorted(sizes.values()) == [4, 4]
    assert set(np.unique(labels)) == {0, 1, 2}


def test_diagonal_connectivity_8_vs_4():
    m = np.zeros((3, 3), dtype=bool)
    m[0, 0] = True
    m[1, 1] = True
    _, sizes8 = label_components(m, connectivity=8)
    _, sizes4 = label_components(m, connectivity=4)
    assert len(sizes8) == 1
    assert len(sizes4) == 2


def test_union_find_merges_u_shape():
    # A U shape exercises the union-find merge path.
    m = np.zeros((5, 5), dtype=bool)
    m[0, :] = True
    m[:, 0] = True
    m[:, 4] = True
    labels, sizes = label_components(m, connectivity=8)
    assert len(sizes) == 1
    assert list(sizes.values())[0] == 13


def test_empty_mask():
    labels, sizes = label_components(np.zeros((3, 3), dtype=bool))
    assert sizes == {}
    assert not labels.any()


def test_remove_small_components():
    m = np.zeros((6, 6), dtype=bool)
    m[0, 0] = True          # speckle
    m[2:5, 2:5] = True      # 3x3 patch
    out = remove_small_components(m, min_pixels=4)
    assert not out[0, 0]
    assert out[2:5, 2:5].all()
    assert out.sum() == 9


def test_remove_small_keeps_everything_when_min_is_one():
    m = np.zeros((3, 3), dtype=bool)
    m[1, 1] = True
    assert remove_small_components(m, min_pixels=1).sum() == 1


def test_bad_connectivity_raises():
    try:
        label_components(np.zeros((2, 2), dtype=bool), connectivity=6)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
