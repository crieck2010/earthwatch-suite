import numpy as np
import pytest

from flood.watermask import (
    apply_cloud_mask,
    build_water_mask,
    component_sizes,
    despeckle,
    label_components,
    water_mask,
)


def test_threshold_boundary():
    arr = np.array([-0.5, 0.0, 0.3, np.nan], dtype=np.float32)
    assert water_mask(arr, 0.0).tolist() == [False, True, True, False]


def test_custom_threshold():
    arr = np.array([-0.5, 0.0, 0.3], dtype=np.float32)
    assert water_mask(arr, 0.2).tolist() == [False, False, True]


def test_nan_never_water():
    arr = np.full((3, 3), np.nan, dtype=np.float32)
    assert not water_mask(arr).any()


def test_cloud_pixels_cleared_not_counted_as_water():
    index = np.array([[0.5, 0.5], [-0.4, 0.5]], dtype=np.float32)
    clouds = np.array([[True, False], [False, False]])
    clean, valid = apply_cloud_mask(water_mask(index), clouds)
    assert clean.tolist() == [[False, True], [False, True]]
    assert valid.tolist() == [[False, True], [True, True]]


def test_cloud_mask_none_means_all_valid():
    mask = np.ones((2, 2), dtype=bool)
    clean, valid = apply_cloud_mask(mask, None)
    assert clean.all() and valid.all()


def test_cloud_shape_mismatch_raises():
    with pytest.raises(ValueError):
        apply_cloud_mask(np.ones((2, 2), bool), np.ones((3, 3), bool))


def test_label_components_two_blobs():
    mask = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=bool
    )
    labels, n = label_components(mask)
    assert n == 2
    assert labels[0, 0] == labels[1, 1] != labels[2, 3]
    assert (labels[~mask] == 0).all()


def test_label_components_4_connectivity_diagonal():
    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    _, n8 = label_components(mask, connectivity=8)
    _, n4 = label_components(mask, connectivity=4)
    assert n8 == 1
    assert n4 == 2


def test_label_components_empty():
    labels, n = label_components(np.zeros((3, 3), bool))
    assert n == 0 and not labels.any()


def test_component_sizes():
    mask = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 1]], dtype=bool)
    labels, n = label_components(mask)
    assert component_sizes(labels, n) == {1: 2, 2: 1}


def test_despeckle_drops_small_keeps_large():
    mask = np.zeros((6, 6), dtype=bool)
    mask[0, 0] = True  # 1-px speckle
    mask[2:5, 2:5] = True  # 3x3 block
    out = despeckle(mask, min_pixels=4)
    assert not out[0, 0]
    assert out[2:5, 2:5].all()


def test_despeckle_min_pixels_one_is_noop():
    mask = np.array([[True, False], [False, False]])
    np.testing.assert_array_equal(despeckle(mask, min_pixels=1), mask)


def test_build_water_mask_pipeline():
    index = np.array(
        [[0.4, 0.4, -0.3], [0.4, -0.3, -0.3], [-0.3, -0.3, 0.4]], dtype=np.float32
    )
    clouds = np.zeros((3, 3), dtype=bool)
    clouds[2, 2] = True  # the lone water pixel is clouded
    water, valid = build_water_mask(index, threshold=0.0, cloud_mask=clouds, min_pixels=1)
    assert water.sum() == 3  # the 2x2-ish block minus nothing; clouded pixel dropped
    assert not valid[2, 2]
