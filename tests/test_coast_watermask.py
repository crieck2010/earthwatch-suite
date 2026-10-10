"""Tests for connected-component labelling and water-mask pipeline."""

import numpy as np
import pytest

from coast.masking import (
    component_sizes,
    label_components,
    remove_small_components,
)
from coast.watermask import water_mask


def test_label_two_components():
    mask = np.zeros((6, 6), dtype=bool)
    mask[0:2, 0:2] = True
    mask[4:6, 4:6] = True
    labels, n = label_components(mask)
    assert n == 2
    assert labels[0, 0] != labels[4, 4]
    assert (labels[~mask] == 0).all()


def test_label_4_vs_8_connectivity():
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = mask[1, 1] = True  # diagonal touch only
    _, n8 = label_components(mask, 8)
    _, n4 = label_components(mask, 4)
    assert n8 == 1
    assert n4 == 2


def test_label_empty():
    labels, n = label_components(np.zeros((4, 4), dtype=bool))
    assert n == 0 and (labels == 0).all()


def test_component_sizes():
    mask = np.zeros((5, 5), dtype=bool)
    mask[0:2, 0:3] = True  # 6 px
    mask[4, 4] = True  # 1 px
    labels, n = label_components(mask)
    sizes = component_sizes(labels, n)
    assert sorted(sizes.values()) == [1, 6]


def test_remove_small_components():
    mask = np.zeros((8, 8), dtype=bool)
    mask[0:3, 0:3] = True  # 9 px kept
    mask[7, 7] = True  # 1 px dropped
    out = remove_small_components(mask, min_pixels=4)
    assert out[0:3, 0:3].all()
    assert not out[7, 7]


def test_water_mask_threshold():
    ndwi = np.array([[-0.2, 0.1], [0.5, -0.1]])
    m = water_mask(ndwi, threshold=0.0, min_pixels=1)
    assert m.water.tolist() == [[False, True], [True, False]]
    assert m.valid.all()


def test_water_mask_cloud_invalid_never_water():
    ndwi = np.full((4, 4), 0.6)  # all water by index
    cloud = np.zeros((4, 4), dtype=bool)
    cloud[0, 0] = True
    m = water_mask(ndwi, threshold=0.0, cloud=cloud, min_pixels=1)
    assert not m.water[0, 0]
    assert not m.valid[0, 0]
    assert m.water[1, 1]
    assert m.cloud_cover == pytest.approx(100.0 / 16)


def test_water_mask_nan_invalid():
    ndwi = np.full((3, 3), 0.5)
    ndwi[1, 1] = np.nan
    m = water_mask(ndwi, threshold=0.0, min_pixels=1)
    assert not m.valid[1, 1]
    assert not m.water[1, 1]


def test_water_mask_despeckle():
    ndwi = np.full((6, 6), -0.3)
    ndwi[0:3, 0:3] = 0.5  # real water body
    ndwi[5, 5] = 0.5  # speckle
    m = water_mask(ndwi, threshold=0.0, min_pixels=4)
    assert m.water[0:3, 0:3].all()
    assert not m.water[5, 5]


def test_water_mask_cloud_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        water_mask(np.zeros((3, 3)), cloud=np.zeros((2, 2), dtype=bool))


def test_water_fraction():
    ndwi = np.array([[-0.2, 0.5], [0.5, -0.2]])
    m = water_mask(ndwi, threshold=0.0, min_pixels=1)
    assert m.water_fraction == pytest.approx(0.5)
