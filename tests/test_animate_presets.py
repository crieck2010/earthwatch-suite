"""Tests for presets and frame geometry."""

import numpy as np
import pytest

from animate.presets import (even_dims, get_preset, list_presets, resize_pad)


def test_preset_table():
    assert get_preset("reel").width == 1080
    assert get_preset("reel").height == 1920
    assert get_preset("wide").width == 1920
    assert get_preset("wide").height == 1080
    assert get_preset("square").width == get_preset("square").height == 1080
    assert get_preset("preview").width == 540


def test_unknown_preset_lists_valid():
    with pytest.raises(ValueError, match="valid presets"):
        get_preset("ultrawide")


def test_list_presets_sorted_names():
    names = [p.name for p in list_presets()]
    assert names == sorted(names)
    assert set(names) >= {"reel", "wide", "square"}


def test_resize_pad_exact_size_noop():
    rng = np.random.default_rng(0)
    f = (rng.random((50, 80, 3)) * 255).astype(np.uint8)
    out = resize_pad(f, 80, 50)
    assert out.shape == (50, 80, 3)
    assert np.array_equal(out, f)


def test_resize_pad_letterbox():
    # wide frame into square canvas: bars top and bottom
    f = np.full((100, 200, 3), 255, dtype=np.uint8)
    out = resize_pad(f, 200, 200)
    assert out.shape == (200, 200, 3)
    assert (out[0, 100] == [0, 0, 0]).all()
    assert (out[100, 100] == [255, 255, 255]).all()


def test_resize_pad_pillarbox_and_color():
    # tall frame into wide canvas with red padding
    f = np.full((200, 100, 3), 255, dtype=np.uint8)
    out = resize_pad(f, 200, 200, pad_color=(255, 0, 0))
    assert (out[100, 0] == [255, 0, 0]).all()
    assert (out[100, 100] == [255, 255, 255]).all()


def test_resize_pad_rgba_input():
    rng = np.random.default_rng(0)
    f = (rng.random((30, 40, 4)) * 255).astype(np.uint8)
    out = resize_pad(f, 80, 60)
    assert out.shape == (60, 80, 3)


def test_resize_pad_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"\(H, W, 3\|4\)"):
        resize_pad(np.zeros((10, 10), dtype=np.uint8), 10, 10)


def test_even_dims():
    assert even_dims(1080, 1920) == (1080, 1920)
    assert even_dims(1079, 1919) == (1080, 1920)
    assert even_dims(65, 97) == (66, 98)
