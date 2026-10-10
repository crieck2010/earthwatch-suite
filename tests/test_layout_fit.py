"""fit_frame: exact output geometry for every mode."""

import numpy as np
import pytest

from layout import fit_frame

pillow = pytest.importorskip("PIL", reason="fit tests need Pillow")


def _frame(w=800, h=600):
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_contain_exact_size_and_letterbox():
    out = fit_frame(_frame(), 1080, 1920, mode="contain")
    assert out.shape == (1920, 1080, 3)
    # corners are the letterbox color
    assert tuple(out[0, 0]) == (10, 10, 14)
    assert tuple(out[-1, -1]) == (10, 10, 14)


def test_cover_exact_size_no_bars():
    frame = _frame()
    out = fit_frame(frame, 1080, 1920, mode="cover")
    assert out.shape == (1920, 1080, 3)
    # center-crop of a scaled cover: middle pixel comes from the frame
    assert out.shape[0] == 1920


def test_blur_fill_exact_size_and_full_bleed():
    out = fit_frame(_frame(), 1080, 1920, mode="blur-fill")
    assert out.shape == (1920, 1080, 3)
    # background fills the whole canvas (corners differ from letterbox)
    assert not np.all(out[0, 0] == np.array([10, 10, 14]))


def test_blur_fill_keeps_sharp_center():
    frame = _frame(400, 1200)  # tall frame -> contain band is wide
    out = fit_frame(frame, 1080, 1920, mode="blur-fill", blur_radius=8.0)
    assert out.shape == (1920, 1080, 3)


def test_rgba_passthrough():
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, (100, 100, 4), dtype=np.uint8)
    out = fit_frame(frame, 200, 300, mode="contain")
    assert out.shape == (300, 200, 4)


def test_bad_mode_fails_fast():
    with pytest.raises(ValueError, match="Unknown fit mode"):
        fit_frame(_frame(), 1080, 1920, mode="ken-burns")


def test_non_uint8_fails_fast():
    with pytest.raises(ValueError, match="uint8"):
        fit_frame(np.zeros((10, 10, 3)), 1080, 1920)


def test_identity_when_already_correct_size():
    frame = _frame(1080, 1920)
    out = fit_frame(frame, 1080, 1920, mode="cover")
    assert out.shape == (1920, 1080, 3)
    np.testing.assert_array_equal(out, frame)
