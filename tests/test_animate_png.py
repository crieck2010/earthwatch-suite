"""Tests for the pure-Python PNG codec."""

import numpy as np
import pytest

from animate import _png


@pytest.mark.parametrize("shape", [(7, 13, 3), (16, 16, 4), (101, 57, 3), (1, 1, 3)])
def test_roundtrip_shapes(shape):
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=shape, dtype=np.uint8)
    _png.write_png("/tmp/anim_png_test.png", img)
    back = _png.read_png("/tmp/anim_png_test.png")
    assert back.shape == img.shape
    assert np.array_equal(back, img)


def test_png_signature(tmp_path):
    p = str(tmp_path / "s.png")
    _png.write_png(p, np.zeros((4, 4, 3), dtype=np.uint8))
    with open(p, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"


def test_png_size_no_decode(tmp_path):
    p = str(tmp_path / "s.png")
    _png.write_png(p, np.zeros((9, 21, 3), dtype=np.uint8))
    assert _png.png_size(p) == (21, 9)


def test_read_matplotlib_png_filters(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grad = np.tile(np.linspace(0, 1, 200), (137, 1))
    p = str(tmp_path / "mpl.png")
    plt.imsave(p, grad, cmap="viridis")  # uses Sub/Up/Average/Paeth filters
    back = _png.read_png(p)
    ref = (plt.imread(p) * 255).astype(np.uint8)
    assert back.shape == ref.shape
    assert np.abs(back.astype(int) - ref.astype(int)).max() <= 1


def test_not_a_png_rejected(tmp_path):
    p = str(tmp_path / "x.png")
    with open(p, "wb") as fh:
        fh.write(b"definitely not png")
    with pytest.raises(ValueError, match="not a PNG"):
        _png.read_png(p)


def test_write_rejects_bad_dtype(tmp_path):
    with pytest.raises(ValueError, match="uint8"):
        _png.write_png(str(tmp_path / "x.png"),
                       np.zeros((4, 4, 3), dtype=float))


def test_write_rejects_bad_shape(tmp_path):
    with pytest.raises(ValueError, match=r"\(H, W, 3\|4\)"):
        _png.write_png(str(tmp_path / "x.png"),
                       np.zeros((4, 4), dtype=np.uint8))
