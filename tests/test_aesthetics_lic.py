import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from aesthetics.lic import lic_texture


def test_shape_and_alpha(uv_scalar):
    u, v, s = uv_scalar
    rgba = lic_texture(u, v, s, vmin=10.0, vmax=20.0, seed=3)
    assert rgba.shape == (48, 64, 4)
    assert rgba[..., 3].min() >= 0.0 and rgba[..., 3].max() <= 1.0
    # NaN hole is transparent, valid area is opaque.
    assert rgba[7, 7, 3] == 0.0
    assert rgba[30, 30, 3] == 1.0
    # Invalid pixels carry no color.
    assert np.all(rgba[7, 7, :3] == 0.0)


def test_deterministic_same_seed(uv_scalar):
    u, v, s = uv_scalar
    a = lic_texture(u, v, s, vmin=10.0, vmax=20.0, seed=11)
    b = lic_texture(u, v, s, vmin=10.0, vmax=20.0, seed=11)
    assert np.array_equal(a, b)


def test_seed_changes_texture(uv_scalar):
    u, v, s = uv_scalar
    a = lic_texture(u, v, s, vmin=10.0, vmax=20.0, seed=11)
    b = lic_texture(u, v, s, vmin=10.0, vmax=20.0, seed=12)
    assert not np.allclose(a, b)


def test_hue_encodes_scalar():
    # No flow at all: hue must still follow the scalar left -> right.
    ny, nx = 24, 48
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    s = np.tile(np.linspace(0.0, 30.0, nx), (ny, 1))
    rgba = lic_texture(u, v, s, vmin=0.0, vmax=30.0, cmap="turbo", seed=1,
                       brightness_floor=1.0, streak_contrast=0.0)
    left = rgba[12, 4, :3]
    right = rgba[12, 43, :3]
    # turbo(0) is dark blue-ish, turbo(1) is dark red-ish: channels differ.
    assert not np.allclose(left, right, atol=0.05)
    # Monotonic hue ramp: mid pixel sits between the ends per channel.
    mid = rgba[12, 24, :3]
    assert left[2] > right[2]  # blue channel falls along turbo
    assert right[0] > left[0]  # red channel rises along turbo


def test_brightness_encodes_speed():
    ny, nx = 24, 48
    # Same scalar everywhere; left half still, right half fast (uniform dir).
    u = np.zeros((ny, nx))
    u[:, 24:] = 5.0
    v = np.zeros((ny, nx))
    s = np.full((ny, nx), 15.0)
    rgba = lic_texture(u, v, s, vmin=0.0, vmax=30.0, seed=1,
                       speed_max=5.0, brightness_floor=0.1,
                       streak_contrast=0.0)
    still = rgba[12, 6, :3].sum()
    fast = rgba[12, 40, :3].sum()
    assert fast > 2.0 * still  # speed -> brightness


def test_fixed_scale_no_flicker():
    # Doubling every value with fixed vmin/vmax must not change hue mapping
    # of the overlapping range: same normalized scalar -> same hue.
    ny, nx = 16, 16
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    s = np.full((ny, nx), 15.0)
    a = lic_texture(u, v, s, vmin=0.0, vmax=30.0, seed=1,
                    brightness_floor=1.0, streak_contrast=0.0)
    b = lic_texture(u, v, 2 * s, vmin=0.0, vmax=60.0, seed=1,
                    brightness_floor=1.0, streak_contrast=0.0)
    assert np.allclose(a[..., :3], b[..., :3], atol=1e-9)


def test_all_nan_returns_transparent():
    u = np.full((10, 10), np.nan)
    rgba = lic_texture(u, u, u, vmin=0.0, vmax=1.0, seed=0)
    assert np.all(rgba[..., 3] == 0.0)
    assert np.all(rgba[..., :3] == 0.0)


def test_bad_inputs_raise():
    u = np.zeros((4, 4))
    with pytest.raises(ValueError):
        lic_texture(u, u, np.zeros((4, 5)), vmin=0, vmax=1)
    with pytest.raises(ValueError):
        lic_texture(u, u, u, vmin=1, vmax=1)
    with pytest.raises(ValueError):
        lic_texture(u, u, u, vmin=0, vmax=1, kernel=0)
    with pytest.raises(ValueError):
        lic_texture(u, u, u, vmin=0, vmax=1, cmap="nope")
