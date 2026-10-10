import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from aesthetics.framing import (
    fit_extent_to_canvas,
    optimal_rotation,
    rotate_frame,
    rotate_frame_fill,
    rotated_render_size,
    tight_bbox,
)


def test_rotate_zero_is_identity():
    rng = np.random.default_rng(0)
    rgba = rng.uniform(0, 1, (90, 60, 4))
    out = rotate_frame(rgba, 0.0)
    assert out.shape == rgba.shape
    assert np.allclose(out, rgba, atol=2.0 / 255.0)


def test_rotate_full_circle_is_identity():
    rng = np.random.default_rng(1)
    rgba = rng.uniform(0, 1, (90, 60, 4))
    out = rotate_frame(rgba, 360.0)
    assert np.allclose(out, rgba, atol=2.0 / 255.0)


def test_rotated_render_size():
    assert rotated_render_size(0.0, (1080, 1920)) == (1080, 1920)
    assert rotated_render_size(90.0, (1080, 1920)) == (1920, 1080)
    # Full auto-fit pattern: oversized render -> rotate with crop -> canvas.
    w, h = rotated_render_size(90.0, (1080, 1920))
    rng = np.random.default_rng(0)
    big = rng.uniform(0, 1, (h, w, 4))
    out = rotate_frame_fill(big, 90.0, (1080, 1920))
    assert out.shape == (1920, 1080, 4)


def test_rotate_frame_fill_keeps_center_content():
    # Bright block at the center of the oversized render stays central.
    w, h = rotated_render_size(90.0, (200, 300))
    rgba = np.zeros((h, w, 4))
    rgba[h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5, :] = 1.0
    out = rotate_frame_fill(rgba, 90.0, (200, 300))
    assert out.shape == (300, 200, 4)
    cy, cx = 150, 100
    assert out[cy - 10:cy + 10, cx - 10:cx + 10, :3].sum() > 0.5 * 300.0


def test_rotate_frame_fill_zero_angle_crops_center():
    rgba = np.zeros((120, 100, 4))
    rgba[50:70, 40:60, :] = 1.0
    out = rotate_frame_fill(rgba, 0.0, (100, 100))
    assert out.shape == (100, 100, 4)
    # Center crop kept the centered block.
    assert out[40:60, 40:60, :3].sum() > 0.5 * (20 * 20 * 3)


def test_rotate_frame_fill_too_small_raises():
    with pytest.raises(ValueError):
        rotate_frame_fill(np.zeros((50, 50, 4)), 90.0, (1080, 1920))


def test_rotate_preserves_shape_and_range():
    rng = np.random.default_rng(2)
    rgba = rng.uniform(0, 1, (100, 70, 4))
    out = rotate_frame(rgba, 37.0)
    assert out.shape == rgba.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_rotate_actually_rotates():
    # Square canvas, off-center block: a 90° in-place rotation keeps the
    # block inside the frame but moves it.
    rgba = np.zeros((100, 100, 4))
    rgba[20:30, 60:70, :] = 1.0
    out = rotate_frame(rgba, 90.0)
    assert out[..., :3].sum() > 0.5 * rgba[..., :3].sum()
    assert not np.allclose(out, rgba, atol=0.05)


def test_optimal_rotation_wide_region():
    # Lake Ontario-like: ~3.8° lon x 0.9° lat at 43.6N on a 1080x1920 canvas.
    ang = optimal_rotation(-79.8, -76.0, 43.2, 44.1, canvas_wh=(1080, 1920))
    assert 80.0 <= ang <= 90.0


def test_optimal_rotation_square_region():
    ang = optimal_rotation(-10.0, 10.0, 40.0, 60.0, canvas_wh=(1080, 1920))
    assert ang <= 5.0  # already fills well north-up


def _area_score(lon0, lon1, lat0, lat1, angle_deg, canvas_wh=(1080, 1920)):
    """Displayed-area score replicated from optimal_rotation's objective."""
    w_px, h_px = canvas_wh
    lat = (lat0 + lat1) / 2.0
    w_km = (lon1 - lon0) * 111.32 * max(np.cos(np.deg2rad(lat)), 1e-6)
    h_km = (lat1 - lat0) * 110.57
    t = np.deg2rad(angle_deg)
    c, s = abs(np.cos(t)), abs(np.sin(t))
    wp = w_km * c + h_km * s
    hp = w_km * s + h_km * c
    return min(w_px / wp, h_px / hp) ** 2


def test_optimal_rotation_is_truly_optimal():
    # For several region shapes, the returned angle must beat both 0° and 90°.
    cases = [
        (-79.8, -76.0, 43.2, 44.1),   # wide (Lake Ontario-like)
        (-1.0, 1.0, 30.0, 60.0),      # tall strip
        (-10.0, 10.0, 40.0, 60.0),    # moderate
        (-125.0, -115.0, 32.0, 42.0),  # square-ish
    ]
    for bbox in cases:
        ang = optimal_rotation(*bbox, canvas_wh=(1080, 1920))
        s = _area_score(*bbox, ang)
        assert s >= _area_score(*bbox, 0.0) - 1e-9
        assert s >= _area_score(*bbox, 90.0) - 1e-9


def test_optimal_rotation_bad_bbox_raises():
    with pytest.raises(ValueError):
        optimal_rotation(5.0, 5.0, 0.0, 1.0)


def test_tight_bbox():
    lons = np.linspace(-80, -76, 9)
    lats = np.linspace(43, 45, 9)
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:5, 3:7] = True
    x0, x1, y0, y1 = tight_bbox(lons, lats, mask, pad_frac=0.0)
    assert x0 == pytest.approx(-78.5)  # lons[3]
    assert x1 == pytest.approx(-77.0)  # lons[6]
    assert y0 == pytest.approx(43.5)   # lats[2]
    assert y1 == pytest.approx(44.0)   # lats[4]


def test_tight_bbox_padding():
    lons = np.linspace(0, 10, 11)
    lats = np.linspace(0, 10, 11)
    mask = np.zeros((11, 11), dtype=bool)
    mask[5, 5] = True
    x0, x1, y0, y1 = tight_bbox(lons, lats, mask, pad_frac=0.1)
    assert (x1 - x0) > 0 and (y1 - y0) > 0  # degenerate span still frames


def test_tight_bbox_empty_mask_raises():
    with pytest.raises(ValueError):
        tight_bbox(np.arange(4), np.arange(4), np.zeros((4, 4), bool))


def test_fit_extent_matches_canvas_aspect():
    bbox = (-79.8, -76.0, 43.2, 44.1)
    out = fit_extent_to_canvas(bbox, (1080, 1920))
    w = out[1] - out[0]
    h = out[3] - out[2]
    assert (w / h) == pytest.approx(1080 / 1920, rel=1e-9)
    # Original bbox still inside.
    assert out[0] <= bbox[0] and out[1] >= bbox[1]
    assert out[2] <= bbox[2] and out[3] >= bbox[3]
