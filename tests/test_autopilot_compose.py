"""Tests for autopilot.compose."""

import numpy as np

from autopilot import score_thumbnail, search_composition


def _black_thumb(h=40, w=60):
    return np.zeros((h, w, 3), dtype=np.float64)


def _half_bright_thumb(h=40, w=60):
    t = np.zeros((h, w, 3))
    t[:, w // 2 :, :] = 0.8
    return t


def _full_frame_gradient_thumb(h=40, w=60):
    # Data across the whole frame: coverage ~1, healthy tonal spread.
    ramp = np.linspace(0.06, 0.95, w)[None, :, None]
    return np.repeat(np.repeat(ramp, h, axis=0), 3, axis=2)


def _flat_gray_thumb(h=40, w=60):
    # Full coverage but zero contrast.
    return np.full((h, w, 3), 0.5)


def test_ranking_is_sane_and_deterministic():
    black = score_thumbnail(_black_thumb())
    half = score_thumbnail(_half_bright_thumb())
    full = score_thumbnail(_full_frame_gradient_thumb())
    gray = score_thumbnail(_flat_gray_thumb())

    # Mostly-black frame: almost no coverage, no contrast.
    assert black["coverage"] < 0.05
    assert black["contrast"] < 0.05

    # Half-bright frame: ~half coverage, saturated contrast (clipped at 1.0).
    assert 0.45 < half["coverage"] < 0.55

    # Full-frame gradient: near-total coverage.
    assert full["coverage"] > 0.95

    # Ranking sanity: full-frame data > half frame > flat gray (coverage but
    # no contrast) > black. Contrast is doing work: flat gray loses to the
    # half-bright frame despite higher coverage.
    assert full["total"] > half["total"] > gray["total"] > black["total"]

    # Deterministic: identical thumbnails score identically.
    assert score_thumbnail(_full_frame_gradient_thumb()) == full


def test_uint8_and_float_inputs_agree():
    f = _half_bright_thumb()
    u8 = (f * 255).astype(np.uint8)
    assert score_thumbnail(f) == score_thumbnail(u8)


def test_label_boxes_reward_dark_backgrounds():
    # Same frame, same box — dark box vs bright box.
    frame = _half_bright_thumb()
    h, w = frame.shape[:2]
    dark_box = [(5, 5, 15, 15)]  # left half is black
    bright_box = [(w - 15, 5, w - 5, 15)]  # right half is bright
    dark = score_thumbnail(frame, label_boxes=dark_box)
    bright = score_thumbnail(frame, label_boxes=bright_box)
    assert dark["label_legibility"] > bright["label_legibility"]
    # No boxes -> neutral 0.5
    assert score_thumbnail(frame)["label_legibility"] == 0.5


def test_weights_override_and_reported():
    s = score_thumbnail(_half_bright_thumb(), weights={"coverage": 1.0, "contrast": 0.0, "label_legibility": 0.0})
    assert s["weights"] == {"coverage": 1.0, "contrast": 0.0, "label_legibility": 0.0}
    assert s["total"] == s["coverage"]


def _stub_render_factory():
    """Stub callback: renders a thumbnail whose brightness encodes the preset."""
    def render(preset, rotation, basemap):
        brightness = {"dim": 0.1, "mid": 0.5, "bright": 0.9}[preset]
        t = np.full((20, 30, 3), brightness)
        # add a bright quadrant for contrast on the "bright" preset
        if preset == "bright":
            t[:10, :15, :] = 0.05
        return t
    return render


def test_search_composition_picks_winner_and_sorts():
    cands = [
        {"preset": "dim", "rotation": 0, "basemap": "dark"},
        {"preset": "mid", "rotation": 0, "basemap": "dark"},
        {"preset": "bright", "rotation": 0, "basemap": "dark", "note": "extra-passthrough"},
    ]
    result = search_composition(_stub_render_factory(), cands)
    assert result["n_candidates"] == 3
    assert result["weights"] == {"coverage": 0.4, "contrast": 0.35, "label_legibility": 0.25}
    totals = [s["total"] for s in result["scores"]]
    assert totals == sorted(totals, reverse=True)
    # "bright" has high coverage AND contrast -> should win over flat mid/dim.
    assert result["winner"]["preset"] == "bright"
    assert result["winner"]["note"] == "extra-passthrough"  # passthrough preserved
    assert len(result["scores"]) == 3


def test_search_composition_propagates_callback_errors():
    def bad_render(preset, rotation, basemap):
        raise RuntimeError("render exploded")

    try:
        search_composition(bad_render, [{"preset": "x", "rotation": 0, "basemap": "y"}])
    except RuntimeError as e:
        assert "render exploded" in str(e)
    else:
        raise AssertionError("callback exception should propagate")


def test_search_composition_empty_candidates_is_honest():
    result = search_composition(_stub_render_factory(), [])
    assert result["winner"] is None
    assert result["scores"] == []
    assert result["n_candidates"] == 0
