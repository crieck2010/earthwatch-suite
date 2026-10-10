"""Tests for render_video(min_duration_s=...) (v0.3.0): frame-hold so
short reels still play a minimum data duration.

Uses tiny synthetic frames and the real backend; skips gracefully
when no ffmpeg backend exists.
"""

import json
import math
import os

import numpy as np
import pytest

from animate import _png
from animate.encode import EncoderUnavailable, resolve_ffmpeg
from animate.pipeline import render_video


@pytest.fixture()
def ffmpeg_exe():
    try:
        return resolve_ffmpeg()
    except EncoderUnavailable:
        pytest.skip("no ffmpeg backend available")


def _manifest(tmp_path, n, fps=10.0):
    for i in range(n):
        p = str(tmp_path / f"f{i}.png")
        img = (np.random.default_rng(i).random((32, 48, 3)) * 255
               ).astype(np.uint8)
        _png.write_png(p, img)
    manifest = {
        "schema": "survey-flow.frame-manifest/1.0",
        "generator": "test",
        "created": "2026-09-29T00:00:00+00:00",
        "fps": fps,
        "size_px": [48, 32],
        "field_source": "synthetic/test",
        "n_frames": n,
        "frames": [
            {"frame_index": i,
             "timestamp_iso": f"2026-09-{16 + i:02d}T00:00:00+00:00",
             "png_path": f"f{i}.png",
             "scalar": "temperature", "scalar_range": [10.0, 20.0],
             "colormap": "cividis", "style": "reel-dark",
             "render_params": {}}
            for i in range(n)
        ],
    }
    mp = str(tmp_path / "manifest.json")
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    return mp


def test_hold_frame_count_math(ffmpeg_exe, tmp_path):
    # 4 frames at 10fps, min 1.0s -> need 10 frames, k=ceil(10/4)=3.
    mp = _manifest(tmp_path, 4, fps=10.0)
    out = str(tmp_path / "hold.mp4")
    result = render_video(mp, out, preset="preview", fps=10.0,
                          min_duration_s=1.0)
    assert result.n_frames == 12
    assert result.params["frame_hold"] == 3
    assert result.params["min_duration_s"] == 1.0


def test_hold_reaches_minimum_duration(ffmpeg_exe, tmp_path):
    # 7 frames at 30fps, min 10s -> need 300, k=ceil(300/7)=43.
    mp = _manifest(tmp_path, 7, fps=30.0)
    out = str(tmp_path / "hold10.mp4")
    result = render_video(mp, out, preset="preview", fps=30.0,
                          min_duration_s=10.0)
    assert result.n_frames == 7 * 43
    assert result.n_frames / result.fps >= 10.0


def test_title_card_keeps_fixed_length_on_top(ffmpeg_exe, tmp_path):
    # Title card is prepended AFTER the hold: 20 card frames (2s at
    # 10fps) + 12 held data frames = 32.
    mp = _manifest(tmp_path, 4, fps=10.0)
    out = str(tmp_path / "titled.mp4")
    result = render_video(mp, out, preset="preview", fps=10.0,
                          title="Quakes", title_seconds=2.0,
                          min_duration_s=1.0)
    assert result.n_frames == 20 + 12


def test_zero_disables_hold(ffmpeg_exe, tmp_path):
    mp = _manifest(tmp_path, 4, fps=10.0)
    out = str(tmp_path / "plain.mp4")
    result = render_video(mp, out, preset="preview", fps=10.0,
                          min_duration_s=0.0)
    assert result.n_frames == 4
    assert "frame_hold" not in result.params


def test_negative_raises(ffmpeg_exe, tmp_path):
    mp = _manifest(tmp_path, 4, fps=10.0)
    with pytest.raises(ValueError, match="min_duration_s"):
        render_video(mp, str(tmp_path / "x.mp4"), preset="preview",
                     fps=10.0, min_duration_s=-1.0)


def test_hold_applies_after_blending(ffmpeg_exe, tmp_path):
    # 4 frames blended with 2 steps -> 4 + 3*2 = 10 frames; then
    # min 2.0s at 10fps needs 20 -> k=2 -> 20 frames total.
    mp = _manifest(tmp_path, 4, fps=10.0)
    out = str(tmp_path / "blend.mp4")
    result = render_video(mp, out, preset="preview", fps=10.0,
                          motion={"smooth": True, "smooth_steps": 2},
                          min_duration_s=2.0)
    assert result.n_frames == 20
    assert result.params["frame_hold"] == 2
