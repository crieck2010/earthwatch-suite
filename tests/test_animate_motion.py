"""Tests for cinematic motion (zoom/pan/smooth) and audio muxing."""

from __future__ import annotations

import json
import math
import os
import struct
import wave

import numpy as np
import pytest

from animate.encode import EncoderUnavailable, resolve_ffmpeg
from animate.motion import (MotionSpec, blend_frames, blend_stamps,
                            coerce_motion, motion_filter)
from animate.pipeline import render_video


@pytest.fixture()
def ffmpeg_exe():
    try:
        return resolve_ffmpeg()
    except EncoderUnavailable:
        pytest.skip("no ffmpeg backend available")


# ---------------------------------------------------------------------------
# MotionSpec validation
# ---------------------------------------------------------------------------

def test_motion_spec_defaults_off():
    spec = MotionSpec()
    assert not spec.animated
    assert not spec.blending


def test_motion_spec_rejects_bad_values():
    with pytest.raises(ValueError):
        MotionSpec(zoom="sideways")
    with pytest.raises(ValueError):
        MotionSpec(pan="nowhere")
    with pytest.raises(ValueError):
        MotionSpec(zoom="in", zoom_speed=1.5)
    with pytest.raises(ValueError):
        MotionSpec(zoom="in", zoom_speed=-0.1)
    with pytest.raises(ValueError):
        MotionSpec(pan="left", pan_speed=2.0)
    with pytest.raises(ValueError):
        MotionSpec(smooth_steps=-1)
    with pytest.raises(ValueError):
        MotionSpec.from_dict({"zoom": "in", "bogus": 1})


def test_coerce_motion():
    assert coerce_motion(None) is None
    spec = MotionSpec(zoom="in")
    assert coerce_motion(spec) is spec
    coerced = coerce_motion({"zoom": "out", "zoom_speed": 0.5})
    assert isinstance(coerced, MotionSpec) and coerced.zoom == "out"
    with pytest.raises(TypeError):
        coerce_motion("zoom in please")


def test_motion_spec_round_trip():
    spec = MotionSpec(zoom="in", zoom_speed=0.5, pan="up-right",
                      pan_speed=0.4, smooth=True, smooth_steps=2)
    assert MotionSpec.from_dict(spec.to_dict()) == spec
    assert spec.animated and spec.blending


# ---------------------------------------------------------------------------
# motion_filter
# ---------------------------------------------------------------------------

def test_motion_filter_none_when_off():
    assert motion_filter(MotionSpec(), 1080, 1920, 30) is None


def test_motion_filter_zoom_in():
    f = motion_filter(MotionSpec(zoom="in", zoom_speed=0.5), 1080, 1920, 30)
    assert f.startswith("zoompan=")
    assert "s=1080x1920" in f
    assert "d=1" in f
    # zoom ramps 1.0 -> 1.3 across the clip
    assert "1.0000+(1.3000-1.0000)*(on/29)" in f


def test_motion_filter_zoom_out_reverses():
    f = motion_filter(MotionSpec(zoom="out", zoom_speed=0.5), 1080, 1920, 30)
    assert "1.3000+(1.0000-1.3000)*(on/29)" in f


def test_motion_filter_pan_only_has_headroom():
    f = motion_filter(MotionSpec(pan="right", pan_speed=1.0), 1080, 1920, 30)
    # constant slight zoom so the window can drift without showing edges
    assert "1.1500+(1.1500-1.1500)" in f
    assert "(iw-iw/zoom)" in f  # pan moves the crop window


def test_motion_filter_single_frame_no_div_zero():
    f = motion_filter(MotionSpec(zoom="in"), 640, 480, 1)
    assert f is not None
    assert "/0)" not in f


def test_motion_filter_all_pan_directions():
    for pan in ("up", "down", "left", "right", "up-left", "up-right",
                "down-left", "down-right"):
        f = motion_filter(MotionSpec(pan=pan), 640, 480, 10)
        assert f is not None, pan


# ---------------------------------------------------------------------------
# blend_frames / blend_stamps
# ---------------------------------------------------------------------------

def test_blend_frames_count_and_endpoints():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.full((4, 4, 3), 100, dtype=np.uint8)
    out = blend_frames([a, b], 3)
    assert len(out) == 2 + 3  # n + (n-1)*steps
    assert out[0] is a and out[-1] is b
    # midpoint is the average
    assert out[2].mean() == pytest.approx(50, abs=1)


def test_blend_frames_noop():
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
    assert blend_frames(frames, 3) == frames
    assert blend_frames(frames * 2, 0) == frames * 2


def test_blend_stamps_alignment():
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)] * 3
    stamps = ["t0", "t1", "t2"]
    assert blend_stamps(stamps, 2) == ["t0", "t0", "t0", "t1", "t1", "t1", "t2"]
    assert len(blend_frames(frames, 2)) == len(blend_stamps(stamps, 2))


# ---------------------------------------------------------------------------
# render_video integration
# ---------------------------------------------------------------------------

def _sine_wav(path, seconds=1.0, rate=8000):
    n = int(seconds * rate)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(
            struct.pack("<h", int(9000 * math.sin(2 * math.pi * 440 * i / rate)))
            for i in range(n)))


def test_render_video_with_motion(ffmpeg_exe, manifest_file, tmp_path):
    out = str(tmp_path / "motion.mp4")
    result = render_video(manifest_file, out, preset="preview",
                          motion={"zoom": "in", "zoom_speed": 0.5,
                                  "pan": "right", "pan_speed": 0.3},
                          ffmpeg_exe=ffmpeg_exe)
    assert os.path.isfile(result.video_path)
    assert result.params["motion"]["zoom"] == "in"
    sidecar = json.load(open(result.sidecar_path))
    assert sidecar["params"]["motion"]["pan"] == "right"


def test_render_video_smooth_increases_frames(ffmpeg_exe, manifest_file,
                                              tmp_path):
    out = str(tmp_path / "smooth.mp4")
    result = render_video(manifest_file, out, preset="preview",
                          motion={"smooth": True, "smooth_steps": 2},
                          ffmpeg_exe=ffmpeg_exe)
    # 2 data frames + 2 blended, no title card requested
    assert result.n_frames == 4


def test_render_video_rejects_bad_motion(ffmpeg_exe, manifest_file, tmp_path):
    with pytest.raises(ValueError):
        render_video(manifest_file, str(tmp_path / "x.mp4"),
                     preset="preview", motion={"zoom": "sideways"},
                     ffmpeg_exe=ffmpeg_exe)


def test_render_video_with_audio(ffmpeg_exe, manifest_file, tmp_path):
    wav = str(tmp_path / "tone.wav")
    _sine_wav(wav)
    out = str(tmp_path / "aud.mp4")
    result = render_video(manifest_file, out, preset="preview",
                          audio_path=wav, ffmpeg_exe=ffmpeg_exe)
    assert os.path.isfile(result.video_path)
    assert result.params["audio"] == "tone.wav"
    # the muxed file must actually contain an audio stream
    import subprocess
    probe = subprocess.run(
        [ffmpeg_exe, "-hide_banner", "-i", result.video_path],
        capture_output=True, text=True)
    assert "Audio:" in probe.stderr


def test_render_video_missing_audio_fails_fast(ffmpeg_exe, manifest_file,
                                               tmp_path):
    with pytest.raises(FileNotFoundError):
        render_video(manifest_file, str(tmp_path / "x.mp4"),
                     preset="preview",
                     audio_path=str(tmp_path / "nope.mp3"),
                     ffmpeg_exe=ffmpeg_exe)


def test_render_video_gif_ignores_audio(ffmpeg_exe, manifest_file, tmp_path):
    wav = str(tmp_path / "tone.wav")
    _sine_wav(wav)
    out = str(tmp_path / "aud.gif")
    result = render_video(manifest_file, out, preset="preview",
                          audio_path=wav, ffmpeg_exe=ffmpeg_exe)
    assert os.path.isfile(result.video_path)
    assert result.params["audio_skipped"] == "gif has no audio track"
