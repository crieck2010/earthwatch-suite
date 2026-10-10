"""Tests for ffmpeg backend resolution and the three encoders.

Encoding tests use tiny synthetic frames and the real backend (the
imageio-ffmpeg wheel's bundled binary when installed, else system
ffmpeg). They skip gracefully when no backend exists.
"""

import os
import subprocess

import numpy as np
import pytest

from animate.encode import (EncoderUnavailable, encode, format_for_path,
                            resolve_ffmpeg, resolve_h264_codec, encode_gif,
                            encode_mp4, encode_webm)


def _frames(n=6, h=48, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.random((h, w, 3)) * 255).astype(np.uint8) for _ in range(n)]


@pytest.fixture()
def ffmpeg_exe():
    try:
        return resolve_ffmpeg()
    except EncoderUnavailable:
        pytest.skip("no ffmpeg backend available")


def _probe(exe, path, *needles):
    r = subprocess.run([exe, "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    err = r.stderr
    return all(n in err for n in needles)


def test_resolve_ffmpeg_returns_executable(ffmpeg_exe):
    assert os.path.isfile(ffmpeg_exe)


def test_resolve_h264_codec_valid(ffmpeg_exe):
    assert resolve_h264_codec(ffmpeg_exe) in ("libx264", "mpeg4")


def test_encoder_unavailable_when_nothing_found(monkeypatch):
    import animate.encode as enc
    monkeypatch.setattr(enc, "shutil",
                        type("S", (), {"which": staticmethod(lambda *_: None)})())
    monkeypatch.setitem(__import__("sys").modules, "imageio_ffmpeg", None)
    with pytest.raises(EncoderUnavailable, match="no ffmpeg backend"):
        enc.resolve_ffmpeg()


def test_encode_mp4(ffmpeg_exe, tmp_path):
    out = str(tmp_path / "t.mp4")
    seen = []
    encode_mp4(iter(_frames()), out, fps=12.0, progress=seen.append)
    assert os.path.getsize(out) > 1000
    assert _probe(ffmpeg_exe, out, "h264", "64x48")
    assert seen == [1, 2, 3, 4, 5, 6]


def test_encode_gif_two_pass(ffmpeg_exe, tmp_path):
    out = str(tmp_path / "t.gif")
    encode_gif(iter(_frames()), out, fps=5.0)
    assert os.path.getsize(out) > 1000
    assert _probe(ffmpeg_exe, out, "gif")


def test_encode_webm(ffmpeg_exe, tmp_path):
    out = str(tmp_path / "t.webm")
    encode_webm(iter(_frames()), out, fps=12.0)
    assert os.path.getsize(out) > 1000
    assert _probe(ffmpeg_exe, out, "vp9")


def test_encode_dispatch(ffmpeg_exe, tmp_path):
    for ext in ("mp4", "gif", "webm"):
        out = str(tmp_path / f"d.{ext}")
        encode(iter(_frames(n=3)), out, ext, fps=5.0)
        assert os.path.getsize(out) > 500


def test_encode_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="unsupported video format"):
        encode(iter(_frames(n=1)), str(tmp_path / "x.avi"), "avi")


def test_encode_rejects_empty(tmp_path):
    with pytest.raises(ValueError, match="at least one frame"):
        encode_mp4(iter([]), str(tmp_path / "x.mp4"))


def test_odd_dimensions_padded_even(ffmpeg_exe, tmp_path):
    out = str(tmp_path / "odd.mp4")
    encode_mp4(iter(_frames(n=3, h=65, w=97)), out)
    assert _probe(ffmpeg_exe, out, "98x66")


def test_format_for_path():
    assert format_for_path("a.mp4") == "mp4"
    assert format_for_path("a.GIF") == "gif"
    assert format_for_path("a.webm") == "webm"
    with pytest.raises(ValueError, match="cannot infer"):
        format_for_path("a.mov")
