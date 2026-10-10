"""End-to-end pipeline and CLI tests (fully offline)."""

import json
import os
import subprocess
import sys

import pytest

from animate.encode import EncoderUnavailable, resolve_ffmpeg
from animate.pipeline import VideoResult, render_video


@pytest.fixture()
def ffmpeg_exe():
    try:
        return resolve_ffmpeg()
    except EncoderUnavailable:
        pytest.skip("no ffmpeg backend available")


def test_render_video_from_manifest(ffmpeg_exe, manifest_file, tmp_path):
    out = str(tmp_path / "reel.mp4")
    result = render_video(manifest_file, out, preset="preview",
                          burn_timestamps_=False)
    assert isinstance(result, VideoResult)
    assert result.fmt == "mp4"
    assert result.preset == "preview"
    assert result.n_frames == 2
    assert (result.width, result.height) == (540, 960)
    assert os.path.getsize(result.video_path) > 1000
    assert os.path.isfile(result.sidecar_path)
    with open(result.sidecar_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["preset"] == "preview"
    assert doc["n_frames"] == 2
    assert doc["video_sha256"]


def test_render_video_gif_and_timestamps(ffmpeg_exe, manifest_file, tmp_path):
    out = str(tmp_path / "r.gif")
    result = render_video(manifest_file, out, preset="preview", fmt="gif",
                          fps=5.0)
    assert result.fmt == "gif"
    assert result.n_frames == 2
    assert os.path.getsize(out) > 1000


def test_render_video_from_png_dir(ffmpeg_exe, tmp_path, tmp_png):
    import shutil
    d = str(tmp_path / "frames")
    os.makedirs(d)
    for i, name in enumerate(("b.png", "a.png")):
        p, _ = tmp_png(name, seed=i)
        shutil.move(p, os.path.join(d, name))
    out = str(tmp_path / "d.mp4")
    result = render_video(d, out, preset="preview", burn_timestamps_=False)
    assert result.n_frames == 2


def test_render_video_with_title(ffmpeg_exe, manifest_file, tmp_path):
    pytest.importorskip("matplotlib")
    out = str(tmp_path / "t.mp4")
    result = render_video(manifest_file, out, preset="preview",
                          title="Demo", subtitle="sub",
                          title_seconds=1.0, fps=10.0,
                          burn_timestamps_=False)
    # 2 content frames + 10 title frames at 10 fps
    assert result.n_frames == 12


def test_render_video_bad_preset(ffmpeg_exe, manifest_file, tmp_path):
    with pytest.raises(ValueError, match="unknown preset"):
        render_video(manifest_file, str(tmp_path / "x.mp4"), preset="nope")


def test_render_video_no_encoder(manifest_file, tmp_path, monkeypatch):
    import animate.pipeline as pl
    def _boom():
        raise EncoderUnavailable("nope")
    monkeypatch.setattr(pl, "resolve_ffmpeg", _boom)
    with pytest.raises(EncoderUnavailable):
        render_video(manifest_file, str(tmp_path / "x.mp4"), preset="preview")


def _cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(os.getcwd(), "src")
    return subprocess.run([sys.executable, "-m", "animate.cli", *args],
                          capture_output=True, text=True, env=env)


def test_cli_presets():
    r = _cli("presets")
    assert r.returncode == 0
    assert "reel" in r.stdout and "wide" in r.stdout and "square" in r.stdout


def test_cli_manifest_info(manifest_file):
    r = _cli("manifest-info", manifest_file)
    assert r.returncode == 0
    info = json.loads(r.stdout)
    assert info["n_frames"] == 2


def test_cli_manifest_info_bad(manifest_file, tmp_path):
    bad = str(tmp_path / "bad.json")
    with open(bad, "w") as fh:
        fh.write("{}")
    r = _cli("manifest-info", bad)
    assert r.returncode == 2


def test_cli_check():
    r = _cli("check")
    assert r.returncode == 0
    assert "ffmpeg:" in r.stdout


def test_cli_demo(ffmpeg_exe, tmp_path):
    out = str(tmp_path / "demo.mp4")
    r = _cli("demo", "--out", out, "--preset", "preview")
    assert r.returncode == 0, r.stderr
    assert os.path.getsize(out) > 1000
    assert os.path.isfile(out + ".provenance.json")


def test_cli_render_manifest(ffmpeg_exe, manifest_file, tmp_path):
    out = str(tmp_path / "cli.mp4")
    r = _cli("render", "--manifest", manifest_file, "--preset", "preview",
             "--out", out, "--no-timestamps")
    assert r.returncode == 0, r.stderr
    assert os.path.getsize(out) > 1000
