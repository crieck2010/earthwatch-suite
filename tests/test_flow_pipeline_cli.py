"""Tests for flow.pipeline and flow.cli (render path needs matplotlib)."""

import json
import os

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from flow.cli import build_parser, main
from flow.fields import TimeVaryingField
from flow.frames import FrameManifest
from flow.pipeline import render_animation

from conftest import make_fake_current_field


def test_render_animation_tiny(tmp_path, fake_cf):
    tvf = TimeVaryingField.from_current_field(fake_cf)
    out = str(tmp_path / "anim")
    manifest = render_animation(tvf, out, n_frames=3, n_particles=20,
                                size=(270, 480), dpi=50, fps=24.0, seed=7)
    assert len(manifest) == 3
    assert manifest.fps == 24.0
    assert manifest.size_px == (270, 480)
    assert manifest.field_source == "test/fake"
    assert os.path.exists(os.path.join(out, "manifest.json"))
    for i in range(3):
        assert os.path.exists(os.path.join(out, "frames", f"frame_{i:04d}.png"))
    # Fixed scalar range across frames (no colormap flicker).
    ranges = {e.scalar_range for e in manifest.entries}
    assert len(ranges) == 1
    # Manifest survives a write/read cycle.
    m2 = FrameManifest.read(os.path.join(out, "manifest.json"))
    assert len(m2) == 3


def test_render_animation_rejects_bad_n_frames(tmp_path, fake_cf):
    tvf = TimeVaryingField.from_current_field(fake_cf)
    with pytest.raises(ValueError):
        render_animation(tvf, str(tmp_path), n_frames=0)
    with pytest.raises(ValueError):
        render_animation(tvf, str(tmp_path), size="portrait")


def test_cli_styles_and_sizes(capsys):
    assert main(["styles"]) == 0
    out = capsys.readouterr().out
    assert "reel-dark" in out
    assert main(["sizes"]) == 0
    assert "reel" in capsys.readouterr().out


def test_cli_render_from_json(tmp_path, fake_cf):
    field_path = str(tmp_path / "field.json")
    with open(field_path, "w") as fh:
        json.dump({
            "u": fake_cf.u.tolist(), "v": fake_cf.v.tolist(),
            "temperature": fake_cf.temperature.tolist(),
            "times": fake_cf.times,
            "lats": fake_cf.lats.tolist(), "lons": fake_cf.lons.tolist(),
            "crs": "EPSG:4326", "source": "test/fake",
        }, fh)
    out = str(tmp_path / "anim")
    rc = main(["render", "--field", field_path, "--out", out,
               "--frames", "2", "--particles", "10",
               "--size", "270x480", "--dpi", "50"])
    assert rc == 0
    assert os.path.exists(os.path.join(out, "manifest.json"))


def test_cli_render_missing_input_returns_2(tmp_path):
    assert main(["render", "--out", str(tmp_path)]) == 2


def test_cli_demo_offline(tmp_path):
    out = str(tmp_path / "demo")
    assert main(["demo", "--out", out, "--frames", "2", "--particles", "10"]) == 0
    assert os.path.exists(os.path.join(out, "manifest.json"))
    assert os.path.exists(os.path.join(out, "frames", "frame_0001.png"))


def test_cli_manifest_rebuild(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(2):
        (d / f"frame_{i:04d}.png").write_bytes(b"fake")
    assert main(["manifest", "--dir", str(d), "--fps", "15"]) == 0
    m = FrameManifest.read(str(d / "manifest.json"))
    assert len(m) == 2 and m.fps == 15.0


def test_cli_manifest_empty_dir_returns_2(tmp_path):
    assert main(["manifest", "--dir", str(tmp_path)]) == 2


def test_parser_has_all_commands():
    parser = build_parser()
    # Trigger subparser registration by parsing each command.
    cases = {
        "render": ["render", "--out", "x"],
        "demo": ["demo", "--out", "x"],
        "styles": ["styles"],
        "sizes": ["sizes"],
        "manifest": ["manifest", "--dir", "x"],
    }
    for cmd, argv in cases.items():
        ns, _ = parser.parse_known_args(argv)
        assert ns.command == cmd


def test_progress_callback_called(tmp_path, fake_cf):
    tvf = TimeVaryingField.from_current_field(fake_cf)
    calls = []
    render_animation(tvf, str(tmp_path), n_frames=3, n_particles=5,
                     size=(135, 240), dpi=50,
                     progress=lambda i, n: calls.append((i, n)))
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "0.1.0" in capsys.readouterr().out
