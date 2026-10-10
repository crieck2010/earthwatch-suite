"""Tests for manifest reading/validation and directory-based manifests."""

import json
import os

import pytest

from animate.manifest import (MANIFEST_SCHEMA_ID, Manifest, frames_from_dir,
                              read_manifest)


def test_read_valid_manifest(manifest_file):
    m = read_manifest(manifest_file)
    assert isinstance(m, Manifest)
    assert len(m) == 2
    assert m.fps == 12.0
    assert m.size_px == (48, 32)
    assert m.field_source == "synthetic/test"
    assert m.frames[0].index == 0
    assert m.frames[1].index == 1
    assert m.frames[0].timestamp_iso == "2026-09-16T00:00:00+00:00"
    assert m.frames[0].scalar_range == (10.0, 20.0)
    assert m.frames[0].colormap == "cividis"
    assert all(os.path.isabs(p) for p in m.png_paths())
    assert all(os.path.isfile(p) for p in m.png_paths())


def test_relative_paths_resolve_against_manifest_dir(manifest_file, tmp_path):
    m = read_manifest(manifest_file)
    assert os.path.dirname(m.png_paths()[0]) == str(tmp_path)


def test_absolute_png_paths_kept(tmp_path, manifest_dict, tmp_png):
    p, _ = tmp_png("abs.png", seed=9)
    manifest_dict["frames"][0]["png_path"] = p
    f = str(tmp_path / "m.json")
    with open(f, "w") as fh:
        json.dump(manifest_dict, fh)
    m = read_manifest(f)
    assert m.png_paths()[0] == p


def test_wrong_schema_rejected(tmp_path, manifest_dict):
    manifest_dict["schema"] = "something-else/3.0"
    f = str(tmp_path / "m.json")
    with open(f, "w") as fh:
        json.dump(manifest_dict, fh)
    with pytest.raises(ValueError, match="not a survey-flow frame manifest"):
        read_manifest(f)


def test_unknown_major_version_rejected(tmp_path, manifest_dict):
    manifest_dict["schema"] = "survey-flow.frame-manifest/2.0"
    f = str(tmp_path / "m.json")
    with open(f, "w") as fh:
        json.dump(manifest_dict, fh)
    with pytest.raises(ValueError, match="unsupported manifest major version"):
        read_manifest(f)


def test_nonsequential_indices_rejected(tmp_path, manifest_dict):
    manifest_dict["frames"][1]["frame_index"] = 7
    f = str(tmp_path / "m.json")
    with open(f, "w") as fh:
        json.dump(manifest_dict, fh)
    with pytest.raises(ValueError, match="sequential"):
        read_manifest(f)


def test_n_frames_mismatch_rejected(tmp_path, manifest_dict):
    manifest_dict["n_frames"] = 99
    f = str(tmp_path / "m.json")
    with open(f, "w") as fh:
        json.dump(manifest_dict, fh)
    with pytest.raises(ValueError, match="n_frames"):
        read_manifest(f)


def test_missing_png_reported(tmp_path, manifest_dict):
    manifest_dict["frames"][0]["png_path"] = "nope/missing.png"
    f = str(tmp_path / "m.json")
    with open(f, "w") as fh:
        json.dump(manifest_dict, fh)
    with pytest.raises(FileNotFoundError, match="missing PNG"):
        read_manifest(f)


def test_missing_required_key_rejected(tmp_path, manifest_dict):
    del manifest_dict["frames"][0]["png_path"]
    f = str(tmp_path / "m.json")
    with open(f, "w") as fh:
        json.dump(manifest_dict, fh)
    with pytest.raises(ValueError, match="missing required key"):
        read_manifest(f)


def test_empty_frames_rejected(tmp_path, manifest_dict):
    manifest_dict["frames"] = []
    manifest_dict["n_frames"] = 0
    f = str(tmp_path / "m.json")
    with open(f, "w") as fh:
        json.dump(manifest_dict, fh)
    with pytest.raises(ValueError, match="no frames"):
        read_manifest(f)


def test_invalid_json_rejected(tmp_path):
    f = str(tmp_path / "m.json")
    with open(f, "w") as fh:
        fh.write("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_manifest(f)


def test_schema_id_constant():
    assert MANIFEST_SCHEMA_ID == "survey-flow.frame-manifest/1.0"


def test_frames_from_dir_sorted(tmp_path, tmp_png):
    d = str(tmp_path / "frames")
    os.makedirs(d)
    names = ["c.png", "a.png", "b.png"]
    for i, n in enumerate(names):
        rng_p, _ = tmp_png(n, seed=i)  # noqa: F841 (writes into tmp_path root)
    # move into subdir in shuffled creation order
    import shutil
    for n in names:
        shutil.move(str(tmp_path / n), os.path.join(d, n))
    m = frames_from_dir(d)
    assert [os.path.basename(p) for p in m.png_paths()] == ["a.png", "b.png", "c.png"]
    assert [f.index for f in m.frames] == [0, 1, 2]


def test_frames_from_dir_timestamps(tmp_path, tmp_png):
    import shutil
    d = str(tmp_path / "fd")
    os.makedirs(d)
    p, _ = tmp_png("x.png", seed=3)
    shutil.move(p, os.path.join(d, "x.png"))
    m = frames_from_dir(d, timestamps=["2026-01-01T00:00:00+00:00"])
    assert m.timestamps() == ["2026-01-01T00:00:00+00:00"]


def test_frames_from_dir_empty_rejected(tmp_path):
    d = str(tmp_path / "empty")
    os.makedirs(d)
    with pytest.raises(ValueError, match="no PNG frames"):
        frames_from_dir(d)


def test_frames_from_dir_not_a_dir(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        frames_from_dir(str(tmp_path / "nope"))
