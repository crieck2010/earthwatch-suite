"""Tests for flow.frames: manifest schema, round-trip, time interpolation."""

import json
import os

import pytest

from flow.frames import (MANIFEST_SCHEMA, FrameEntry, FrameManifest,
                         interpolate_times)


def _entry(i=0):
    return FrameEntry(frame_index=i, timestamp_iso="2026-09-16T00:00:00+00:00",
                      png_path=f"frames/frame_{i:04d}.png",
                      scalar_range=(12.0, 20.0))


def test_entry_round_trip():
    e = _entry(3)
    e2 = FrameEntry.from_dict(e.to_dict())
    assert e2.frame_index == 3
    assert e2.scalar_range == (12.0, 20.0)
    assert e2.png_path == "frames/frame_0003.png"


def test_entry_rejects_missing_keys():
    with pytest.raises(ValueError):
        FrameEntry.from_dict({"frame_index": 0})


def test_manifest_enforces_sequential_index():
    m = FrameManifest()
    m.add(_entry(0))
    with pytest.raises(ValueError):
        m.add(_entry(5))


def test_manifest_write_read_round_trip(tmp_path):
    m = FrameManifest(fps=24.0, size_px=(1080, 1920), field_source="test")
    for i in range(3):
        m.add(_entry(i))
    path = str(tmp_path / "manifest.json")
    m.write(path, generator="survey-flow 0.1.0")
    m2 = FrameManifest.read(path)
    assert len(m2) == 3
    assert m2.fps == 24.0
    assert m2.size_px == (1080, 1920)
    assert m2.field_source == "test"
    assert m2.generator == "survey-flow 0.1.0"
    assert m2.entries[2].frame_index == 2


def test_manifest_stores_relative_paths(tmp_path):
    m = FrameManifest()
    abs_png = str(tmp_path / "frames" / "frame_0000.png")
    m.add(FrameEntry(frame_index=0, timestamp_iso="t", png_path=abs_png))
    out = str(tmp_path / "out" / "manifest.json")
    m.write(out)
    with open(out, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["frames"][0]["png_path"] == "../frames/frame_0000.png"
    m2 = FrameManifest.read(out)
    resolved = m2.png_paths(os.path.dirname(out))
    assert resolved[0] == os.path.normpath(abs_png)


def test_manifest_rejects_unknown_schema(tmp_path):
    path = str(tmp_path / "bad.json")
    with open(path, "w") as fh:
        json.dump({"schema": "something-else/1.0", "frames": []}, fh)
    with pytest.raises(ValueError):
        FrameManifest.read(path)
    with open(path, "w") as fh:
        json.dump({"schema": "survey-flow.frame-manifest/2.0", "frames": []},
                  fh)
    with pytest.raises(ValueError):
        FrameManifest.read(path)


def test_manifest_schema_constant():
    assert MANIFEST_SCHEMA == "survey-flow.frame-manifest/1.0"


def test_interpolate_times_even():
    times = ["2026-09-16T00:00:00+00:00", "2026-09-16T02:00:00+00:00"]
    out = interpolate_times(times, 5)
    assert len(out) == 5
    assert out[0] == times[0]
    assert out[-1] == times[1]
    assert out[2] == "2026-09-16T01:00:00+00:00"


def test_interpolate_times_single_and_fallback():
    out = interpolate_times(["2026-09-16T00:00:00+00:00"], 3)
    assert out == ["2026-09-16T00:00:00+00:00"] * 3
    out = interpolate_times(["a", "b"], 4)
    assert out == ["a", "a", "b", "b"]
    with pytest.raises(ValueError):
        interpolate_times([], 3)
    with pytest.raises(ValueError):
        interpolate_times(["a"], 0)


def test_png_paths_without_base_dir():
    m = FrameManifest()
    m.add(_entry(0))
    assert m.png_paths() == ["frames/frame_0000.png"]
    assert len(m) == 1


def test_manifest_to_dict_schema_fields(tmp_path):
    m = FrameManifest()
    m.add(_entry(0))
    d = m.to_dict(generator="survey-flow 0.1.0")
    assert d["schema"] == MANIFEST_SCHEMA
    assert d["n_frames"] == 1
    assert d["generator"] == "survey-flow 0.1.0"
    assert "created" in d
