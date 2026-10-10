"""Runner tests with injected fakes — no peer modules, no network."""

import json
import os

import pytest

from reels.recipe import ReelRecipe
from reels.runner import run_recipe, RunState, STATE_FILE


class FakeField:
    def __init__(self, payload="{}"):
        self.payload = payload

    def to_json(self, path):
        with open(path, "w") as fh:
            fh.write(self.payload)
        return path


def _recipe(**kw):
    data = {"name": "t", "source": "synthetic",
            "bbox": [-87.9, 41.6, -86.7, 42.9], "n_frames": 4,
            "output": "t.mp4"}
    data.update(kw)
    return ReelRecipe.from_dict(data)


def _fakes(calls, fail_at=None):
    def fetch_fn(recipe, stage_dir, cache, rhash):
        calls.append("fetch")
        if fail_at == "fetch":
            raise RuntimeError("boom-fetch")
        p = os.path.join(stage_dir, "field.json")
        FakeField('{"ok": true}').to_json(p)
        return FakeField(), p

    def render_fn(recipe, field_obj, stage_dir, progress=None):
        calls.append("render")
        if fail_at == "render":
            raise RuntimeError("boom-render")
        if progress:
            progress(4, 4)
        mp = os.path.join(stage_dir, "manifest.json")
        with open(mp, "w") as fh:
            json.dump({"schema": "survey-flow.frame-manifest/1.0",
                       "frames": []}, fh)
        return mp

    def encode_fn(recipe, manifest_path, stage_dir, progress=None):
        calls.append("encode")
        if fail_at == "encode":
            raise RuntimeError("boom-encode")
        vp = os.path.join(stage_dir, "t.mp4")
        with open(vp, "wb") as fh:
            fh.write(b"fake-video")
        return vp

    return fetch_fn, render_fn, encode_fn


def test_full_run(tmp_path):
    calls = []
    fetch_fn, render_fn, encode_fn = _fakes(calls)
    work = str(tmp_path / "work")
    res = run_recipe(_recipe(), work, fetch_fn=fetch_fn,
                     render_fn=render_fn, encode_fn=encode_fn)
    assert res.ok, res.errors
    assert res.stages_run == ["fetch", "render", "encode", "deliver"]
    assert res.stages_skipped == []
    assert os.path.isfile(res.video_path)
    assert os.path.isfile(res.summary_path)
    with open(res.summary_path) as fh:
        summary = json.load(fh)
    assert summary["recipe_name"] == "t"
    assert summary["site_id"] == ""
    assert summary["video_path"] == res.video_path


def test_resume_skips_done(tmp_path):
    calls = []
    fetch_fn, render_fn, encode_fn = _fakes(calls)
    work = str(tmp_path / "work")
    r1 = run_recipe(_recipe(), work, fetch_fn=fetch_fn,
                    render_fn=render_fn, encode_fn=encode_fn)
    assert r1.ok
    calls.clear()
    r2 = run_recipe(_recipe(), work, fetch_fn=fetch_fn,
                    render_fn=render_fn, encode_fn=encode_fn)
    assert r2.ok
    assert r2.stages_run == []
    assert set(r2.stages_skipped) == {"fetch", "render", "encode", "deliver"}
    assert calls == []


def test_recipe_change_reruns(tmp_path):
    calls = []
    fetch_fn, render_fn, encode_fn = _fakes(calls)
    work = str(tmp_path / "work")
    assert run_recipe(_recipe(), work, fetch_fn=fetch_fn,
                      render_fn=render_fn, encode_fn=encode_fn).ok
    r2 = run_recipe(_recipe(n_frames=8), work, fetch_fn=fetch_fn,
                    render_fn=render_fn, encode_fn=encode_fn)
    assert r2.ok
    assert r2.stages_run == ["fetch", "render", "encode", "deliver"]


def test_failure_stops_pipeline(tmp_path):
    calls = []
    fetch_fn, render_fn, encode_fn = _fakes(calls, fail_at="render")
    work = str(tmp_path / "work")
    res = run_recipe(_recipe(), work, fetch_fn=fetch_fn,
                     render_fn=render_fn, encode_fn=encode_fn)
    assert not res.ok
    assert res.stages_run == ["fetch"]
    assert any("render" in e for e in res.errors)
    # state records the failure
    with open(os.path.join(work, STATE_FILE)) as fh:
        state = RunState.from_dict(json.load(fh))
    assert state.stages["render"].status == "failed"
    assert state.stages["fetch"].status == "done"


def test_resume_after_failure(tmp_path):
    calls = []
    fns = _fakes(calls, fail_at="render")
    work = str(tmp_path / "work")
    res = run_recipe(_recipe(), work, fetch_fn=fns[0],
                     render_fn=fns[1], encode_fn=fns[2])
    assert not res.ok
    calls.clear()
    good = _fakes(calls)
    res2 = run_recipe(_recipe(), work, fetch_fn=good[0],
                      render_fn=good[1], encode_fn=good[2],
                      load_field_fn=lambda p: FakeField(open(p).read()))
    assert res2.ok, res2.errors
    assert res2.stages_skipped == ["fetch"]
    assert "render" in res2.stages_run


def test_progress_callback(tmp_path):
    calls = []
    fetch_fn, render_fn, encode_fn = _fakes(calls)
    seen = []
    res = run_recipe(_recipe(), str(tmp_path / "w"), fetch_fn=fetch_fn,
                     render_fn=render_fn, encode_fn=encode_fn,
                     progress=lambda s, d, t: seen.append((s, d, t)))
    assert res.ok
    assert ("render", 4, 4) in seen


def test_run_state_round_trip():
    s = RunState(recipe_name="x", recipe_sha="y")
    assert RunState.from_dict(s.to_dict()).recipe_sha == "y"
