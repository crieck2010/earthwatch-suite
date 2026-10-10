import json
import os

import pytest

from reels.deliver import finalize, SUMMARY_SCHEMA
from reels.recipe import ReelRecipe


def _recipe(**kw):
    data = {"name": "demo", "source": "synthetic",
            "bbox": [-87.9, 41.6, -86.7, 42.9],
            "site_id": "lake-test", "site_name": "Lake Test",
            "output": "demo.mp4"}
    data.update(kw)
    return ReelRecipe.from_dict(data)


def _video(tmp_path, name="v.mp4"):
    p = tmp_path / name
    p.write_bytes(b"video-bytes")
    return str(p)


def test_finalize(tmp_path):
    video = _video(tmp_path)
    # provenance sidecar alongside the video gets published too
    with open(video + ".provenance.json", "w") as fh:
        json.dump({"ok": True}, fh)
    done = finalize(_recipe(), video, str(tmp_path / "work"))
    summary_path = done.summary_path
    assert os.path.isfile(summary_path)
    assert done.video_path.endswith("demo.mp4")
    with open(summary_path) as fh:
        s = json.load(fh)
    assert s["schema"] == SUMMARY_SCHEMA
    # mirrors survey-monitor RunSummary fields
    for key in ("site_id", "site_name", "timestamp", "passes_processed",
                "passes_skipped", "alerts_fired", "alerts", "errors"):
        assert key in s, key
    assert s["site_id"] == "lake-test"
    assert s["passes_processed"] == s["n_frames"] == 48
    assert s["video_path"].endswith("demo.mp4")
    assert os.path.isfile(s["video_path"])
    assert s["provenance_sidecar"].endswith(".provenance.json")
    assert os.path.isfile(s["provenance_sidecar"])
    assert len(s["video_sha256"]) == 64


def test_finalize_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        finalize(_recipe(), str(tmp_path / "nope.mp4"), str(tmp_path / "w"))


def test_finalize_custom_out_dir(tmp_path):
    video = _video(tmp_path)
    out = tmp_path / "pub"
    done = finalize(_recipe(), video, str(tmp_path / "w"), out_dir=str(out))
    assert done.summary_path.startswith(str(out))
