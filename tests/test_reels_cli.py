import json
import os

import pytest

from reels import _peers
from reels.cli import build_parser, main


def test_peer_missing_message():
    with pytest.raises(_peers.PeerMissing) as ei:
        _peers._load("no_such_module_xyz.thing", "some-pkg")
    assert "pip install some-pkg" in str(ei.value)


def test_recipe_new_and_show(tmp_path):
    out = str(tmp_path / "r.yaml")
    assert main(["recipe-new", "--out", out]) == 0
    assert os.path.isfile(out)
    assert main(["recipe-show", out]) == 0


def test_recipe_show_invalid(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: ''\n")
    assert main(["recipe-show", str(bad)]) == 1


def test_from_site(tmp_path):
    pytest.importorskip("yaml")
    import yaml

    site = tmp_path / "site.yaml"
    site.write_text(yaml.safe_dump({
        "site_id": "s1", "name": "S1",
        "geometry": {"bbox": [-87.9, 41.6, -86.7, 42.9]},
        "currents": {"source": "noaa-ofs", "ofs_code": "LMHOFS"},
        "animation": {"n_frames": 6},
    }))
    out = str(tmp_path / "r.yaml")
    assert main(["from-site", str(site), "--out", out]) == 0
    assert os.path.isfile(out)


def test_cache_clear(tmp_path):
    from reels.cache import Cache

    work = str(tmp_path / "w")
    Cache(os.path.join(work, ".cache")).store_field("ab" * 32, "{}")
    assert main(["cache-clear", "--work-dir", work]) == 0
    assert not os.path.exists(os.path.join(work, ".cache", "fields"))


def test_demo_without_peers_reports_error(tmp_path, monkeypatch):
    # peers absent -> demo fails with a clear message, exit 1 (no crash)
    import reels._peers as peers

    def _boom(*a, **k):
        raise peers.PeerMissing("nope")

    monkeypatch.setattr(peers, "current_field_cls", _boom)
    rc = main(["demo", "--work-dir", str(tmp_path / "w")])
    assert rc == 1
