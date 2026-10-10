import os

import pytest

from reels.sites import (load_site_config, find_site_configs,
                         recipe_from_site, recipe_from_site_file,
                         animation_block_spec)

yaml = pytest.importorskip("yaml")


def _write_site(tmp_path, currents=None, animation=None, **kw):
    site = {
        "site_id": "lake-test",
        "name": "Lake Test",
        "geometry": {"bbox": [-87.9, 41.6, -86.7, 42.9]},
    }
    site.update(kw)
    if currents is not None:
        site["currents"] = currents
    if animation is not None:
        site["animation"] = animation
    p = tmp_path / "site.yaml"
    p.write_text(yaml.safe_dump(site))
    return str(p)


def test_from_site_currents_block(tmp_path):
    p = _write_site(tmp_path, currents={
        "source": "noaa-ofs", "ofs_code": "LMHOFS",
        "cycle": "00", "hours": [0, 12]})
    r = recipe_from_site_file(p)
    assert r.source == "noaa-ofs" and r.ofs_code == "LMHOFS"
    assert r.hours == [0, 12]
    assert r.bbox == [-87.9, 41.6, -86.7, 42.9]
    assert r.site_id == "lake-test"


def test_from_site_animation_block(tmp_path):
    p = _write_site(tmp_path,
                    currents={"source": "noaa-ofs", "ofs_code": "GLOFS"},
                    animation={"n_frames": 24, "flow_style": "reel-light",
                               "animate_preset": "wide", "fps": 24.0,
                               "title": "Custom", "output": "custom.mp4"})
    r = recipe_from_site_file(p)
    assert (r.n_frames, r.flow_style, r.animate_preset, r.fps) == \
        (24, "reel-light", "wide", 24.0)
    assert r.title == "Custom" and r.output == "custom.mp4"


def test_from_site_defaults(tmp_path):
    # no currents/animation blocks at all -> still a valid recipe
    p = _write_site(tmp_path)
    r = recipe_from_site_file(p)
    assert r.source == "noaa-ofs" and r.n_frames == 48
    assert r.title == "Lake Test surface currents"


def test_overrides_win(tmp_path):
    p = _write_site(tmp_path, currents={"source": "noaa-ofs"})
    r = recipe_from_site(load_site_config(p), {"n_frames": 10})
    assert r.n_frames == 10


def test_cmems_site(tmp_path):
    p = _write_site(tmp_path, currents={
        "source": "cmems", "preset": "global-physics-daily",
        "start": "2026-09-16T00:00:00", "end": "2026-09-18T00:00:00"})
    r = recipe_from_site_file(p)
    assert r.source == "cmems" and r.cmems_preset == "global-physics-daily"


def test_find_site_configs(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "site.yaml").write_text("site_id: a\n")
    (tmp_path / "b").mkdir()
    found = find_site_configs(str(tmp_path))
    assert found == [str(tmp_path / "a" / "site.yaml")]
    assert find_site_configs(str(tmp_path / "nope")) == []


def test_real_coast_site():
    coast = "/home/hatch/workspace/survey-sites/sites/coast/site.yaml"
    if not os.path.isfile(coast):
        pytest.skip("survey-sites checkout not present")
    r = recipe_from_site_file(coast)
    assert r.name  # builds even without currents/animation blocks


def test_animation_block_spec():
    spec = animation_block_spec()
    assert spec["block"] == "animation"
    assert "animate_preset" in spec["fields"]
    assert spec["fields"]["n_frames"]["default"] == 48
