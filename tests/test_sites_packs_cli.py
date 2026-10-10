"""Scaffold + packs + CLI tests."""

import json
import os
import subprocess
import sys

import pytest
import yaml

from sites.packs import list_packs, load_pack
from sites.scaffold import TEMPLATES, scaffold, write_config
from sites.validate import validate_config, validate_file

PACK_ROOT = os.path.join(os.path.dirname(__file__), "..", "sites")


# --- scaffold ------------------------------------------------------------


def test_scaffold_unknown_template():
    with pytest.raises(ValueError):
        scaffold("x", "X", template="nope")


def test_scaffold_write_and_validate_roundtrip(tmp_path):
    for template in TEMPLATES:
        path = str(tmp_path / f"{template}.yaml")
        write_config(scaffold(f"rt-{template}", f"RT {template}", template), path)
        assert validate_file(path) == [], f"template {template} round-trip failed"


def test_scaffold_json_roundtrip(tmp_path):
    path = str(tmp_path / "rt.json")
    write_config(scaffold("rt-json", "RT JSON", "farm"), path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["site_id"] == "rt-json"
    assert validate_config(data) == []


def test_scaffold_coast_has_transects():
    cfg = scaffold("c", "C", template="coast")
    assert len(cfg["transects"]) == 2
    assert cfg["engines"]["coast"]["transect_ids"] == ["shore-a", "shore-b"]


# --- packs ---------------------------------------------------------------


def test_list_packs_finds_four():
    packs = list_packs(PACK_ROOT)
    assert sorted(p.site_id for p in packs) == [
        "black-rock-quarry", "green-acre-farm", "harbor-cove", "pine-ridge-forest",
    ]


def test_all_shipped_packs_validate():
    for pack in list_packs(PACK_ROOT):
        assert pack.valid, f"{pack.site_id}: {pack.errors}"


def test_pack_geojson_sidecars_match_config():
    for pack in list_packs(PACK_ROOT):
        zones_path = os.path.join(pack.directory, "zones.geojson")
        with open(zones_path, encoding="utf-8") as fh:
            gj = json.load(fh)
        assert gj["type"] == "FeatureCollection"
        cfg_ids = [z["zone_id"] for z in pack.config.get("zones", [])]
        gj_ids = [f["properties"]["zone_id"] for f in gj["features"]]
        assert gj_ids == cfg_ids, f"{pack.site_id}: sidecar/config zone mismatch"
        fp_path = os.path.join(pack.directory, "footprint.geojson")
        with open(fp_path, encoding="utf-8") as fh:
            fp = json.load(fh)
        assert fp["geometry"] == pack.config["geometry"]


def test_pack_demo_scripts_exist_and_executable():
    for pack in list_packs(PACK_ROOT):
        demo = os.path.join(pack.directory, "demo.sh")
        assert os.path.isfile(demo), f"{pack.site_id}: demo.sh missing"
        assert os.access(demo, os.X_OK), f"{pack.site_id}: demo.sh not executable"


def test_load_pack_missing_config(tmp_path):
    pack = load_pack(str(tmp_path))
    assert not pack.valid
    assert any("no site.yaml" in e for e in pack.errors)


# --- CLI -----------------------------------------------------------------


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "sites.cli", *args],
        capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), ".."),
        env={**os.environ, "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "src")},
    )


def test_cli_validate_ok():
    r = run_cli("validate", os.path.join(PACK_ROOT, "quarry", "site.yaml"))
    assert r.returncode == 0, r.stderr
    assert "valid site config" in r.stdout


def test_cli_validate_bad(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("site_id: 'Bad!'\n")
    r = run_cli("validate", str(p))
    assert r.returncode == 1


def test_cli_new_and_list(tmp_path):
    out = str(tmp_path / "fresh.yaml")
    r = run_cli("new", "fresh-site", "--name", "Fresh", "--template", "farm", "--out", out)
    assert r.returncode == 0, r.stderr
    r = run_cli("validate", out)
    assert r.returncode == 0
    r = run_cli("list", "--pack-dir", PACK_ROOT)
    assert r.returncode == 0 and "harbor-cove" in r.stdout


def test_cli_show_and_stories():
    r = run_cli("show", os.path.join(PACK_ROOT, "coast"))
    assert r.returncode == 0 and "shore-a" in r.stdout
    r = run_cli("stories", "--pack-dir", PACK_ROOT)
    assert r.returncode == 0 and "demo.sh" in r.stdout
