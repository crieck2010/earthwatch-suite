"""End-to-end CLI tests in --synthetic mode (no network, no data files)."""

import csv
import json
import os

import pytest

from elev3d.cli import main
from elev3d.synthetic import CUT_M3, FILL_M3


def test_cli_diff_synthetic(capsys):
    assert main(["diff", "--synthetic", "--noise-floor", "0.08"]) == 0
    out = capsys.readouterr().out
    summary = json.loads(out)
    assert summary["valid_pixels"] == 10000
    assert summary["changed_pixels"] > 2000


def test_cli_volumes_synthetic(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["volumes", "--synthetic", "--noise-floor", "0.08",
                 "--rmse1", "0.03", "--rmse2", "0.03",
                 "--out", "volumes.csv"]) == 0
    with open("volumes.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3  # cut-pit, fill-mound, stable-road
    by_id = {r["zone_id"]: r for r in rows}
    assert float(by_id["cut-pit"]["cut_m3"]) == pytest.approx(CUT_M3, rel=0.05)
    assert float(by_id["fill-mound"]["fill_m3"]) == pytest.approx(FILL_M3, rel=0.05)
    assert by_id["cut-pit"]["confidence"] == "high"
    # re-run is a no-op (dedupe)
    assert main(["volumes", "--synthetic", "--noise-floor", "0.08",
                 "--out", "volumes.csv"]) == 0
    with open("volumes.csv", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 3


def test_cli_polygons_synthetic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["polygons", "--synthetic", "--threshold", "0.08",
                 "--out", "polys.geojson"]) == 0
    with open("polys.geojson", encoding="utf-8") as fh:
        gj = json.load(fh)
    assert len(gj["features"]) == 2
    assert {f["properties"]["dominant"] for f in gj["features"]} == {"cut", "fill"}


def test_cli_coreg_synthetic_passes(capsys):
    assert main(["coreg", "--synthetic"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True


def test_cli_demo(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["demo"]) == 0
    assert os.path.exists("demo_volumes.csv")
    assert os.path.exists("demo_polygons.geojson")
    out = capsys.readouterr().out
    assert "cut=" in out and "fill=" in out
