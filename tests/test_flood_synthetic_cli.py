"""End-to-end CLI tests in --synthetic mode (no network)."""

import csv
import json
import os

from flood.cli import main
from flood.synthetic import PLANTED_FLOODS


def test_demo_detects_planted_flood(tmp_path):
    outdir = str(tmp_path / "demo")
    assert main(["demo", "--out-dir", outdir]) == 0
    for name in ("zones.geojson", "timeseries.csv", "delta.csv",
                 "flood_polygons.geojson", "watermask.npy", "watermask.qml"):
        assert os.path.exists(os.path.join(outdir, name)), name
    with open(os.path.join(outdir, "delta.csv")) as fh:
        rows = list(csv.DictReader(fh))
    planted = [
        r for r in rows
        if r["zone_id"] == "river-reach" and r["date"] == "2024-04-15"
    ]
    assert len(planted) == 1
    assert planted[0]["flooded"] == "true"
    assert float(planted[0]["delta_ha"]) > 0


def test_demo_upland_never_floods(tmp_path):
    outdir = str(tmp_path / "demo")
    assert main(["demo", "--out-dir", outdir]) == 0
    with open(os.path.join(outdir, "delta.csv")) as fh:
        rows = list(csv.DictReader(fh))
    upland = [r for r in rows if r["zone_id"] == "upland-control"]
    assert upland and not any(r["flooded"] == "true" for r in upland)


def test_timeseries_synthetic(tmp_path):
    out = str(tmp_path / "ts.csv")
    assert main(["timeseries", "--synthetic", "--out", out]) == 0
    with open(out) as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    assert "water_ha" in rows[0] and "water_fraction" in rows[0]
    # 3 zones x 36 months x 1 index
    assert len(rows) == 3 * 36


def test_watermask_synthetic(tmp_path):
    out = str(tmp_path / "mask.npy")
    assert main(["watermask", "--synthetic", "--out", out]) == 0
    assert os.path.exists(out)
    assert os.path.exists(str(tmp_path / "mask.qml"))


def test_polygons_synthetic(tmp_path):
    out = str(tmp_path / "polys.geojson")
    assert main(["polygons", "--synthetic", "--out", out]) == 0
    with open(out) as fh:
        data = json.load(fh)
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) >= 1


def test_delta_cli(tmp_path):
    ts = str(tmp_path / "ts.csv")
    assert main(["timeseries", "--synthetic", "--out", ts]) == 0
    out = str(tmp_path / "delta.csv")
    assert main(["delta", "--timeseries", ts, "--out", out]) == 0
    assert os.path.exists(out)
    assert os.path.exists(str(tmp_path / "delta.json"))


def test_indices_cli(capsys):
    assert main(["indices"]) == 0
    captured = capsys.readouterr()
    assert "ndwi" in captured.out and "mndwi" in captured.out


def test_planted_floods_documented():
    assert ("river-reach", "2024-04-15", 1.0) in PLANTED_FLOODS
