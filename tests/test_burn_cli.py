"""End-to-end CLI tests in --synthetic mode (no network)."""

import json
import os

import numpy as np
import pytest

from burn.cli import main


def test_demo_end_to_end(tmp_path):
    out = str(tmp_path / "demo")
    assert main(["demo", "--out-dir", out]) == 0
    for name in ("dnbr.npy", "severity.npy", "severity.qml", "dnbr.qml",
                 "disturbance_polygons.geojson"):
        assert os.path.exists(os.path.join(out, name)), name
    sev = np.load(os.path.join(out, "severity.npy"))
    assert set(np.unique(sev)) <= {0, 1, 2, 3, 4, 5, 255}
    with open(os.path.join(out, "disturbance_polygons.geojson"), encoding="utf-8") as fh:
        fc = json.load(fh)
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) >= 1
    total_ha = sum(f["properties"]["area_ha"] for f in fc["features"])
    assert total_ha > 0


def test_dnbr_then_severity_pipeline(tmp_path):
    dpath = str(tmp_path / "d.npy")
    spath = str(tmp_path / "s.npy")
    assert main(["dnbr", "--synthetic", "--out", dpath]) == 0
    assert main(["severity", "--dnbr", dpath, "--out", spath,
                 "--qml", str(tmp_path / "s.qml")]) == 0
    sev = np.load(spath)
    assert 4 in np.unique(sev)  # planted high-severity core present


def test_dnbr_relative_flag(tmp_path):
    dpath = str(tmp_path / "rd.npy")
    assert main(["dnbr", "--synthetic", "--relative", "--out", dpath]) == 0
    assert os.path.exists(dpath)


def test_timeseries_synthetic(tmp_path):
    csv_path = str(tmp_path / "ts.csv")
    assert main(["timeseries", "--synthetic", "--out", csv_path]) == 0
    assert main(["timeseries", "--synthetic", "--out", csv_path]) == 0  # idempotent
    with open(csv_path, encoding="utf-8") as fh:
        lines = fh.read().strip().splitlines()
    assert len(lines) == 4  # header + 3 zones
    assert lines[0].startswith("zone_id,zone_name,pass_id")


def test_disturbance_synthetic(tmp_path):
    jpath = str(tmp_path / "events.json")
    assert main(["disturbance", "--synthetic", "--out", jpath]) == 0
    with open(jpath, encoding="utf-8") as fh:
        events = json.load(fh)
    assert len(events) == 1
    assert events[0]["pixels"] == 144


def test_polygons_synthetic(tmp_path):
    gpath = str(tmp_path / "p.geojson")
    assert main(["polygons", "--synthetic", "--out", gpath]) == 0
    with open(gpath, encoding="utf-8") as fh:
        fc = json.load(fh)
    assert len(fc["features"]) >= 1


def test_indices_lists():
    assert main(["indices"]) == 0


def test_dnbr_requires_rasters_without_synthetic():
    with pytest.raises(SystemExit):
        main(["dnbr", "--out", "x.npy"])
