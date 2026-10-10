"""CLI end-to-end tests (synthetic mode: no network, no data files)."""

import csv
import json
import os

import pytest

from coast.cli import main


def test_cli_demo_end_to_end(tmp_path):
    out = str(tmp_path / "demo")
    assert main(["demo", "--out-dir", out]) == 0
    for name in ("positions.csv", "rates.csv", "storm.csv", "transects.geojson"):
        assert os.path.exists(os.path.join(out, name)), name
    # rates recover the planted trends
    with open(os.path.join(out, "rates.csv")) as fh:
        rows = {r["transect_id"]: r for r in csv.DictReader(fh)}
    assert float(rows["shore-a"]["rate_m_per_yr"]) == pytest.approx(-2.0, abs=0.6)
    assert float(rows["shore-b"]["rate_m_per_yr"]) == pytest.approx(3.0, abs=0.6)
    # storm bite recovered on shore-a
    with open(os.path.join(out, "storm.csv")) as fh:
        srows = {r["transect_id"]: r for r in csv.DictReader(fh)}
    assert float(srows["shore-a"]["displacement_m"]) == pytest.approx(-8.0, abs=4.0)
    # shorelines are valid GeoJSON
    shore_dir = os.path.join(out, "shorelines")
    files = sorted(f for f in os.listdir(shore_dir) if f.endswith(".geojson"))
    assert len(files) == 7
    obj = json.load(open(os.path.join(shore_dir, files[0])))
    assert obj["type"] == "FeatureCollection" and obj["features"]


def test_cli_shoreline_then_transects_then_rates(tmp_path):
    shore_dir = str(tmp_path / "shorelines")
    assert main(["shoreline", "--synthetic", "--out-dir", shore_dir]) == 0
    tpath = str(tmp_path / "transects.geojson")
    from coast import synthetic as syn
    from coast.geojson import write_geojson

    write_geojson(tpath, [
        {
            "type": "Feature",
            "geometry": f["geometry"],
            "properties": f["properties"],
        }
        for f in syn.transects_geojson_dict()["features"]
    ])
    pos = str(tmp_path / "positions.csv")
    assert main([
        "transects", "--shorelines", shore_dir, "--transects", tpath,
        "--out", pos, "--site-id", "demo-cove",
    ]) == 0
    with open(pos) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 14  # 2 transects x 7 passes
    rates = str(tmp_path / "rates.csv")
    assert main(["rates", "--positions", pos, "--out", rates]) == 0
    storm = str(tmp_path / "storm.csv")
    assert main(["storm", "--positions", pos, "--event", "2021-06-15", "--out", storm]) == 0
    # re-running transects is a no-op (dedupe)
    assert main([
        "transects", "--shorelines", shore_dir, "--transects", tpath,
        "--out", pos, "--site-id", "demo-cove",
    ]) == 0
    with open(pos) as fh:
        assert len(list(csv.DictReader(fh))) == 14


def test_cli_indices_lists_ndwi_mndwi(capsys):
    assert main(["indices"]) == 0
    out = capsys.readouterr().out
    assert "ndwi" in out and "mndwi" in out
