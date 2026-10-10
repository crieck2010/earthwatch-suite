"""Tests for the synthetic demo world and the CLI end-to-end."""

import csv
import os

import numpy as np
import pytest

from thermal import synthetic as syn
from thermal.cli import main
from thermal.qml import lst_qml


def test_synthetic_raster_uhi_gradient():
    arr, transform = syn.synthetic_lst_raster()
    rows, cols = arr.shape
    west = arr[:, : cols // 2].mean()
    east = arr[:, cols // 2 :].mean()
    assert east - west == pytest.approx(syn.PLANTED_UHI_C, abs=1.0)
    assert transform[0] == 30.0  # honest 30 m pixels


def test_lst_qml_has_diverging_stops():
    qml = lst_qml()
    assert "singlebandpseudocolor" in qml
    assert "40" in qml  # hot stop present


def test_demo_report_artifacts(tmp_path):
    report = syn.demo_report(str(tmp_path))
    for key in ("timeseries", "uhi", "anomalies", "heatwaves"):
        assert os.path.exists(report[key])
    assert report["n_heatwaves"] == "1"
    assert int(report["n_hot"]) >= 1


def test_cli_demo_end_to_end(tmp_path, capsys):
    out = str(tmp_path / "demo")
    rc = main(["demo", "--out-dir", out])
    assert rc == 0
    assert os.path.exists(os.path.join(out, "lst_timeseries.csv"))
    assert os.path.exists(os.path.join(out, "uhi.csv"))
    assert os.path.exists(os.path.join(out, "heatwaves.csv"))
    captured = capsys.readouterr()
    assert "survey-thermal demo" in captured.out


def test_cli_timeseries_uhi_heatwave_chain(tmp_path, capsys):
    ts_path = str(tmp_path / "ts.csv")
    uhi_path = str(tmp_path / "uhi.csv")
    hw_path = str(tmp_path / "hw.csv")
    assert main(["timeseries", "--synthetic", "--out", ts_path]) == 0
    assert main(["uhi", "--timeseries", ts_path, "--reference", "meadow",
                 "--out", uhi_path]) == 0
    assert main(["heatwave", "--timeseries", ts_path, "--out", hw_path]) == 0
    with open(hw_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["zone_id"] == "downtown"


def test_cli_anomalies_chain(tmp_path):
    ts_path = str(tmp_path / "ts.csv")
    an_path = str(tmp_path / "an.csv")
    assert main(["timeseries", "--synthetic", "--out", ts_path]) == 0
    assert main(["anomalies", "--timeseries", ts_path, "--out", an_path]) == 0
    with open(an_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    hot = [r for r in rows if r["hot"] == "True"]
    assert len(hot) >= 1


def test_cli_lst_synthetic(tmp_path):
    out = str(tmp_path / "lst.tif")
    rc = main(["lst", "--synthetic", "--source", "st", "--out", out])
    assert rc == 0
    # rasterio not installed here -> numpy fallback
    assert os.path.exists(out + ".npy")
    arr = np.load(out + ".npy")
    assert np.isfinite(arr).all()
    assert 10.0 < arr.mean() < 45.0


def test_cli_uhi_auto_reference(tmp_path, capsys):
    ts_path = str(tmp_path / "ts.csv")
    uhi_path = str(tmp_path / "uhi.csv")
    assert main(["timeseries", "--synthetic", "--out", ts_path]) == 0
    assert main(["uhi", "--timeseries", ts_path, "--out", uhi_path]) == 0
    captured = capsys.readouterr()
    assert "auto-selected reference zone: meadow" in captured.out
