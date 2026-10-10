"""Tests for vigor classification, QML, and the synthetic pipeline."""

import numpy as np
import pytest

from vegetation import qml, vigor
from vegetation.synthetic import synthetic_bands, synthetic_index_raster


def test_classify_boundaries():
    arr = np.array([[-0.5, 0.0, 0.199, 0.2, 0.399, 0.4, 0.599, 0.6, 0.799, 0.8, 1.5]])
    out = vigor.classify(arr)
    assert out.tolist()[0] == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_classify_nodata():
    arr = np.array([[0.7, np.nan]])
    out = vigor.classify(arr)
    assert out[0, 0] == 4
    assert out[0, 1] == vigor.NODATA_CLASS


def test_classify_custom_thresholds():
    bands = (
        (0, "low", float("-inf"), 0.5, (0, 0, 0, 255)),
        (1, "high", 0.5, float("inf"), (255, 255, 255, 255)),
    )
    out = vigor.classify(np.array([[0.4, 0.6]]), bands=bands)
    assert out.tolist()[0] == [0, 1]


def test_histogram_fractions_sum():
    classes = vigor.classify(synthetic_index_raster())
    hist = vigor.class_histogram(classes)
    assert abs(sum(h["fraction"] for h in hist) - 1.0) < 1e-9
    labels = {h["label"] for h in hist}
    assert {"Water / shadow", "Bare soil", "Stressed", "Vigorous"} <= labels


def test_qml_has_all_classes():
    text = qml.vigor_qml()
    for class_id, label in [(0, "Water / shadow"), (5, "Very dense")]:
        assert f'value="{class_id}"' in text
        assert label in text
    assert 'type="paletted"' in text


def test_write_qml(tmp_path):
    path = qml.write_vigor_qml(str(tmp_path / "v.qml"))
    assert path.endswith(".qml")
    assert "<qgis" in open(path).read()


def test_write_class_npy_roundtrip(tmp_path):
    classes = vigor.classify(synthetic_index_raster())
    path = vigor.write_class_npy(classes, str(tmp_path / "c.npy"))
    back = np.load(path)
    assert (back == classes).all()
    assert back.dtype == np.uint8


def test_geotiff_needs_rasterio():
    # transform given but rasterio may be absent: either works or clear error
    pytest.importorskip("rasterio")
    classes = vigor.classify(synthetic_index_raster(shape=(8, 8)))
    import tempfile, os

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.tif")
        vigor.write_class_geotiff(classes, p, transform=(1, 0, 0, 0, -1, 0), crs="EPSG:4326")
        assert os.path.exists(p)


def test_synthetic_bands_consistent_with_ndvi():
    bands = synthetic_bands()
    assert set(bands) == {"nir", "red", "green", "blue"}
    for b in bands.values():
        assert np.isfinite(b).all()


def test_cli_demo_end_to_end(tmp_path):
    from vegetation.cli import main

    out = tmp_path / "demo"
    rc = main(["demo", "--out-dir", str(out), "--seed", "7"])
    assert rc == 0
    for name in ("zones.geojson", "timeseries.csv", "anomalies.csv",
                 "season.json", "vigor_classes.npy", "vigor_classes.qml"):
        assert (out / name).exists(), name


def test_cli_indices_lists():
    from vegetation.cli import main

    assert main(["indices"]) == 0
