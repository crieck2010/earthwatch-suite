"""Validator unit tests: every rule, valid and invalid."""

import pytest

from sites.scaffold import scaffold
from sites.validate import ValidationError, validate_config, validate_file


def good():
    return scaffold("t-site", "T Site", template="quarry")


def test_valid_quarry_config():
    assert validate_config(good()) == []


def test_bad_site_id():
    cfg = good()
    cfg["site_id"] = "Bad ID!"
    errors = validate_config(cfg)
    assert any("site_id" in e for e in errors)


def test_unclosed_ring_rejected():
    cfg = good()
    cfg["geometry"] = {"type": "Polygon",
                       "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}
    assert any("closed" in e for e in validate_config(cfg))


def test_inverted_bbox_rejected():
    cfg = good()
    cfg["geometry"] = {"bbox": [1.0, 1.0, 0.0, 0.0]}
    assert any("bbox" in e for e in validate_config(cfg))


def test_unknown_index_rejected():
    cfg = good()
    cfg["indices"] = ["ndvi", "dem-diff"]
    cfg["thresholds"]["dem-diff"] = {"warning": {"absolute_delta": 1.0}}
    errors = validate_config(cfg)
    assert any("dem-diff" in e for e in errors)


def test_missing_threshold_for_index():
    cfg = good()
    del cfg["thresholds"]["ndvi"]
    assert any("thresholds" in e for e in validate_config(cfg))


def test_warning_needs_a_condition():
    cfg = good()
    cfg["thresholds"]["ndvi"]["warning"] = {}
    assert any("warning" in e for e in validate_config(cfg))


def test_bad_direction_rejected():
    cfg = good()
    cfg["thresholds"]["ndvi"]["direction"] = "sideways"
    assert any("direction" in e for e in validate_config(cfg))


def test_fixed_baseline_needs_reference_pass():
    cfg = good()
    cfg["baseline"] = {"strategy": "fixed", "window_passes": 5, "min_passes": 2}
    assert any("reference_pass" in e for e in validate_config(cfg))


def test_cloud_cover_bounds():
    cfg = good()
    cfg["max_cloud_cover"] = 150.0
    assert any("max_cloud_cover" in e for e in validate_config(cfg))


def test_duplicate_zone_ids_rejected():
    cfg = good()
    cfg["zones"].append(dict(cfg["zones"][0]))
    assert any("duplicate" in e for e in validate_config(cfg))


def test_bad_zone_kind_rejected():
    cfg = good()
    cfg["zones"][0]["kind"] = "volcano"
    assert any("kind" in e for e in validate_config(cfg))


def test_dem3d_stable_zone_must_exist():
    cfg = good()
    cfg["engines"]["dem3d"]["stable_zone"] = "no-such-zone"
    assert any("stable_zone" in e for e in validate_config(cfg))


def test_thermal_reference_zone_must_exist():
    cfg = scaffold("t-farm", "T Farm", template="farm")
    cfg["engines"]["thermal"]["reference_zone"] = "no-such-zone"
    assert any("reference_zone" in e for e in validate_config(cfg))


def test_transect_must_be_linestring():
    cfg = scaffold("t-coast", "T Coast", template="coast")
    cfg["transects"][0]["geometry"] = {"type": "Polygon", "coordinates": []}
    assert any("LineString" in e for e in validate_config(cfg))


def test_duplicate_transect_ids_rejected():
    cfg = scaffold("t-coast", "T Coast", template="coast")
    cfg["transects"].append(dict(cfg["transects"][0]))
    assert any("duplicate" in e for e in validate_config(cfg))


def test_alerts_severity_min_enum():
    cfg = good()
    cfg["alerts"]["severity_min"] = "panic"
    assert any("severity_min" in e for e in validate_config(cfg))


def test_license_plan_enum():
    cfg = good()
    cfg["license"]["plan"] = "enterprise"
    assert any("license.plan" in e or "plan" in e for e in validate_config(cfg))


def test_non_dict_config_rejected():
    assert validate_config(["not", "a", "dict"])


def test_validate_file_missing(tmp_path):
    errors = validate_file(str(tmp_path / "nope.yaml"))
    assert errors  # file-not-found surfaces as a parse error


def test_validate_file_bad_yaml(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("site_id: [unclosed\n")
    errors = validate_file(str(p))
    assert errors
