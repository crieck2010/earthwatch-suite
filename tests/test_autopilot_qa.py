"""Tests for autopilot.qa."""

import numpy as np
import pytest

from autopilot import QAError, QAReport, qa_field, repair


def _field():
    # 4 timesteps, 6h cadence; step 1 all-NaN; step 2 out-of-range spike;
    # a 24h gap before step 3.
    speed = np.ones((4, 3, 3)) * 2.0
    speed[1] = np.nan
    speed[2] = 1e6
    return {
        "grids": {"speed": speed},
        "times": [
            "2026-01-01T00:00:00",
            "2026-01-01T06:00:00",
            "2026-01-01T12:00:00",
            "2026-01-02T12:00:00",
        ],
        "lats": np.linspace(40, 45, 3),
        "lons": np.linspace(-75, -70, 3),
    }


def test_detects_all_nan_out_of_range_and_gap():
    report = qa_field(_field(), expected_cadence_hours=6)
    assert not report.ok
    types = {(i["type"], i.get("var"), i.get("index")) for i in report.issues}
    assert ("all_nan", "speed", 1) in types
    assert ("out_of_range", "speed", 2) in types
    gap = next(i for i in report.issues if i["type"] == "temporal_gap")
    assert gap["index"] == 3
    assert gap["gap_hours"] == 24.0
    assert gap["expected_hours"] == 6.0
    assert report.n_timesteps == 4
    assert report.checked_vars == ["speed"]


def test_clean_field_passes():
    f = _field()
    f["grids"]["speed"] = np.ones((4, 3, 3)) * 2.0
    f["times"] = [
        "2026-01-01T00:00:00",
        "2026-01-01T06:00:00",
        "2026-01-01T12:00:00",
        "2026-01-01T18:00:00",
    ]
    report = qa_field(f, expected_cadence_hours=6)
    assert report.ok
    assert isinstance(report, QAReport)


def test_duck_typed_to_dict():
    class Field:
        def to_dict(self):
            return _field()

    report = qa_field(Field(), expected_cadence_hours=6)
    assert not report.ok


def test_median_cadence_default():
    # No expected_cadence_hours: median diff (6h) used; 24h gap still flagged.
    report = qa_field(_field())
    assert any(i["type"] == "temporal_gap" for i in report.issues)


def test_repair_drop_removes_flagged_steps_and_records_actions():
    f = _field()
    report = qa_field(f, expected_cadence_hours=6)
    new_field, updated = repair(report, f, policy="drop")
    assert new_field["grids"]["speed"].shape[0] == 2  # dropped steps 1 and 2
    assert list(new_field["times"]) == ["2026-01-01T00:00:00", "2026-01-02T12:00:00"]
    dropped = [a for a in updated.actions if a["action"] == "dropped_timestep"]
    assert sorted(a["index"] for a in dropped) == [1, 2]
    assert all(a["reason"] for a in dropped)  # reasons recorded
    assert dropped[0]["time"] == "2026-01-01T06:00:00"


def test_repair_interpolate_fixes_isolated_step():
    speed = np.ones((5, 2, 2)) * 4.0
    speed[2] = np.nan  # isolated bad single step
    f = {
        "grids": {"speed": speed},
        "times": [f"2026-01-01T{h:02d}:00:00" for h in (0, 6, 12, 18, 23)],
        "lats": np.array([40.0, 41.0]),
        "lons": np.array([-75.0, -74.0]),
    }
    report = qa_field(f, expected_cadence_hours=6)
    new_field, updated = repair(report, f, policy="interpolate")
    assert np.allclose(new_field["grids"]["speed"][2], 4.0)
    interp = [a for a in updated.actions if a["action"] == "interpolated_timestep"]
    assert len(interp) == 1 and interp[0]["index"] == 2


def test_repair_interpolate_leaves_runs_alone():
    speed = np.ones((5, 2, 2)) * 4.0
    speed[2] = np.nan
    speed[3] = np.nan  # a run of two — not isolated
    f = {
        "grids": {"speed": speed},
        "times": [f"2026-01-01T{h:02d}:00:00" for h in (0, 6, 12, 18, 23)],
        "lats": np.array([40.0, 41.0]),
        "lons": np.array([-75.0, -74.0]),
    }
    report = qa_field(f, expected_cadence_hours=6)
    new_field, updated = repair(report, f, policy="interpolate")
    assert np.all(np.isnan(new_field["grids"]["speed"][2]))
    assert updated.actions == []  # nothing interpolated, honestly reported


def test_repair_fail_raises_and_passes_clean():
    f = _field()
    report = qa_field(f, expected_cadence_hours=6)
    with pytest.raises(QAError):
        repair(report, f, policy="fail")

    clean = _field()
    clean["grids"]["speed"] = np.ones((4, 3, 3)) * 2.0
    clean["times"] = [f"2026-01-01T{h:02d}:00:00" for h in (0, 6, 12, 18)]
    ok_report = qa_field(clean, expected_cadence_hours=6)
    new_field, updated = repair(ok_report, clean, policy="fail")
    assert new_field["grids"]["speed"].shape[0] == 4


def test_repair_unknown_policy_raises():
    f = _field()
    report = qa_field(f)
    with pytest.raises(ValueError):
        repair(report, f, policy="yeet")


def test_outlier_fraction_threshold_respected():
    # 1 bad cell out of 100 = 1% ; default threshold is 0.01 (strict >).
    speed = np.ones((2, 10, 10)) * 2.0
    speed[1, 0, 0] = 1e6
    f = {"grids": {"speed": speed}, "times": ["2026-01-01T00:00:00", "2026-01-01T06:00:00"]}
    r_default = qa_field(f, expected_cadence_hours=6)
    assert not any(i["type"] == "out_of_range" for i in r_default.issues)
    r_strict = qa_field(f, expected_cadence_hours=6, outlier_fraction_threshold=0.001)
    assert any(i["type"] == "out_of_range" for i in r_strict.issues)
