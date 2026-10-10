"""Tests for autopilot.density."""

from autopilot import strand_count_for_bbox


def test_north_america_calibration():
    # (-165,15,-55,72): area = 110 * 57 = 6270 deg^2; 6270 * 0.5 = 3135.
    r = strand_count_for_bbox((-165, 15, -55, 72))
    assert r["count"] == 3135
    assert r["area_deg2"] == 6270.0
    assert r["density_per_deg2"] == 3135 / 6270.0
    assert r["method"] == "area"
    assert r["clamped"] is False


def test_tiny_bbox_clamps_to_min():
    r = strand_count_for_bbox((0, 0, 1, 1))
    assert r["count"] == 500
    assert r["clamped"] is True


def test_huge_bbox_clamps_to_max():
    r = strand_count_for_bbox((-180, -90, 180, 90))
    assert r["count"] == 6000
    assert r["area_deg2"] == 64800.0
    assert r["clamped"] is True


def test_custom_density_and_bounds():
    r = strand_count_for_bbox((-165, 15, -55, 72), base_density=1.0)
    assert r["count"] == 6000  # raw 6270 -> clamped to max
    assert r["clamped"] is True


def test_deterministic():
    a = strand_count_for_bbox((-120, 25, -70, 50))
    b = strand_count_for_bbox((-120, 25, -70, 50))
    assert a == b
