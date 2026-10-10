"""Tests for labels_for_bbox: selection, ranking, determinism, edges."""

from gazetteer import labels_for_bbox

LAKE_ONTARIO = (-80.5, -76.0, 43.0, 44.6)


def test_output_schema_exact():
    labels = labels_for_bbox(*LAKE_ONTARIO, max_labels=3)
    assert len(labels) == 3
    for label in labels:
        assert set(label.keys()) == {"x", "y", "text", "priority"}
        assert isinstance(label["text"], str) and label["text"]
        assert isinstance(label["priority"], int)


def test_place_labels_contract():
    # x=lon, y=lat in data coordinates; priority higher wins ties.
    labels = labels_for_bbox(*LAKE_ONTARIO, max_labels=8)
    for label in labels:
        assert -80.5 <= label["x"] <= -76.0
        assert 43.0 <= label["y"] <= 44.6


def test_ranking_order():
    labels = labels_for_bbox(*LAKE_ONTARIO, max_labels=8)
    names = [l["text"] for l in labels]
    assert names[0] == "Toronto"  # biggest city in the bbox
    priorities = [l["priority"] for l in labels]
    assert priorities == sorted(priorities, reverse=True)
    # reference-reel cities are all present in the top set
    for expected in ("Toronto", "Rochester", "Kingston", "Oshawa"):
        assert expected in names


def test_max_labels_respected():
    assert len(labels_for_bbox(*LAKE_ONTARIO, max_labels=2)) == 2
    assert labels_for_bbox(*LAKE_ONTARIO, max_labels=0) == []


def test_determinism():
    a = labels_for_bbox(*LAKE_ONTARIO, max_labels=8)
    b = labels_for_bbox(*LAKE_ONTARIO, max_labels=8)
    assert a == b


def test_min_population_filter():
    all_labels = labels_for_bbox(*LAKE_ONTARIO, max_labels=50)
    big = labels_for_bbox(*LAKE_ONTARIO, max_labels=50, min_population=1_000_000)
    assert 0 < len(big) < len(all_labels)
    assert {l["text"] for l in big} <= {l["text"] for l in all_labels}


def test_kinds_filter():
    cities = labels_for_bbox(*LAKE_ONTARIO, max_labels=50, kinds=("city",))
    towns = labels_for_bbox(*LAKE_ONTARIO, max_labels=50, kinds=("town",))
    assert cities and towns
    assert {l["text"] for l in cities}.isdisjoint({l["text"] for l in towns})


def test_empty_bbox_returns_empty():
    # Mid-ocean, no populated places.
    assert labels_for_bbox(10.0, 10.5, 60.0, 60.1) == []


def test_antimeridian_bbox():
    # Fiji straddles 180/-180; lon0 > lon1 means "wrap".
    labels = labels_for_bbox(177.0, -178.0, -19.0, -16.0, max_labels=5)
    names = [l["text"] for l in labels]
    assert "Suva" in names
    assert labels[0]["text"] == "Suva"  # capital ranks first


def test_antimeridian_no_wrap_confusion():
    # Same region expressed WITHOUT wrap must not leak across the globe.
    wrapped = labels_for_bbox(177.0, -178.0, -19.0, -16.0, max_labels=50)
    normal = labels_for_bbox(-178.0, 177.0, -19.0, -16.0, max_labels=50)
    assert len(wrapped) < len(normal)  # normal spans nearly the whole planet


def test_swapped_latitudes_normalized():
    a = labels_for_bbox(-80.5, -76.0, 43.0, 44.6, max_labels=5)
    b = labels_for_bbox(-80.5, -76.0, 44.6, 43.0, max_labels=5)
    assert a == b


def test_global_bbox_top_is_megacity():
    labels = labels_for_bbox(-180.0, 180.0, -90.0, 90.0, max_labels=1)
    assert len(labels) == 1
    # The single highest-priority place on Earth should be enormous.
    assert labels[0]["priority"] >= 60
