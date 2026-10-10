"""Duck-typed interop: the canvas dict survey-viz consumes."""

import pytest

from layout import to_viz_canvas, validate_canvas


def test_canvas_shape():
    c = to_viz_canvas("tiktok")
    assert c["platform"] == "tiktok"
    assert c["width"] == 1080 and c["height"] == 1920
    assert set(c["regions"]) == {"title", "map", "chart", "caption", "footer"}
    for rect in c["regions"].values():
        assert isinstance(rect, list) and len(rect) == 4
    assert isinstance(c["unsafe"], list)


def test_canvas_content_filtering():
    c = to_viz_canvas("square", content=("title", "map", "footer"))
    assert set(c["regions"]) == {"title", "map", "footer"}


def test_canvas_is_json_round_trippable():
    import json
    c = to_viz_canvas("youtube-shorts")
    assert validate_canvas(json.loads(json.dumps(c))) == c


def test_validate_rejects_garbage():
    with pytest.raises(ValueError, match="missing"):
        validate_canvas({"width": 1080, "height": 1920})
    with pytest.raises(ValueError, match="positive ints"):
        validate_canvas({"width": -1, "height": 1920, "regions": {"map": [0, 0, 1, 1]}})
    with pytest.raises(ValueError, match="unknown region"):
        validate_canvas({"width": 1080, "height": 1920,
                         "regions": {"hologram": [0, 0, 1, 1]}})
    with pytest.raises(ValueError, match="escapes the canvas"):
        validate_canvas({"width": 1080, "height": 1920,
                         "regions": {"map": [0.9, 0.9, 0.5, 0.5]}})


def test_unknown_platform_fails_fast():
    with pytest.raises(ValueError, match="Unknown platform"):
        to_viz_canvas("vine")
