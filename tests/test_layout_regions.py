"""Region safety: nothing lands under platform chrome.

The core guarantee of this engine, tested per platform: every curated
region stays inside the canvas, clears every unsafe rect, and doesn't
collide with another region.
"""

import pytest

from layout import get_platform, layout_for, list_platforms
from layout.geometry import contains, overlaps, within_canvas
from layout.platforms import QUAKE_REGION_NAMES
from layout.regions import validate_layout


@pytest.mark.parametrize("key", [
    p for p in
    ["tiktok", "instagram-reel", "youtube-shorts", "x-portrait",
     "square", "widescreen", "legacy"]
])
def test_curated_regions_validate_clean(key):
    plat = get_platform(key)
    assert validate_layout(plat, dict(plat.regions)) == []


@pytest.mark.parametrize("key", ["tiktok", "instagram-reel", "youtube-shorts",
                                 "x-portrait", "square", "widescreen",
                                 "legacy"])
def test_quake_regions_validate_clean(key):
    plat = get_platform(key)
    assert validate_layout(plat, dict(plat.quake_regions)) == []


def test_quake_flavor_has_ranking():
    regions = layout_for("tiktok", flavor="quake")
    assert set(regions) == set(QUAKE_REGION_NAMES)
    assert "ranking" in regions


def test_bad_flavor_fails_fast():
    with pytest.raises(ValueError, match="Unknown flavor"):
        layout_for("tiktok", flavor="imax")


def test_layout_for_validates_before_returning():
    regions = layout_for("tiktok")
    plat = get_platform("tiktok")
    assert validate_layout(plat, regions) == []


def test_content_filtering_is_dataset_aware():
    # categorical variables draw no time-series chart
    regions = layout_for("tiktok", content=("title", "map", "caption", "footer"))
    assert set(regions) == {"title", "map", "caption", "footer"}


def test_unknown_content_fails_fast():
    with pytest.raises(ValueError, match="Unknown content"):
        layout_for("tiktok", content=("title", "hologram"))


def test_tiktok_title_clears_status_bar():
    plat = get_platform("tiktok")
    title = plat.regions["title"]
    top_unsafe = plat.unsafe[0]  # status bar / For You tabs
    assert not overlaps(title, top_unsafe)
    # title sits fully below the unsafe band
    _, tb, _, th = top_unsafe
    _, yb, _, yh = title
    assert yb + yh <= tb + 1e-9


def test_tiktok_footer_and_caption_clear_bottom_chrome():
    plat = get_platform("tiktok")
    bottom_unsafe = plat.unsafe[2]
    _, bb, _, bh = bottom_unsafe
    for name in ("footer", "caption", "chart"):
        _, yb, _, _ = plat.regions[name]
        assert yb >= bb + bh - 1e-9, name
        assert not overlaps(plat.regions[name], bottom_unsafe)


def test_tiktok_map_clears_action_rail():
    plat = get_platform("tiktok")
    rail = plat.unsafe[1]
    assert not overlaps(plat.regions["map"], rail)


def test_legacy_reproduces_survey_viz_hardcoded_layout():
    # survey-viz <= 0.17 render.py hardcoded rects (the escape hatch).
    regions = layout_for("legacy")
    assert regions["title"] == (0.00, 0.90, 1.00, 0.10)
    assert regions["map"] == (0.04, 0.36, 0.92, 0.52)
    assert regions["chart"] == (0.08, 0.08, 0.84, 0.24)
    assert regions["caption"] == (0.00, 0.065, 1.00, 0.075)
    assert regions["footer"] == (0.00, 0.00, 1.00, 0.06)


def test_validator_catches_chrome_overlap():
    plat = get_platform("tiktok")
    bad = dict(plat.regions)
    bad["title"] = (0.0, 0.95, 1.0, 0.05)  # under the status bar
    problems = validate_layout(plat, bad)
    assert any("unsafe" in p for p in problems)


def test_validator_catches_region_collision():
    plat = get_platform("square")
    bad = dict(plat.regions)
    bad["chart"] = plat.regions["map"]  # deliberate collision
    problems = validate_layout(plat, bad)
    assert any("overlap" in p for p in problems)


def test_edge_touching_is_not_overlap():
    # footer top edge exactly meets the bottom-chrome edge on 9:16
    plat = get_platform("tiktok")
    assert validate_layout(plat, dict(plat.regions)) == []


def test_caption_may_float_over_regions_but_not_chrome():
    plat = get_platform("square")
    regions = dict(plat.regions)
    # caption deliberately overlapping the chart: allowed (floating chip)
    assert validate_layout(plat, regions) == []
    # ... but never over an unsafe rect
    tiktok = get_platform("tiktok")
    bad = dict(tiktok.regions)
    bad["caption"] = (0.0, 0.0, 1.0, 0.10)  # into the bottom chrome
    problems = validate_layout(tiktok, bad)
    assert any("unsafe" in p for p in problems)
