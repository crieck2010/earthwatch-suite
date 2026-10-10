"""render_caption tests: captions must contain the computed numbers."""

import re

import numpy as np
import pytest

from narrate import render_caption, story_facts


def _facts(region_name="the North Sound"):
    nt, ny, nx = 2, 3, 4
    u10 = np.full((nt, ny, nx), 5.0)
    v10 = np.zeros((nt, ny, nx))
    u10[1, 2, 3] = 10.0
    return story_facts(
        {
            "grids": {"u10": u10, "v10": v10, "t2m": np.full((nt, ny, nx), 10.0)},
            "air_temperature": np.full((nt, ny, nx), 50.0),
            "temperature_unit": "°F",
            "times": ["2026-09-28T00:00:00Z", "2026-09-29T00:00:00Z"],
            "lats": np.linspace(40.0, 42.0, ny),
            "lons": np.linspace(-10.0, -7.0, nx),
        },
        region_name=region_name,
    )


def test_instagram_contains_peak_knots():
    caption = render_caption(_facts())
    peak_kt = 10.0 * 1.9438444924
    assert f"{peak_kt:.0f}" in caption  # "19"
    assert re.search(r"\b19 knots\b", caption)


def test_instagram_contains_peak_location_and_time():
    caption = render_caption(_facts())
    assert "42.00°N" in caption
    assert "7.00°W" in caption
    assert "2026-09-29T00:00:00Z" in caption


def test_instagram_contains_mean_and_direction():
    facts = _facts()
    caption = render_caption(facts)
    mean_kt = facts["mean_speed_knots"]
    assert f"{mean_kt:.0f}" in caption
    assert "easterly" in caption


def test_instagram_uses_caller_region_name_verbatim():
    caption = render_caption(_facts(region_name="the Tacoma Narrows"))
    assert "the Tacoma Narrows" in caption


def test_instagram_falls_back_to_generic_area_without_region():
    caption = render_caption(_facts(region_name=""))
    assert "the Tacoma Narrows" not in caption
    assert "the area" in caption


def test_instagram_temperature_appears_when_present():
    caption = render_caption(_facts())
    assert "50.0" in caption and "°F" in caption


def test_instagram_sentence_count():
    caption = render_caption(_facts())
    sentences = [s for s in caption.split(". ") if s.strip()]
    assert 2 <= len(sentences) <= 4
    # No sentence is a fragment: each ends with terminal punctuation.
    assert caption.rstrip().endswith(".")


def test_instagram_has_no_hype_adjectives():
    caption = render_caption(_facts()).lower()
    for hype in ("whopping", "massive", "incredible", "extreme", "wild"):
        assert hype not in caption


def test_email_style_is_single_paragraph():
    caption = render_caption(_facts(), style="email")
    assert "\n" not in caption
    peak_kt = 10.0 * 1.9438444924
    assert f"{peak_kt:.0f}" in caption
    assert "easterly" in caption
    assert "the North Sound" in caption
    assert "42.00°N" in caption


def test_unknown_style_raises():
    with pytest.raises(ValueError, match="unknown caption style"):
        render_caption(_facts(), style="tiktok")


def test_no_temperature_omits_temperature_sentence():
    facts = _facts()
    facts["temperature"] = None
    caption = render_caption(facts)
    assert "temperatures ranged" not in caption.lower()
    assert "°F" not in caption


def test_no_dominant_direction_renders_honestly():
    facts = _facts()
    facts["direction"] = {
        "bearing_deg": None,
        "label": None,
        "note": facts["direction"]["note"],
    }
    caption = render_caption(facts)
    assert "No dominant direction" in caption


def test_peak_location_uses_hemisphere_suffixes():
    """Regression: negative lon must render as °W, not °E."""
    facts = _facts()
    for style in ("instagram", "email"):
        caption = render_caption(facts, style=style)
        assert "°W" in caption
        assert "°E" not in caption
