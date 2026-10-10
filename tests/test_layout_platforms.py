"""Platform preset sanity: aspects, canvas bounds, key handling."""

import pytest

from layout import get_platform, list_platforms
from layout.geometry import within_canvas
from layout.platforms import PLATFORMS, QUAKE_REGION_NAMES, REGION_NAMES


def test_all_platforms_advertised():
    assert set(list_platforms()) == set(PLATFORMS)


def test_aspects():
    assert get_platform("tiktok").aspect == "9:16"
    assert get_platform("instagram-reel").aspect == "9:16"
    assert get_platform("youtube-shorts").aspect == "9:16"
    assert get_platform("x-portrait").aspect == "4:5"
    assert get_platform("square").aspect == "1:1"
    assert get_platform("widescreen").aspect == "16:9"


def test_canvas_sizes():
    assert (get_platform("tiktok").width, get_platform("tiktok").height) == (1080, 1920)
    assert (get_platform("square").width, get_platform("square").height) == (1080, 1080)
    assert (get_platform("widescreen").width, get_platform("widescreen").height) == (1920, 1080)


def test_every_platform_has_every_region():
    for key in list_platforms():
        plat = get_platform(key)
        assert set(plat.regions) == set(REGION_NAMES), key
        assert plat.quake_regions is not None, key
        assert set(plat.quake_regions) == set(QUAKE_REGION_NAMES), key


def test_unsafe_rects_inside_canvas():
    for key in list_platforms():
        for u in get_platform(key).unsafe:
            assert within_canvas(u), (key, u)


def test_unknown_platform_fails_fast():
    with pytest.raises(ValueError, match="Unknown platform"):
        get_platform("myspace")


def test_nine_sixteen_share_union_safe_regions():
    r = get_platform("tiktok").regions
    assert get_platform("instagram-reel").regions == r
    assert get_platform("youtube-shorts").regions == r
    # ... while carrying their own chrome measurements
    assert get_platform("tiktok").unsafe != get_platform("instagram-reel").unsafe
