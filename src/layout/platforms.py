"""Platform presets: canvas size + the screen regions each platform's own
chrome covers.

All rects are figure fractions ``(left, bottom, width, height)``,
origin bottom-left. The chrome measurements are community-measured
approximations (see docs/PLATFORMS.md for sources and the honesty
notes) — platforms change their chrome, so treat them as guidance and
override per need.

The 9:16 templates (TikTok / Instagram Reels / YouTube Shorts) share one
region set that clears the UNION of the three platforms' chrome, so a
reel laid out for one is safe on all three.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .geometry import Rect, from_top_rect

# Region names every platform template provides.
REGION_NAMES = ("title", "map", "chart", "caption", "footer")

# The earthquake renderer draws an extra "largest events" ranking panel;
# platforms carry a dedicated six-region set for that flavor.
QUAKE_REGION_NAMES = ("title", "map", "ranking", "chart", "caption", "footer")


@dataclass(frozen=True)
class Platform:
    """One distribution target."""
    key: str
    label: str
    width: int
    height: int
    # Screen regions the platform's own UI covers (unsafe for content).
    unsafe: Tuple[Rect, ...]
    # Curated content regions, keyed by REGION_NAMES entries.
    regions: Dict[str, Rect]
    # Optional six-region set for the earthquake "quake" flavor
    # (adds the largest-events ranking panel). Falls back to
    # ``regions`` when None — callers then derive/omit ranking.
    quake_regions: Optional[Dict[str, Rect]] = None
    notes: str = ""

    @property
    def aspect(self) -> str:
        from math import gcd
        g = gcd(self.width, self.height)
        return f"{self.width // g}:{self.height // g}"

    def safe_rect(self, name: str) -> Rect:
        return self.regions[name]


def _tiktok_unsafe() -> Tuple[Rect, ...]:
    return (
        from_top_rect(0.00, 0.00, 1.00, 0.078),   # status bar / For You tabs
        from_top_rect(0.856, 0.417, 0.144, 0.333),  # right action rail
        from_top_rect(0.00, 0.8125, 1.00, 0.1875),  # caption / song / progress
    )


def _reels_unsafe() -> Tuple[Rect, ...]:
    return (
        from_top_rect(0.00, 0.00, 1.00, 0.083),   # status bar + reels header
        from_top_rect(0.875, 0.443, 0.125, 0.297),  # right action rail
        from_top_rect(0.00, 0.781, 1.00, 0.219),   # caption / audio / progress
    )


def _shorts_unsafe() -> Tuple[Rect, ...]:
    return (
        from_top_rect(0.00, 0.00, 1.00, 0.0625),  # status bar
        from_top_rect(0.866, 0.427, 0.134, 0.302),  # right action rail
        from_top_rect(0.00, 0.802, 1.00, 0.198),   # title / channel / progress
    )


# One 9:16 region set, safe against the union of the three 9:16 chromes
# (top 0.083, bottom 0.219, rail x >= 0.856 / y 0.245–0.755).
_NINE_SIXTEEN_REGIONS: Dict[str, Rect] = {
    "title": (0.02, 0.815, 0.82, 0.095),
    "map": (0.03, 0.48, 0.80, 0.32),
    "chart": (0.05, 0.31, 0.76, 0.15),
    "caption": (0.02, 0.245, 0.82, 0.055),
    "footer": (0.00, 0.219, 0.856, 0.02),
}

# Quake flavor for 9:16: the map/chart span is split three ways to fit
# the largest-events ranking panel between them.
_NINE_SIXTEEN_QUAKE_REGIONS: Dict[str, Rect] = {
    "title": (0.02, 0.815, 0.82, 0.095),
    "map": (0.03, 0.57, 0.80, 0.23),
    "ranking": (0.03, 0.43, 0.80, 0.12),
    "chart": (0.05, 0.31, 0.76, 0.10),
    "caption": (0.02, 0.245, 0.82, 0.055),
    "footer": (0.00, 0.219, 0.856, 0.02),
}


def _x_portrait_unsafe() -> Tuple[Rect, ...]:
    return (
        from_top_rect(0.00, 0.00, 1.00, 0.0815),  # status / nav
        from_top_rect(0.00, 0.911, 1.00, 0.0889),  # reply / action bar
    )


PLATFORMS: Dict[str, Platform] = {
    "tiktok": Platform(
        key="tiktok", label="TikTok", width=1080, height=1920,
        unsafe=_tiktok_unsafe(), regions=dict(_NINE_SIXTEEN_REGIONS),
        quake_regions=dict(_NINE_SIXTEEN_QUAKE_REGIONS),
        notes="9:16 vertical. Regions clear the union of TikTok / Reels / "
              "Shorts chrome, so one render is safe on all three.",
    ),
    "instagram-reel": Platform(
        key="instagram-reel", label="Instagram Reels", width=1080, height=1920,
        unsafe=_reels_unsafe(), regions=dict(_NINE_SIXTEEN_REGIONS),
        quake_regions=dict(_NINE_SIXTEEN_QUAKE_REGIONS),
        notes="9:16 vertical. Deepest bottom chrome of the three 9:16 "
              "platforms (~21.9%); the shared region set clears it.",
    ),
    "youtube-shorts": Platform(
        key="youtube-shorts", label="YouTube Shorts", width=1080, height=1920,
        unsafe=_shorts_unsafe(), regions=dict(_NINE_SIXTEEN_REGIONS),
        quake_regions=dict(_NINE_SIXTEEN_QUAKE_REGIONS),
        notes="9:16 vertical. Shares the union-safe 9:16 region set.",
    ),
    "x-portrait": Platform(
        key="x-portrait", label="X (portrait 4:5)", width=1080, height=1350,
        unsafe=_x_portrait_unsafe(),
        regions={
            "title": (0.02, 0.83, 0.96, 0.085),
            "map": (0.03, 0.44, 0.94, 0.36),
            "chart": (0.05, 0.20, 0.90, 0.20),
            "caption": (0.02, 0.115, 0.96, 0.07),
            "footer": (0.00, 0.0891, 1.00, 0.0249),
        },
        quake_regions={
            "title": (0.02, 0.83, 0.96, 0.085),
            "map": (0.03, 0.54, 0.94, 0.27),
            "ranking": (0.03, 0.40, 0.94, 0.12),
            "chart": (0.05, 0.20, 0.90, 0.18),
            "caption": (0.02, 0.115, 0.96, 0.07),
            "footer": (0.00, 0.0891, 1.00, 0.0249),
        },
        notes="4:5 portrait — the tallest X will show un-cropped in feed.",
    ),
    "square": Platform(
        key="square", label="Square 1:1", width=1080, height=1080,
        unsafe=(),
        regions={
            "title": (0.00, 0.88, 1.00, 0.12),
            "map": (0.04, 0.40, 0.92, 0.44),
            "chart": (0.08, 0.10, 0.84, 0.24),
            "caption": (0.00, 0.06, 1.00, 0.08),
            "footer": (0.00, 0.00, 1.00, 0.055),
        },
        quake_regions={
            "title": (0.00, 0.88, 1.00, 0.12),
            "map": (0.04, 0.52, 0.92, 0.32),
            "ranking": (0.04, 0.36, 0.92, 0.14),
            "chart": (0.08, 0.10, 0.84, 0.22),
            "caption": (0.00, 0.06, 1.00, 0.08),
            "footer": (0.00, 0.00, 1.00, 0.055),
        },
        notes="1:1 feed post. No platform chrome inside the canvas; full "
              "bleed is safe.",
    ),
    "widescreen": Platform(
        key="widescreen", label="Widescreen 16:9", width=1920, height=1080,
        unsafe=(),
        regions={
            "title": (0.02, 0.86, 0.96, 0.12),
            "map": (0.02, 0.12, 0.62, 0.70),
            "chart": (0.67, 0.12, 0.31, 0.40),
            "caption": (0.67, 0.56, 0.31, 0.12),
            "footer": (0.00, 0.00, 1.00, 0.05),
        },
        quake_regions={
            "title": (0.02, 0.86, 0.96, 0.12),
            "map": (0.02, 0.12, 0.62, 0.70),
            "ranking": (0.67, 0.66, 0.31, 0.16),
            "chart": (0.67, 0.30, 0.31, 0.32),
            "caption": (0.67, 0.16, 0.31, 0.10),
            "footer": (0.00, 0.00, 1.00, 0.05),
        },
        notes="16:9 landscape (YouTube main). Map left, chart right column, "
              "caption chip floats over the right column's top.",
    ),
    # Baseline: reproduces survey-viz's historical hardcoded layout
    # exactly (1080x1920, title [0,0.90,1,0.10], map
    # [0.04,0.36,0.92,0.52], chart [0.08,0.08,0.84,0.24], caption
    # [0,0.065,1,0.075], footer [0,0,1,0.06]). Escape hatch for
    # pixel-identical legacy renders.
    "legacy": Platform(
        key="legacy", label="Legacy (survey-viz default)", width=1080,
        height=1920, unsafe=(),
        regions={
            "title": (0.00, 0.90, 1.00, 0.10),
            "map": (0.04, 0.36, 0.92, 0.52),
            "chart": (0.08, 0.08, 0.84, 0.24),
            "caption": (0.00, 0.065, 1.00, 0.075),
            "footer": (0.00, 0.00, 1.00, 0.06),
        },
        quake_regions={
            "title": (0.00, 0.90, 1.00, 0.10),
            "map": (0.04, 0.52, 0.92, 0.36),
            "ranking": (0.04, 0.30, 0.92, 0.20),
            "chart": (0.08, 0.08, 0.84, 0.18),
            "caption": (0.00, 0.065, 1.00, 0.075),
            "footer": (0.00, 0.00, 1.00, 0.06),
        },
        notes="Pixel-identical to survey-viz <= 0.17 renders. No safe-zone "
              "correction — the TikTok status bar and bottom caption zone "
              "overlap this layout's title/footer.",
    ),
}


def list_platforms() -> List[str]:
    """Platform keys in a stable, UI-friendly order."""
    return ["tiktok", "instagram-reel", "youtube-shorts", "x-portrait",
            "square", "widescreen", "legacy"]


def get_platform(key: str) -> Platform:
    """Fetch a platform preset by key (fail fast on typos)."""
    try:
        return PLATFORMS[key]
    except KeyError:
        raise ValueError(
            f"Unknown platform {key!r} (expected one of {list_platforms()})"
        ) from None
