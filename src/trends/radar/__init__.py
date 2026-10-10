"""Mapping-visualization trend radar (v0.2.0).

Reads the official/public surfaces of Reddit, TikTok Creative Center, and
YouTube Data API v3 for map/data-viz content trends, tags items with
deterministic format heuristics (:mod:`trends.radar.formats`), aggregates
them into ranked per-format signals (:mod:`trends.radar.signals`), and
renders a Markdown digest (:mod:`trends.radar.digest`) plus machine-
readable priors for survey-autopilot (:mod:`trends.radar.priors`).

Legal/ethical contract (baked in, documented in README + docs/SOURCES.md):

* Official/public routes only; metadata only — thumbnails and videos are
  never downloaded or rehosted, only linked.
* Rate limits respected; responses cached in-memory per run.
* Unreachable sources produce an honest "unavailable" note and a nonzero
  exit — never invented items.
* Every digest links back to the originals and carries the
  inspiration-not-replication note.
"""
from .digest import INSPIRATION_NOTE, render_radar_digest
from .formats import FALLBACK_TAG, TAG_RULES, TAG_VOCABULARY, tag_item
from .priors import priors_from_signals
from .signals import build_signals
from .sources import fetch_reddit, fetch_tiktok_videos, fetch_youtube_videos

__all__ = [
    "fetch_reddit",
    "fetch_tiktok_videos",
    "fetch_youtube_videos",
    "tag_item",
    "TAG_RULES",
    "TAG_VOCABULARY",
    "FALLBACK_TAG",
    "build_signals",
    "render_radar_digest",
    "INSPIRATION_NOTE",
    "priors_from_signals",
]
