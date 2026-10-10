"""Trending-video discovery via the official YouTube Data API v3 (keyed).

Endpoint: ``https://www.googleapis.com/youtube/v3/search`` with
``part=snippet&type=video``, ordered by view count over the trailing
window (``publishedAfter``).

Credential rule: the YouTube Data API v3 requires a free API key. When
none is provided — neither the ``api_key`` argument nor the
``YOUTUBE_API_KEY`` environment variable — this module raises
:class:`CredentialsMissingError` **before making any request**: nothing
is fetched, nothing is sent, and no partial data is returned. The error
message says exactly how to get a key.

Quotas: the free tier grants 10,000 units/day; one ``search.list`` call
costs 100 units, so a digest build spends ~100 units. The per-run
in-memory cache in :mod:`._http` prevents re-polling inside one run.

Note: the ``search`` response carries no view/like/comment statistics
(those need a second, costlier ``videos.list`` call), so ``metrics`` is
empty and ``thumbnail_url`` is the snippet thumbnail URL only — it is
never downloaded or rehosted.
"""
from __future__ import annotations

import os
import urllib.parse
from datetime import datetime, timezone
from typing import Optional

from ...models import CredentialsMissingError, SourceUnparsableError
from ..formats import tag_item
from ._http import REQUEST_TIMEOUT, get_json

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

_HEADERS = {
    "User-Agent": "survey-trends/0.2.0 (github.com/crieck2010/survey-trends)",
    "Accept": "application/json",
}

_MISSING_KEY_MESSAGE = (
    "YouTube Data API v3 requires an API key, and none was provided. "
    "Nothing was fetched — no request was made.\n"
    "Get a free key (10,000 quota units/day; one radar fetch costs ~100):\n"
    "  1. Go to the Google Cloud Console: https://console.cloud.google.com\n"
    "  2. Create (or select) a project.\n"
    "  3. APIs & Services > Library > enable 'YouTube Data API v3'.\n"
    "  4. APIs & Services > Credentials > Create Credentials > API key.\n"
    "Then either set the YOUTUBE_API_KEY environment variable or pass\n"
    "api_key= to fetch_youtube_videos."
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_search(payload: object, fetched_at: str) -> list[dict]:
    """Parse a YouTube ``search.list`` payload into TrendItem dicts."""
    if not isinstance(payload, dict):
        raise SourceUnparsableError(
            "youtube: search response was not a JSON object "
            f"({type(payload).__name__})"
        )
    if "error" in payload:
        # Quota/auth errors surface here when a key IS provided.
        error = payload["error"]
        message = error.get("message", "") if isinstance(error, dict) else ""
        from ...models import SourceUnreachableError

        raise SourceUnreachableError(
            f"youtube: API error: {message or error}; no data was fabricated."
        )
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise SourceUnparsableError(
            "youtube: search response has no items array; shape not recognized"
        )
    items: list[dict] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        video_id = raw.get("id") or {}
        video_id = video_id.get("videoId") if isinstance(video_id, dict) else None
        snippet = raw.get("snippet")
        if not video_id or not isinstance(snippet, dict):
            continue
        title = str(snippet.get("title") or "").strip()
        if not title or title == "Private video":
            continue
        channel = snippet.get("channelTitle")
        thumbnails = snippet.get("thumbnails") or {}
        thumb_url = None
        if isinstance(thumbnails, dict):
            for pref in ("medium", "default", "high"):
                thumb = thumbnails.get(pref)
                if isinstance(thumb, dict) and str(thumb.get("url", "")).startswith(
                    "http"
                ):
                    thumb_url = thumb["url"]
                    break
        item = {
            "id": f"youtube:{video_id}",
            "source": "youtube",
            "title": title,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "author": str(channel).strip() if isinstance(channel, str) and channel else None,
            "published_at": snippet.get("publishedAt")
            if isinstance(snippet.get("publishedAt"), str)
            else None,
            # search.list carries no statistics; only keys the source
            # provides are included (none here).
            "metrics": {},
            "tags": [],
            "thumbnail_url": thumb_url,  # URL only — never downloaded
            "fetched_at": fetched_at,
        }
        item["tags"] = tag_item(item)
        items.append(item)
    return items


def fetch_youtube_videos(
    query: str,
    api_key: Optional[str] = None,
    published_after: Optional[str] = None,
    max_results: int = 25,
    order: str = "viewCount",
) -> list[dict]:
    """Search YouTube for ``query`` as TrendItem dicts (official API).

    Raises :class:`CredentialsMissingError` when no API key is available
    (nothing is fetched), :class:`SourceUnreachableError` on HTTP/quota
    failures, and :class:`SourceUnparsableError` on unrecognized payloads.
    """
    key = api_key or os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise CredentialsMissingError(_MISSING_KEY_MESSAGE)
    params = {
        "part": "snippet",
        "type": "video",
        "q": query,
        "maxResults": max(1, min(50, max_results)),
        "order": order,
        "key": key,
    }
    if published_after:
        params["publishedAfter"] = published_after
    url = SEARCH_URL + "?" + urllib.parse.urlencode(params)
    payload = get_json(
        url, headers=_HEADERS, timeout=REQUEST_TIMEOUT, source_name="youtube"
    )
    return _parse_search(payload, _now_iso())
