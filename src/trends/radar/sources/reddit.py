"""Fetch map-viz posts from Reddit's public JSON API (keyless).

Endpoint shape: ``https://www.reddit.com/r/{sub}/{sort}.json?limit=N&t=week``.

Rate-limit politeness: a 2-second sleep between subreddit requests, an
honest ``User-Agent`` identifying this project, and the per-run in-memory
cache in :mod:`._http` so no URL is fetched twice per digest build.

Honest-access contract: HTTP errors (including 429) raise
:class:`SourceUnreachableError`; non-JSON or unrecognized payloads raise
:class:`SourceUnparsableError`. This module never synthesizes posts.

Probe note (2026-10-01, plain server): ``www.reddit.com`` answered the
public JSON endpoint with HTTP 403 (anti-bot block page); ``old.reddit.com``
302-redirected into the same block. Reddit now requires OAuth for
server-side API access, so from most plain servers this source reports
"unavailable" via ``SourceUnreachableError`` — honestly, with the 403
named. It will work wherever Reddit still serves the public JSON
endpoint to the caller's network.
"""
from __future__ import annotations

import time as time_module
from datetime import datetime, timezone
from typing import Optional, Sequence

from ...models import SourceUnparsableError
from ..formats import tag_item
from ._http import REQUEST_TIMEOUT, get_json

USER_AGENT = "survey-trends/0.2.0 (github.com/crieck2010/survey-trends)"
REDDIT_URL = "https://www.reddit.com/r/{sub}/{sort}.json?limit={limit}&t={time}"
#: Pause between subreddit requests so one digest build stays polite.
REQUEST_GAP_SECONDS = 2


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_to_iso(value) -> Optional[str]:
    try:
        stamp = float(value)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(stamp, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _parse_listing(payload: object, sub: str, fetched_at: str) -> list[dict]:
    """Parse one Reddit ``.json`` listing into TrendItem dicts.

    Strict: the envelope must be ``{"data": {"children": [...]}}``; a
    child is usable only if it carries a title. Skips unusable children
    rather than guessing at them.
    """
    if not isinstance(payload, dict):
        raise SourceUnparsableError(
            f"reddit r/{sub}: listing was not a JSON object "
            f"({type(payload).__name__})"
        )
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("children"), list):
        raise SourceUnparsableError(
            f"reddit r/{sub}: listing has no data.children array; "
            "shape not recognized"
        )
    items: list[dict] = []
    for child in data["children"]:
        if not isinstance(child, dict):
            continue
        post = child.get("data")
        if not isinstance(post, dict):
            continue
        title = str(post.get("title") or "").strip()
        if not title:
            continue  # a listing row without a title is not usable
        permalink = str(post.get("permalink") or "").strip()
        url = f"https://www.reddit.com{permalink}" if permalink else ""
        author = post.get("author")
        author = str(author).strip() if isinstance(author, str) and author else None
        metrics = {}
        score = post.get("score")
        if isinstance(score, int) and not isinstance(score, bool):
            metrics["score"] = score
        comments = post.get("num_comments")
        if isinstance(comments, int) and not isinstance(comments, bool):
            metrics["comments"] = comments
        thumbnail = post.get("thumbnail")
        thumbnail_url = (
            thumbnail
            if isinstance(thumbnail, str) and thumbnail.startswith("http")
            else None
        )
        item = {
            "id": f"reddit:{post.get('id', '')}",
            "source": "reddit",
            "title": title,
            "url": url,
            "author": author,
            "published_at": _utc_to_iso(post.get("created_utc")),
            "metrics": metrics,
            "tags": [],
            "thumbnail_url": thumbnail_url,
            "fetched_at": fetched_at,
        }
        item["tags"] = tag_item(item)
        items.append(item)
    return items


def fetch_reddit(
    subreddits: Sequence[str] = ("MapPorn", "dataisbeautiful"),
    sort: str = "hot",
    time: str = "week",  # noqa: A002 - matches Reddit's own `t` parameter name
    limit: int = 25,
) -> list[dict]:
    """Fetch recent top posts from ``subreddits`` as TrendItem dicts.

    Raises :class:`SourceUnreachableError` on HTTP/network failure
    (including 429) and :class:`SourceUnparsableError` on unrecognized
    payloads. Never returns invented posts.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    fetched_at = _now_iso()
    items: list[dict] = []
    subs = [s.strip() for s in subreddits if s and s.strip()]
    for index, sub in enumerate(subs):
        if index > 0:
            time_module.sleep(REQUEST_GAP_SECONDS)
        url = REDDIT_URL.format(sub=sub, sort=sort, limit=limit, time=time)
        payload = get_json(
            url, headers=headers, timeout=REQUEST_TIMEOUT, source_name="reddit"
        )
        items.extend(_parse_listing(payload, sub, fetched_at))
    return items
