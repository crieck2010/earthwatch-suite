"""Trending-*video* discovery via TikTok Creative Center's public surfaces.

What this covers (exactly)
--------------------------
The *video*-trend side of TikTok's Creative Center — i.e. which trending
video entities exist — as distinct from the *music* chart that
``trends.fetch`` reads. It tries the known Creative Center video/trend
endpoints in order (the probe list from ``docs/SOURCES.md``, verified
2026-09-29 from a plain server):

1. Health probe: ``CreativeOne/Report/GetTopContentsOverview`` — a tiny
   unsigned endpoint. If it does not answer ``StatusCode: 0``, the
   Creative Center backend is unreachable and there is no point trying
   further.
2. ``CreativeOne/Report/CreativeCenterGetTopContentsList`` — the
   top-contents (video) endpoint. Probe: HTTP 200, ``StatusCode: 0``,
   but **zero video entities** for US/GB across periods and labels —
   deprecated or gated.
3. ``creative_radar_api/v1/popular_trend/list`` — the legacy trend-list
   endpoint. Probe: HTTP 200 but ``{"code": 40101, "msg": "no
   permission"}`` — login-walled.

When none of these yield video entities, this module raises
:class:`SourceUnreachableError` with the per-attempt detail. It NEVER
falls back to unofficial scraping: no ``www.tiktok.com/api/*`` calls
(they need msToken/X-Bogus request signing), no login sessions, no
headless browsers, no third-party proxies. See ``docs/SOURCES.md`` for
the full probe history and rationale.

Status: as of 2026-09-29 there is **no verifiable keyless path** from a
plain server to TikTok's trending-video entities, so
:func:`fetch_tiktok_videos` is expected to raise today. The radar CLI
treats that as an honest "unavailable" source (exit 2), not a crash.
``VIDEO_ENDPOINTS`` is the single place to add a working endpoint if a
future probe finds one — add a fixture regression test when you do.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from ...models import SourceUnreachableError
from ..formats import tag_item
from ._http import REQUEST_TIMEOUT, get_json

#: Health probe (tiny, unsigned). Verified 2026-09-29: HTTP 200,
#: {"BaseResp": {"StatusCode": 0, ...}}.
HEALTH_URL = "https://ads.tiktok.com/CreativeOne/Report/GetTopContentsOverview"

#: Candidate trending-video endpoints, tried in order. ``{region}`` is the
#: ISO country code, ``{period}`` the day window, ``{limit}`` the page size.
VIDEO_ENDPOINTS: tuple[str, ...] = (
    # Top-contents (video) endpoint: probe showed HTTP 200 but zero
    # entities — kept so a future re-open is picked up automatically.
    "https://ads.tiktok.com/CreativeOne/Report/CreativeCenterGetTopContentsList"
    "?country_code={region}&period={period}&limit={limit}",
    # Legacy popular-trend endpoint: probe showed 40101 "no permission".
    "https://ads.tiktok.com/creative_radar_api/v1/popular_trend/list"
    "?period={period}&country_code={region}&page=1&limit={limit}",
)

_REQUEST_HEADERS = {
    "User-Agent": "survey-trends/0.2.0 (github.com/crieck2010/survey-trends)",
    "Accept": "application/json",
    "Referer": "https://ads.tiktok.com/",
    "Origin": "https://ads.tiktok.com",
    "Accept-Language": "en-US,en;q=0.9",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _health_ok() -> bool:
    try:
        payload = get_json(
            HEALTH_URL,
            headers=_REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
            source_name="tiktok-creative-center",
        )
    except Exception:
        return False
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("BaseResp"), dict)
        and payload["BaseResp"].get("StatusCode") == 0
    )


def _entity_list(payload: Any) -> list[dict]:
    """Extract the video-entity list from a Creative Center payload.

    Returns [] when the payload is well-formed but carries no entities
    (the probe's zero-entity case), which the caller treats as
    "endpoint answered but yielded nothing usable" — not as data.
    """
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    for key in ("list", "entities", "contents", "items"):
        value = data.get(key)
        if isinstance(value, list):
            return [e for e in value if isinstance(e, dict)]
    return []


def _permission_denied(payload: Any) -> Optional[str]:
    """Return a 'code: msg' string when the payload is a permission denial."""
    if isinstance(payload, dict) and payload.get("code") in (40101, 401, 403):
        msg = payload.get("msg") or payload.get("message") or "permission denied"
        return f"{payload['code']}: {msg}"
    return None


def _first_present(mapping: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_int(mapping: dict, keys: tuple[str, ...]) -> Optional[int]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _item_from_entity(entity: dict, fetched_at: str) -> Optional[dict]:
    """Build a TrendItem from one video entity; None when unusable."""
    title = _first_present(
        entity, ("title", "name", "desc", "description", "caption", "video_name")
    )
    if not title:
        return None
    video_id = _first_present(entity, ("id", "video_id", "item_id", "aweme_id"))
    url = _first_present(entity, ("url", "share_url", "video_url", "link"))
    if not url and video_id:
        url = f"https://www.tiktok.com/@{video_id}"
    metrics: dict[str, int] = {}
    for key, names in (
        ("views", ("play_count", "view_count", "views")),
        ("likes", ("like_count", "likes", "digg_count")),
        ("comments", ("comment_count", "comments")),
    ):
        value = _first_int(entity, names)
        if value is not None:
            metrics[key] = value
    item = {
        "id": f"tiktok:{video_id or title[:40]}",
        "source": "tiktok",
        "title": title,
        "url": url,
        "author": _first_present(entity, ("author", "author_name", "nickname")) or None,
        "published_at": None,  # Creative Center entities carry no public timestamp
        "metrics": metrics,
        "tags": [],
        "thumbnail_url": None,  # never download or rehost thumbnails
        "fetched_at": fetched_at,
    }
    item["tags"] = tag_item(item)
    return item


def fetch_tiktok_videos(
    region: str = "US", period: int = 7, limit: int = 25
) -> list[dict]:
    """Discover trending video entities from Creative Center (keyless).

    Returns TrendItem dicts when an endpoint yields video entities.
    Raises :class:`SourceUnreachableError` when the backend is
    unreachable or no endpoint returns usable video entities. Never
    synthesizes videos and never uses unofficial scraping.
    """
    fetched_at = _now_iso()
    if not _health_ok():
        raise SourceUnreachableError(
            "TikTok Creative Center did not answer the health probe "
            f"({HEALTH_URL}); the video-trend backend is unreachable from here."
        )
    attempts: list[str] = []
    for template in VIDEO_ENDPOINTS:
        url = template.format(region=region, period=period, limit=limit)
        try:
            payload = get_json(
                url,
                headers=_REQUEST_HEADERS,
                timeout=REQUEST_TIMEOUT,
                source_name="tiktok-creative-center",
            )
        except SourceUnreachableError as exc:
            attempts.append(f"{url} -> {exc}")
            continue
        entities = _entity_list(payload)
        if not entities:
            denied = _permission_denied(payload)
            if denied:
                attempts.append(f"{url} -> login-walled ({denied})")
            else:
                attempts.append(
                    f"{url} -> responded but returned zero video entities "
                    "(deprecated or gated)"
                )
            continue
        items = [
            item
            for item in (_item_from_entity(e, fetched_at) for e in entities[:limit])
            if item is not None
        ]
        if not items:
            attempts.append(
                f"{url} -> {len(entities)} entities but none had a usable title"
            )
            continue
        return items
    detail = "; ".join(attempts) if attempts else "no video endpoint configured"
    raise SourceUnreachableError(
        "No TikTok Creative Center endpoint returned trending video entities. "
        f"Attempts: {detail}"
    )
