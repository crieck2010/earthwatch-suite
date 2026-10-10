"""Fetch TikTok's official Creative Center popular-music chart (keyless).

Strategy
--------
TikTok publishes a "popular music" chart inside Creative Center. The page is
JS-heavy, but the chart data has historically been served by plain JSON
endpoints that need no API key, login, or cookies. This module tries the
known endpoints in order:

1. Health check: ``CreativeOne/Report/GetTopContentsOverview`` — a tiny
   unsigned endpoint. If this fails, the Creative Center backend is
   unreachable from here and there is no point trying further.
2. Music chart: ``creative_radar_api/v1/popular_trend/music/list`` — the
   legacy popular-music endpoint (region + 7-day period parameters).

Honest-access contract: every failure raises :class:`SourceUnreachableError`
or :class:`SourceUnparsableError`. This module NEVER synthesizes song
titles, ranks, or counts. See ``docs/SOURCES.md`` for the probe history
(what was tried, what responded, and when).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Sequence

import requests

from .models import (
    SongRank,
    SourceUnparsableError,
    SourceUnreachableError,
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
REQUEST_TIMEOUT = 25  # seconds per request; the VM has been slow to TikTok
MAX_SONGS = 30

#: Tiny unsigned Creative Center endpoint used as a reachability probe.
#: Verified 2026-09-29: HTTP 200, {"BaseResp": {"StatusCode": 0, ...}}.
OVERVIEW_URL = "https://ads.tiktok.com/CreativeOne/Report/GetTopContentsOverview"

#: Candidate popular-music endpoints, tried in order. ``{region}`` is the
#: ISO country code (e.g. "US").
MUSIC_ENDPOINTS: Sequence[str] = (
    "https://ads.tiktok.com/creative_radar_api/v1/popular_trend/music/list"
    "?period=7&country_code={region}&page=1&limit=30",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Referer": "https://ads.tiktok.com/",
            "Origin": "https://ads.tiktok.com",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return s


def _overview_ok(session: requests.Session) -> bool:
    """True when the Creative Center backend answers the health probe."""
    try:
        resp = session.get(OVERVIEW_URL, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        return False
    if resp.status_code != 200:
        return False
    try:
        payload = resp.json()
    except ValueError:
        return False
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("BaseResp"), dict)
        and payload["BaseResp"].get("StatusCode") == 0
    )


def _first_present(mapping: dict, keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _first_int(mapping: dict, keys: Sequence[str]) -> Optional[int]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            digits = "".join(ch for ch in value if ch.isdigit())
            if digits:
                return int(digits)
    return None


def _normalize_trend(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    mapping = {
        "up": "up", "rise": "up", "rising": "up", "1": "up",
        "down": "down", "fall": "down", "falling": "down", "-1": "down",
        "new": "new", "0": "same", "same": "same", "steady": "same",
    }
    return mapping.get(text)


def _parse_music_payload(
    payload: object, source: str, fetched_at: str
) -> list[SongRank]:
    """Parse a Creative Center music-chart payload into SongRank records.

    Strict: the envelope must be a dict whose ``data`` holds a ``list``.
    Anything else raises :class:`SourceUnparsableError` rather than being
    guessed at.
    """
    if not isinstance(payload, dict):
        raise SourceUnparsableError(
            f"music endpoint returned a non-object payload ({type(payload).__name__})"
        )
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("list"), list):
        raise SourceUnparsableError(
            "music endpoint payload has no data.list array; shape not recognized"
        )
    songs: list[SongRank] = []
    for index, item in enumerate(data["list"][:MAX_SONGS], start=1):
        if not isinstance(item, dict):
            continue
        title = _first_present(item, ("title", "name", "song_name", "music_name"))
        if not title:
            continue  # a chart row without a title is not usable; skip it
        artist = _first_present(item, ("artist", "author", "singer", "artist_name"))
        uses = _first_int(item, ("use_count", "video_count", "posts", "play_count", "count"))
        trend = _normalize_trend(item.get("trend", item.get("rank_change")))
        rank = _first_int(item, ("rank", "ranking", "position")) or index
        songs.append(
            SongRank(
                rank=rank,
                title=title,
                artist=artist,
                uses=uses,
                trend=trend,
                source=source,
                fetched_at=fetched_at,
            )
        )
    if not songs:
        raise SourceUnparsableError(
            "music endpoint returned an empty or title-less list; nothing verifiable to report"
        )
    songs.sort(key=lambda s: s.rank)
    return songs


def fetch_trending_songs(
    region: str = "US",
    session: Optional[requests.Session] = None,
) -> list[SongRank]:
    """Fetch the trailing-7-day popular-music chart for ``region``.

    Returns a list of :class:`SongRank` (rank-sorted). Raises
    :class:`SourceUnreachableError` when the source cannot be reached or no
    endpoint returns song data, and :class:`SourceUnparsableError` when a
    payload arrives in an unrecognized shape. Never returns invented data.
    """
    own_session = session is None
    session = session or _session()
    fetched_at = _now_iso()
    attempts: list[str] = []
    try:
        if not _overview_ok(session):
            raise SourceUnreachableError(
                "TikTok Creative Center did not answer the health probe "
                f"({OVERVIEW_URL}); the chart backend is unreachable from here."
            )
        for template in MUSIC_ENDPOINTS:
            url = template.format(region=region)
            try:
                resp = session.get(url, timeout=REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                attempts.append(f"{url} -> network error: {exc}")
                continue
            if resp.status_code != 200:
                attempts.append(f"{url} -> HTTP {resp.status_code}")
                continue
            try:
                payload = resp.json()
            except ValueError:
                attempts.append(f"{url} -> HTTP 200 but body was not JSON")
                continue
            try:
                return _parse_music_payload(payload, "tiktok-creative-center", fetched_at)
            except SourceUnparsableError as exc:
                attempts.append(f"{url} -> unparsable: {exc}")
                continue
        detail = "; ".join(attempts) if attempts else "no music endpoint configured"
        raise SourceUnreachableError(
            "No Creative Center music endpoint returned usable song data. "
            f"Attempts: {detail}"
        )
    finally:
        if own_session:
            session.close()
