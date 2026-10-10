"""Render the weekly trending-sounds digest as Markdown.

Two renderers:

* :func:`build_digest` — the real digest from verified :class:`SongRank`
  records.
* :func:`build_unavailable_digest` — the honest fallback: a clear
  "trending data unavailable this week" notice plus the evergreen
  Instagram in-app guidance. Never lists songs it did not verify.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from .models import SongRank

#: Title/artist keywords that tend to suit map/data-viz reels: calm,
#: wordless-leaning beds that sit under narration or on-screen text.
_GOOD_FIT_KEYWORDS = (
    "instrumental", "ambient", "lofi", "lo-fi", "cinematic", "orchestral",
    "piano", "strings", "chill", "dreamy", "ethereal", "atmospheric",
    "calm", "peaceful", "acoustic", "jazz", "bossa", "lounge", "downtempo",
    "meditation", "sleep", "study", "focus",
)

#: Markers that suggest a track needs a closer look before it goes under a
#: data-viz reel (lyrics-forward, meme, or challenge-driven sounds).
_CAUTION_KEYWORDS = (
    "dance", "challenge", "meme", "remix", "sped up", "slowed",
    "explicit", "diss", "freestyle",
)


def fit_note(song: SongRank) -> str:
    """One-line fit heuristic for a data-viz/map reel.

    Keyword-based only — a starting filter, not a verdict. Always advises a
    listen before publishing.
    """
    haystack = f"{song.title} {song.artist}".lower()
    if any(kw in haystack for kw in _GOOD_FIT_KEYWORDS):
        return "likely fit — calm/wordless-leaning bed for map reels"
    if any(kw in haystack for kw in _CAUTION_KEYWORDS):
        return "check first — meme/dance-leaning; may clash with data-viz"
    return "unclassified — listen and check lyrics/vibe before use"


def _week_range(fetched_at: str) -> str:
    """'Sep 22 – Sep 29, 2026' style trailing-7-day range from a fetch stamp."""
    try:
        end = datetime.strptime(fetched_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return "trailing 7 days"
    start = end.replace(hour=0, minute=0, second=0)
    from datetime import timedelta

    start = end - timedelta(days=6)
    if start.year == end.year:
        return f"{start.strftime('%b %d')} – {end.strftime('%b %d, %Y')}"
    return f"{start.strftime('%b %d, %Y')} – {end.strftime('%b %d, %Y')}"


def _fmt_uses(uses) -> str:
    if uses is None:
        return "—"
    if uses >= 1_000_000:
        return f"{uses / 1_000_000:.1f}M"
    if uses >= 1_000:
        return f"{uses / 1_000:.1f}K"
    return str(uses)


def _trend_arrow(trend) -> str:
    return {"up": "▲", "down": "▼", "new": "★ new", "same": "–"}.get(trend or "", "—")


INSTAGRAM_GUIDANCE = """\
## Instagram: find trending audio in-app (no API exists)

Instagram offers no API for trending-audio discovery, so this part is
manual — about two minutes, once a week:

1. **Reels music library → Trending tab.** Start a Reel, tap the music
   icon, open the **Trending** tab. This is Instagram's own official
   trending-audio list, tailored to your account.
2. **Professional Dashboard → Trending Audio.** Profile → Professional
   Dashboard → Tips and Resources → **Trending Audio** for the dedicated
   surface (most detailed on US accounts).
3. **The ↗ arrow signal.** While scrolling Reels, look next to the track
   name: an upward arrow (↗) instead of the music-note icon means
   Instagram is actively amplifying videos using that sound. Tap the
   track name to see usage: under ~1,000 uses is an early trend, over
   ~100,000 is maturing toward saturation.

Reminder: trending in-app sounds cannot be attached through the
publishing APIs — they only exist inside Instagram/TikTok's apps. To use
one, license it (or a royalty-free soundalike) and mux it into the MP4
before publishing, exactly as reel-studio's audio step already does.
"""


def build_digest(
    songs: Sequence[SongRank], fetched_at: str, region: str = "US"
) -> str:
    """Render the weekly digest Markdown from verified song records."""
    songs = sorted(songs, key=lambda s: s.rank)
    shown = songs[:20]
    week = _week_range(fetched_at)
    lines = [
        f"# Weekly Trending Sounds — TikTok ({region})",
        "",
        f"Week of {week} · trailing 7 days",
        f"Fetched {fetched_at} from the TikTok Creative Center popular-music chart.",
        "",
        "## Top tracks",
        "",
        "| # | Title | Artist | Videos using it | Trend | Fit for map reels |",
        "|---|-------|--------|-----------------|-------|-------------------|",
    ]
    for song in shown:
        lines.append(
            f"| {song.rank} | {song.title} | {song.artist or '—'} | "
            f"{_fmt_uses(song.uses)} | {_trend_arrow(song.trend)} | "
            f"{fit_note(song)} |"
        )
    good = [s for s in shown if fit_note(s).startswith("likely fit")]
    lines += [
        "",
        "## Fit notes for data-viz reels",
        "",
        "Keyword heuristic only — always listen before publishing, and use",
        "audio you own or that is royalty-free/licensed for commercial use.",
        "",
    ]
    if good:
        lines.append(
            "Tracks whose titles/artists suggest a calm, wordless-leaning bed "
            "that sits well under map reels:"
        )
        lines.append("")
        for song in good[:8]:
            lines.append(f"- **{song.title}** — {song.artist or 'unknown artist'}")
        lines.append("")
    else:
        lines += [
            "No track this week clearly matched the calm/instrumental profile —",
            "this week's chart leans lyrics- or meme-forward. The Instagram",
            "in-app Trending tab (below) is the better hunting ground this week.",
            "",
        ]
    lines += [
        INSTAGRAM_GUIDANCE,
        "",
        "---",
        "_Ranks reflect the TikTok Creative Center popular-music chart "
        f"({region}, trailing 7 days), fetched {fetched_at}. Counts are as "
        "reported by the source; missing values are shown as —. Nothing in "
        "this digest is estimated or invented._",
        "",
    ]
    return "\n".join(lines)


def build_unavailable_digest(reason: str, checked_at: str) -> str:
    """Honest fallback digest: no songs are listed because none were verified."""
    return "\n".join(
        [
            "# Weekly Trending Sounds — unavailable this week",
            "",
            "**Trending data unavailable this week — source unreachable.**",
            "",
            f"Checked {checked_at}. What happened: {reason}",
            "",
            "No song titles, ranks, or counts are listed below because none",
            "could be verified against TikTok's official chart this week.",
            "This digest never estimates or invents chart data.",
            "",
            INSTAGRAM_GUIDANCE,
            "",
            "---",
            "_The TikTok section will return automatically once the Creative "
            "Center chart endpoint responds again. See docs/SOURCES.md for "
            "the source details and probe history._",
            "",
        ]
    )
