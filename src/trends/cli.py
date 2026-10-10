"""Command-line interface for survey-trends.

* ``survey-trends digest --markdown`` — fetch the chart and print the weekly
  digest as Markdown to stdout. Exit 0 with data; exit 2 when the source is
  unreachable (the unavailable notice is still printed to stdout so a
  scheduled email job has a meaningful body); exit 1 on unexpected errors.
* ``survey-trends fetch --json`` — print the raw verified song records as
  JSON (machine-readable). Same exit-code contract.
* ``survey-trends radar --markdown`` / ``--json`` — build the
  map-viz trend radar from Reddit, TikTok, and YouTube. Exit 0 = digest
  rendered from all sources; exit 2 = at least one source was unavailable
  (the digest still renders from what responded, with an honest
  "unavailable" section — never invented); exit 1 = hard error.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from . import __version__
from .digest import build_digest, build_unavailable_digest
from .fetch import _now_iso, fetch_trending_songs
from .models import TrendsError
from .radar import (
    build_signals,
    fetch_reddit,
    fetch_tiktok_videos,
    fetch_youtube_videos,
    priors_from_signals,
    render_radar_digest,
)

EXIT_OK = 0
EXIT_SOURCE_UNAVAILABLE = 2
EXIT_ERROR = 1


def _cmd_digest(args: argparse.Namespace) -> int:
    checked_at = _now_iso()
    try:
        songs = fetch_trending_songs(region=args.region)
    except TrendsError as exc:
        sys.stdout.write(build_unavailable_digest(str(exc), checked_at))
        return EXIT_SOURCE_UNAVAILABLE
    except Exception as exc:  # defensive: never half-print a digest
        sys.stderr.write(f"survey-trends: unexpected error: {exc}\n")
        return EXIT_ERROR
    sys.stdout.write(build_digest(songs, songs[0].fetched_at, region=args.region))
    return EXIT_OK


def _cmd_fetch(args: argparse.Namespace) -> int:
    try:
        songs = fetch_trending_songs(region=args.region)
    except TrendsError as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": str(exc)}) + "\n")
        return EXIT_SOURCE_UNAVAILABLE
    except Exception as exc:
        sys.stderr.write(f"survey-trends: unexpected error: {exc}\n")
        return EXIT_ERROR
    sys.stdout.write(
        json.dumps(
            {"ok": True, "songs": [s.to_dict() for s in songs]}, indent=2
        )
        + "\n"
    )
    return EXIT_OK


def _run_radar(args: argparse.Namespace) -> int:
    checked_at = _now_iso()
    items: list[dict] = []
    coverage: list[dict] = []
    sources_ok = True

    def _run(label: str, fetcher, ok_detail: str) -> None:
        nonlocal sources_ok
        try:
            got = fetcher()
        except TrendsError as exc:
            sources_ok = False
            coverage.append(
                {"source": label, "status": "unavailable", "n_items": 0,
                 "detail": str(exc)}
            )
        else:
            items.extend(got)
            coverage.append(
                {"source": label, "status": "ok", "n_items": len(got),
                 "detail": ok_detail}
            )

    subs = [s.strip() for s in args.subreddits.split(",") if s.strip()]
    _run(
        "reddit",
        lambda: fetch_reddit(subreddits=tuple(subs), limit=args.limit),
        f"r/{', r/'.join(subs)} (hot, trailing week) via the public JSON API, keyless",
    )
    _run(
        "tiktok",
        lambda: fetch_tiktok_videos(limit=args.limit),
        "Creative Center public video-trend endpoints, keyless",
    )
    _run(
        "youtube",
        lambda: fetch_youtube_videos(args.query, max_results=args.limit),
        f"official YouTube Data API v3 search for {args.query!r}",
    )

    signals = build_signals(items, window_days=7)
    as_json = args.json and not args.markdown
    if as_json:
        payload = {
            "ok": sources_ok,
            "fetched_at": checked_at,
            "signals": signals,
            "coverage": coverage,
            "priors": priors_from_signals(signals),
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        sys.stdout.write(
            render_radar_digest(signals, coverage, fetched_at=checked_at)
        )
    return EXIT_OK if sources_ok else EXIT_SOURCE_UNAVAILABLE


def _cmd_radar(args: argparse.Namespace) -> int:
    try:
        return _run_radar(args)
    except Exception as exc:  # defensive: never half-print a digest
        sys.stderr.write(f"survey-trends: unexpected error: {exc}\n")
        return EXIT_ERROR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="survey-trends",
        description="Weekly trending-sounds digest from TikTok's official chart.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    digest = sub.add_parser("digest", help="Print the weekly digest as Markdown.")
    digest.add_argument(
        "--markdown",
        action="store_true",
        help="Emit Markdown to stdout (the scheduled-email mode).",
    )
    digest.add_argument("--region", default="US", help="Chart region (default: US).")
    digest.set_defaults(func=_cmd_digest)

    fetch = sub.add_parser("fetch", help="Print raw verified song records.")
    fetch.add_argument("--json", action="store_true", help="Emit JSON to stdout.")
    fetch.add_argument("--region", default="US", help="Chart region (default: US).")
    fetch.set_defaults(func=_cmd_fetch)

    radar = sub.add_parser(
        "radar",
        help="Build the map-viz trend radar from Reddit, TikTok, and YouTube.",
    )
    radar.add_argument(
        "--markdown",
        action="store_true",
        help="Emit the radar digest as Markdown to stdout (default).",
    )
    radar.add_argument(
        "--json",
        action="store_true",
        help="Emit signals, coverage, and priors as JSON to stdout.",
    )
    radar.add_argument(
        "--subreddits",
        default="MapPorn,dataisbeautiful",
        help="Comma-separated subreddits to poll (default: MapPorn,dataisbeautiful).",
    )
    radar.add_argument(
        "--query",
        default="map data visualization",
        help="YouTube search query (default: %(default)r).",
    )
    radar.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Max items per source (default: 25).",
    )
    radar.set_defaults(func=_cmd_radar)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
