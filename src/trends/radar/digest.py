"""Render the mapping-visualization trend radar as Markdown.

:func:`render_radar_digest` takes ranked ``signals`` (from
:func:`trends.radar.signals.build_signals`) and ``source_coverage`` — a
list of ``{"source", "status", "n_items", "detail"}`` dicts where
``status`` is ``"ok"`` or ``"unavailable"`` — and returns the digest
string.

The digest names every source that failed and why (never inventing
items to fill the gap), links every exemplar back to its original, and
closes with the "Inspiration, not replication" legal note.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping, Optional, Sequence


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt_int(value) -> str:
    try:
        return f"{int(round(value)):,}"
    except (TypeError, ValueError):
        return "—"


INSPIRATION_NOTE = """\
## Inspiration, not replication

This radar surfaces **links and metadata only** — titles, sources, and
engagement counts — so you can study what formats are resonating and
bring those *techniques* into your own original work. Styles and
techniques (particle flows, globe spins, timelapse pacing) are not
copyrightable; a specific person's video, image, or caption is.
**Reposting someone's reel, or cloning it frame-for-frame, is
infringement.** Every exemplar above links to the original: watch it,
learn the technique, then make something that is yours.
"""


def render_radar_digest(
    signals: Sequence[Mapping],
    source_coverage: Sequence[Mapping],
    fetched_at: Optional[str] = None,
) -> str:
    """Render the trend-radar Markdown digest."""
    fetched_at = fetched_at or _now_iso()
    signals = list(signals)
    window = signals[0].get("window_days", 7) if signals else 7
    lines = [
        "# 🛰️ Map-Viz Trend Radar",
        "",
        f"Fetched {fetched_at} · trailing {window}-day window · "
        f"{len(signals)} format{'s' if len(signals) != 1 else ''} detected.",
        "",
    ]
    if signals:
        lines += ["## Ranked formats", ""]
        for rank, sig in enumerate(signals, start=1):
            lines += [
                f"### {rank}. {sig.get('tag', 'other')}",
                "",
                f"- Items: {sig.get('n_items', 0)} "
                f"({(sig.get('share') or 0):.0%} of this pull) · "
                f"Engagement: {_fmt_int(sig.get('engagement'))}",
                f"- Takeaway: {sig.get('takeaway', '')}",
            ]
            exemplars = sig.get("exemplars") or []
            if exemplars:
                lines.append("- Exemplars (links to originals):")
                for ex in exemplars:
                    title = ex.get("title") or "(untitled)"
                    url = ex.get("url") or ""
                    source = ex.get("source") or "unknown"
                    if url:
                        lines.append(f"  - [{title}]({url}) ({source})")
                    else:
                        lines.append(f"  - {title} ({source})")
            lines.append("")
    else:
        lines += [
            "## Ranked formats",
            "",
            "No verified trend items this pull — every source was "
            "unavailable or returned nothing usable. Nothing below is "
            "estimated or invented; check the coverage table for why.",
            "",
        ]
    lines += ["## Source coverage", ""]
    if source_coverage:
        lines += [
            "| Source | Status | Items | Detail |",
            "|--------|--------|-------|--------|",
        ]
        for cov in source_coverage:
            status = cov.get("status", "unavailable")
            mark = "✅" if status == "ok" else "❌"
            detail = str(cov.get("detail", "")).replace("\n", " ")
            lines.append(
                f"| {cov.get('source', '?')} | {mark} {status} | "
                f"{cov.get('n_items', 0)} | {detail} |"
            )
        lines.append("")
    else:
        lines += ["No sources were queried.", ""]
    lines += [
        INSPIRATION_NOTE,
        "",
        "---",
        "_Engagement sums the metrics each source provides (scores, views, "
        "likes, comments) — different sources use different currencies, so "
        "it is a within-pull ranking input, not a comparable quantity. "
        "Format tags are keyword heuristics (see trends.radar.formats), "
        "not verdicts._",
        "",
    ]
    return "\n".join(lines)
