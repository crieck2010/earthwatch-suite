"""Aggregate tagged TrendItems into ranked per-format signals.

:func:`build_signals` groups items by their format tags and emits one
signal record per format:

* ``tag`` — the format tag (see :mod:`trends.radar.formats`).
* ``n_items`` — items carrying this tag (an item may count toward
  several tags).
* ``engagement`` — sum of the item's available numeric metrics (score,
  views, likes, comments — whichever keys the source provided; ``None``
  values are ignored). Cross-source by construction: a Reddit score and
  a TikTok view count are different currencies, so engagement is a
  within-pull ranking input, not a comparable quantity.
* ``share`` — ``n_items / total items`` in this pull.
* ``velocity`` — relative engagement change vs ``baseline`` (a
  ``{tag: previous engagement}`` mapping from an earlier pull), or
  ``None`` when no baseline was provided.
* ``takeaway`` — a plain-language sentence for the digest.
* ``exemplars`` — up to 3 highest-engagement items: title + url +
  source (links only — never content).
* ``window_days`` — echoed from the call for downstream consumers.

Ranking: engagement descending, tag name ascending as the deterministic
tiebreak. No randomness, no network, no UI imports.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from .formats import FALLBACK_TAG


def _engagement(item: dict) -> float:
    total = 0.0
    metrics = item.get("metrics") or {}
    for value in metrics.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            total += value
    return total


def _fmt_int(value: float) -> str:
    return f"{int(round(value)):,}"


def _takeaway(
    rank: int, tag: str, n_items: int, share: float,
    engagement: float, velocity: Optional[float],
) -> str:
    if rank == 1:
        lead = f"{tag} leads this pull"
    elif rank == 2:
        lead = f"{tag} is the runner-up format"
    elif rank == 3:
        lead = f"{tag} rounds out the top three"
    else:
        lead = f"{tag} is present but quieter"
    sentence = (
        f"{lead}: {n_items} item{'s' if n_items != 1 else ''} "
        f"({share:.0%} of the pull), engagement {_fmt_int(engagement)}."
    )
    if velocity is not None:
        direction = "accelerating" if velocity > 0 else "cooling"
        sentence += f" {direction} vs the previous pull ({velocity:+.0%})."
    return sentence


def build_signals(
    items: Sequence[dict],
    baseline: Optional[Mapping[str, float]] = None,
    window_days: int = 7,
) -> list[dict]:
    """Aggregate tagged items into ranked per-format signals."""
    items = list(items)
    total = len(items)
    by_tag: dict[str, list[dict]] = {}
    for item in items:
        tags = item.get("tags") or [FALLBACK_TAG]
        for tag in tags:
            by_tag.setdefault(tag, []).append(item)
    # Engagement per tag, then deterministic rank order.
    engagement = {
        tag: sum(_engagement(item) for item in tagged)
        for tag, tagged in by_tag.items()
    }
    ranked = sorted(by_tag, key=lambda t: (-engagement[t], t))
    signals: list[dict] = []
    for rank, tag in enumerate(ranked, start=1):
        tagged = by_tag[tag]
        eng = engagement[tag]
        n_items = len(tagged)
        share = (n_items / total) if total else 0.0
        velocity: Optional[float] = None
        if baseline is not None:
            prev = baseline.get(tag)
            if isinstance(prev, (int, float)) and not isinstance(prev, bool) and prev:
                velocity = (eng - prev) / prev
        exemplars = sorted(tagged, key=lambda i: (-_engagement(i), i.get("title") or ""))
        signals.append(
            {
                "tag": tag,
                "n_items": n_items,
                "engagement": eng,
                "share": share,
                "velocity": velocity,
                "takeaway": _takeaway(rank, tag, n_items, share, eng, velocity),
                "exemplars": [
                    {
                        "title": e.get("title", ""),
                        "url": e.get("url", ""),
                        "source": e.get("source", ""),
                    }
                    for e in exemplars[:3]
                ],
                "window_days": window_days,
            }
        )
    return signals
