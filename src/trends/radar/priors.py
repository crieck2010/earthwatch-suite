"""Convert ranked radar signals into priors for survey-autopilot.

Schema (exact — a sibling task wires this into survey-autopilot)::

    {
      "version": 1,
      "generated_at": "<ISO-8601 UTC str>",
      "window_days": <int>,
      "priors": {"<tag>": <multiplier float>, ...},
      "notes": ["<takeaway str>", ...],
    }

Multiplier rule (deterministic): the top-ranked format gets 1.3, the
runner-up 1.15, third place 1.05, and every other present format 1.0.
Only formats with **at least 2 items** are included (a single data point
is not a trend). Multipliers are clamped to [0.25, 4.0].

``notes`` carries the takeaway sentences of the included formats, in
rank order. ``window_days`` is read from the signals (emitted by
:func:`trends.radar.signals.build_signals`); 7 when no signals exist.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping, Sequence

PRIOR_VERSION = 1
MIN_ITEMS_FOR_PRIOR = 2
_MULTIPLIERS = (1.3, 1.15, 1.05)
_DEFAULT_MULTIPLIER = 1.0
_CLAMP_LO, _CLAMP_HI = 0.25, 4.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clamp(value: float) -> float:
    return max(_CLAMP_LO, min(_CLAMP_HI, value))


def priors_from_signals(signals: Sequence[Mapping]) -> dict:
    """Build the priors payload from ranked signals. Deterministic."""
    signals = list(signals)
    eligible = [s for s in signals if (s.get("n_items") or 0) >= MIN_ITEMS_FOR_PRIOR]
    priors: dict[str, float] = {}
    notes: list[str] = []
    for rank, sig in enumerate(eligible, start=1):
        tag = sig.get("tag", "other")
        multiplier = (
            _MULTIPLIERS[rank - 1] if rank <= len(_MULTIPLIERS) else _DEFAULT_MULTIPLIER
        )
        priors[tag] = _clamp(float(multiplier))
        takeaway = sig.get("takeaway")
        if isinstance(takeaway, str) and takeaway:
            notes.append(takeaway)
    window_days = signals[0].get("window_days", 7) if signals else 7
    return {
        "version": PRIOR_VERSION,
        "generated_at": _now_iso(),
        "window_days": window_days,
        "priors": priors,
        "notes": notes,
    }
