"""Priority scoring and kind classification for gazetteer places.

Priority formula (documented contract — do not change without a minor
version bump, downstream caches and label layouts depend on it)::

    priority = round(10 * log10(pop + 1)) - 2 * scalerank + capital_bonus

where ``capital_bonus`` is 10 for any FEATURECLA containing "capital"
(national capitals outrank same-size towns) and 0 otherwise.

Rationale: population spans six orders of magnitude, so it enters on a
log scale; Natural Earth ``scalerank`` (0 = most important .. 10 = least)
is the cartographer's own importance judgment and enters linearly;
capitals get a bump because reference reels label them even when small
(e.g. KINGSTON on the Lake Ontario reel).

``priority`` is an ``int``; higher wins. survey-aesthetics'
``place_labels`` resolves ties by ``priority``, so this module also
defines the total tie-break order: ``(-priority, name)`` — deterministic
for every input.

Kind mapping (drives the ``kinds=`` filter):

* "capital" in FEATURECLA (case-insensitive)  -> "city"
* POP_MAX >= 100_000                           -> "city"
* FEATURECLA in {Scientific station, Meteorological Station,
                 Historic place}                -> "other"
* everything else                              -> "town"
"""

from __future__ import annotations

import math

CITY_POP_THRESHOLD = 100_000
CAPITAL_BONUS = 10
OTHER_CLASSES = frozenset(
    {"Scientific station", "Meteorological Station", "Historic place"}
)

#: Kinds selected by default in :func:`gazetteer.labels_for_bbox`.
DEFAULT_KINDS = ("city", "town")


def is_capital(featurecla: str) -> bool:
    """True when the Natural Earth feature class names a capital."""
    return "capital" in (featurecla or "").lower()


def kind_of(featurecla: str, pop: int) -> str:
    """Classify a place as ``"city"``, ``"town"`` or ``"other"``."""
    featurecla = featurecla or ""
    if is_capital(featurecla):
        return "city"
    if featurecla in OTHER_CLASSES:
        return "other"
    if pop >= CITY_POP_THRESHOLD:
        return "city"
    return "town"


def priority_of(pop: int, scalerank: int, featurecla: str) -> int:
    """Priority score for one place. Higher = more label-worthy."""
    pop = max(int(pop or 0), 0)
    scalerank = int(scalerank if scalerank is not None else 10)
    score = round(10.0 * math.log10(pop + 1)) - 2 * scalerank
    if is_capital(featurecla):
        score += CAPITAL_BONUS
    return int(score)


def sort_key(place) -> tuple:
    """Total order used for ranking: ``(-priority, name)``.

    ``name`` is the ASCII label text; the tuple is a strict total order
    because names are unique per record in the bundled dataset, so
    sorting is deterministic for every input.
    """
    return (-place.priority, place.name)
