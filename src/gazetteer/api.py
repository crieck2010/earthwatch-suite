"""Public selection API: bbox labels and name search.

Output contract for :func:`labels_for_bbox` — exactly what
survey-aesthetics' ``place_labels`` consumes::

    [{"x": lon, "y": lat, "text": "NAME", "priority": int}, ...]

(``place_labels`` treats ``x``/``y`` as data coordinates and ``priority``
as the tie-break, higher winning.)

Collision contract: this engine returns *ranked candidates*. Final
on-canvas decluttering (greedy compass-offset placement, dropping labels
that fit nowhere) stays in survey-aesthetics' ``place_labels`` — it is
not duplicated here. Callers that want fewer labels than requested
should shrink ``max_labels``, not reimplement placement.

Call-once-per-reel rule: labels depend only on the region bbox, not on
time, so survey-viz should call :func:`labels_for_bbox` once per reel
and reuse the list for every frame.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from gazetteer.index import query_bbox
from gazetteer.places import Place, load_places
from gazetteer.scoring import DEFAULT_KINDS, sort_key


def labels_for_bbox(
    lon0: float,
    lon1: float,
    lat0: float,
    lat1: float,
    *,
    max_labels: int = 8,
    min_population: int = 0,
    kinds: Sequence[str] = DEFAULT_KINDS,
) -> List[Dict]:
    """Top-N ranked place labels inside a lon/lat bbox.

    ``lon0 > lon1`` selects a bbox crossing the antimeridian. Candidates
    are filtered by ``kinds`` and ``min_population``, ranked by
    ``(-priority, name)`` (see :mod:`gazetteer.scoring`), and the first
    ``max_labels`` are returned in ranked order. An empty bbox region
    returns ``[]``. Deterministic for every input.
    """
    kinds = tuple(kinds)
    candidates = [
        p
        for p in query_bbox(lon0, lon1, lat0, lat1)
        if p.kind in kinds and p.pop >= min_population
    ]
    candidates.sort(key=sort_key)
    top = candidates[: max(0, int(max_labels))]
    return [
        {"x": p.lon, "y": p.lat, "text": p.name, "priority": p.priority}
        for p in top
    ]


def _match_quality(place: Place, q: str) -> Optional[int]:
    """0 = prefix match, 1 = substring match, None = no match.

    Matches against both the ASCII label and the local (diacritic) name.
    """
    for name in (place.name, place.name_local):
        low = name.lower()
        if low.startswith(q):
            return 0
        if q in low:
            return 1
    return None


def search(query: str, *, limit: int = 5) -> List[Dict]:
    """Find places by name for explicit user label requests.

    Case-insensitive; prefix matches rank above substring matches, then
    by ``(-priority, name)``. Returns the :func:`labels_for_bbox` label
    dicts plus ``adm0``/``adm1``/``kind``/``pop`` context so a UI (e.g.
    reel-studio) can disambiguate ("Rochester, United States, New York"
    vs "Rochester, United Kingdom, Medway").
    """
    q = (query or "").strip().lower()
    if not q:
        return []
    scored: List[Tuple[int, Tuple, Place]] = []
    for place in load_places():
        quality = _match_quality(place, q)
        if quality is not None:
            scored.append((quality, sort_key(place), place))
    scored.sort(key=lambda s: (s[0], s[1]))
    out = []
    for _, _, p in scored[: max(0, int(limit))]:
        out.append(
            {
                "x": p.lon,
                "y": p.lat,
                "text": p.name,
                "priority": p.priority,
                "adm0": p.adm0,
                "adm1": p.adm1,
                "kind": p.kind,
                "pop": p.pop,
            }
        )
    return out
