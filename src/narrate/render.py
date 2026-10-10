"""Render mapped.earth-style narrative captions from fact dicts.

The voice: concrete numbers, no hype adjectives, specific but generic.
Place names come only from ``facts["region_name"]`` — nothing is ever
invented. Numbers are never rounded beyond one decimal and never
editorialized ("nearly", "a whopping") — the honesty rules that make a
caption auditable against the data.
"""

from __future__ import annotations

import datetime as _dt
from typing import Dict, List, Tuple


def _kt(knots: float) -> str:
    """Knot value for captions: whole knots >= 3, else one decimal."""
    if knots >= 3:
        return f"{knots:.0f}"
    return f"{knots:.1f}"


def _subject(kind: str) -> str:
    return "the wind" if kind == "wind" else "the surface water"


def _time_window(facts: Dict) -> str:
    span = facts["time_span"]
    return f"{span['start']} to {span['end']}"


def _lat(lat: float) -> str:
    """Format a latitude with the correct hemisphere suffix."""
    return f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}"


def _lon(lon: float) -> str:
    """Format a longitude with the correct hemisphere suffix."""
    return f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"


def _peak_clause(facts: Dict) -> str:
    peak = facts["peak"]
    subj = _subject(facts["field_kind"])
    return (f"At its strongest, {subj} there ran at "
            f"{_kt(peak['value_knots'])} knots, "
            f"at {_lat(peak['lat'])}, {_lon(peak['lon'])} "
            f"on {peak['time']}.")


def _mean_clause(facts: Dict) -> str:
    subj = _subject(facts["field_kind"])
    span = facts["time_span"]
    region = facts.get("region_name") or "the area"
    return (f"Across {region}, {span['n_timesteps']} {span['cadence']} samples "
            f"averaged {_kt(facts['mean_speed_knots'])} knots.")


def _direction_clause(facts: Dict) -> str:
    d = facts["direction"]
    if d["label"] is None:
        return "No dominant direction — the flow averaged out."
    return f"The flow was mostly {d['label']}."


def _temperature_clause(facts: Dict) -> str:
    t = facts.get("temperature")
    if t is None:
        return ""
    noun = "Air temperatures" if facts["field_kind"] == "wind" else "Water temperatures"
    return f"{noun} ranged from {t['min']:.1f} to {t['max']:.1f} {t['unit']}."


def render_caption(facts: Dict, style: str = "instagram") -> str:
    """Render a 2–4 sentence caption from a :func:`story_facts` dict.

    Parameters
    ----------
    facts:
        The fact dict returned by :func:`narrate.facts.story_facts`.
    style:
        ``"instagram"`` — 2–4 sentences in the mapped.earth caption
        voice (concrete numbers, no hype adjectives, place label from
        ``region_name`` only, never invented).
        ``"email"`` — one compact paragraph for a daily-reel email body:
        same facts, tighter phrasing, still full sentences.

    Raises
    ------
    ValueError
        Unknown style.
    KeyError
        Missing expected fact keys (not a :func:`story_facts` dict).
    """
    if style == "instagram":
        return _render_instagram(facts)
    if style == "email":
        return _render_email(facts)
    raise ValueError(
        f"unknown caption style {style!r}; expected 'instagram' or 'email'.")


def _render_instagram(facts: Dict) -> str:
    sentences = [_mean_clause(facts), _peak_clause(facts), _direction_clause(facts)]
    temp = _temperature_clause(facts)
    if temp:
        sentences.append(temp)
    return " ".join(sentences)


def _render_email(facts: Dict) -> str:
    peak = facts["peak"]
    span = facts["time_span"]
    subj = _subject(facts["field_kind"])
    region = facts.get("region_name") or "the area"
    d = facts["direction"]
    direction_bit = f"Mostly {d['label']} flow" if d["label"] else "No dominant direction"
    parts = [
        f"{region} — {subj} from {span['start']} to {span['end']} "
        f"({span['n_timesteps']} {span['cadence']} samples): "
        f"averaged {_kt(facts['mean_speed_knots'])} knots, "
        f"peaked at {_kt(peak['value_knots'])} knots "
        f"({_lat(peak['lat'])}, {_lon(peak['lon'])}, {peak['time']}).",
        direction_bit + ".",
    ]
    temp = _temperature_clause(facts)
    if temp:
        parts.append(temp)
    return " ".join(parts)


def _parse_iso(value) -> _dt.datetime | None:
    """Parse an ISO-8601 timestamp; return None when unparsable."""
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value
    if isinstance(value, _dt.date):
        return _dt.datetime(value.year, value.month, value.day)
    s = str(value).strip()
    if not s:
        return None
    try:
        return _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _peak_fraction(facts: Dict) -> float:
    """Fraction of the time span at which the peak occurs.

    Computed from the ISO timestamps in ``facts["time_span"]`` and
    ``facts["peak"]["time"]``. Returns 0.5 when the position cannot be
    derived (unparsable timestamps, single timestep, or a span of zero
    or negative length) and clamps the result to [0.0, 1.0].
    """
    span = facts.get("time_span") or {}
    peak = facts.get("peak") or {}
    start = _parse_iso(span.get("start"))
    end = _parse_iso(span.get("end"))
    peak_time = _parse_iso(peak.get("time"))
    if start is None or end is None or peak_time is None:
        return 0.5
    try:
        total = (end - start).total_seconds()
        offset = (peak_time - start).total_seconds()
    except TypeError:
        # Mixed naive/aware timestamps: retry with tzinfo stripped.
        try:
            start = start.replace(tzinfo=None)
            end = end.replace(tzinfo=None)
            peak_time = peak_time.replace(tzinfo=None)
            total = (end - start).total_seconds()
            offset = (peak_time - start).total_seconds()
        except Exception:
            return 0.5
    if total <= 0:
        return 0.5
    return min(1.0, max(0.0, float(offset / total)))


def _validate_beat_facts(facts: Dict) -> None:
    """Raise ValueError with a clear message for missing structure."""
    if not isinstance(facts, dict):
        raise ValueError(
            f"facts must be a dict as returned by story_facts, "
            f"got {type(facts).__name__}.")
    if facts.get("field_kind") not in ("wind", "current"):
        raise ValueError(
            "facts['field_kind'] must be 'wind' or 'current' "
            "(as returned by story_facts), got "
            f"{facts.get('field_kind')!r}.")
    peak = facts.get("peak")
    if not isinstance(peak, dict):
        raise ValueError(
            "facts['peak'] must be a dict with value_knots/lat/lon/time "
            "(as returned by story_facts).")
    for key in ("value_knots", "lat", "lon"):
        if key not in peak or peak[key] is None:
            raise ValueError(
                f"facts['peak']['{key}'] is required for headline beats.")
        try:
            float(peak[key])
        except (TypeError, ValueError):
            raise ValueError(
                f"facts['peak']['{key}'] must be numeric, "
                f"got {peak[key]!r}.") from None
    span = facts.get("time_span")
    if not isinstance(span, dict):
        raise ValueError(
            "facts['time_span'] must be a dict with start/end "
            "(as returned by story_facts).")
    for key in ("start", "end"):
        if key not in span or span[key] is None or str(span[key]).strip() == "":
            raise ValueError(
                f"facts['time_span']['{key}'] is required for headline beats.")
    if "mean_speed_knots" not in facts or facts["mean_speed_knots"] is None:
        raise ValueError(
            "facts['mean_speed_knots'] is required for headline beats.")
    try:
        float(facts["mean_speed_knots"])
    except (TypeError, ValueError):
        raise ValueError(
            f"facts['mean_speed_knots'] must be numeric, "
            f"got {facts['mean_speed_knots']!r}.") from None


def headline_beats(facts: Dict, *, max_beats: int = 4) -> List[Tuple[float, str]]:
    """Draft headline text swaps for an animated reel.

    Drafts headline text swaps for an animated reel (the mapped.earth
    "Where Italy lives" pattern: the title changes at story moments),
    consuming ONLY the dict returned by :func:`story_facts` — never the
    raw field. The beats are deterministic: no RNG, no AI, no invented
    claims.

    These are DRAFTS for human approval/editing — review and edit the
    text before putting it on a reel. Fractions are approximate
    placements along the reel timeline, not frame-exact cues. The
    peak and closing lines only restate computed facts (peak speed and
    location, mean speed, dominant direction, temperature range) —
    no causal claims are made.

    Beat design:

    1. Opening beat at 0.0: region + subject framing, e.g.
       ``"North America — the wind in motion"`` (currents use
       ``"the surface water"``). Uses ``region_name`` verbatim when
       present; falls back to the bare subject when empty.
    2. Peak beat at the fraction where ``peak.time`` falls inside
       ``time_span`` (computed from the ISO timestamps; 0.5 when that
       cannot be derived): quotes the peak in knots — the same unit
       :func:`render_caption` uses for both wind and currents — with
       hemisphere-correct lat/lon via the shared ``_lat``/``_lon``
       helpers, e.g. ``"Peak: 19 knots at 42.00°N, 7.00°W"``.
    3. Closing beat at 0.9: a synthesis line restating the mean speed
       and, when present, the dominant direction.
    4. Optional mid beat at 0.65, only when ``max_beats >= 4`` and the
       facts carry a meaningful temperature range (>= 1.0 in the
       reported unit) that adds information. Never padded with filler;
       fewer beats is fine.

    Parameters
    ----------
    facts:
        The fact dict returned by :func:`narrate.facts.story_facts`.
        ``region_name`` and ``temperature`` are optional; missing or
        empty values still yield valid beats (at minimum the opening).
    max_beats:
        Maximum number of beats to return. Must be >= 1.
        ``max_beats=1`` returns just the opening beat.

    Returns
    -------
    list of (float, str)
        ``(t_fraction, text)`` pairs sorted ascending by fraction,
        fractions in [0.0, 1.0], at most ``max_beats`` entries.

    Raises
    ------
    ValueError
        ``max_beats`` is not an integer >= 1, or ``facts`` is missing
        required structure (``field_kind``, ``peak`` with
        ``value_knots``/``lat``/``lon``, ``time_span`` with
        ``start``/``end``, or ``mean_speed_knots``).
    """
    if isinstance(max_beats, bool) or not isinstance(max_beats, int):
        raise ValueError(
            f"max_beats must be an integer >= 1, got {max_beats!r}.")
    if max_beats < 1:
        raise ValueError(
            f"max_beats must be >= 1, got {max_beats}.")
    _validate_beat_facts(facts)

    kind = facts["field_kind"]
    subject = _subject(kind)

    region_raw = facts.get("region_name") or ""
    region = str(region_raw).strip()
    if region:
        opening = f"{region} — {subject} in motion"
    else:
        opening = f"{subject[0].upper()}{subject[1:]} in motion"

    if max_beats == 1:
        return [(0.0, opening)]

    peak = facts["peak"]
    peak_fraction = _peak_fraction(facts)
    peak_text = (
        f"Peak: {_kt(float(peak['value_knots']))} knots at "
        f"{_lat(float(peak['lat']))}, {_lon(float(peak['lon']))}"
    )

    mean_knots = float(facts["mean_speed_knots"])
    direction = facts.get("direction")
    direction_label = None
    if isinstance(direction, dict):
        direction_label = direction.get("label")
    if direction_label:
        closing_text = (
            f"Average: {_kt(mean_knots)} knots, mostly {direction_label}"
        )
    else:
        closing_text = f"Average: {_kt(mean_knots)} knots"

    beats: List[Tuple[float, str]] = [(0.0, opening), (peak_fraction, peak_text)]

    if max_beats >= 3:
        beats.append((0.9, closing_text))

    if max_beats >= 4:
        temp = facts.get("temperature")
        if isinstance(temp, dict):
            t_min = temp.get("min")
            t_max = temp.get("max")
            t_unit = temp.get("unit")
            if t_min is not None and t_max is not None and t_unit:
                try:
                    t_min_f = float(t_min)
                    t_max_f = float(t_max)
                except (TypeError, ValueError):
                    t_min_f = t_max_f = None  # type: ignore[assignment]
                if t_min_f is not None and t_max_f is not None:
                    if abs(t_max_f - t_min_f) >= 1.0:
                        noun = (
                            "Air temperatures" if kind == "wind"
                            else "Water temperatures"
                        )
                        mid_text = (
                            f"{noun} ranged from {t_min_f:.1f} to "
                            f"{t_max_f:.1f} {t_unit}"
                        )
                        beats.append((0.65, mid_text))

    beats.sort(key=lambda item: item[0])
    # Defensive clamp: fractions already in range, but guarantee it.
    return [(min(1.0, max(0.0, float(frac))), str(text))
            for frac, text in beats[:max_beats]]
