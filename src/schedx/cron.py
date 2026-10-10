"""Cron expression parsing and next-occurrence computation (stdlib-only).

Supports the standard 5-field cron ``minute hour dom month dow``:

* ``*`` — every value
* ``*/n`` — every n units
* ``a-b`` — inclusive range
* ``a,b,c`` — list
* combinations: ``1-10/2,30,45``
* month and weekday names: ``jan``–``dec``, ``mon``–``sun`` (case-insensitive)

Day-of-month vs day-of-week follow the classic cron rule: when both
fields are restricted (not ``*``), a day matches if *either* matches;
when one is ``*``, only the other decides.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from typing import List, Optional, Set

_MONTH_NAMES = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_DOW_NAMES = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5,
              "sat": 6, "sun": 0}

# (min, max) per field: minute, hour, dom, month, dow (dow 0=Monday..6=Sunday;
# 7 is accepted as an alias for Sunday on input).


class ScheduleError(ValueError):
    """A schedule expression is invalid."""


def _expand_names(field: str, names: dict) -> str:
    def _sub(match: "re.Match") -> str:
        return str(names[match.group(0).lower()])
    return re.sub(r"[a-zA-Z]+", _sub, field)


def _parse_field(text: str, lo: int, hi: int, field_name: str) -> Set[int]:
    values: Set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise ScheduleError(
                f"cron: empty element in {field_name} field {text!r}")
        step = 1
        if "/" in part:
            part, step_text = part.split("/", 1)
            try:
                step = int(step_text)
            except ValueError:
                raise ScheduleError(
                    f"cron: bad step {step_text!r} in {field_name} field") from None
            if step < 1:
                raise ScheduleError(
                    f"cron: step must be >= 1 in {field_name} field")
        if part == "*":
            start, end = lo, hi
        elif "-" in part:
            a, b = part.split("-", 1)
            try:
                start, end = int(a), int(b)
            except ValueError:
                raise ScheduleError(
                    f"cron: bad range {part!r} in {field_name} field") from None
        else:
            try:
                start = end = int(part)
            except ValueError:
                raise ScheduleError(
                    f"cron: bad value {part!r} in {field_name} field") from None
        if not (lo <= start <= hi and lo <= end <= hi):
            raise ScheduleError(
                f"cron: {field_name} value out of range [{lo}-{hi}]: {part!r}")
        if start > end:
            raise ScheduleError(
                f"cron: reversed range {part!r} in {field_name} field")
        values.update(range(start, end + 1, step))
    return values


@dataclass(frozen=True)
class CronSpec:
    """A parsed 5-field cron expression."""
    minute: frozenset
    hour: frozenset
    dom: frozenset
    month: frozenset
    dow: frozenset  # Monday=0..Sunday=6
    dom_star: bool
    dow_star: bool
    raw: str

    def matches(self, moment: _dt.datetime) -> bool:
        """Whether ``moment`` (tz-aware) matches this schedule."""
        if moment.minute not in self.minute:
            return False
        if moment.hour not in self.hour:
            return False
        if moment.month not in self.month:
            return False
        dom_match = moment.day in self.dom
        dow_match = moment.weekday() in self.dow
        if self.dom_star and self.dow_star:
            day_ok = True
        elif self.dom_star:
            day_ok = dow_match
        elif self.dow_star:
            day_ok = dom_match
        else:
            day_ok = dom_match or dow_match
        return day_ok


def parse_cron(expr: str) -> CronSpec:
    """Parse a 5-field cron expression; raise :class:`ScheduleError`."""
    if not isinstance(expr, str):
        raise ScheduleError(f"cron: expression must be a string, got {expr!r}")
    fields = expr.strip().split()
    if len(fields) != 5:
        raise ScheduleError(
            f"cron: expected 5 fields (minute hour dom month dow), "
            f"got {len(fields)} in {expr!r}")
    minute_f, hour_f, dom_f, month_f, dow_f = fields
    month_f = _expand_names(month_f, _MONTH_NAMES)
    dow_f = _expand_names(dow_f, _DOW_NAMES)
    minute = _parse_field(minute_f, 0, 59, "minute")
    hour = _parse_field(hour_f, 0, 23, "hour")
    dom = _parse_field(dom_f, 1, 31, "day of month")
    month = _parse_field(month_f, 1, 12, "month")
    dow = _parse_field(dow_f, 0, 7, "day of week")
    # Cron numbers: 0 and 7 are Sunday, 1 Monday .. 6 Saturday.
    # Internally Monday=0 .. Sunday=6 (datetime.weekday()).
    dow = {6 if d in (0, 7) else d - 1 for d in dow}
    return CronSpec(
        minute=frozenset(minute), hour=frozenset(hour),
        dom=frozenset(dom), month=frozenset(month),
        dow=frozenset(dow),
        dom_star=dom_f.strip() == "*", dow_star=dow_f.strip() == "*",
        raw=expr.strip())


def next_cron_occurrence(spec: CronSpec,
                         after: _dt.datetime,
                         limit_days: int = 366) -> Optional[_dt.datetime]:
    """Next datetime strictly after ``after`` matching ``spec``.

    ``after`` must be tz-aware. Returns None when nothing matches
    within ``limit_days`` (e.g. Feb 30th).
    """
    if after.tzinfo is None:
        raise ScheduleError("next_cron_occurrence: 'after' must be tz-aware")
    candidate = (after.replace(second=0, microsecond=0)
                 + _dt.timedelta(minutes=1))
    deadline = after + _dt.timedelta(days=limit_days)
    while candidate <= deadline:
        if spec.matches(candidate):
            return candidate
        candidate += _dt.timedelta(minutes=1)
    return None


def prev_cron_occurrence(spec: CronSpec,
                         now: _dt.datetime,
                         limit_days: int = 370) -> Optional[_dt.datetime]:
    """Latest datetime at or before ``now`` matching ``spec``.

    ``now`` must be tz-aware. Returns None when nothing matches
    within ``limit_days`` back.
    """
    if now.tzinfo is None:
        raise ScheduleError("prev_cron_occurrence: 'now' must be tz-aware")
    candidate = now.replace(second=0, microsecond=0)
    deadline = now - _dt.timedelta(days=limit_days)
    while candidate >= deadline:
        if spec.matches(candidate):
            return candidate
        candidate -= _dt.timedelta(minutes=1)
    return None


def describe_cron(expr: str) -> str:
    """Plain-words description of a cron expression (best effort)."""
    spec = parse_cron(expr)
    parts: List[str] = []

    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    time_part = None
    minutes = sorted(spec.minute)
    hours = sorted(spec.hour)
    if len(minutes) == 1 and len(hours) == 1:
        time_part = f"at {hours[0]:02d}:{minutes[0]:02d}"
    elif minutes == list(range(0, 60)):
        time_part = "every minute" if hours == list(range(0, 24)) else \
            f"every minute during hour(s) {', '.join(map(str, hours))}"
    else:
        time_part = (f"at minute(s) {', '.join(map(str, minutes))} "
                     f"of hour(s) {', '.join(map(str, hours))}")
    parts.append(time_part)

    if not spec.dom_star or not spec.dow_star:
        if not spec.dow_star:
            days = ", ".join(dow_names[d] for d in sorted(spec.dow))
            parts.append(f"on {days}")
        if not spec.dom_star:
            parts.append(f"on day(s) {', '.join(map(str, sorted(spec.dom)))} "
                         "of the month")
    months = sorted(spec.month)
    if months != list(range(1, 13)):
        parts.append(f"in {', '.join(month_names[m - 1] for m in months)}")
    return " ".join(parts)
