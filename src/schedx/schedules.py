"""Schedule DSL parsing and due-time computation.

DSL (validated by :func:`parse_schedule`):

* ``cron:<5 fields>`` — e.g. ``cron:0 6 * * mon``
* ``every:<n><s|m|h|d|w>`` — e.g. ``every:30m``, ``every:6h``, ``every:1w``
* ``once:<ISO datetime>`` — e.g. ``once:2026-10-01T06:00`` (naive =
  the job's timezone; an offset is honored)

:func:`next_run` answers "when should this job run next, given what
already happened" — the single function the runner ticks on.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from typing import Optional, Union

from .cron import CronSpec, ScheduleError, next_cron_occurrence, parse_cron

_EVERY_RE = re.compile(r"^every:(\d+)([smhdw])$")
_EVERY_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


@dataclass(frozen=True)
class CronSchedule:
    spec: CronSpec


@dataclass(frozen=True)
class IntervalSchedule:
    seconds: int


@dataclass(frozen=True)
class OnceSchedule:
    at: _dt.datetime  # tz-aware after binding to the job's tz


Schedule = Union[CronSchedule, IntervalSchedule, OnceSchedule]


def parse_schedule(text: str) -> Schedule:
    """Parse a schedule DSL string; raise :class:`ScheduleError`."""
    if not isinstance(text, str) or not text.strip():
        raise ScheduleError(
            "schedule: must be one of 'cron:<5 fields>', "
            "'every:<n>s|m|h|d|w', 'once:<ISO datetime>'")
    text = text.strip()
    if text.startswith("cron:"):
        return CronSchedule(parse_cron(text[len("cron:"):]))
    match = _EVERY_RE.match(text)
    if match:
        amount, unit = int(match.group(1)), match.group(2)
        if amount < 1:
            raise ScheduleError(
                f"schedule: interval must be >= 1, got {text!r}")
        return IntervalSchedule(amount * _EVERY_SECONDS[unit])
    if text.startswith("once:"):
        raw = text[len("once:"):]
        try:
            at = _dt.datetime.fromisoformat(raw)
        except ValueError:
            raise ScheduleError(
                f"schedule: bad ISO datetime in {text!r} "
                f"(e.g. 'once:2026-10-01T06:00')") from None
        return OnceSchedule(at)
    raise ScheduleError(
        f"schedule: unknown kind in {text!r} — use 'cron:...', "
        f"'every:...' or 'once:...'")


def _bind_once(schedule: OnceSchedule,
               tz: _dt.tzinfo) -> _dt.datetime:
    at = schedule.at
    if at.tzinfo is None:
        return at.replace(tzinfo=tz)
    return at.astimezone(tz)


def next_run(schedule: Schedule,
             tz: _dt.tzinfo,
             after: _dt.datetime,
             last_run: Optional[_dt.datetime] = None) -> Optional[_dt.datetime]:
    """Next scheduled time strictly after ``after`` (tz-aware).

    * cron: the next cron occurrence after ``after``.
    * interval: ``last_run + interval`` when ``last_run`` is known and
      in the future of ``after``; otherwise the next interval boundary
      after ``after`` (anchored at the Unix epoch — deterministic).
    * once: the one time, or None when it already passed.

    Returns None when there is no future occurrence (a spent ``once``,
    or a cron with no match in a year).
    """
    if after.tzinfo is None:
        raise ScheduleError("next_run: 'after' must be tz-aware")
    after = after.astimezone(tz)
    if isinstance(schedule, CronSchedule):
        return next_cron_occurrence(schedule.spec, after)
    if isinstance(schedule, OnceSchedule):
        at = _bind_once(schedule, tz)
        return at if at > after else None
    # IntervalSchedule
    step = _dt.timedelta(seconds=schedule.seconds)
    if last_run is not None:
        last_run = last_run.astimezone(tz)
        candidate = last_run + step
        if candidate > after:
            return candidate
    # Anchor at the epoch for determinism: the first boundary > after.
    epoch = _dt.datetime(1970, 1, 1, tzinfo=tz)
    elapsed = (after - epoch).total_seconds()
    n = int(elapsed // schedule.seconds) + 1
    return epoch + _dt.timedelta(seconds=n * schedule.seconds)


def describe_schedule(text: str) -> str:
    """Plain-words description of a schedule DSL string."""
    schedule = parse_schedule(text)
    if isinstance(schedule, CronSchedule):
        from .cron import describe_cron
        return "Cron " + describe_cron(schedule.spec.raw)
    if isinstance(schedule, IntervalSchedule):
        s = schedule.seconds
        for unit, name in ((604800, "week"), (86400, "day"),
                           (3600, "hour"), (60, "minute"), (1, "second")):
            if s % unit == 0:
                n = s // unit
                plural = "s" if n != 1 else ""
                return f"Every {n} {name}{plural}"
    at = schedule.at
    return f"Once at {at.isoformat()}"


def latest_occurrence(schedule: Schedule,
                      tz: _dt.tzinfo,
                      now: _dt.datetime,
                      last_occurrence: Optional[_dt.datetime] = None
                      ) -> Optional[_dt.datetime]:
    """Latest scheduled time at or before ``now`` (tz-aware).

    * cron: the most recent matching minute at/before ``now``.
    * interval: ``last_occurrence + k*step`` for the largest ``k >= 1``
      still at/before ``now``; None when ``last_occurrence`` is None
      or no full interval has elapsed.
    * once: the one time, when at/before ``now``.

    The caller decides whether that occurrence was already handled
    (``occurrence <= last_occurrence``) or missed.
    """
    from .cron import prev_cron_occurrence
    if now.tzinfo is None:
        raise ScheduleError("latest_occurrence: 'now' must be tz-aware")
    now = now.astimezone(tz)
    if isinstance(schedule, CronSchedule):
        return prev_cron_occurrence(schedule.spec, now)
    if isinstance(schedule, OnceSchedule):
        at = _bind_once(schedule, tz)
        return at if at <= now else None
    if last_occurrence is None:
        return None
    base = last_occurrence.astimezone(tz)
    elapsed = (now - base).total_seconds()
    k = int(elapsed // schedule.seconds)
    if k < 1:
        return None
    return base + _dt.timedelta(seconds=k * schedule.seconds)
