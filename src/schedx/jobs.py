"""Scheduled jobs: the :class:`Job` model and the :class:`JobStore`.

A job is data: a reel description plus the settings a run needs, a
schedule expression, and run policies. Jobs live as JSON files — one
per job — in a jobs directory, so they are diffable, backup-friendly,
and editable by hand. :class:`JobStore` validates every file on load
and names the file when one is bad.

The schedule DSL (see :mod:`schedx.schedules`):

* ``cron:<5 fields>`` — e.g. ``cron:0 6 * * mon`` (Mondays 06:00)
* ``every:<n><s|m|h|d|w>`` — e.g. ``every:6h``, ``every:1w``
* ``once:<ISO datetime>`` — e.g. ``once:2026-10-01T06:00`` (in the
  job's timezone)

``settings`` is a plain dict handed to the executor untouched — for
the reel-studio binding these are the run settings (``motion``,
``audio_path``, ``story_captions``, ``cmap``, ``platform``,
``style_preset``, ``stride_days``, ``stride_hours``).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .cron import ScheduleError, parse_cron
from .schedules import parse_schedule

_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

#: Missed-run policies. ``skip`` (default): a run missed while the
#: runner was down is skipped, never backfilled. ``run-once``: the
#: next tick runs it once, then the schedule resumes.
CATCHUP_POLICIES = ("skip", "run-once")


@dataclass(frozen=True)
class Job:
    """One scheduled generation job.

    Fields:
        name: slug, e.g. ``"weekly-sst"``.
        description: the reel description, parsed by survey-viz at
            run time (exactly what step 1 of reel-studio accepts).
        schedule: schedule DSL string (see module docstring).
        settings: run settings dict, passed to the executor untouched.
        enabled: disabled jobs are listed but never run.
        timezone: IANA name (``"America/New_York"``) or ``"local"``
            (the machine's local zone).
        catchup: ``"skip"`` or ``"run-once"`` for missed runs.
        retries: immediate in-run retries on executor failure (0-5).
        retry_delay_seconds: wait between retries.
        tags: free-form tags for grouping.
    """

    name: str
    description: str
    schedule: str
    settings: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True
    timezone: str = "local"
    catchup: str = "skip"
    retries: int = 0
    retry_delay_seconds: int = 60
    tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        errors = _validate(self)
        if errors:
            raise ScheduleError(
                f"job {self.name!r} is invalid:\n  - "
                + "\n  - ".join(errors))


def _validate(job: Job) -> List[str]:
    errors: List[str] = []
    if not isinstance(job.name, str) or not _NAME_RE.match(job.name):
        errors.append(
            f"name: {job.name!r} must be a lowercase hyphenated slug")
    if not isinstance(job.description, str) or not job.description.strip():
        errors.append("description: must be a non-empty string")
    try:
        parse_schedule(job.schedule)
    except ScheduleError as exc:
        errors.append(f"schedule: {exc}")
    if not isinstance(job.settings, Mapping):
        errors.append("settings: must be a mapping")
    if not isinstance(job.enabled, bool):
        errors.append(f"enabled: expected bool, got {job.enabled!r}")
    try:
        _tzinfo(job.timezone)
    except ScheduleError as exc:
        errors.append(f"timezone: {exc}")
    if job.catchup not in CATCHUP_POLICIES:
        errors.append(
            f"catchup: {job.catchup!r} not in {CATCHUP_POLICIES}")
    if not isinstance(job.retries, int) or not (0 <= job.retries <= 5):
        errors.append(f"retries: expected int 0-5, got {job.retries!r}")
    if (not isinstance(job.retry_delay_seconds, int)
            or job.retry_delay_seconds < 0):
        errors.append("retry_delay_seconds: expected int >= 0, "
                      f"got {job.retry_delay_seconds!r}")
    return errors


def validate_job(job: Job) -> List[str]:
    """Non-raising validation; empty list means valid."""
    return _validate(job)


def _tzinfo(name: str) -> _dt.tzinfo:
    """Resolve a job timezone name to a tzinfo."""
    if name == "local":
        return _dt.datetime.now().astimezone().tzinfo
    try:
        from zoneinfo import ZoneInfo
    except ImportError:  # pragma: no cover - py3.9 fallback
        raise ScheduleError(
            f"timezone: zoneinfo unavailable, use 'local' (got {name!r})")
    try:
        return ZoneInfo(name)
    except Exception:
        raise ScheduleError(
            f"timezone: unknown IANA zone {name!r} (or use 'local')") from None


def job_timezone(job: Job) -> _dt.tzinfo:
    """The tzinfo a job's schedule is evaluated in."""
    return _tzinfo(job.timezone)


# --- JSON -------------------------------------------------------------------

_JOB_FIELDS = ("name", "description", "schedule", "settings", "enabled",
               "timezone", "catchup", "retries", "retry_delay_seconds",
               "tags")


def job_from_dict(data: Mapping[str, Any], source: str = "<dict>") -> Job:
    """Build a :class:`Job` from a plain dict (parsed JSON)."""
    if not isinstance(data, Mapping):
        raise ScheduleError(f"{source}: job must be a JSON object")
    unknown = sorted(set(data) - set(_JOB_FIELDS))
    if unknown:
        raise ScheduleError(
            f"{source}: unknown job field(s): {', '.join(unknown)} "
            f"(known: {', '.join(_JOB_FIELDS)})")
    try:
        return Job(
            name=data.get("name", ""),
            description=data.get("description", ""),
            schedule=data.get("schedule", ""),
            settings=(data.get("settings")
                      if data.get("settings") is not None else {}),
            enabled=data.get("enabled", True),
            timezone=data.get("timezone", "local"),
            catchup=data.get("catchup", "skip"),
            retries=data.get("retries", 0),
            retry_delay_seconds=data.get("retry_delay_seconds", 60),
            tags=tuple(data.get("tags", ()) or ()),
        )
    except ScheduleError as exc:
        raise ScheduleError(f"{source}: {exc}") from None


def job_to_dict(job: Job) -> Dict[str, Any]:
    """Serialize a :class:`Job` to a JSON-safe dict."""
    return {
        "name": job.name,
        "description": job.description,
        "schedule": job.schedule,
        "settings": dict(job.settings),
        "enabled": job.enabled,
        "timezone": job.timezone,
        "catchup": job.catchup,
        "retries": job.retries,
        "retry_delay_seconds": job.retry_delay_seconds,
        "tags": list(job.tags),
    }


# --- store ------------------------------------------------------------------

class JobStore:
    """A directory of ``<name>.json`` job files.

    The directory is created on first write. Every read validates;
    a bad file raises :class:`ScheduleError` naming the file.
    """

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(os.path.expanduser(path))

    def _fpath(self, name: str) -> str:
        return os.path.join(self.path, f"{name}.json")

    def list_jobs(self) -> List[Job]:
        """All jobs, sorted by name."""
        if not os.path.isdir(self.path):
            return []
        jobs = []
        for fname in sorted(os.listdir(self.path)):
            if not fname.endswith(".json") or fname.startswith("."):
                continue
            jobs.append(self.get_job(fname[:-5]))
        return jobs

    def get_job(self, name: str) -> Job:
        fpath = self._fpath(name)
        if not os.path.isfile(fpath):
            raise ScheduleError(f"no such job: {name!r}")
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            raise ScheduleError(f"{fpath}: cannot parse JSON: {exc}") from None
        job = job_from_dict(data, source=fpath)
        if job.name != name:
            raise ScheduleError(
                f"{fpath}: file name implies job {name!r} but the file "
                f"declares {job.name!r}")
        return job

    def save_job(self, job: Job) -> str:
        """Validate and write ``job``; returns the file path."""
        if not isinstance(job, Job):
            raise ScheduleError(
                f"save_job: expected a Job, got {type(job).__name__}")
        os.makedirs(self.path, exist_ok=True)
        fpath = self._fpath(job.name)
        tmp = fpath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(job_to_dict(job), fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, fpath)  # atomic write
        return fpath

    def delete_job(self, name: str) -> None:
        fpath = self._fpath(name)
        if not os.path.isfile(fpath):
            raise ScheduleError(f"no such job: {name!r}")
        os.remove(fpath)

    # -- run state (separate from definitions) --------------------------------

    def _state_path(self) -> str:
        return os.path.join(self.path, ".state.json")

    def load_state(self) -> Dict[str, Dict[str, Any]]:
        """``{job_name: {last_run, last_status, next_run}}`` (may be empty)."""
        sp = self._state_path()
        if not os.path.isfile(sp):
            return {}
        try:
            with open(sp, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def save_state(self, state: Dict[str, Dict[str, Any]]) -> None:
        os.makedirs(self.path, exist_ok=True)
        tmp = self._state_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, self._state_path())
