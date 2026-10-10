"""Running due jobs: :class:`Runner`, the run ledger, and executors.

The runner is deliberately dumb: every tick it asks "what is due?",
runs it through an **executor**, and appends the outcome to an
append-only JSONL ledger. What "running" means is the executor's
business — a callable ``(job, run_dir) -> dict`` — so this engine
never imports reel-studio, subprocess policy, or any render stack.
The reel-studio binding lives in reel-studio's ``studio/scheduler.py``.

Missed runs (the machine was off — this is a local scheduler, there
is no cloud): each job's ``catchup`` policy decides. ``"skip"``
(default) records one ``skipped`` entry and moves on — a missed weekly
reel is never backfilled. ``"run-once"`` executes it at the next tick,
then the schedule resumes.

Only one runner tick may hold the jobs directory at a time
(portable lockfile); a second concurrent tick raises
:class:`RunnerBusy` instead of double-running a job.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .cron import ScheduleError
from .jobs import Job, JobStore, job_timezone
from .schedules import (IntervalSchedule, OnceSchedule, _bind_once,
                        latest_occurrence, next_run, parse_schedule)

#: A run missed by more than this is "missed" (catchup policy applies);
#: inside it, the run just goes ahead.
CATCHUP_GRACE_SECONDS = 300

#: Executor protocol: (job, run_dir) -> result dict. May raise; the
#: runner records the failure and retries per the job's policy.
Executor = Callable[[Job, str], Dict[str, Any]]


class RunnerBusy(RuntimeError):
    """Another runner tick holds the jobs directory lock."""


@dataclass
class RunRecord:
    """One ledger entry: a single execution attempt of a job."""
    job: str
    attempt: int
    started_at: str
    finished_at: str
    status: str  # "ok" | "failed" | "skipped"
    run_dir: str
    detail: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job": self.job,
            "attempt": self.attempt,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "run_dir": self.run_dir,
            "detail": self.detail,
            "error": self.error,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RunRecord":
        return RunRecord(
            job=data["job"], attempt=data.get("attempt", 1),
            started_at=data.get("started_at", ""),
            finished_at=data.get("finished_at", ""),
            status=data.get("status", ""),
            run_dir=data.get("run_dir", ""),
            detail=data.get("detail") or {},
            error=data.get("error"))


class Runner:
    """Ticks a :class:`JobStore`, executing due jobs via ``executor``."""

    def __init__(self,
                 store: JobStore,
                 executor: Executor,
                 out_root: str,
                 sleeper: Callable[[float], None] = time.sleep) -> None:
        self.store = store
        self.executor = executor
        self.out_root = os.path.abspath(os.path.expanduser(out_root))
        self._sleep = sleeper

    # -- ledger -----------------------------------------------------------

    def _ledger_path(self) -> str:
        return os.path.join(self.store.path, "ledger.jsonl")

    def _append_ledger(self, record: RunRecord) -> None:
        os.makedirs(self.store.path, exist_ok=True)
        with open(self._ledger_path(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def ledger_tail(self, n: int = 20) -> List[RunRecord]:
        """The last ``n`` ledger entries (oldest first)."""
        path = self._ledger_path()
        if not os.path.isfile(path):
            return []
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        records = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                records.append(RunRecord.from_dict(json.loads(line)))
        return records

    # -- lock -------------------------------------------------------------

    def _lock_path(self) -> str:
        return os.path.join(self.store.path, ".runner.lock")

    def _acquire(self) -> int:
        os.makedirs(self.store.path, exist_ok=True)
        try:
            return os.open(self._lock_path(),
                           os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RunnerBusy(
                "another runner tick is already running "
                f"(lock: {self._lock_path()})") from None

    # -- tick -------------------------------------------------------------

    def tick(self, now: Optional[_dt.datetime] = None) -> List[RunRecord]:
        """Run every due job once. Returns the records written."""
        now = now or _dt.datetime.now(_dt.timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=_dt.timezone.utc)
        fd = self._acquire()
        try:
            return self._tick_locked(now)
        finally:
            os.close(fd)
            try:
                os.remove(self._lock_path())
            except OSError:
                pass

    def _tick_locked(self, now: _dt.datetime) -> List[RunRecord]:
        state = self.store.load_state()
        records: List[RunRecord] = []
        for job in self.store.list_jobs():
            if not job.enabled:
                continue
            record = self._maybe_run(job, state, now)
            if record is not None:
                records.append(record)
        self.store.save_state(state)
        return records

    def _maybe_run(self, job: Job,
                   state: Dict[str, Dict[str, Any]],
                   now: _dt.datetime) -> Optional[RunRecord]:
        tz = job_timezone(job)
        now_tz = now.astimezone(tz)
        schedule = parse_schedule(job.schedule)
        job_state = state.setdefault(job.name, {})
        # last_run stores the latest *handled* occurrence (ran or
        # skipped) — the anchor the cadence is measured from.
        last_occ = _parse_iso(job_state.get("last_run"))

        if isinstance(schedule, OnceSchedule):
            at = _bind_once(schedule, tz)
            if last_occ is not None:
                job_state["next_run"] = None
                return None  # already ran
            if at <= now_tz:
                return self._execute(job, job_state, now_tz, at)
            job_state["next_run"] = at.isoformat()
            return None

        if last_occ is None:
            # First sight of this job: assume the runner has been
            # watching since the grace window opened, so a just-passed
            # cron occurrence still fires, ancient ones don't, and a
            # new interval job's cadence anchors here.
            if isinstance(schedule, IntervalSchedule):
                last_occ = now_tz
            else:
                last_occ = (now_tz - _dt.timedelta(
                    seconds=CATCHUP_GRACE_SECONDS))
            job_state["last_run"] = last_occ.isoformat()

        occ = latest_occurrence(schedule, tz, now_tz, last_occ)
        if occ is None or occ <= last_occ:
            self._update_next(job, schedule, tz, now_tz, last_occ,
                              job_state)
            return None

        # occ is an unhandled occurrence at/before now.
        missed_by = (now_tz - occ).total_seconds()
        if missed_by <= CATCHUP_GRACE_SECONDS or \
                job.catchup == "run-once":
            return self._execute(job, job_state, now_tz, occ)
        # "skip": record the miss once, then resume the schedule —
        # a missed weekly reel is never backfilled.
        record = RunRecord(
            job=job.name, attempt=0,
            started_at=now_tz.isoformat(), finished_at=now_tz.isoformat(),
            status="skipped", run_dir="",
            detail={"missed_run": occ.isoformat(),
                    "missed_by_seconds": round(missed_by),
                    "catchup": "skip"})
        self._append_ledger(record)
        job_state["last_run"] = occ.isoformat()
        job_state["last_status"] = "skipped"
        self._update_next(job, schedule, tz, now_tz, occ, job_state)
        return record

    def _update_next(self, job: Job, schedule, tz: _dt.tzinfo,
                     now_tz: _dt.datetime,
                     last_occ: Optional[_dt.datetime],
                     job_state: Dict[str, Any]) -> None:
        nr = next_run(schedule, tz, now_tz, last_occ)
        job_state["next_run"] = nr.isoformat() if nr else None

    def _execute(self, job: Job,
                 job_state: Dict[str, Any],
                 now_tz: _dt.datetime,
                 occurrence: _dt.datetime) -> RunRecord:
        stamp = now_tz.strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(self.out_root, job.name, stamp)
        os.makedirs(run_dir, exist_ok=True)

        last_error: Optional[str] = None
        detail: Dict[str, Any] = {}
        attempts = 1 + job.retries
        for attempt in range(1, attempts + 1):
            started = _dt.datetime.now(_dt.timezone.utc).isoformat()
            try:
                detail = dict(self.executor(job, run_dir) or {})
                status = "ok"
                last_error = None
            except Exception as exc:  # noqa: BLE001 - the ledger records it
                status = "failed"
                last_error = f"{type(exc).__name__}: {exc}"
                detail = {"traceback": traceback.format_exc(limit=5)}
            finished = _dt.datetime.now(_dt.timezone.utc).isoformat()
            if status == "ok":
                break
            if attempt < attempts and job.retry_delay_seconds:
                self._sleep(job.retry_delay_seconds)

        record = RunRecord(
            job=job.name, attempt=attempt,
            started_at=started, finished_at=finished,
            status=status, run_dir=run_dir,
            detail=detail, error=last_error)
        self._append_ledger(record)
        # Anchor the cadence at the occurrence that just ran, not at
        # wall-clock time, so grace-window delays don't drift the
        # schedule.
        job_state["last_run"] = occurrence.isoformat()
        job_state["last_status"] = status
        self._update_next(job, parse_schedule(job.schedule),
                          job_timezone(job), now_tz, occurrence,
                          job_state)
        return record


def _parse_iso(value: Any) -> Optional[_dt.datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        at = _dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if at.tzinfo is None:
        at = at.replace(tzinfo=_dt.timezone.utc)
    return at
