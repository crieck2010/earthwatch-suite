"""``python -m schedx`` — manage jobs and tick the runner headlessly.

Commands:
    list                 list jobs with next run times
    next                 show the next run of every enabled job
    add FILE.json        add a job from a JSON file
    remove NAME          delete a job
    enable NAME          enable a job
    disable NAME         disable a job
    validate [FILE]      validate one file or every job in the dir
    run-once             tick the runner once (needs --executor)
    ledger [-n N]        show the last N ledger entries
    describe SCHEDULE    plain-words description of a schedule string

``--executor`` is a dotted path (``pkg.mod:func`` or ``pkg.mod.func``)
to an ``(job, run_dir) -> dict`` callable; builtin shorthands are
``null`` (dry-run) and ``shell`` (``settings.command``).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import json
import os
import sys
from typing import Callable

from .cron import ScheduleError
from .executors import null_executor, shell_executor
from .jobs import Job, JobStore, job_from_dict, job_to_dict, job_timezone
from .runner import Executor, Runner
from .schedules import describe_schedule, next_run, parse_schedule

_BUILTIN_EXECUTORS = {"null": null_executor, "shell": shell_executor}


def _resolve_executor(spec: str) -> Executor:
    if spec in _BUILTIN_EXECUTORS:
        return _BUILTIN_EXECUTORS[spec]
    target = spec.replace(":", ".")
    module_name, _, func_name = target.rpartition(".")
    if not module_name or not func_name:
        raise ScheduleError(
            f"bad executor {spec!r}: use 'null', 'shell', or a dotted "
            f"path like 'mymod:run_job'")
    module = importlib.import_module(module_name)
    func = getattr(module, func_name, None)
    if not callable(func):
        raise ScheduleError(
            f"executor {spec!r}: {func_name} is not callable")
    return func


def _jobs_dir(args: argparse.Namespace) -> str:
    return os.path.abspath(os.path.expanduser(
        args.jobs_dir or os.environ.get("SCHEDX_JOBS_DIR", "~/.schedx/jobs")))


def _out_root(args: argparse.Namespace) -> str:
    return os.path.abspath(os.path.expanduser(
        args.out_root or os.environ.get("SCHEDX_OUT_ROOT", "~/.schedx/runs")))


def _fmt_next(store: JobStore, job: Job,
              now: _dt.datetime) -> str:
    tz = job_timezone(job)
    schedule = parse_schedule(job.schedule)
    state = store.load_state().get(job.name, {})
    last_run = None
    if state.get("last_run"):
        try:
            last_run = _dt.datetime.fromisoformat(state["last_run"])
        except ValueError:
            last_run = None
    nr = next_run(schedule, tz, now.astimezone(tz), last_run)
    return nr.isoformat() if nr else "(none — schedule spent)"


def cmd_list(args: argparse.Namespace) -> int:
    store = JobStore(_jobs_dir(args))
    now = _dt.datetime.now(_dt.timezone.utc)
    jobs = store.list_jobs()
    if not jobs:
        print(f"no jobs in {_jobs_dir(args)}")
        return 0
    for job in jobs:
        flag = "" if job.enabled else " [disabled]"
        try:
            nxt = _fmt_next(store, job, now)
        except ScheduleError as exc:  # pragma: no cover - validated on load
            nxt = f"(invalid: {exc})"
        print(f"{job.name}{flag}\n  schedule: {job.schedule}\n"
              f"  next run: {nxt}")
    return 0


def cmd_next(args: argparse.Namespace) -> int:
    store = JobStore(_jobs_dir(args))
    now = _dt.datetime.now(_dt.timezone.utc)
    for job in store.list_jobs():
        if job.enabled:
            print(f"{job.name}: {_fmt_next(store, job, now)}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    store = JobStore(_jobs_dir(args))
    with open(args.file, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    job = job_from_dict(data, source=args.file)
    fpath = store.save_job(job)
    print(f"added job {job.name!r} -> {fpath}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    store = JobStore(_jobs_dir(args))
    store.delete_job(args.name)
    print(f"removed job {args.name!r}")
    return 0


def _toggle(args: argparse.Namespace, enabled: bool) -> int:
    store = JobStore(_jobs_dir(args))
    job = store.get_job(args.name)
    updated = job_from_dict({**job_to_dict(job), "enabled": enabled},
                            source=f"<toggle {args.name}>")
    store.save_job(updated)
    print(f"job {args.name!r} {'enabled' if enabled else 'disabled'}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            job_from_dict(json.load(fh), source=args.file)
        print(f"{args.file}: valid")
        return 0
    store = JobStore(_jobs_dir(args))
    jobs = store.list_jobs()  # raises naming the bad file
    print(f"{len(jobs)} job(s) valid in {_jobs_dir(args)}")
    return 0


def cmd_run_once(args: argparse.Namespace) -> int:
    if not args.executor:
        print("error: run-once needs --executor "
              "('null', 'shell', or a dotted path)",
              file=sys.stderr)
        return 2
    store = JobStore(_jobs_dir(args))
    runner = Runner(store, _resolve_executor(args.executor),
                    _out_root(args))
    records = runner.tick()
    if not records:
        print("no jobs due")
        return 0
    for record in records:
        extra = f": {record.error}" if record.error else ""
        print(f"{record.job}: {record.status} "
              f"(attempt {record.attempt}){extra}")
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    store = JobStore(_jobs_dir(args))
    runner = Runner(store, null_executor, _out_root(args))
    for record in runner.ledger_tail(args.n):
        print(f"{record.finished_at} {record.job} {record.status} "
              f"attempt={record.attempt} dir={record.run_dir or '-'}")
        if record.error:
            print(f"    error: {record.error}")
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    print(describe_schedule(args.schedule))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="schedx", description="Scheduled generation jobs (survey-schedule)")
    parser.add_argument("--jobs-dir", default=None,
                        help="jobs directory (default: $SCHEDX_JOBS_DIR or "
                             "~/.schedx/jobs)")
    parser.add_argument("--out-root", default=None,
                        help="run output root (default: $SCHEDX_OUT_ROOT or "
                             "~/.schedx/runs)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list jobs with next run times")
    sub.add_parser("next", help="next run of every enabled job")

    p = sub.add_parser("add", help="add a job from a JSON file")
    p.add_argument("file")

    p = sub.add_parser("remove", help="delete a job")
    p.add_argument("name")

    p = sub.add_parser("enable", help="enable a job")
    p.add_argument("name")

    p = sub.add_parser("disable", help="disable a job")
    p.add_argument("name")

    p = sub.add_parser("validate", help="validate job file(s)")
    p.add_argument("file", nargs="?")

    p = sub.add_parser("run-once", help="tick the runner once")
    p.add_argument("--executor", default=None,
                   help="'null', 'shell', or dotted path to a callable")

    p = sub.add_parser("ledger", help="show recent ledger entries")
    p.add_argument("-n", type=int, default=20)

    p = sub.add_parser("describe", help="describe a schedule string")
    p.add_argument("schedule")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            return cmd_list(args)
        if args.command == "next":
            return cmd_next(args)
        if args.command == "add":
            return cmd_add(args)
        if args.command == "remove":
            return cmd_remove(args)
        if args.command == "enable":
            return _toggle(args, True)
        if args.command == "disable":
            return _toggle(args, False)
        if args.command == "validate":
            return cmd_validate(args)
        if args.command == "run-once":
            return cmd_run_once(args)
        if args.command == "ledger":
            return cmd_ledger(args)
        if args.command == "describe":
            return cmd_describe(args)
    except ScheduleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    raise AssertionError("unreachable")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
