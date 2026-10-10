"""survey-schedule: scheduled generation jobs for the survey/reel stack.

A ``Job`` is data — a reel description, run settings, and a schedule
expression — stored as JSON in a jobs directory. :class:`Runner`
ticks the directory, executes due jobs through an executor callable,
and appends every outcome to a JSONL ledger.

The engine never imports survey-viz, survey-layout, survey-style, or
reel-studio: the executor is duck-typed ``(job, run_dir) -> dict``,
and the reel-studio binding lives in reel-studio's
``studio/scheduler.py``.
"""

from .cron import (CronSpec, ScheduleError, describe_cron,
                   next_cron_occurrence, parse_cron,
                   prev_cron_occurrence)
from .executors import null_executor, shell_executor
from .jobs import (CATCHUP_POLICIES, Job, JobStore, job_from_dict,
                   job_to_dict, job_timezone, validate_job)
from .runner import Executor, RunRecord, Runner, RunnerBusy
from .schedules import (CronSchedule, IntervalSchedule, OnceSchedule,
                        describe_schedule, latest_occurrence, next_run,
                        parse_schedule)

__version__ = "0.1.0"
SCHEDULE_SCHEMA_VERSION = "1.0"

__all__ = [
    "SCHEDULE_SCHEMA_VERSION",
    "CATCHUP_POLICIES",
    "CronSchedule",
    "CronSpec",
    "Executor",
    "IntervalSchedule",
    "Job",
    "JobStore",
    "OnceSchedule",
    "RunRecord",
    "Runner",
    "RunnerBusy",
    "ScheduleError",
    "describe_cron",
    "describe_schedule",
    "job_from_dict",
    "job_to_dict",
    "job_timezone",
    "latest_occurrence",
    "next_cron_occurrence",
    "next_run",
    "null_executor",
    "parse_cron",
    "parse_schedule",
    "prev_cron_occurrence",
    "shell_executor",
    "validate_job",
]
