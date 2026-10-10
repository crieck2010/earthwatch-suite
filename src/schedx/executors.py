"""Ready-made executors for the :class:`schedx.runner.Runner`.

An executor is ``(job, run_dir) -> dict``. These two cover testing
and shelling out; the reel-studio binding is its own function in
reel-studio's ``studio/scheduler.py`` (this engine never imports
reel-studio).
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict

from .cron import ScheduleError
from .jobs import Job


def null_executor(job: Job, run_dir: str) -> Dict[str, Any]:
    """Dry-run executor: records success without doing anything.

    Useful for testing schedules end to end. Never fails.
    """
    marker = os.path.join(run_dir, "DRY_RUN.txt")
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write(f"dry run of job {job.name}\n")
    return {"status": "ok", "dry_run": True, "run_dir": run_dir}


def shell_executor(job: Job, run_dir: str) -> Dict[str, Any]:
    """Run ``job.settings["command"]`` in a shell.

    The command is a format template with ``{job}``, ``{name}``,
    ``{run_dir}`` and ``{description}`` available. Its working
    directory is ``run_dir``. Non-zero exit raises
    :class:`~schedx.cron.ScheduleError` so the runner records a
    failed attempt (and retries per the job's policy).
    """
    command = (job.settings or {}).get("command")
    if not command or not isinstance(command, str):
        raise ScheduleError(
            f"job {job.name!r}: shell executor needs "
            f"settings.command (a string)")
    rendered = command.format(job=job.name, name=job.name,
                              run_dir=run_dir,
                              description=job.description)
    proc = subprocess.run(rendered, shell=True, cwd=run_dir,
                          capture_output=True, text=True, timeout=86400)
    tail = (proc.stdout or "")[-2000:] + (proc.stderr or "")[-2000:]
    if proc.returncode != 0:
        raise ScheduleError(
            f"job {job.name!r}: command exited {proc.returncode}\n{tail}")
    return {"status": "ok", "returncode": proc.returncode,
            "output_tail": tail[-2000:]}
