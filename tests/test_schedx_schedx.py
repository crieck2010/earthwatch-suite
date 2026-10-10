"""Tests for survey-schedule (schedx). Run with pytest."""

import datetime as dt
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schedx import (CATCHUP_POLICIES, Job, JobStore, RunRecord, Runner,
                    RunnerBusy, ScheduleError, describe_cron,
                    describe_schedule, job_from_dict, job_to_dict,
                    next_run, null_executor, parse_cron, parse_schedule,
                    shell_executor, validate_job)
from schedx.runner import CATCHUP_GRACE_SECONDS

UTC = dt.timezone.utc


def utc(*args):
    return dt.datetime(*args, tzinfo=UTC)


def make_job(**over):
    base = dict(name="weekly-sst",
                description="weekly sea surface temperature of Lake Michigan",
                schedule="cron:0 6 * * mon",
                timezone="UTC")
    base.update(over)
    return Job(**base)


# --- cron parsing -----------------------------------------------------------

class TestCronParsing:
    def test_star(self):
        spec = parse_cron("* * * * *")
        assert spec.minute == frozenset(range(60))

    def test_step(self):
        spec = parse_cron("*/15 * * * *")
        assert spec.minute == frozenset([0, 15, 30, 45])

    def test_range_and_list(self):
        spec = parse_cron("0 9-17 * * 1,3,5")
        assert spec.hour == frozenset(range(9, 18))
        assert spec.dow == frozenset([0, 2, 4])

    def test_names(self):
        spec = parse_cron("0 6 * jan mon")
        assert spec.month == frozenset([1])
        assert spec.dow == frozenset([0])

    def test_sunday_alias_7(self):
        assert parse_cron("0 6 * * 7").dow == frozenset([6])

    def test_bad_field_count(self):
        with pytest.raises(ScheduleError):
            parse_cron("0 6 * *")

    def test_out_of_range(self):
        with pytest.raises(ScheduleError):
            parse_cron("99 * * * *")

    def test_reversed_range(self):
        with pytest.raises(ScheduleError):
            parse_cron("10-5 * * * *")

    def test_bad_step(self):
        with pytest.raises(ScheduleError):
            parse_cron("*/0 * * * *")

    def test_next_weekly(self):
        # Sunday 2026-09-27 12:00 UTC -> Monday 2026-09-28 06:00 UTC
        spec = parse_cron("0 6 * * mon")
        nxt = spec_next(spec, utc(2026, 9, 27, 12, 0))
        assert nxt == utc(2026, 9, 28, 6, 0)

    def test_next_same_day_later(self):
        spec = parse_cron("30 6 * * *")
        assert spec_next(spec, utc(2026, 9, 28, 5, 0)) == \
            utc(2026, 9, 28, 6, 30)

    def test_next_same_day_passed_goes_tomorrow(self):
        spec = parse_cron("30 6 * * *")
        assert spec_next(spec, utc(2026, 9, 28, 7, 0)) == \
            utc(2026, 9, 29, 6, 30)

    def test_dom_dow_or_semantics(self):
        # 13th of month OR Friday
        spec = parse_cron("0 12 13 * fri")
        # 2026-10-02 is a Friday -> matches
        assert spec_next(spec, utc(2026, 10, 1, 0, 0)) == \
            utc(2026, 10, 2, 12, 0)

    def test_dom_only_when_dow_star(self):
        spec = parse_cron("0 12 13 * *")
        assert spec_next(spec, utc(2026, 10, 1, 0, 0)) == \
            utc(2026, 10, 13, 12, 0)

    def test_impossible_returns_none(self):
        assert spec_next(parse_cron("0 0 30 2 *"),
                         utc(2026, 1, 1)) is None

    def test_describe(self):
        text = describe_cron("0 6 * * mon")
        assert "06:00" in text and "Mon" in text


def spec_next(spec, after):
    from schedx import next_cron_occurrence
    return next_cron_occurrence(spec, after)


# --- schedule DSL -----------------------------------------------------------

class TestScheduleDSL:
    def test_every(self):
        s = parse_schedule("every:6h")
        assert s.seconds == 21600

    def test_every_bad_unit(self):
        with pytest.raises(ScheduleError):
            parse_schedule("every:6x")

    def test_every_zero(self):
        with pytest.raises(ScheduleError):
            parse_schedule("every:0h")

    def test_once_naive_binds_job_tz(self):
        s = parse_schedule("once:2026-10-01T06:00")
        tz = dt.timezone(dt.timedelta(hours=-4), "EDT")
        nr = next_run(s, tz, utc(2026, 9, 28, 0, 0))
        assert nr == dt.datetime(2026, 10, 1, 6, 0, tzinfo=tz)

    def test_once_spent(self):
        s = parse_schedule("once:2026-01-01T06:00")
        assert next_run(s, UTC, utc(2026, 9, 28)) is None

    def test_once_with_offset(self):
        s = parse_schedule("once:2026-10-01T06:00+02:00")
        nr = next_run(s, UTC, utc(2026, 9, 28))
        assert nr == utc(2026, 10, 1, 4, 0)

    def test_unknown_kind(self):
        with pytest.raises(ScheduleError):
            parse_schedule("sometimes")

    def test_cron_kind(self):
        s = parse_schedule("cron:0 6 * * *")
        assert next_run(s, UTC, utc(2026, 9, 28, 7, 0)) == \
            utc(2026, 9, 29, 6, 0)

    def test_interval_from_last_run(self):
        s = parse_schedule("every:1h")
        last = utc(2026, 9, 28, 6, 0)
        assert next_run(s, UTC, utc(2026, 9, 28, 6, 30),
                        last) == utc(2026, 9, 28, 7, 0)

    def test_describe_every(self):
        assert describe_schedule("every:1w") == "Every 1 week"
        assert describe_schedule("every:30m") == "Every 30 minutes"

    def test_describe_once(self):
        assert "Once at" in describe_schedule("once:2026-10-01T06:00")


# --- job validation ---------------------------------------------------------

class TestJobValidation:
    def test_valid(self):
        assert validate_job(make_job()) == []

    def test_bad_name(self):
        with pytest.raises(ScheduleError):
            make_job(name="Bad Name!")

    def test_empty_description(self):
        with pytest.raises(ScheduleError):
            make_job(description="   ")

    def test_bad_schedule(self):
        with pytest.raises(ScheduleError):
            make_job(schedule="weekly-ish")

    def test_bad_timezone(self):
        with pytest.raises(ScheduleError):
            make_job(timezone="Mars/Olympus")

    def test_bad_catchup(self):
        with pytest.raises(ScheduleError):
            make_job(catchup="eventually")

    def test_bad_retries(self):
        with pytest.raises(ScheduleError):
            make_job(retries=9)

    def test_settings_not_mapping(self):
        with pytest.raises(ScheduleError):
            make_job(settings=["nope"])

    def test_unknown_field_in_dict(self):
        with pytest.raises(ScheduleError):
            job_from_dict({"name": "x", "bogus": 1}, source="t.json")

    def test_round_trip(self):
        job = make_job(tags=("ocean",), retries=2)
        assert job_from_dict(job_to_dict(job)) == job


# --- store ------------------------------------------------------------------

class TestJobStore:
    def test_save_list_get_delete(self, tmp_path):
        store = JobStore(str(tmp_path / "jobs"))
        job = make_job()
        fpath = store.save_job(job)
        assert os.path.isfile(fpath)
        assert [j.name for j in store.list_jobs()] == ["weekly-sst"]
        assert store.get_job("weekly-sst") == job
        store.delete_job("weekly-sst")
        assert store.list_jobs() == []

    def test_missing_job(self, tmp_path):
        with pytest.raises(ScheduleError):
            JobStore(str(tmp_path)).get_job("nope")

    def test_name_mismatch(self, tmp_path):
        store = JobStore(str(tmp_path))
        os.makedirs(store.path, exist_ok=True)
        data = job_to_dict(make_job(name="aaa"))
        with open(os.path.join(store.path, "bbb.json"), "w") as fh:
            json.dump(data, fh)
        with pytest.raises(ScheduleError, match="bbb.json"):
            store.get_job("bbb")

    def test_bad_json_names_file(self, tmp_path):
        store = JobStore(str(tmp_path))
        os.makedirs(store.path, exist_ok=True)
        bad = os.path.join(store.path, "bad.json")
        with open(bad, "w") as fh:
            fh.write("{not json")
        with pytest.raises(ScheduleError, match="bad.json"):
            store.get_job("bad")

    def test_state_round_trip(self, tmp_path):
        store = JobStore(str(tmp_path))
        assert store.load_state() == {}
        store.save_state({"a": {"last_run": "x"}})
        assert store.load_state() == {"a": {"last_run": "x"}}


# --- runner -----------------------------------------------------------------

def _runner(tmp_path, executor, **job_over):
    store = JobStore(str(tmp_path / "jobs"))
    job = make_job(**job_over)
    store.save_job(job)
    runner = Runner(store, executor, str(tmp_path / "runs"),
                    sleeper=lambda s: None)
    return store, runner, job


class TestRunner:
    def test_due_job_runs_and_ledgers(self, tmp_path):
        calls = []

        def exe(job, run_dir):
            calls.append((job.name, run_dir))
            os.makedirs(run_dir, exist_ok=True)
            return {"video": "out.mp4"}

        store, runner, job = _runner(tmp_path, exe)
        # Monday 2026-09-28 06:01 UTC: the 06:00 run is due (within grace).
        records = runner.tick(utc(2026, 9, 28, 6, 1))
        assert len(records) == 1
        rec = records[0]
        assert rec.status == "ok" and rec.job == "weekly-sst"
        assert rec.detail["video"] == "out.mp4"
        assert calls and calls[0][0] == "weekly-sst"
        # Ledger persisted.
        assert len(runner.ledger_tail()) == 1
        # Second tick: nothing due.
        assert runner.tick(utc(2026, 9, 28, 6, 2)) == []

    def test_not_due_no_run(self, tmp_path):
        store, runner, job = _runner(tmp_path, null_executor)
        assert runner.tick(utc(2026, 9, 28, 5, 0)) == []
        assert runner.ledger_tail() == []

    def test_disabled_job_never_runs(self, tmp_path):
        store, runner, job = _runner(tmp_path, null_executor,
                                     enabled=False)
        assert runner.tick(utc(2026, 9, 28, 6, 1)) == []

    def test_missed_run_skip_policy(self, tmp_path):
        store, runner, job = _runner(tmp_path, null_executor)
        # Monday 2026-09-28 06:01 UTC: runs, anchoring the schedule.
        assert len(runner.tick(utc(2026, 9, 28, 6, 1))) == 1
        # Runner "down" for a week; next tick Mon 2026-10-05 12:00 UTC:
        # the 06:00 occurrence was missed -> one skipped entry, then
        # the schedule resumes (never backfilled).
        records = runner.tick(utc(2026, 10, 5, 12, 0))
        assert len(records) == 1
        assert records[0].status == "skipped"
        assert records[0].detail["catchup"] == "skip"
        assert records[0].detail["missed_run"].startswith(
            "2026-10-05T06:00")
        assert runner.tick(utc(2026, 10, 5, 12, 1)) == []
        state = store.load_state()["weekly-sst"]
        assert state["next_run"].startswith("2026-10-12T06:00")

    def test_missed_run_run_once_policy(self, tmp_path):
        ran = []
        store, runner, job = _runner(
            tmp_path, lambda j, d: ran.append(j.name) or {},
            catchup="run-once")
        assert len(runner.tick(utc(2026, 9, 28, 6, 1))) == 1
        records = runner.tick(utc(2026, 10, 5, 12, 0))
        assert [r.status for r in records] == ["ok"]
        assert ran == ["weekly-sst", "weekly-sst"]

    def test_failure_recorded_and_retried(self, tmp_path):
        attempts = []

        def flaky(job, run_dir):
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("boom")
            return {"ok": True}

        store, runner, job = _runner(tmp_path, flaky, retries=2,
                                     retry_delay_seconds=0)
        records = runner.tick(utc(2026, 9, 28, 6, 1))
        assert records[0].status == "ok"
        assert records[0].attempt == 3
        assert len(attempts) == 3

    def test_failure_exhausts_retries(self, tmp_path):
        def always(job, run_dir):
            raise RuntimeError("nope")

        store, runner, job = _runner(tmp_path, always, retries=1,
                                     retry_delay_seconds=0)
        records = runner.tick(utc(2026, 9, 28, 6, 1))
        rec = records[0]
        assert rec.status == "failed"
        assert rec.attempt == 2
        assert "nope" in rec.error
        assert "traceback" in rec.detail

    def test_once_job_runs_exactly_once(self, tmp_path):
        store, runner, job = _runner(
            tmp_path, null_executor, schedule="once:2026-09-28T06:00")
        assert len(runner.tick(utc(2026, 9, 28, 6, 1))) == 1
        assert runner.tick(utc(2026, 9, 28, 7, 0)) == []
        assert runner.tick(utc(2026, 9, 29, 6, 0)) == []

    def test_once_job_future_not_due(self, tmp_path):
        store, runner, job = _runner(
            tmp_path, null_executor, schedule="once:2026-10-01T06:00")
        assert runner.tick(utc(2026, 9, 28, 6, 1)) == []

    def test_interval_first_run_one_interval_out(self, tmp_path):
        store, runner, job = _runner(tmp_path, null_executor,
                                     schedule="every:1h")
        assert runner.tick(utc(2026, 9, 28, 6, 0)) == []
        state = store.load_state()["weekly-sst"]
        assert state["next_run"].startswith("2026-09-28T07:00")

    def test_interval_runs_from_last_run(self, tmp_path):
        store, runner, job = _runner(tmp_path, null_executor,
                                     schedule="every:1h")
        runner.tick(utc(2026, 9, 28, 6, 0))  # arms first run at 07:00
        records = runner.tick(utc(2026, 9, 28, 7, 1))  # within grace
        assert [r.status for r in records] == ["ok"]

    def test_lock_prevents_double_tick(self, tmp_path):
        store = JobStore(str(tmp_path / "jobs"))
        store.save_job(make_job())
        runner = Runner(store, null_executor, str(tmp_path / "runs"))
        open(os.path.join(store.path, ".runner.lock"), "w").close()
        with pytest.raises(RunnerBusy):
            runner.tick(utc(2026, 9, 28, 6, 1))

    def test_grace_boundary(self, tmp_path):
        store, runner, job = _runner(tmp_path, null_executor)
        just_inside = utc(2026, 9, 28, 6, 0) + dt.timedelta(
            seconds=CATCHUP_GRACE_SECONDS - 1)
        records = runner.tick(just_inside)
        assert [r.status for r in records] == ["ok"]


# --- executors --------------------------------------------------------------

class TestExecutors:
    def test_null(self, tmp_path):
        out = null_executor(make_job(), str(tmp_path))
        assert out["dry_run"] is True

    def test_shell(self, tmp_path):
        job = make_job(settings={"command": "echo hello-{name} > out.txt"})
        out = shell_executor(job, str(tmp_path))
        assert out["returncode"] == 0
        assert (tmp_path / "out.txt").read_text().strip() == \
            "hello-weekly-sst"

    def test_shell_failure(self, tmp_path):
        job = make_job(settings={"command": "exit 3"})
        with pytest.raises(ScheduleError):
            shell_executor(job, str(tmp_path))

    def test_shell_no_command(self, tmp_path):
        with pytest.raises(ScheduleError):
            shell_executor(make_job(), str(tmp_path))


# --- CLI --------------------------------------------------------------------

class TestCLI:
    def _setup(self, tmp_path, monkeypatch):
        jobs = str(tmp_path / "jobs")
        monkeypatch.setenv("SCHEDX_JOBS_DIR", jobs)
        monkeypatch.setenv("SCHEDX_OUT_ROOT", str(tmp_path / "runs"))
        return jobs

    def test_add_list_next_validate(self, tmp_path, monkeypatch,
                                    capsys):
        from schedx.cli import main
        jobs = self._setup(tmp_path, monkeypatch)
        data = job_to_dict(make_job())
        f = tmp_path / "job.json"
        f.write_text(json.dumps(data))
        assert main(["--jobs-dir", jobs, "add", str(f)]) == 0
        assert main(["--jobs-dir", jobs, "list"]) == 0
        out = capsys.readouterr().out
        assert "weekly-sst" in out and "cron:0 6 * * mon" in out
        assert main(["--jobs-dir", jobs, "validate"]) == 0
        assert main(["--jobs-dir", jobs, "describe",
                     "every:1w"]) == 0
        assert "Every 1 week" in capsys.readouterr().out

    def test_run_once_needs_executor(self, tmp_path, monkeypatch,
                                     capsys):
        from schedx.cli import main
        jobs = self._setup(tmp_path, monkeypatch)
        assert main(["--jobs-dir", jobs, "run-once"]) == 2

    def test_enable_disable_remove(self, tmp_path, monkeypatch, capsys):
        from schedx.cli import main
        jobs = self._setup(tmp_path, monkeypatch)
        f = tmp_path / "job.json"
        f.write_text(json.dumps(job_to_dict(make_job())))
        main(["--jobs-dir", jobs, "add", str(f)])
        assert main(["--jobs-dir", jobs, "disable", "weekly-sst"]) == 0
        assert "disabled" in capsys.readouterr().out
        assert main(["--jobs-dir", jobs, "enable", "weekly-sst"]) == 0
        assert main(["--jobs-dir", jobs, "remove", "weekly-sst"]) == 0
        assert main(["--jobs-dir", jobs, "list"]) == 0
        assert "no jobs" in capsys.readouterr().out

    def test_run_once_with_null_executor(self, tmp_path, monkeypatch,
                                         capsys):
        from schedx.cli import main
        jobs = self._setup(tmp_path, monkeypatch)
        f = tmp_path / "job.json"
        f.write_text(json.dumps(job_to_dict(
            make_job(schedule="once:2020-01-01T00:00"))))
        main(["--jobs-dir", jobs, "add", str(f)])
        assert main(["--jobs-dir", jobs, "run-once",
                     "--executor", "null"]) == 0
        assert "weekly-sst: ok" in capsys.readouterr().out
        assert main(["--jobs-dir", jobs, "ledger", "-n", "5"]) == 0
        assert "weekly-sst ok" in capsys.readouterr().out
