"""Tests for survey-alerts. No network, no third-party deps required."""

import builtins
import json
import os
import sys

import pytest

_REAL_IMPORT = builtins.__import__

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alerts import (
    AlertRecord,
    DigestState,
    build_report,
    dedupe,
    read_alerts,
    select_new,
)
from alerts import html as html_mod
from alerts.digest import split_by_site
from alerts.emailer import LoggingNotifier, SmtpConfig, compose_email
from alerts.errors import AlertsError
from alerts.pdf import pdf_available, render_pdf

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
ALERTS = os.path.join(FIXTURES, "alerts.jsonl")


def read_all(**kwargs):
    records, skipped = read_alerts(ALERTS, **kwargs)
    return records, skipped


# ---------------------------------------------------------------- reader

def test_reader_parses_all_valid_lines():
    records, skipped = read_all()
    assert len(records) == 6  # 5 unique + 1 deliberate duplicate
    assert skipped == 1  # the non-JSON line


def test_reader_skips_corrupt_line_but_keeps_rest():
    records, skipped = read_all()
    assert skipped == 1
    assert {r.alert_id for r in records} >= {
        "quarry-north:pass-2026-09-06:ndvi:critical:0",
        "wetland-east:pass-2026-09-11:ndwi:warning:0",
    }


def test_reader_tolerates_unknown_extra_fields():
    records, _ = read_all()
    rec = next(r for r in records
               if r.alert_id == "wetland-east:pass-2026-09-06:ndwi:warning:0")
    assert rec.extras["future_field"] == "kept-verbatim"
    assert rec.extras["note"].startswith("schema additions")


def test_reader_extras_round_trip():
    records, _ = read_all()
    rec = next(r for r in records if r.extras)
    assert AlertRecord.from_dict(rec.to_dict()).extras == rec.extras


def test_reader_missing_optional_fields_degrade_gracefully():
    rec = AlertRecord.from_dict({"alert_id": "x:1", "site_id": "x"})
    assert rec.value is None and rec.z_score is None and rec.pass_date == ""
    assert rec.severity == "info"


def test_reader_requires_alert_id():
    with pytest.raises(AlertsError):
        AlertRecord.from_dict({"site_id": "x"})


def test_reader_rejects_non_object():
    with pytest.raises(AlertsError):
        AlertRecord.from_dict([1, 2, 3])


def test_reader_normalizes_unknown_severity_to_info():
    rec = AlertRecord.from_dict({"alert_id": "x:1", "severity": "EMERGENCY"})
    assert rec.severity == "info"


def test_reader_min_severity_filter():
    records, _ = read_all(min_severity="warning")
    assert records and all(r.severity_rank() >= 1 for r in records)
    assert not any(r.severity == "info" for r in records)


def test_reader_since_filter():
    records, _ = read_all(since="2026-09-06T00:00:00+00:00")
    assert all(r.timestamp >= "2026-09-06T00:00:00+00:00" for r in records)
    assert len(records) == 4  # the two 09-06 and two 09-11 events


def test_reader_missing_file():
    with pytest.raises(AlertsError):
        read_alerts("/nonexistent/alerts.jsonl")


def test_dedupe_keeps_last_and_first_seen_order():
    records, _ = read_all()
    deduped = dedupe(records)
    assert len(deduped) == 5
    ids = [r.alert_id for r in deduped]
    assert ids[0] == "quarry-north:pass-2026-09-01:ndvi:warning:0"
    assert len(set(ids)) == 5


# ---------------------------------------------------------------- model

def _report(**kwargs):
    records, _ = read_all()
    return build_report(dedupe(records), generated_at="2026-09-12T00:00:00+00:00",
                        **kwargs)


def test_report_counts_and_worst():
    report = _report()
    assert report.total() == 5
    assert report.worst == "critical"
    assert report.by_severity == {"info": 1, "warning": 3, "critical": 1}
    assert report.by_index == {"ndvi": 2, "ndwi": 3}


def test_report_sorts_worst_first():
    report = _report()
    assert report.events[0].severity == "critical"
    assert report.events[-1].severity == "info"


def test_report_headline_mentions_critical():
    assert "CRITICAL" in _report().headline


def test_report_headline_empty():
    report = build_report([], generated_at="2026-09-12T00:00:00+00:00")
    assert report.total() == 0
    assert "No alerts" in report.headline
    assert report.worst is None


def test_report_id_deterministic():
    assert _report().report_id == _report().report_id


def test_report_id_changes_with_input():
    records, _ = read_all()
    assert (build_report(dedupe(records)).report_id
            != build_report(dedupe(records)[:2]).report_id)


def test_summary_lines_plain_english():
    lines = _report().summary_lines
    assert len(lines) == 5
    veg = next(line for line in lines
               if "NDVI" in line and "2026-09-01" in line)
    assert "dropped" in veg and "\u03c3" in veg and "2026-09-01" in veg
    assert veg.endswith(".")


def test_summary_line_handles_missing_delta():
    rec = AlertRecord.from_dict({"alert_id": "x:1", "site_id": "x",
                                 "index": "ndvi"})
    line = build_report([rec]).summary_lines[0]
    assert "changed" in line


# ---------------------------------------------------------------- digest

def test_split_by_site_groups_and_orders():
    records, _ = read_all()
    groups = split_by_site(dedupe(records))
    assert [g[0] for g in groups] == ["quarry-north", "wetland-east"]
    assert len(groups[0][1]) == 3 and len(groups[1][1]) == 2


def test_select_new_filters_delivered_and_severity(tmp_path):
    state_path = str(tmp_path / "state.json")
    state = DigestState.load(state_path)
    records, _ = read_all()
    records = dedupe(records)
    new = select_new(records, state, min_severity="warning")
    assert len(new) == 4  # 3 warning + 1 critical, no info
    state.mark_delivered("quarry-north",
                         [r for r in new if r.site_id == "quarry-north"],
                         "2026-09-12T00:00:00+00:00")
    state.save()
    reloaded = DigestState.load(state_path)
    again = select_new(records, reloaded, min_severity="warning")
    assert {r.site_id for r in again} == {"wetland-east"}
    assert len(again) == 2


def test_digest_state_round_trip(tmp_path):
    path = str(tmp_path / "state.json")
    state = DigestState.load(path)  # missing file -> empty, no error
    assert state.delivered == {}
    state.mark_delivered("s1", [], "2026-09-12T00:00:00+00:00")
    state.save()
    assert DigestState.load(path).last_digest["s1"] == "2026-09-12T00:00:00+00:00"


def test_digest_state_rejects_corrupt_file(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json")
    with pytest.raises(AlertsError):
        DigestState.load(str(path))


def test_select_new_since_filter(tmp_path):
    state = DigestState.load(str(tmp_path / "s.json"))
    records, _ = read_all()
    new = select_new(dedupe(records), state, min_severity="info",
                     since="2026-09-11T00:00:00+00:00")
    assert len(new) == 2  # the two 2026-09-11 events


def test_select_new_idempotent_on_repeat(tmp_path):
    path = str(tmp_path / "s.json")
    state = DigestState.load(path)
    records, _ = read_all()
    records = dedupe(records)
    first = select_new(records, state, min_severity="info")
    for site_id, group in split_by_site(first):
        state.mark_delivered(site_id, group, "2026-09-12T00:00:00+00:00")
    state.save()
    second = select_new(records, DigestState.load(path), min_severity="info")
    assert second == []


# ---------------------------------------------------------------- render

def test_render_html_deterministic():
    a = html_mod.render_html(_report())
    b = html_mod.render_html(_report())
    assert a == b
    assert "<!DOCTYPE html>" in a and "@page" in a


def test_render_html_contains_alert_table():
    out = html_mod.render_html(_report())
    assert "quarry-north" in out or "Quarry north pit" in out
    assert "CRITICAL" in out and "NDVI" in out


def test_render_html_escapes_injection():
    rec = AlertRecord.from_dict({
        "alert_id": "x:1", "site_id": "x",
        "site_name": "<script>alert(1)</script>", "index": "ndvi"})
    out = html_mod.render_html(build_report([rec]))
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_render_text_contains_summary():
    out = html_mod.render_text(_report())
    assert "Site alert report" in out
    assert "What changed:" in out
    assert "critical: 1" in out


def test_render_pdf_without_reportlab_or_with_it(tmp_path):
    report = _report()
    path = str(tmp_path / "report.pdf")
    if pdf_available():
        assert render_pdf(report, path) == path
        assert os.path.getsize(path) > 1000
    else:
        with pytest.raises(AlertsError, match="reportlab"):
            render_pdf(report, path)


# ---------------------------------------------------------------- email

def test_compose_email_multipart():
    report = _report()
    msg = compose_email(report, ["ops@example.com"], "alerts@example.com")
    assert msg["To"] == "ops@example.com"
    assert msg["From"] == "alerts@example.com"
    assert "[CRITICAL]" in msg["Subject"]
    assert msg.is_multipart()
    parts = {p.get_content_type() for p in msg.iter_parts()}
    assert {"text/plain", "text/html"} <= parts


def test_compose_email_requires_recipients_and_from():
    report = _report()
    with pytest.raises(AlertsError):
        compose_email(report, [], "alerts@example.com")
    with pytest.raises(AlertsError):
        compose_email(report, ["a@b.c"], "")


def test_smtp_config_from_env(monkeypatch):
    monkeypatch.setenv("SURVEY_ALERTS_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SURVEY_ALERTS_SMTP_PORT", "2525")
    monkeypatch.setenv("SURVEY_ALERTS_SMTP_USER", "u")
    monkeypatch.setenv("SURVEY_ALERTS_SMTP_PASSWORD", "s3cret")
    monkeypatch.setenv("SURVEY_ALERTS_SMTP_FROM", "alerts@example.com")
    cfg = SmtpConfig.from_env()
    assert (cfg.host, cfg.port, cfg.user, cfg.from_addr) == (
        "smtp.example.com", 2525, "u", "alerts@example.com")
    redacted = cfg.redacted()
    assert "s3cret" not in redacted  # credentials never leak into logs
    assert "smtp.example.com" in redacted


def test_smtp_config_defaults_and_validation():
    cfg = SmtpConfig(host="", from_addr="")
    with pytest.raises(AlertsError, match="SMTP config incomplete"):
        cfg.validate()


def test_logging_notifier_records_and_writes_files(tmp_path):
    notifier = LoggingNotifier(out_dir=str(tmp_path))
    report = _report()
    out = notifier(report)
    assert out.endswith(".html") and os.path.isfile(out)
    assert os.path.isfile(out.replace(".html", ".txt"))
    assert notifier.sent == [report]


# ---------------------------------------------------------------- plugin

def _fake_monitor_module(monkeypatch):
    """Inject a stub `monitor` module exposing register_notifier."""
    import types
    registry = {}

    def register_notifier(name, fn):
        registry[name] = fn

    module = types.ModuleType("monitor")
    module.register_notifier = register_notifier
    module._registry = registry
    monkeypatch.setitem(sys.modules, "monitor", module)
    return registry


def test_install_notifiers_registers_with_monitor(monkeypatch):
    from alerts import plugin
    registry = _fake_monitor_module(monkeypatch)
    result = plugin.install_notifiers()
    assert result["installed"] is True
    assert set(result["notifiers"]) == {
        "survey-alerts:email", "survey-alerts:pdf"}
    assert set(registry) == set(result["notifiers"])


def test_install_notifiers_fails_soft_without_monitor(monkeypatch):
    from alerts import plugin
    monkeypatch.delitem(sys.modules, "monitor", raising=False)
    # ensure no real monitor package shadows the absence
    monkeypatch.setattr("builtins.__import__", _no_monitor_import)
    try:
        result = plugin.install_notifiers()
    finally:
        monkeypatch.undo()
    assert result["installed"] is False
    assert "reason" in result


def _no_monitor_import(name, *args, **kwargs):
    if name == "monitor" or name.startswith("monitor."):
        raise ImportError("No module named 'monitor'")
    return _REAL_IMPORT(name, *args, **kwargs)


def test_email_notifier_normalizes_dict_events(monkeypatch):
    from alerts import plugin
    sent = {}

    class FakeSMTP:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            pass

        def login(self, *a):
            pass

        def send_message(self, msg):
            sent["subject"] = msg["Subject"]

    monkeypatch.setattr("smtplib.SMTP", FakeSMTP)
    cfg = SmtpConfig(host="smtp.example.com", from_addr="a@b.c")
    records, _ = read_all()
    plugin.email_notifier([r.to_dict() for r in dedupe(records)[:1]],
                          to=["ops@example.com"], config=cfg)
    assert "[WARNING]" in sent["subject"]


def test_email_notifier_rejects_bad_event_type():
    from alerts import plugin
    cfg = SmtpConfig(host="smtp.example.com", from_addr="a@b.c")
    with pytest.raises(AlertsError):
        plugin.email_notifier([object()], to=["a@b.c"], config=cfg)


def test_pdf_notifier_writes_report_file(tmp_path):
    from alerts import plugin
    records, _ = read_all()
    path = plugin.pdf_notifier([r.to_dict() for r in dedupe(records)[:2]],
                               out_dir=str(tmp_path))
    assert os.path.isfile(path)
    assert path.endswith(".pdf" if pdf_available() else ".html")


# ---------------------------------------------------------------- CLI

def _run_cli(*argv, cwd=None):
    from alerts.cli import main
    old = os.getcwd()
    try:
        if cwd:
            os.chdir(cwd)
        return main(list(argv))
    finally:
        os.chdir(old)


def test_cli_render_html_to_file(tmp_path):
    out = str(tmp_path / "report.html")
    rc = _run_cli("render", "--alerts", ALERTS, "--format", "html",
                  "--out", out, "--min-severity", "warning")
    assert rc == 0
    text = open(out, encoding="utf-8").read()
    assert "Quarry north pit" in text and "CRITICAL" in text


def test_cli_render_text_to_stdout(capsys):
    rc = _run_cli("render", "--alerts", ALERTS, "--format", "text",
                  "--min-severity", "critical")
    assert rc == 0
    out = capsys.readouterr().out
    assert "CRITICAL" in out and "WARNING" not in out.split("Alert detail:")[1]


def test_cli_render_idempotent(tmp_path):
    a, b = str(tmp_path / "a.html"), str(tmp_path / "b.html")
    _run_cli("render", "--alerts", ALERTS, "--out", a)
    _run_cli("render", "--alerts", ALERTS, "--out", b)
    assert open(a).read() == open(b).read()


def test_cli_send_dry_run(capsys):
    rc = _run_cli("send", "--alerts", ALERTS, "--to", "ops@example.com",
                  "--from", "alerts@example.com", "--dry-run")
    assert rc == 0
    err = capsys.readouterr().err
    assert "dry run" in err and "ops@example.com" in err


def test_cli_digest_multi_site_file_is_idempotent(tmp_path):
    # one file carrying two sites' records: each site must be marked under
    # its own site_id, or the second site gets re-delivered on re-run
    sites = tmp_path / "sites"
    (sites / "mixed").mkdir(parents=True)
    records, _ = read_all()
    with open(sites / "mixed" / "alerts.jsonl", "w") as fh:
        for r in dedupe(records):
            fh.write(json.dumps(r.to_dict()) + "\n")
    state = str(tmp_path / "state.json")
    out_dir = str(tmp_path / "reports")
    assert _run_cli("digest", "--sites-dir", str(sites), "--state", state,
                    "--out-dir", out_dir) == 0
    assert len(os.listdir(out_dir)) == 4  # 2 sites x (html + txt)
    n_before = len(os.listdir(out_dir))
    assert _run_cli("digest", "--sites-dir", str(sites), "--state", state,
                    "--out-dir", out_dir) == 0
    assert len(os.listdir(out_dir)) == n_before  # nothing re-delivered
    saved = json.load(open(state))
    assert set(saved["delivered"]) == {"quarry-north", "wetland-east"}


def test_cli_digest_end_to_end(tmp_path):
    sites = tmp_path / "sites"
    (sites / "quarry-north").mkdir(parents=True)
    (sites / "wetland-east").mkdir(parents=True)
    records, _ = read_all()
    qn = [r for r in dedupe(records) if r.site_id == "quarry-north"]
    we = [r for r in dedupe(records) if r.site_id == "wetland-east"]
    for site_id, group in (("quarry-north", qn), ("wetland-east", we)):
        with open(sites / site_id / "alerts.jsonl", "w") as fh:
            for r in group:
                fh.write(json.dumps(r.to_dict()) + "\n")
    state = str(tmp_path / "state.json")
    out_dir = str(tmp_path / "reports")
    rc = _run_cli("digest", "--sites-dir", str(sites), "--state", state,
                  "--out-dir", out_dir)
    assert rc == 0
    assert len(os.listdir(out_dir)) == 4  # 2 sites x (html + txt)
    # second run: nothing new -> no new state churn, still rc 0
    rc = _run_cli("digest", "--sites-dir", str(sites), "--state", state,
                  "--out-dir", out_dir)
    assert rc == 0
    saved = json.load(open(state))
    assert set(saved["delivered"]) == {"quarry-north", "wetland-east"}


def test_cli_render_missing_file_returns_error(capsys):
    rc = _run_cli("render", "--alerts", "/nonexistent.jsonl")
    assert rc == 2
    assert "error" in capsys.readouterr().err


def test_cli_notifiers_without_monitor(capsys, monkeypatch):
    monkeypatch.delitem(sys.modules, "monitor", raising=False)
    monkeypatch.setattr("builtins.__import__", _no_monitor_import)
    try:
        rc = _run_cli("notifiers")
    finally:
        monkeypatch.undo()
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["installed"] is False
