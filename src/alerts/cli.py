"""Command-line interface for survey-alerts.

Commands:

* ``render`` — build a site report from an ``alerts.jsonl`` file and write
  it out (HTML, plain text, or PDF when reportlab is installed). Never
  sends anything: the safe preview path.
* ``send`` — render and email the report via SMTP (config from flags or
  ``SURVEY_ALERTS_SMTP_*`` env vars). ``--dry-run`` composes without
  sending.
* ``digest`` — walk a directory of per-site alert logs (the layout
  survey-monitor writes: ``<sites-dir>/<site_id>/alerts.jsonl``), select
  only undelivered alerts via a state file, and deliver per-site reports.
* ``notifiers`` — register this module's notifiers with survey-monitor's
  ``register_notifier`` hook (needs survey-monitor installed) and list them.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from typing import List, Optional

from . import html as _html
from .digest import DigestState, select_new, split_by_site
from .emailer import LoggingNotifier, SmtpConfig, compose_email, send_email
from .errors import AlertsError
from .model import build_report
from .pdf import pdf_available, render_pdf
from .plugin import install_notifiers
from .reader import dedupe, read_alerts

_LOG = logging.getLogger("alerts.cli")


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _read(path: str, min_severity: str,
          since: Optional[str]) -> "list":
    records, skipped = read_alerts(path, min_severity=min_severity,
                                   since=since)
    records = dedupe(records)
    if skipped:
        _LOG.warning("skipped %d corrupt line(s) in %s", skipped, path)
    return records


def _write_report(report, fmt: str, out: Optional[str]) -> str:
    if fmt == "pdf":
        if not pdf_available():
            raise AlertsError(
                "PDF rendering needs reportlab "
                "('pip install \"survey-alerts[pdf]\"'); use --format html "
                "and print to PDF from a browser instead")
        path = out or f"{report.site_id or 'site'}-{report.report_id}.pdf"
        return render_pdf(report, path)
    if fmt == "text":
        text = _html.render_text(report)
    else:
        text = _html.render_html(report)
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text)
        return out
    sys.stdout.write(text)
    if fmt == "html" and not text.endswith("\n"):
        sys.stdout.write("\n")
    return "<stdout>"


def cmd_render(a: argparse.Namespace) -> int:
    records = _read(a.alerts, a.min_severity, a.since)
    report = build_report(records, generated_at=_utcnow())
    path = _write_report(report, a.format, a.out)
    print(f"rendered {report.total()} alert(s), report {report.report_id} -> {path}",
          file=sys.stderr)
    return 0


def cmd_send(a: argparse.Namespace) -> int:
    records = _read(a.alerts, a.min_severity, a.since)
    report = build_report(records, generated_at=_utcnow())
    config = SmtpConfig.from_env(
        host=a.smtp_host, port=a.smtp_port, user=a.smtp_user,
        password=a.smtp_password, from_addr=a.from_,
        use_tls=(not a.no_tls))
    recipients = [r.strip() for r in a.to.split(",") if r.strip()]
    msg = compose_email(report, recipients, config.from_addr or
                        os.environ.get("SURVEY_ALERTS_SMTP_FROM", ""))
    if a.dry_run:
        print(f"dry run: would send via {config.redacted()} "
              f"to {', '.join(recipients)}", file=sys.stderr)
        print(f"subject: {msg['Subject']}", file=sys.stderr)
        return 0
    summary = send_email(msg, config)
    print(f"sent {report.total()} alert(s) to {', '.join(recipients)} "
          f"via {summary}", file=sys.stderr)
    return 0


def _site_alert_files(sites_dir: str) -> List[tuple]:
    found = []
    if not os.path.isdir(sites_dir):
        raise AlertsError(f"sites directory not found: {sites_dir}")
    for site_id in sorted(os.listdir(sites_dir)):
        path = os.path.join(sites_dir, site_id, "alerts.jsonl")
        if os.path.isfile(path):
            found.append((site_id, path))
    return found


def cmd_digest(a: argparse.Namespace) -> int:
    state = DigestState.load(a.state)
    now = _utcnow()
    delivered_any = 0
    sites_hit = 0
    for _dir_site, path in _site_alert_files(a.sites_dir):
        records = _read(path, a.min_severity, a.since)
        new = select_new(records, state, min_severity=a.min_severity,
                         since=a.since)
        # a file may carry several sites' records: report and mark each
        # site under its own site_id so re-delivery can never happen
        for site_id, group in split_by_site(new):
            report = build_report(group, generated_at=now)
            sites_hit += 1
            if a.email_to:
                config = SmtpConfig.from_env(
                    host=a.smtp_host, port=a.smtp_port, user=a.smtp_user,
                    password=a.smtp_password, from_addr=a.from_,
                    use_tls=(not a.no_tls))
                recipients = [r.strip() for r in a.email_to.split(",")
                              if r.strip()]
                msg = compose_email(report, recipients, config.from_addr or
                                    os.environ.get("SURVEY_ALERTS_SMTP_FROM", ""))
                if a.dry_run:
                    print(f"dry run: would email site {site_id} "
                          f"({report.total()} alerts) to "
                          f"{', '.join(recipients)}", file=sys.stderr)
                else:
                    send_email(msg, config)
                    print(f"emailed site {site_id} ({report.total()} alerts)",
                          file=sys.stderr)
            if a.out_dir:
                notifier = LoggingNotifier(out_dir=a.out_dir)
                path_written = notifier(report)
                print(f"wrote {path_written}", file=sys.stderr)
            if not a.dry_run:
                state.mark_delivered(site_id, group, now)
            delivered_any += report.total()
    if not a.dry_run:
        state.save()
    print(f"digest complete: {delivered_any} new alert(s) across "
          f"{sites_hit} site(s); state -> {a.state}", file=sys.stderr)
    return 0


def cmd_notifiers(a: argparse.Namespace) -> int:
    result = install_notifiers()
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="survey-alerts",
        description="Turn survey-monitor alert events into site reports "
                    "and email digests.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="verbose logging")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("render", help="render a report without sending it")
    s.add_argument("--alerts", required=True,
                   help="path to alerts.jsonl")
    s.add_argument("--format", choices=["html", "text", "pdf"],
                   default="html", help="output format (default: html)")
    s.add_argument("--out", default=None,
                   help="output file (default: stdout)")
    s.add_argument("--min-severity", default="info",
                   choices=["info", "warning", "critical"],
                   help="minimum severity to include")
    s.add_argument("--since", default=None,
                   help="only alerts with timestamp >= this ISO-8601 value")
    s.set_defaults(func=cmd_render)

    s = sub.add_parser("send", help="render and email a report via SMTP")
    s.add_argument("--alerts", required=True, help="path to alerts.jsonl")
    s.add_argument("--to", required=True,
                   help="recipient address(es), comma-separated")
    s.add_argument("--from", dest="from_", default=None,
                   help="From address (or SURVEY_ALERTS_SMTP_FROM)")
    s.add_argument("--smtp-host", default=None)
    s.add_argument("--smtp-port", type=int, default=None)
    s.add_argument("--smtp-user", default=None)
    s.add_argument("--smtp-password", default=None,
                   help="or SURVEY_ALERTS_SMTP_PASSWORD")
    s.add_argument("--no-tls", action="store_true",
                   help="disable STARTTLS")
    s.add_argument("--min-severity", default="info",
                   choices=["info", "warning", "critical"])
    s.add_argument("--since", default=None)
    s.add_argument("--dry-run", action="store_true",
                   help="compose only; do not send")
    s.set_defaults(func=cmd_send)

    s = sub.add_parser("digest",
                       help="digest new alerts across a sites directory")
    s.add_argument("--sites-dir", required=True,
                   help="<sites-dir>/<site_id>/alerts.jsonl layout")
    s.add_argument("--state", default="digest-state.json",
                   help="digest state file (default: digest-state.json)")
    s.add_argument("--min-severity", default="warning",
                   choices=["info", "warning", "critical"],
                   help="minimum severity to digest (default: warning)")
    s.add_argument("--since", default=None)
    s.add_argument("--out-dir", default=None,
                   help="write per-site reports here (html+txt)")
    s.add_argument("--email-to", default=None,
                   help="email each site report here (comma-separated)")
    s.add_argument("--from", dest="from_", default=None)
    s.add_argument("--smtp-host", default=None)
    s.add_argument("--smtp-port", type=int, default=None)
    s.add_argument("--smtp-user", default=None)
    s.add_argument("--smtp-password", default=None)
    s.add_argument("--no-tls", action="store_true")
    s.add_argument("--dry-run", action="store_true",
                   help="select only; do not send or mark delivered")
    s.set_defaults(func=cmd_digest)

    s = sub.add_parser("notifiers",
                       help="register notifiers with survey-monitor")
    s.set_defaults(func=cmd_notifiers)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args)
    except AlertsError as exc:
        print(f"survey-alerts: error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0
