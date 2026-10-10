"""survey-monitor notifier plug-ins.

survey-monitor *detects* change; it calls notifiers registered under
:meth:`monitor.notify.register_notifier` after each run. This module
implements the two notifiers survey-monitor's docs anticipated:

* ``"survey-alerts:email"`` — build the report model and deliver it by
  SMTP (configuration from ``SURVEY_ALERTS_SMTP_*`` env vars; see
  :mod:`alerts.emailer`). Without SMTP configuration the notifier raises a
  clear error instead of silently dropping the alerts.
* ``"survey-alerts:pdf"`` — render the report (HTML, or PDF when reportlab
  is installed) into ``SURVEY_ALERTS_OUT_DIR`` (default
  ``./alert-reports``), one file per run.

survey-monitor is imported **lazily** and the install fails soft: importing
this module, or calling :func:`install_notifiers`, without survey-monitor
installed simply reports ``installed: False`` — survey-alerts always works
standalone on a path to an ``alerts.jsonl`` file.

A notifier receives a list of survey-monitor ``AlertEvent`` objects *or*
plain dicts (the CLI and the tests pass dicts); both are normalized through
:func:`alerts.reader.AlertRecord.from_dict`.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

from . import html as _html
from .emailer import SmtpConfig, compose_email, send_email
from .errors import AlertsError
from .model import build_report
from .pdf import pdf_available, render_pdf
from .reader import AlertRecord

_LOG = logging.getLogger("alerts.plugin")

EMAIL_NOTIFIER_NAME = "survey-alerts:email"
PDF_NOTIFIER_NAME = "survey-alerts:pdf"


def _normalize(events: Iterable[Any]) -> List[AlertRecord]:
    records: List[AlertRecord] = []
    for event in events:
        if hasattr(event, "to_dict") and callable(event.to_dict):
            data = event.to_dict()
        elif isinstance(event, Mapping):
            data = dict(event)
        else:
            raise AlertsError(
                f"notifier received unsupported event type: {type(event)!r}")
        records.append(AlertRecord.from_dict(data))
    return records


def email_notifier(events: Iterable[Any],
                   to: Optional[List[str]] = None,
                   config: Optional[SmtpConfig] = None) -> str:
    """survey-monitor notifier: email the run's alerts as a site report."""
    records = _normalize(events)
    report = build_report(
        records,
        generated_at=_dt.datetime.now(
            _dt.timezone.utc).isoformat(timespec="seconds"))
    cfg = config or SmtpConfig.from_env()
    cfg.validate()
    recipients = to or [r.strip() for r in
                        os.environ.get("SURVEY_ALERTS_TO", "").split(",")
                        if r.strip()]
    msg = compose_email(report, recipients, cfg.from_addr)
    summary = send_email(msg, cfg)
    _LOG.info("emailed %d alerts for site %s via %s",
              report.total(), report.site_id, summary)
    return summary


def pdf_notifier(events: Iterable[Any],
                 out_dir: Optional[str] = None) -> str:
    """survey-monitor notifier: render the run's alerts to a report file."""
    records = _normalize(events)
    report = build_report(
        records,
        generated_at=_dt.datetime.now(
            _dt.timezone.utc).isoformat(timespec="seconds"))
    directory = out_dir or os.environ.get("SURVEY_ALERTS_OUT_DIR",
                                          "alert-reports")
    os.makedirs(directory, exist_ok=True)
    base = os.path.join(directory,
                        f"{report.site_id or 'site'}-{report.report_id}")
    if pdf_available():
        path = render_pdf(report, base + ".pdf")
    else:
        path = base + ".html"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_html.render_html(report))
    _LOG.info("wrote %d-alert report to %s", report.total(), path)
    return path


def install_notifiers(to: Optional[List[str]] = None,
                       config: Optional[SmtpConfig] = None,
                       out_dir: Optional[str] = None) -> Dict[str, Any]:
    """Register survey-alerts notifiers with survey-monitor's registry.

    Returns ``{"installed": True, "notifiers": [...]}`` on success, or
    ``{"installed": False, "reason": ...}`` when survey-monitor is not
    importable — never raises for a missing optional dependency.
    """
    try:
        from monitor import register_notifier  # type: ignore
    except ImportError:
        return {"installed": False,
                "reason": "survey-monitor is not installed; "
                          "survey-alerts still works standalone on "
                          "alerts.jsonl files"}

    def _email(events: Iterable[Any]) -> None:
        email_notifier(events, to=to, config=config)

    def _pdf(events: Iterable[Any]) -> None:
        pdf_notifier(events, out_dir=out_dir)

    register_notifier(EMAIL_NOTIFIER_NAME, _email)
    register_notifier(PDF_NOTIFIER_NAME, _pdf)
    _LOG.info("registered notifiers: %s, %s",
              EMAIL_NOTIFIER_NAME, PDF_NOTIFIER_NAME)
    return {"installed": True,
            "notifiers": [EMAIL_NOTIFIER_NAME, PDF_NOTIFIER_NAME]}
