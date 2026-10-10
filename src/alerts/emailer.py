"""Email composition and delivery (stdlib-only).

The same :class:`SiteReport` model feeds both parts of a multipart message:
plain text (always readable, the log-friendly fallback) and HTML (the
pretty report). SMTP delivery uses the standard library ``smtplib``; server
details and credentials come from explicit arguments or environment
variables — they are never hardcoded and never logged.

Environment variables (all optional; explicit arguments win):

* ``SURVEY_ALERTS_SMTP_HOST`` — SMTP server hostname
* ``SURVEY_ALERTS_SMTP_PORT`` — port (default 587)
* ``SURVEY_ALERTS_SMTP_USER`` / ``SURVEY_ALERTS_SMTP_PASSWORD`` — credentials
* ``SURVEY_ALERTS_SMTP_FROM`` — From: address
* ``SURVEY_ALERTS_SMTP_TLS`` — "0" to disable STARTTLS (default enabled)
"""

from __future__ import annotations

import logging
import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import List, Optional

from . import html as _html
from .errors import AlertsError
from .model import SiteReport

_LOG = logging.getLogger("alerts.email")


@dataclass
class SmtpConfig:
    """SMTP connection details. ``password`` is never logged or rendered."""

    host: str
    port: int = 587
    user: str = ""
    password: str = ""
    from_addr: str = ""
    use_tls: bool = True

    def redacted(self) -> str:
        who = f"{self.user}@" if self.user else ""
        return (f"smtp://{who}{self.host}:{self.port} "
                f"from={self.from_addr or '(unset)'} tls={self.use_tls}")

    @classmethod
    def from_env(cls,
                 host: Optional[str] = None,
                 port: Optional[int] = None,
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 from_addr: Optional[str] = None,
                 use_tls: Optional[bool] = None) -> "SmtpConfig":
        """Build config: explicit args win, then ``SURVEY_ALERTS_SMTP_*``."""
        env_port = os.environ.get("SURVEY_ALERTS_SMTP_PORT", "")
        env_tls = os.environ.get("SURVEY_ALERTS_SMTP_TLS", "")
        return cls(
            host=host or os.environ.get("SURVEY_ALERTS_SMTP_HOST", ""),
            port=port if port is not None
            else (int(env_port) if env_port.isdigit() else 587),
            user=user if user is not None
            else os.environ.get("SURVEY_ALERTS_SMTP_USER", ""),
            password=password if password is not None
            else os.environ.get("SURVEY_ALERTS_SMTP_PASSWORD", ""),
            from_addr=from_addr or os.environ.get("SURVEY_ALERTS_SMTP_FROM", ""),
            use_tls=use_tls if use_tls is not None
            else env_tls.strip() != "0",
        )

    def validate(self) -> None:
        missing = [name for name, value in
                   (("host", self.host), ("from_addr", self.from_addr))
                   if not value]
        if missing:
            raise AlertsError(
                f"SMTP config incomplete (missing: {', '.join(missing)}); "
                f"set SURVEY_ALERTS_SMTP_HOST / SURVEY_ALERTS_SMTP_FROM "
                f"or pass --smtp-host / --from")


def compose_email(report: SiteReport, to: List[str],
                  from_addr: str) -> EmailMessage:
    """Build a multipart (plain + HTML) message from a site report."""
    if not to:
        raise AlertsError("no recipients: pass --to or set recipients")
    if not from_addr:
        raise AlertsError("no From address: pass --from or set "
                          "SURVEY_ALERTS_SMTP_FROM")
    msg = EmailMessage()
    worst = (report.worst or "info").upper()
    msg["Subject"] = (f"[{worst}] Site alert report — {report.site_name} "
                      f"({report.total()} alert{'s' if report.total() != 1 else ''})")
    msg["From"] = from_addr
    msg["To"] = ", ".join(to)
    msg.set_content(_html.render_text(report))
    msg.add_alternative(_html.render_html(report), subtype="html")
    return msg


def send_email(msg: EmailMessage, config: SmtpConfig) -> str:
    """Deliver ``msg`` via SMTP. Returns a redacted connection summary."""
    config.validate()
    summary = config.redacted()
    _LOG.info("sending email via %s to %s", summary, msg["To"])
    with smtplib.SMTP(config.host, config.port, timeout=30) as smtp:
        if config.use_tls:
            smtp.starttls()
        if config.user:
            # password is passed straight to the server, never logged
            smtp.login(config.user, config.password)
        smtp.send_message(msg)
    return summary


class LoggingNotifier:
    """Test/ops notifier: records what *would* be sent, sends nothing.

    Useful for cron dry-runs and for the survey-monitor ``logging``-style
    workflow. Optionally also writes the rendered report to files.
    """

    def __init__(self, out_dir: Optional[str] = None) -> None:
        self.out_dir = out_dir
        self.sent: List[SiteReport] = []

    def __call__(self, report: SiteReport) -> str:
        self.sent.append(report)
        _LOG.info("NOTIFY %s %s: %s", report.worst, report.site_id,
                  report.headline)
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            base = os.path.join(self.out_dir,
                                f"{report.site_id}-{report.report_id}")
            with open(base + ".html", "w", encoding="utf-8") as fh:
                fh.write(_html.render_html(report))
            with open(base + ".txt", "w", encoding="utf-8") as fh:
                fh.write(_html.render_text(report))
            return base + ".html"
        return "logged"
