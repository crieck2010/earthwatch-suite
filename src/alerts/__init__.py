"""survey-alerts: alert-report delivery engine for earthwatch-suite.

Consumes the alert events that ``survey-monitor`` writes (``alerts.jsonl``,
per the stable schema in survey-monitor's ``docs/ALERT_SCHEMA.md``) and turns
them into human-readable deliverables: HTML reports (print-to-PDF ready),
plain-text/HTML email bodies, and per-site digests.

Engine-first: everything here except ``alerts/cli.py`` is stdlib-only pure
logic — no UI imports, no network except SMTP delivery, no hard third-party
dependencies.
"""

from .errors import AlertsError
from .reader import AlertRecord, dedupe, read_alerts
from .model import SiteReport, build_report
from .digest import DigestState, select_new

__all__ = [
    "AlertsError",
    "AlertRecord",
    "dedupe",
    "read_alerts",
    "SiteReport",
    "build_report",
    "DigestState",
    "select_new",
]

__version__ = "0.1.0"
