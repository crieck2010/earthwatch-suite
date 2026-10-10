"""survey-monitor: scheduled site-monitoring engine for satellite remote sensing.

Engine-first design: everything under ``monitor`` except ``monitor.cli`` is
pure logic with no UI or network imports. The CLI is a thin wrapper.

Pipeline (headless, cron-friendly)::

    site config  ->  list passes  ->  per-pass index metrics
                 ->  rolling/fixed baseline  ->  threshold evaluation
                 ->  alert events (JSONL)  ->  notifier hooks

Public surface::

    from monitor import SiteConfig, load_state, run_site, run_all
    from monitor import AlertEvent, Severity
    from monitor import register_notifier

Sister modules ``survey-imagery`` (acquisition) and ``survey-change``
(change statistics) are *optional*: they are imported lazily and only
when a real satellite acquisition run is requested. Everything else —
config parsing, scheduling, threshold evaluation, alert writing —
runs on the standard library alone.
"""

from __future__ import annotations

from .alerts import (
    AlertEvent,
    RunSummary,
    append_alerts,
    alerts_to_geojson,
    read_alerts,
    summarize_alerts,
    write_summary,
)
from .config import (
    AlertLevel,
    BaselineConfig,
    IndexThreshold,
    MonitorError,
    SiteConfig,
    load_site_config,
    save_site_config,
)
from .evaluate import evaluate_pass
from .notify import Notifier, get_notifier, list_notifiers, register_notifier
from .pipeline import run_all, run_site
from .scheduler import PassInfo, new_passes
from .state import SiteState, load_state, save_state

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AlertEvent",
    "AlertLevel",
    "BaselineConfig",
    "IndexThreshold",
    "MonitorError",
    "Notifier",
    "PassInfo",
    "RunSummary",
    "SiteConfig",
    "SiteState",
    "alerts_to_geojson",
    "append_alerts",
    "evaluate_pass",
    "get_notifier",
    "list_notifiers",
    "load_site_config",
    "load_state",
    "new_passes",
    "read_alerts",
    "register_notifier",
    "run_all",
    "run_site",
    "save_site_config",
    "save_state",
    "summarize_alerts",
    "write_summary",
]
