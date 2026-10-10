"""Notifier hooks: delivery is a plug-in, not a feature.

survey-monitor *detects* change; it does not send email or SMS. After a run
fires alerts, the pipeline calls every notifier named in the site config.
Notifiers are plain callables::

    def my_notifier(events: list[AlertEvent]) -> None: ...

Register them in-process::

    from monitor import register_notifier
    register_notifier("email-pdf", survey_alerts.deliver)

``survey-alerts`` (planned sibling module) will implement real delivery —
email and PDF reports — against exactly this interface, with no changes
needed here. Until then the built-in ``logging`` notifier records what
would have been sent.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, List

from .alerts import AlertEvent
from .config import MonitorError

#: A notifier receives the alerts fired by one run and delivers them.
Notifier = Callable[[List[AlertEvent]], None]

_REGISTRY: Dict[str, Notifier] = {}

_LOG = logging.getLogger("monitor.notify")


def register_notifier(name: str, notifier: Notifier) -> None:
    """Register a delivery callable under ``name`` (overwrites silently)."""
    if not callable(notifier):
        raise MonitorError(f"notifier {name!r} is not callable")
    _REGISTRY[name] = notifier


def get_notifier(name: str) -> Notifier:
    """Return the notifier registered as ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise MonitorError(
            f"no notifier registered as {name!r}; "
            f"available: {sorted(_REGISTRY) or 'none'} "
            f"(did you forget register_notifier?)") from exc


def list_notifiers() -> List[str]:
    return sorted(_REGISTRY)


def logging_notifier(events: List[AlertEvent]) -> None:
    """Built-in stub: logs what would have been delivered."""
    for event in events:
        _LOG.info("ALERT %s %s/%s: %s=%.3f delta=%+.3f z=%+.2f (pass %s)",
                  event.severity.upper(), event.site_id, event.index,
                  event.metric, event.value, event.delta, event.z_score,
                  event.pass_id)


def notify_all(names: Iterable[str],
               events: List[AlertEvent]) -> Dict[str, str]:
    """Call each named notifier with ``events``.

    Returns ``{name: "ok"}`` or ``{name: "error: ..."}`` per notifier — one
    failing delivery never blocks the others.
    """
    results: Dict[str, str] = {}
    for name in names:
        try:
            get_notifier(name)(list(events))
        except Exception as exc:  # noqa: BLE001 - delivery is best-effort
            results[name] = f"error: {exc}"
        else:
            results[name] = "ok"
    return results


register_notifier("logging", logging_notifier)
