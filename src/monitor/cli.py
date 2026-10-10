"""Thin command-line interface for survey-monitor.

Every subcommand is a thin wrapper over the pure engine: argument parsing
lives here, all computation lives in the engine modules. Run
``survey-monitor --help`` or ``python -m monitor --help`` for usage.

Cron recipe (runs every satellite pass window, e.g. twice daily)::

    0 6,18 * * * /usr/bin/survey-monitor run --config-dir ~/monitor/sites \\
        --store-dir ~/monitor/state >> ~/monitor/cron.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .alerts import read_alerts
from .config import MonitorError, load_site_config, save_site_config, template_config
from .notify import list_notifiers
from .pipeline import run_all, run_site
from .state import load_state


def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--config-dir", default="sites",
                    help="directory of site config files (default: ./sites)")
    sp.add_argument("--store-dir", default="monitor-state",
                    help="directory of per-site state (default: ./monitor-state)")


def cmd_add_site(a: argparse.Namespace) -> int:
    if a.template:
        config = template_config(a.site_id or "demo-site")
        path = os.path.join(a.config_dir, f"{config.site_id}.json")
        save_site_config(config, path)
        print(f"wrote starter config: {path}")
        print("edit it, then run: survey-monitor run --site", config.site_id)
        return 0
    if not a.config:
        print("add-site needs --config <file> or --template",
              file=sys.stderr)
        return 2
    config = load_site_config(a.config)
    path = os.path.join(a.config_dir, f"{config.site_id}.json")
    save_site_config(config, path)
    print(f"added site {config.site_id!r} ({config.name}) -> {path}")
    return 0


def cmd_list_sites(a: argparse.Namespace) -> int:
    rows = []
    for fname in sorted(os.listdir(a.config_dir) if
                        os.path.isdir(a.config_dir) else []):
        if not fname.endswith((".json", ".yaml", ".yml")):
            continue
        path = os.path.join(a.config_dir, fname)
        try:
            config = load_site_config(path)
        except MonitorError as exc:
            rows.append((fname, "INVALID", str(exc), "", ""))
            continue
        state = load_state(a.store_dir, config.site_id)
        rows.append((config.site_id, config.name,
                     ",".join(config.indices),
                     str(len(state.passes)),
                     state.last_run or "-"))
    if not rows:
        print(f"no site configs in {a.config_dir} "
              f"(try: survey-monitor add-site --template)")
        return 0
    print(f"{'SITE':<24}{'NAME':<28}{'INDICES':<22}{'PASSES':<8}LAST RUN")
    for site_id, name, indices, npasses, last_run in rows:
        print(f"{site_id:<24}{name[:27]:<28}{indices[:21]:<22}"
              f"{npasses:<8}{last_run}")
    return 0


def _print_summary(summary, verbose: bool) -> None:
    print(f"[{summary.site_id}] {summary.passes_processed} pass(es) "
          f"processed, {summary.passes_skipped} skipped, "
          f"{summary.alerts_fired} alert(s)")
    if summary.alerts_fired and verbose:
        by_sev = summary.alerts.get("by_severity", {})
        print("  by severity:", ", ".join(f"{k}={v}" for k, v in by_sev.items()
                                          if v))
    for err in summary.errors:
        print(f"  error: {err}", file=sys.stderr)


def cmd_run(a: argparse.Namespace) -> int:
    log = print if a.verbose else None
    try:
        if a.site:
            summary = run_site(a.site, a.config_dir, a.store_dir,
                               since=a.since, synthetic=a.synthetic,
                               progress=log)
            _print_summary(summary, a.verbose)
        else:
            summaries = run_all(a.config_dir, a.store_dir,
                                since=a.since, synthetic=a.synthetic,
                                progress=log)
            if not summaries:
                print(f"no site configs in {a.config_dir}")
                return 0
            for summary in summaries:
                _print_summary(summary, a.verbose)
    except MonitorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_alerts(a: argparse.Namespace) -> int:
    events = read_alerts(a.store_dir, a.site, severity=a.severity,
                         since=a.since)
    if a.format == "json":
        print(json.dumps([e.to_dict() for e in events], indent=2))
        return 0
    if not events:
        print(f"no alerts for site {a.site!r}"
              + (f" at severity {a.severity}" if a.severity else ""))
        return 0
    print(f"{'TIME':<22}{'SEVERITY':<10}{'INDEX':<8}{'DELTA':<10}"
          f"{'Z':<8}PASS")
    for e in events:
        print(f"{e.timestamp:<22}{e.severity:<10}{e.index:<8}"
              f"{e.delta:+.3f}   {e.z_score:+.2f}  {e.pass_id}")
    if a.verbose:
        print(f"\n{len(events)} alert(s); full log: "
              f"{os.path.join(a.store_dir, a.site, 'alerts.jsonl')}")
    return 0


def cmd_notifiers(_a: argparse.Namespace) -> int:
    names = list_notifiers()
    print("registered notifiers:", ", ".join(names) if names else "(none)")
    print("register more in-process with monitor.register_notifier(name, fn)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="survey-monitor",
        description="Scheduled satellite site-monitoring engine: per-pass "
                    "index tracking with threshold alerts.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("add-site",
                       help="add a site config (from file or a starter template)")
    _add_common(s)
    s.add_argument("--config", help="JSON/YAML site config to install")
    s.add_argument("--template", action="store_true",
                   help="write a starter config to edit")
    s.add_argument("--site-id", help="site id for --template")
    s.set_defaults(func=cmd_add_site)

    s = sub.add_parser("list-sites", help="list configured sites and state")
    _add_common(s)
    s.set_defaults(func=cmd_list_sites)

    s = sub.add_parser("run", help="run the monitor (one site or all)")
    _add_common(s)
    s.add_argument("--site", help="run only this site id (default: all sites)")
    s.add_argument("--since", help="only consider passes after this ISO date")
    s.add_argument("--synthetic", action="store_true",
                   help="use the seeded synthetic provider (demo/tests; "
                        "no network, no survey-imagery needed)")
    s.add_argument("--verbose", "-v", action="store_true")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("alerts", help="show a site's alert log")
    _add_common(s)
    s.add_argument("--site", required=True, help="site id")
    s.add_argument("--severity", choices=["info", "warning", "critical"])
    s.add_argument("--since", help="only alerts at/after this ISO timestamp")
    s.add_argument("--format", choices=["table", "json"], default="table")
    s.add_argument("--verbose", "-v", action="store_true")
    s.set_defaults(func=cmd_alerts)

    s = sub.add_parser("notifiers", help="list registered notifier hooks")
    s.set_defaults(func=cmd_notifiers)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except MonitorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
