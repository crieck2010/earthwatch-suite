"""Run orchestration: one site or every site, headless and cron-friendly.

``run_site`` is the whole engine in one call: load config, load state,
list passes, evaluate the new ones, fire alerts, notify, persist.
``run_all`` walks a directory of site configs. Every run writes
``summary.json`` and appends to ``alerts.jsonl`` + ``alerts.geojson``.
"""

from __future__ import annotations

import datetime as _dt
import glob
import os
from typing import Callable, Dict, List, Optional, Tuple

from .acquire import ImageryPassProvider, PassProvider
from .alerts import (RunSummary, append_alerts, make_event, read_alerts,
                     summarize_alerts, write_alerts_geojson, write_summary)
from .config import MonitorError, SiteConfig, load_site_config
from .evaluate import evaluate_pass
from .notify import notify_all
from .scheduler import PassInfo, new_passes
from .state import SiteState, baseline_stats, load_state, save_state

Progress = Callable[[str], None]


def _null_progress(_msg: str) -> None:
    pass


def _provider_for(config: SiteConfig, synthetic: bool) -> PassProvider:
    if synthetic:
        from .acquire import SyntheticPassProvider
        return SyntheticPassProvider(config)
    return ImageryPassProvider(config)


def _baseline_map(state: SiteState, config: SiteConfig,
                  exclude_pass_id: Optional[str] = None
                  ) -> Dict[str, Tuple[float, float]]:
    """Baseline (mean, std) per index, excluding the pass under evaluation."""
    saved = state.passes
    if exclude_pass_id is not None:
        state = SiteState(
            site_id=state.site_id,
            passes=[p for p in saved if p.pass_id != exclude_pass_id],
            streaks=dict(state.streaks),
            last_alert_pass=dict(state.last_alert_pass),
            last_severity=dict(state.last_severity),
            last_run=state.last_run,
        )
    out: Dict[str, Tuple[float, float]] = {}
    for index in config.indices:
        stats = baseline_stats(state, index, config.baseline)
        if stats is not None:
            out[index] = stats
    return out


def run_site(site_id: str, config_dir: str, store_dir: str,
             provider: Optional[PassProvider] = None,
             since: Optional[str] = None,
             synthetic: bool = False,
             progress: Optional[Progress] = None) -> RunSummary:
    """Run the monitor for one site. Returns the run summary."""
    log = progress or _null_progress
    config_path = os.path.join(config_dir, f"{site_id}.json")
    if not os.path.isfile(config_path):
        config_path = os.path.join(config_dir, f"{site_id}.yaml")
    if not os.path.isfile(config_path):
        config_path = os.path.join(config_dir, f"{site_id}.yml")
    if not os.path.isfile(config_path):
        raise MonitorError(f"no site config for {site_id!r} in {config_dir} "
                           f"(expected {site_id}.json/.yaml)")
    config = load_site_config(config_path)
    if config.site_id != site_id:
        raise MonitorError(f"config file {config_path} has site_id "
                           f"{config.site_id!r}, expected {site_id!r}")

    state = load_state(store_dir, site_id)
    summary = RunSummary(site_id=site_id, site_name=config.name)

    if provider is None:
        provider = _provider_for(config, synthetic)

    try:
        candidates = provider.list_passes(since=since)
    except MonitorError as exc:
        summary.errors.append(str(exc))
        write_summary(store_dir, summary)
        return summary

    passes = new_passes(state.known_pass_ids(), candidates,
                        max_cloud_cover=config.max_cloud_cover)
    log(f"{site_id}: {len(passes)} new pass(es)")

    fired = []
    seq = len(read_alerts(store_dir, site_id))
    for n, info in enumerate(passes, start=1):
        log(f"  pass {n}/{len(passes)}: {info.pass_id} ({info.date})")
        try:
            metrics = provider.metrics(info.pass_id)
        except MonitorError as exc:
            summary.passes_skipped += 1
            summary.errors.append(f"{info.pass_id}: {exc}")
            continue
        metrics = {k: metrics[k] for k in config.indices if k in metrics}
        if not metrics:
            summary.passes_skipped += 1
            continue

        baselines = _baseline_map(state, config)
        candidates_hit = evaluate_pass(metrics, baselines, config.thresholds)

        # Record the pass first so the state always advances, even when the
        # baseline was too thin to evaluate this pass.
        state.record_pass(info.pass_id, info.date, metrics)
        summary.passes_processed += 1

        for cand in candidates_hit:
            threshold = config.thresholds[cand.index]
            streak = state.streaks.get(cand.index, 0)
            if cand.severity in ("warning", "critical"):
                streak += 1
            else:  # info watch events never build persistence streaks
                streak = 0
            state.streaks[cand.index] = streak

            if cand.severity == "info":
                fired.append(make_event(config.site_id, config.name,
                                        config.metric, cand, info.pass_id,
                                        info.date, seq=seq))
                seq += 1
                continue

            if streak < threshold.persistence:
                continue
            if config.cooldown_passes:
                last_id = state.last_alert_pass.get(cand.index)
                if last_id is not None:
                    idx_now = [p.pass_id for p in state.passes].index(
                        info.pass_id)
                    idx_last = [p.pass_id for p in state.passes].index(last_id) \
                        if last_id in state.known_pass_ids() else -1
                    if 0 <= idx_now - idx_last - 1 < config.cooldown_passes:
                        continue
            fired.append(make_event(config.site_id, config.name,
                                    config.metric, cand, info.pass_id,
                                    info.date, seq=seq))
            seq += 1
            state.streaks[cand.index] = 0  # reset after firing
            state.last_alert_pass[cand.index] = info.pass_id
            state.last_severity[cand.index] = cand.severity

    if fired:
        append_alerts(store_dir, site_id, fired)
        # Keep the QGIS-friendly view in sync with the log.
        write_alerts_geojson(store_dir, site_id,
                             read_alerts(store_dir, site_id),
                             config.geometry)
        if config.notifiers:
            results = notify_all(config.notifiers, fired)
            failed = [f"{k}: {v}" for k, v in results.items()
                      if not v.startswith("ok")]
            summary.errors.extend(failed)
        summary.alerts_fired = len(fired)
        summary.alerts = summarize_alerts(fired)

    state.last_run = _dt.datetime.now(_dt.timezone.utc).isoformat(
        timespec="seconds")
    save_state(store_dir, state)
    write_summary(store_dir, summary)
    log(f"{site_id}: {summary.passes_processed} processed, "
        f"{summary.alerts_fired} alert(s)")
    return summary


def run_all(config_dir: str, store_dir: str,
            since: Optional[str] = None,
            synthetic: bool = False,
            progress: Optional[Progress] = None) -> List[RunSummary]:
    """Run the monitor for every ``*.json``/``*.yaml`` config in ``config_dir``."""
    paths = sorted(glob.glob(os.path.join(config_dir, "*.json"))
                   + glob.glob(os.path.join(config_dir, "*.yaml"))
                   + glob.glob(os.path.join(config_dir, "*.yml")))
    summaries: List[RunSummary] = []
    for path in paths:
        site_id = os.path.splitext(os.path.basename(path))[0]
        try:
            summaries.append(run_site(site_id, config_dir, store_dir,
                                      since=since, synthetic=synthetic,
                                      progress=progress))
        except MonitorError as exc:
            summary = RunSummary(site_id=site_id, site_name=site_id)
            summary.errors.append(str(exc))
            summaries.append(summary)
    return summaries
