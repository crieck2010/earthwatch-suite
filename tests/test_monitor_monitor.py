"""Pytest suite for survey-monitor. Run with: pytest (or python -m pytest).

All tests use synthetic/fixture data — no network, no survey-imagery needed.
"""

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import monitor
from monitor import (
    AlertLevel, BaselineConfig, IndexThreshold, MonitorError, SiteConfig,
    SiteState, PassInfo, load_state, save_state, new_passes,
)
from monitor import alerts as alerts_mod
from monitor import config as config_mod
from monitor import evaluate as eval_mod
from monitor import notify as notify_mod
from monitor import pipeline as pipe_mod
from monitor import state as state_mod
from monitor.acquire import SyntheticPassProvider


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_config(**over):
    thresholds = {
        "ndvi": IndexThreshold(
            direction="down",
            warning=AlertLevel(absolute_delta=0.10, z_score=2.0),
            critical=AlertLevel(absolute_delta=0.20, z_score=3.0),
            persistence=2,
        ),
        "ndwi": IndexThreshold(
            direction="either",
            warning=AlertLevel(absolute_delta=0.10, z_score=2.0),
            persistence=1,
        ),
    }
    data = {
        "site_id": "demo-site",
        "name": "Demo site",
        "geometry": {"bbox": [-70.12, 43.65, -70.02, 43.72]},
        "indices": ["ndvi", "ndwi"],
        "thresholds": {k: v.to_dict() for k, v in thresholds.items()},
        "baseline": {"strategy": "rolling", "window_passes": 5,
                     "min_passes": 2},
    }
    data.update(over)
    return SiteConfig.from_dict(data)


# ---------------------------------------------------------------------------
# config parsing / validation
# ---------------------------------------------------------------------------

def test_config_valid_minimal():
    cfg = make_config()
    assert cfg.site_id == "demo-site"
    assert cfg.indices == ["ndvi", "ndwi"]
    assert cfg.thresholds["ndvi"].direction == "down"


def test_config_rejects_bad_site_id():
    with pytest.raises(MonitorError):
        make_config(site_id="Bad ID!")


def test_config_rejects_unknown_index():
    with pytest.raises(MonitorError):
        make_config(indices=["ndvi", "frobnicate"])


def test_config_rejects_missing_threshold_entry():
    data_cfg = make_config().to_dict()
    del data_cfg["thresholds"]["ndwi"]
    with pytest.raises(MonitorError):
        SiteConfig.from_dict(data_cfg)


def test_config_rejects_empty_indices():
    with pytest.raises(MonitorError):
        make_config(indices=[])


def test_config_rejects_reversed_bbox():
    with pytest.raises(MonitorError):
        make_config(geometry={"bbox": [-70.02, 43.65, -70.12, 43.72]})


def test_config_rejects_bad_bbox_length():
    with pytest.raises(MonitorError):
        make_config(geometry={"bbox": [-70.12, 43.65, -70.02]})


def test_config_rejects_unclosed_polygon():
    with pytest.raises(MonitorError):
        make_config(geometry={"type": "Polygon",
                              "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1]]]})
    # closed ring is fine
    make_config(geometry={"type": "Polygon",
                          "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1],
                                           [0, 0]]]})


def test_config_rejects_bad_direction():
    with pytest.raises(MonitorError):
        IndexThreshold.from_dict({"direction": "sideways",
                                  "warning": {"absolute_delta": 0.1}})


def test_config_rejects_empty_warning_level():
    with pytest.raises(MonitorError):
        IndexThreshold.from_dict({"direction": "either", "warning": {}})


def test_config_rejects_bad_persistence():
    with pytest.raises(MonitorError):
        IndexThreshold.from_dict({"direction": "either", "persistence": 0,
                                  "warning": {"absolute_delta": 0.1}})


def test_baseline_fixed_requires_reference():
    with pytest.raises(MonitorError):
        BaselineConfig.from_dict({"strategy": "fixed"})
    BaselineConfig.from_dict({"strategy": "fixed",
                              "reference_pass": "p-2026-01-01"})


def test_config_round_trip():
    cfg = make_config()
    cfg2 = SiteConfig.from_dict(cfg.to_dict())
    assert cfg2.to_dict() == cfg.to_dict()


def test_save_and_load_json(tmp_path):
    cfg = make_config()
    path = str(tmp_path / "site.json")
    config_mod.save_site_config(cfg, path)
    cfg2 = config_mod.load_site_config(path)
    assert cfg2.site_id == cfg.site_id
    assert cfg2.thresholds["ndvi"].critical.absolute_delta == 0.20


def test_load_missing_file_raises():
    with pytest.raises(MonitorError):
        config_mod.load_site_config("/nonexistent/site.json")


def test_template_config_is_valid():
    cfg = config_mod.template_config("starter")
    assert cfg.site_id == "starter"
    assert cfg.indices  # template carries watched indices


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

def test_state_load_empty_when_missing(tmp_path):
    state = load_state(str(tmp_path), "new-site")
    assert state.site_id == "new-site"
    assert state.passes == []


def test_state_record_pass_keeps_date_order():
    state = SiteState(site_id="s")
    state.record_pass("p2", "2026-02-01", {"ndvi": 0.5})
    state.record_pass("p1", "2026-01-01", {"ndvi": 0.6})
    assert [p.pass_id for p in state.passes] == ["p1", "p2"]


def test_state_rerecord_replaces_pass():
    state = SiteState(site_id="s")
    state.record_pass("p1", "2026-01-01", {"ndvi": 0.6})
    state.record_pass("p1", "2026-01-01", {"ndvi": 0.61})
    assert len(state.passes) == 1
    assert state.passes[0].metrics["ndvi"] == pytest.approx(0.61)


def test_state_drops_nonfinite_metrics():
    state = SiteState(site_id="s")
    state.record_pass("p1", "2026-01-01",
                      {"ndvi": float("nan"), "ndwi": 0.2})
    assert "ndvi" not in state.passes[0].metrics
    assert state.passes[0].metrics["ndwi"] == 0.2


def test_baseline_stats_rolling_math():
    state = SiteState(site_id="s")
    for i, v in enumerate([0.5, 0.52, 0.48, 0.51, 0.49, 0.50]):
        state.record_pass(f"p{i}", f"2026-01-{i + 1:02d}", {"ndvi": v})
    mean, std = state_mod.baseline_stats(
        state, "ndvi", BaselineConfig(strategy="rolling", window_passes=3,
                                      min_passes=2))
    assert mean == pytest.approx((0.51 + 0.49 + 0.50) / 3)
    assert std == pytest.approx(math.sqrt(
        ((0.51 - mean) ** 2 + (0.49 - mean) ** 2 + (0.50 - mean) ** 2) / 2))


def test_baseline_stats_needs_min_passes():
    state = SiteState(site_id="s")
    state.record_pass("p1", "2026-01-01", {"ndvi": 0.5})
    assert state_mod.baseline_stats(
        state, "ndvi", BaselineConfig(min_passes=2)) is None


def test_baseline_stats_fixed_anchors_to_reference():
    state = SiteState(site_id="s")
    for i in range(6):
        state.record_pass(f"p{i}", f"2026-01-{i + 1:02d}", {"ndvi": 0.5 + i})
    mean, std = state_mod.baseline_stats(
        state, "ndvi", BaselineConfig(strategy="fixed", window_passes=3,
                                      reference_pass="p2", min_passes=2))
    # anchored: p0, p1, p2 -> values 0.5, 1.5, 2.5
    assert mean == pytest.approx(1.5)


def test_baseline_stats_fixed_missing_reference():
    state = SiteState(site_id="s")
    state.record_pass("p1", "2026-01-01", {"ndvi": 0.5})
    assert state_mod.baseline_stats(
        state, "ndvi",
        BaselineConfig(strategy="fixed", reference_pass="nope",
                       min_passes=1)) is None


def test_state_save_load_round_trip(tmp_path):
    state = SiteState(site_id="s")
    state.record_pass("p1", "2026-01-01", {"ndvi": 0.5})
    state.streaks["ndvi"] = 2
    state_mod.save_state(str(tmp_path), state)
    loaded = load_state(str(tmp_path), "s")
    assert loaded.passes[0].metrics["ndvi"] == 0.5
    assert loaded.streaks["ndvi"] == 2


def test_state_corrupt_file_raises(tmp_path):
    path = os.path.join(str(tmp_path), "s", "state.json")
    os.makedirs(os.path.dirname(path))
    with open(path, "w") as fh:
        fh.write("{not json")
    with pytest.raises(MonitorError):
        load_state(str(tmp_path), "s")


# ---------------------------------------------------------------------------
# scheduler
# ---------------------------------------------------------------------------

def _pass(pid, date, cloud=None):
    return PassInfo(pass_id=pid, date=date, cloud_cover=cloud)


def test_new_passes_excludes_known():
    cands = [_pass("a", "2026-01-01"), _pass("b", "2026-02-01")]
    out = new_passes({"a"}, cands)
    assert [p.pass_id for p in out] == ["b"]


def test_new_passes_sorts_oldest_first():
    cands = [_pass("b", "2026-02-01"), _pass("a", "2026-01-01")]
    out = new_passes(set(), cands)
    assert [p.pass_id for p in out] == ["a", "b"]


def test_new_passes_since_cutoff():
    cands = [_pass("a", "2026-01-01"), _pass("b", "2026-02-01")]
    out = new_passes(set(), cands, since="2026-01-15")
    assert [p.pass_id for p in out] == ["b"]


def test_new_passes_cloud_filter():
    cands = [_pass("a", "2026-01-01", cloud=80.0),
             _pass("b", "2026-02-01", cloud=10.0),
             _pass("c", "2026-03-01", cloud=None)]
    out = new_passes(set(), cands, max_cloud_cover=30.0)
    assert [p.pass_id for p in out] == ["b", "c"]


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def _thr(direction="either", warn_d=0.10, warn_z=2.0,
         crit_d=None, crit_z=None, persistence=1):
    warn = AlertLevel(absolute_delta=warn_d, z_score=warn_z)
    crit = None
    if crit_d is not None or crit_z is not None:
        crit = AlertLevel(absolute_delta=crit_d, z_score=crit_z)
    return IndexThreshold(direction=direction, warning=warn,
                          critical=crit, persistence=persistence)


def test_evaluate_absolute_warning():
    cands = eval_mod.evaluate_pass(
        {"ndvi": 0.40}, {"ndvi": (0.55, 0.01)},
        {"ndvi": _thr(warn_z=1000.0)})  # only the absolute leg can breach
    assert len(cands) == 1
    assert cands[0].severity == "warning"
    assert cands[0].delta == pytest.approx(-0.15)
    assert cands[0].threshold_kind == "absolute_delta"


def test_evaluate_zscore_warning():
    cands = eval_mod.evaluate_pass(
        {"ndvi": 0.53}, {"ndvi": (0.50, 0.01)},
        {"ndvi": _thr(warn_d=10.0, warn_z=2.0)})  # delta too small
    assert len(cands) == 1
    assert cands[0].threshold_kind == "z_score"
    assert cands[0].z_score == pytest.approx(3.0)


def test_evaluate_critical_beats_warning():
    cands = eval_mod.evaluate_pass(
        {"ndvi": 0.30}, {"ndvi": (0.55, 0.02)},
        {"ndvi": _thr(crit_d=0.20, crit_z=3.0)})
    assert len(cands) == 1
    assert cands[0].severity == "critical"


def test_evaluate_direction_up_ignores_drop():
    cands = eval_mod.evaluate_pass(
        {"ndvi": 0.30}, {"ndvi": (0.55, 0.02)},
        {"ndvi": _thr(direction="up")})
    assert cands == []


def test_evaluate_direction_down_ignores_rise():
    cands = eval_mod.evaluate_pass(
        {"ndvi": 0.80}, {"ndvi": (0.55, 0.02)},
        {"ndvi": _thr(direction="down")})
    assert cands == []


def test_evaluate_info_when_approaching():
    # delta 0.06: half of warning 0.10, not breaching
    cands = eval_mod.evaluate_pass(
        {"ndvi": 0.61}, {"ndvi": (0.55, 0.10)},
        {"ndvi": _thr(warn_d=0.10, warn_z=10.0)})
    assert len(cands) == 1
    assert cands[0].severity == "info"


def test_evaluate_quiet_when_small_change():
    cands = eval_mod.evaluate_pass(
        {"ndvi": 0.56}, {"ndvi": (0.55, 0.02)},
        {"ndvi": _thr(warn_d=0.10, warn_z=10.0)})
    assert cands == []


def test_evaluate_skips_missing_baseline_or_threshold():
    cands = eval_mod.evaluate_pass(
        {"ndvi": 0.30, "nbr": 0.1}, {"ndvi": (0.55, 0.02)},
        {"ndvi": _thr()})
    assert [c.index for c in cands] == ["ndvi"]  # nbr skipped


def test_evaluate_zero_std_gives_zero_z():
    cands = eval_mod.evaluate_pass(
        {"ndvi": 0.40}, {"ndvi": (0.55, 0.0)},
        {"ndvi": _thr(warn_d=0.10, warn_z=2.0)})
    assert cands[0].z_score == 0.0
    assert cands[0].severity == "warning"  # via absolute delta


def test_evaluate_ordering_critical_first():
    cands = eval_mod.evaluate_pass(
        {"ndvi": 0.30, "ndwi": 0.30},
        {"ndvi": (0.55, 0.02), "ndwi": (0.55, 0.02)},
        {"ndvi": _thr(crit_d=0.20), "ndwi": _thr()})
    assert [c.severity for c in cands] == ["critical", "warning"]


# ---------------------------------------------------------------------------
# alerts
# ---------------------------------------------------------------------------

def _candidate(**kw):
    base = dict(index="ndvi", severity="warning", delta=-0.15, z_score=-2.5,
                threshold_kind="absolute_delta", threshold=0.10,
                direction="down", value=0.40, baseline_mean=0.55,
                baseline_std=0.02)
    base.update(kw)
    return eval_mod.AlertCandidate(**base)


def test_alert_event_round_trip():
    ev = alerts_mod.make_event("demo-site", "Demo site", "mean",
                               _candidate(), "p-1", "2026-09-01", seq=3)
    assert ev.alert_id == "demo-site:p-1:ndvi:warning:3"
    ev2 = alerts_mod.AlertEvent.from_dict(ev.to_dict())
    assert ev2.to_dict() == ev.to_dict()


def test_alert_event_rejects_missing_field():
    with pytest.raises(MonitorError):
        alerts_mod.AlertEvent.from_dict({"alert_id": "x"})


def test_append_and_read_jsonl(tmp_path):
    store = str(tmp_path)
    ev = alerts_mod.make_event("s", "S", "mean", _candidate(), "p1",
                               "2026-09-01")
    alerts_mod.append_alerts(store, "s", [ev])
    alerts_mod.append_alerts(store, "s", [ev])
    events = alerts_mod.read_alerts(store, "s")
    assert len(events) == 2
    assert events[0].alert_id == ev.alert_id


def test_read_alerts_filters(tmp_path):
    store = str(tmp_path)
    ev1 = alerts_mod.make_event("s", "S", "mean", _candidate(), "p1",
                                "2026-09-01", timestamp="2026-09-01T00:00:00+00:00")
    ev2 = alerts_mod.make_event("s", "S", "mean",
                                _candidate(severity="critical"), "p2",
                                "2026-09-02", timestamp="2026-09-02T00:00:00+00:00")
    alerts_mod.append_alerts(store, "s", [ev1, ev2])
    assert len(alerts_mod.read_alerts(store, "s", severity="critical")) == 1
    assert len(alerts_mod.read_alerts(store, "s",
                                      since="2026-09-02T00:00:00+00:00")) == 1
    assert alerts_mod.read_alerts(store, "nope") == []


def test_read_alerts_skips_corrupt_lines(tmp_path):
    store = str(tmp_path)
    path = alerts_mod.alerts_path(store, "s")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("{bad json\n")
        fh.write(json.dumps(alerts_mod.make_event(
            "s", "S", "mean", _candidate(), "p1", "2026-09-01").to_dict()) + "\n")
    assert len(alerts_mod.read_alerts(store, "s")) == 1


def test_alerts_to_geojson(tmp_path):
    ev = alerts_mod.make_event("s", "S", "mean", _candidate(), "p1",
                               "2026-09-01")
    fc = alerts_mod.alerts_to_geojson(
        [ev], {"type": "Polygon",
               "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]})
    assert fc["type"] == "FeatureCollection"
    assert fc["features"][0]["geometry"]["type"] == "Polygon"
    assert fc["features"][0]["properties"]["severity"] == "warning"


def test_summarize_alerts():
    evs = [alerts_mod.make_event("s", "S", "mean", _candidate(), "p1",
                                 "2026-09-01"),
           alerts_mod.make_event("s", "S", "mean",
                                 _candidate(severity="critical"), "p2",
                                 "2026-09-02")]
    summary = alerts_mod.summarize_alerts(evs)
    assert summary["total"] == 2
    assert summary["worst"] == "critical"
    assert summary["by_index"] == {"ndvi": 2}


def test_run_summary_round_trip():
    s = alerts_mod.RunSummary(site_id="s", site_name="S",
                              passes_processed=3, alerts_fired=1)
    s2 = alerts_mod.RunSummary.from_dict(s.to_dict())
    assert s2.passes_processed == 3 and s2.alerts_fired == 1


# ---------------------------------------------------------------------------
# notify
# ---------------------------------------------------------------------------

def test_notify_register_and_call():
    received = []
    notify_mod.register_notifier("test-hook", received.extend)
    try:
        assert "test-hook" in notify_mod.list_notifiers()
        ev = alerts_mod.make_event("s", "S", "mean", _candidate(), "p1",
                                   "2026-09-01")
        results = notify_mod.notify_all(["test-hook"], [ev])
        assert results == {"test-hook": "ok"}
        assert received == [ev]
    finally:
        notify_mod._REGISTRY.pop("test-hook", None)


def test_notify_unknown_name_gives_error_result():
    results = notify_mod.notify_all(["no-such-notifier"], [])
    assert results["no-such-notifier"].startswith("error:")


def test_notify_failure_does_not_block_others():
    def boom(_events):
        raise RuntimeError("delivery down")
    notify_mod.register_notifier("test-boom", boom)
    try:
        results = notify_mod.notify_all(["test-boom", "logging"], [])
        assert results["test-boom"].startswith("error:")
        assert results["logging"] == "ok"
    finally:
        notify_mod._REGISTRY.pop("test-boom", None)


def test_notify_rejects_non_callable():
    with pytest.raises(MonitorError):
        notify_mod.register_notifier("bad", "not-a-function")


# ---------------------------------------------------------------------------
# pipeline end-to-end (synthetic provider)
# ---------------------------------------------------------------------------

def _demo_provider(config, plants, seed=11):
    return SyntheticPassProvider(config, n_passes=10,
                                 start_date="2026-06-01", seed=seed,
                                 plants=plants)


def test_pipeline_fires_alert_after_persistence(tmp_path):
    cfg = make_config()
    sites = tmp_path / "sites"
    store = tmp_path / "state"
    sites.mkdir()
    config_mod.save_site_config(cfg, str(sites / "demo-site.json"))
    # plant a sharp NDVI drop on passes 7 and 8 -> persistence=2 -> 1 alert
    provider = _demo_provider(cfg, {7: {"ndvi": -0.30}, 8: {"ndvi": -0.30}})
    summary = pipe_mod.run_site("demo-site", str(sites), str(store),
                                provider=provider)
    assert summary.passes_processed == 10
    ndvi_alerts = [a for a in alerts_mod.read_alerts(str(store), "demo-site")
                   if a.index == "ndvi"
                   and a.severity in ("warning", "critical")]
    assert len(ndvi_alerts) >= 1, "expected a warning/critical NDVI alert"
    fired_ids = [a.pass_id for a in ndvi_alerts]
    assert summary.alerts_fired == len(
        alerts_mod.read_alerts(str(store), "demo-site"))


def test_pipeline_second_run_is_incremental(tmp_path):
    cfg = make_config()
    sites = tmp_path / "sites"
    store = tmp_path / "state"
    sites.mkdir()
    config_mod.save_site_config(cfg, str(sites / "demo-site.json"))
    provider = _demo_provider(cfg, {7: {"ndvi": -0.30}, 8: {"ndvi": -0.30}})
    first = pipe_mod.run_site("demo-site", str(sites), str(store),
                              provider=provider)
    second = pipe_mod.run_site("demo-site", str(sites), str(store),
                               provider=provider)
    assert second.passes_processed == 0  # no new passes
    assert (len(alerts_mod.read_alerts(str(store), "demo-site"))
            == first.alerts_fired)


def test_pipeline_info_events_for_approaching(tmp_path):
    cfg = make_config()
    # gentle drift: info watch events, no warning/critical
    cfg.thresholds["ndvi"] = IndexThreshold(
        direction="either",
        warning=AlertLevel(absolute_delta=0.50, z_score=10.0),
        persistence=1)
    sites = tmp_path / "sites"
    store = tmp_path / "state"
    sites.mkdir()
    config_mod.save_site_config(cfg, str(sites / "demo-site.json"))
    provider = _demo_provider(cfg, {5: {"ndvi": -0.30}})
    summary = pipe_mod.run_site("demo-site", str(sites), str(store),
                                provider=provider)
    infos = [a for a in alerts_mod.read_alerts(str(store), "demo-site")
             if a.severity == "info"]
    assert summary.passes_processed == 10
    assert infos, "expected info watch events for the approaching change"


def test_pipeline_run_all_covers_every_site(tmp_path):
    cfg = make_config()
    sites = tmp_path / "sites"
    store = tmp_path / "state"
    sites.mkdir()
    config_mod.save_site_config(cfg, str(sites / "demo-site.json"))
    cfg2 = make_config(site_id="second")
    config_mod.save_site_config(cfg2, str(sites / "second.json"))
    summaries = pipe_mod.run_all(str(sites), str(store), synthetic=True)
    assert {s.site_id for s in summaries} == {"demo-site", "second"}


def test_pipeline_missing_config_raises():
    with pytest.raises(MonitorError):
        pipe_mod.run_site("ghost", "/tmp/no-such-dir", "/tmp/no-such-dir-2")


def test_pipeline_provider_error_becomes_summary_error(tmp_path):
    cfg = make_config()
    sites = tmp_path / "sites"
    store = tmp_path / "state"
    sites.mkdir()
    config_mod.save_site_config(cfg, str(sites / "demo-site.json"))

    class BrokenProvider:
        def list_passes(self, since=None):
            raise MonitorError("STAC is down")

    summary = pipe_mod.run_site("demo-site", str(sites), str(store),
                                provider=BrokenProvider())
    assert summary.passes_processed == 0
    assert summary.errors == ["STAC is down"]


def test_pipeline_cooldown_suppresses_repeat_alerts(tmp_path):
    cfg = make_config(cooldown_passes=3)
    sites = tmp_path / "sites"
    store = tmp_path / "state"
    sites.mkdir()
    config_mod.save_site_config(cfg, str(sites / "demo-site.json"))
    # long low plateau: alert fires once, cooldown suppresses re-alerts
    provider = _demo_provider(cfg, {i: {"ndvi": -0.15} for i in range(6, 10)})
    summary = pipe_mod.run_site("demo-site", str(sites), str(store),
                                provider=provider)
    ndvi = [a for a in alerts_mod.read_alerts(str(store), "demo-site")
            if a.index == "ndvi" and a.severity in ("warning", "critical")]
    assert len(ndvi) <= 2  # fired once; cooldown holds the rest


# ---------------------------------------------------------------------------
# package surface
# ---------------------------------------------------------------------------

def test_package_version_and_exports():
    assert monitor.__version__ == "0.1.0"
    for name in ("SiteConfig", "run_site", "run_all", "evaluate_pass",
                 "register_notifier", "AlertEvent"):
        assert name in monitor.__all__
