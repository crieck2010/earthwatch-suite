"""Site configuration schema for survey-monitor.

A site is described by one YAML or JSON file (see ``examples/site_config.yaml``).
Parsing and validation here are pure logic — no network, no UI.

The full field reference lives in ``docs/SITE_CONFIG.md``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Indices survey-monitor can watch. Mirrors the spectral-index registry of
#: ``survey-imagery`` (``imagery.indices``); validated locally so this engine
#: never needs to import survey-imagery.
KNOWN_INDICES = (
    "ndvi", "evi", "savi", "ndwi", "ndmi", "ndbi", "nbr", "bsi",
)

#: Baseline strategies.
BASELINE_STRATEGIES = ("rolling", "fixed")

#: Threshold alert directions.
DIRECTIONS = ("up", "down", "either")

_SITE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,63}$")


class MonitorError(Exception):
    """Raised for any configuration, state, or pipeline error in survey-monitor."""


# ---------------------------------------------------------------------------
# Alert levels and per-index thresholds
# ---------------------------------------------------------------------------


@dataclass
class AlertLevel:
    """One severity level of a per-index threshold.

    A pass breaches the level when *either* condition holds (after the
    direction filter is applied):

    * ``|delta| >= absolute_delta`` — raw change in index units, or
    * ``|z| >= z_score`` — change standardised against the baseline spread.
    """

    absolute_delta: Optional[float] = None
    z_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"absolute_delta": self.absolute_delta, "z_score": self.z_score}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlertLevel":
        return cls(
            absolute_delta=data.get("absolute_delta"),
            z_score=data.get("z_score"),
        )


@dataclass
class IndexThreshold:
    """Watch specification for one spectral index.

    ``direction`` selects which sign of change matters: ``"up"`` (index
    rising, e.g. vegetation gain), ``"down"`` (index falling, e.g. clearing),
    or ``"either"``. ``persistence`` is the number of *consecutive* breaching
    passes required before an alert fires — it is the main defence against
    single-pass noise.
    """

    direction: str = "either"
    warning: AlertLevel = field(default_factory=AlertLevel)
    critical: Optional[AlertLevel] = None
    persistence: int = 1

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "direction": self.direction,
            "warning": self.warning.to_dict(),
            "persistence": self.persistence,
        }
        if self.critical is not None:
            d["critical"] = self.critical.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexThreshold":
        direction = data.get("direction", "either")
        if direction not in DIRECTIONS:
            raise MonitorError(f"threshold direction must be one of {DIRECTIONS}, "
                               f"got {direction!r}")
        persistence = data.get("persistence", 1)
        if not isinstance(persistence, int) or persistence < 1:
            raise MonitorError("threshold persistence must be an integer >= 1")
        warning = AlertLevel.from_dict(data.get("warning") or {})
        if warning.absolute_delta is None and warning.z_score is None:
            raise MonitorError("threshold 'warning' needs at least one of "
                               "'absolute_delta' or 'z_score'")
        critical_data = data.get("critical")
        critical = AlertLevel.from_dict(critical_data) if critical_data else None
        return cls(direction=direction, warning=warning,
                   critical=critical, persistence=persistence)


# ---------------------------------------------------------------------------
# Baseline strategy
# ---------------------------------------------------------------------------


@dataclass
class BaselineConfig:
    """How the comparison baseline is formed.

    * ``rolling`` — mean/std over the trailing ``window_passes`` processed
      passes (excludes the pass being evaluated). Adapts to slow drift.
    * ``fixed`` — statistics frozen from the single ``reference_pass`` (a
      ``pass_id`` recorded by an earlier run). Anchored, auditable.
    """

    strategy: str = "rolling"
    window_passes: int = 5
    reference_pass: Optional[str] = None
    min_passes: int = 2  # passes required before any baseline exists

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "window_passes": self.window_passes,
            "reference_pass": self.reference_pass,
            "min_passes": self.min_passes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineConfig":
        strategy = data.get("strategy", "rolling")
        if strategy not in BASELINE_STRATEGIES:
            raise MonitorError(
                f"baseline strategy must be one of {BASELINE_STRATEGIES}, "
                f"got {strategy!r}")
        window = data.get("window_passes", 5)
        if not isinstance(window, int) or window < 1:
            raise MonitorError("baseline window_passes must be an integer >= 1")
        min_passes = data.get("min_passes", 2)
        if not isinstance(min_passes, int) or min_passes < 1:
            raise MonitorError("baseline min_passes must be an integer >= 1")
        reference = data.get("reference_pass")
        if strategy == "fixed" and not reference:
            raise MonitorError("baseline strategy 'fixed' requires "
                               "'reference_pass' (a pass_id from a prior run)")
        return cls(strategy=strategy, window_passes=window,
                   reference_pass=reference, min_passes=min_passes)


# ---------------------------------------------------------------------------
# Site configuration
# ---------------------------------------------------------------------------


@dataclass
class SiteConfig:
    """One monitored site. ``geometry`` is either a GeoJSON geometry dict or
    ``{"bbox": [min_lon, min_lat, max_lon, max_lat]}``.
    """

    site_id: str
    name: str
    geometry: Dict[str, Any]
    indices: List[str]
    thresholds: Dict[str, IndexThreshold]
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    collection: str = "sentinel-2-l2a"
    api_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    max_cloud_cover: float = 30.0
    metric: str = "mean"           # which zonal statistic is watched
    cooldown_passes: int = 0      # quiet passes required before re-alerting
    notifiers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "site_id": self.site_id,
            "name": self.name,
            "geometry": self.geometry,
            "indices": list(self.indices),
            "thresholds": {k: v.to_dict() for k, v in self.thresholds.items()},
            "baseline": self.baseline.to_dict(),
            "collection": self.collection,
            "api_url": self.api_url,
            "max_cloud_cover": self.max_cloud_cover,
            "metric": self.metric,
            "cooldown_passes": self.cooldown_passes,
            "notifiers": list(self.notifiers),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SiteConfig":
        site_id = data.get("site_id", "")
        if not isinstance(site_id, str) or not _SITE_ID_RE.match(site_id):
            raise MonitorError(
                "site_id must be lowercase alphanumeric with dashes/underscores "
                f"(got {site_id!r})")
        name = data.get("name") or site_id
        geometry = data.get("geometry") or {}
        _validate_geometry(geometry)
        indices = data.get("indices") or []
        if not indices:
            raise MonitorError("site needs at least one index in 'indices'")
        unknown = [i for i in indices if i not in KNOWN_INDICES]
        if unknown:
            raise MonitorError(f"unknown indices {unknown}; known: "
                               f"{list(KNOWN_INDICES)}")
        thresholds = {}
        for index in indices:
            raw = (data.get("thresholds") or {}).get(index)
            if raw is None:
                raise MonitorError(
                    f"index {index!r} has no entry in 'thresholds'")
            thresholds[index] = IndexThreshold.from_dict(raw)
        baseline = BaselineConfig.from_dict(data.get("baseline") or {})
        max_cloud = data.get("max_cloud_cover", 30.0)
        if not (0.0 <= float(max_cloud) <= 100.0):
            raise MonitorError("max_cloud_cover must be between 0 and 100")
        cooldown = data.get("cooldown_passes", 0)
        if not isinstance(cooldown, int) or cooldown < 0:
            raise MonitorError("cooldown_passes must be an integer >= 0")
        return cls(
            site_id=site_id,
            name=name,
            geometry=geometry,
            indices=list(indices),
            thresholds=thresholds,
            baseline=baseline,
            collection=data.get("collection", "sentinel-2-l2a"),
            api_url=data.get("api_url",
                             "https://planetarycomputer.microsoft.com/api/stac/v1"),
            max_cloud_cover=float(max_cloud),
            metric=data.get("metric", "mean"),
            cooldown_passes=cooldown,
            notifiers=list(data.get("notifiers") or []),
        )


def _validate_geometry(geometry: Dict[str, Any]) -> None:
    """Accept a GeoJSON geometry dict or ``{"bbox": [minx, miny, maxx, maxy]}``."""
    if not isinstance(geometry, dict):
        raise MonitorError("geometry must be a dict (GeoJSON geometry or bbox)")
    if "bbox" in geometry:
        bbox = geometry["bbox"]
        if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                or not all(isinstance(v, (int, float)) for v in bbox)):
            raise MonitorError("geometry.bbox must be [min_lon, min_lat, "
                               "max_lon, max_lat]")
        minx, miny, maxx, maxy = bbox
        if not (minx < maxx and miny < maxy):
            raise MonitorError("geometry.bbox must satisfy "
                               "min_lon < max_lon and min_lat < max_lat")
        if not (-180 <= minx <= 180 and -180 <= maxx <= 180
                and -90 <= miny <= 90 and -90 <= maxy <= 90):
            raise MonitorError("geometry.bbox out of lon/lat range")
        return
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if gtype == "Polygon":
        if not (isinstance(coords, list) and coords and
                isinstance(coords[0], list) and len(coords[0]) >= 4):
            raise MonitorError("Polygon geometry needs a closed linear ring "
                               "(>= 4 positions)")
        ring = coords[0]
        if ring[0] != ring[-1]:
            raise MonitorError("Polygon ring must be closed (first == last "
                               "position)")
    elif gtype == "MultiPolygon":
        if not (isinstance(coords, list) and coords and
                isinstance(coords[0], list) and coords[0] and
                isinstance(coords[0][0], list)):
            raise MonitorError("MultiPolygon geometry has invalid coordinates")
    else:
        raise MonitorError(
            f"geometry must be a GeoJSON Polygon/MultiPolygon or a bbox dict "
            f"(got type {gtype!r})")


# ---------------------------------------------------------------------------
# File IO
# ---------------------------------------------------------------------------


def load_site_config(path: str) -> SiteConfig:
    """Load and validate a site config from a JSON or YAML file."""
    if not os.path.isfile(path):
        raise MonitorError(f"site config not found: {path}")
    _, ext = os.path.splitext(path.lower())
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if ext in (".yaml", ".yml"):
        try:
            import yaml  # lazy optional dependency
        except ImportError as exc:
            raise MonitorError(
                "YAML site configs need PyYAML: pip install 'survey-monitor[yaml]'"
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise MonitorError(f"site config {path} must be a JSON/YAML object")
    return SiteConfig.from_dict(data)


def save_site_config(config: SiteConfig, path: str) -> str:
    """Write a site config as JSON; returns the path written."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config.to_dict(), fh, indent=2)
        fh.write("\n")
    return path


def template_config(site_id: str = "demo-site") -> SiteConfig:
    """Return a starter SiteConfig a user can edit and save."""
    return SiteConfig(
        site_id=site_id,
        name="Demo site",
        geometry={"bbox": [-70.12, 43.65, -70.02, 43.72]},
        indices=["ndvi", "ndwi"],
        thresholds={
            "ndvi": IndexThreshold(
                direction="down",
                warning=AlertLevel(absolute_delta=0.10, z_score=2.0),
                critical=AlertLevel(absolute_delta=0.20, z_score=3.0),
                persistence=2,
            ),
            "ndwi": IndexThreshold(
                direction="either",
                warning=AlertLevel(absolute_delta=0.10, z_score=2.0),
                critical=AlertLevel(absolute_delta=0.20, z_score=3.0),
                persistence=2,
            ),
        },
        baseline=BaselineConfig(strategy="rolling", window_passes=5),
    )
