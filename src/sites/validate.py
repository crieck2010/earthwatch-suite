"""Structural validation of site configs against the canonical schema.

Stdlib-only on purpose (no ``jsonschema`` dependency): the validator
implements the rules in ``schema/site-config.schema.json`` directly and
reports failures as human-readable dotted-path errors. The JSON Schema
file remains the normative contract; this module is its executable twin.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    import yaml

    _HAVE_YAML = True
except ImportError:  # pragma: no cover
    _HAVE_YAML = False

SCHEMA_VERSION = "1.0.0"

_SITE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,63}$")

#: Indices survey-monitor can watch. Mirrors survey-imagery's registry and
#: survey-monitor's KNOWN_INDICES — kept as a literal so this package never
#: needs to import either engine.
KNOWN_INDICES = ("ndvi", "evi", "savi", "ndwi", "ndmi", "ndbi", "nbr", "bsi")

BASELINE_STRATEGIES = ("rolling", "fixed")
DIRECTIONS = ("up", "down", "either")
SEVERITIES = ("info", "warning", "critical")
PLANS = ("trial", "site", "unlimited")
ZONE_KINDS = (
    "pit", "stockpile", "field", "orchard", "pasture", "forest",
    "burn", "water", "wetland", "urban", "reference", "custom",
)


class ValidationError(Exception):
    """One or more schema violations. ``errors`` lists them all."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


class _Ctx:
    """Collects dotted-path errors."""

    def __init__(self) -> None:
        self.errors: List[str] = []

    def err(self, path: str, msg: str) -> None:
        self.errors.append(f"{path}: {msg}")

    def check_type(self, path: str, value: Any, types, what: str) -> bool:
        if not isinstance(value, types):
            self.err(path, f"expected {what}, got {type(value).__name__}")
            return False
        return True


def _validate_geometry(ctx: _Ctx, path: str, geom: Any) -> None:
    if not ctx.check_type(path, geom, dict, "object"):
        return
    if "bbox" in geom:
        bbox = geom["bbox"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or not all(
            isinstance(v, (int, float)) for v in bbox
        ):
            ctx.err(path, "bbox must be [min_lon, min_lat, max_lon, max_lat]")
            return
        if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            ctx.err(path, "bbox min must be less than max")
        return
    gtype = geom.get("type")
    if gtype not in ("Polygon", "MultiPolygon"):
        ctx.err(
            path,
            "geometry must be a GeoJSON Polygon/MultiPolygon or a "
            f"bbox object (got type={gtype!r})",
        )
        return
    coords = geom.get("coordinates")
    if not isinstance(coords, list) or not coords:
        ctx.err(path, "coordinates must be a non-empty array")
        return
    rings = coords if gtype == "Polygon" else [r for poly in coords for r in poly]
    for i, ring in enumerate(rings):
        if not isinstance(ring, list) or len(ring) < 4:
            ctx.err(path, f"ring {i}: needs at least 4 positions")
            continue
        if ring[0] != ring[-1]:
            ctx.err(path, f"ring {i}: first and last position must match (closed ring)")
        for j, pos in enumerate(ring):
            if (
                not isinstance(pos, (list, tuple))
                or len(pos) < 2
                or not all(isinstance(v, (int, float)) for v in pos[:2])
            ):
                ctx.err(path, f"ring {i} position {j}: expected [lon, lat] numbers")
                break


def _validate_id(ctx: _Ctx, path: str, value: Any) -> None:
    if not ctx.check_type(path, value, str, "string"):
        return
    if not _SITE_ID_RE.match(value):
        ctx.err(
            path,
            "must be lowercase alphanumeric with dashes/underscores "
            f"(got {value!r})",
        )


def _validate_threshold(ctx: _Ctx, path: str, thr: Any) -> None:
    if not ctx.check_type(path, thr, dict, "object"):
        return
    direction = thr.get("direction", "either")
    if direction not in DIRECTIONS:
        ctx.err(f"{path}.direction", f"must be one of {DIRECTIONS} (got {direction!r})")
    persistence = thr.get("persistence", 1)
    if not isinstance(persistence, int) or persistence < 1:
        ctx.err(f"{path}.persistence", "must be an integer >= 1")
    warning = thr.get("warning") or {}
    if not isinstance(warning, dict) or (
        warning.get("absolute_delta") is None and warning.get("z_score") is None
    ):
        ctx.err(
            f"{path}.warning",
            "needs at least one of 'absolute_delta' or 'z_score'",
        )
    for level in ("warning", "critical"):
        lvl = thr.get(level)
        if lvl is None:
            continue
        if not isinstance(lvl, dict):
            ctx.err(f"{path}.{level}", "must be an object")
            continue
        for key in ("absolute_delta", "z_score"):
            val = lvl.get(key)
            if val is not None and (
                not isinstance(val, (int, float)) or val < 0
            ):
                ctx.err(f"{path}.{level}.{key}", "must be a number >= 0")


def _validate_zones(ctx: _Ctx, zones: Any) -> None:
    if not ctx.check_type("zones", zones, list, "array"):
        return
    seen = set()
    for i, zone in enumerate(zones):
        path = f"zones[{i}]"
        if not isinstance(zone, dict):
            ctx.err(path, "must be an object")
            continue
        zid = zone.get("zone_id")
        _validate_id(ctx, f"{path}.zone_id", zid)
        if isinstance(zid, str):
            if zid in seen:
                ctx.err(f"{path}.zone_id", f"duplicate zone_id {zid!r}")
            seen.add(zid)
        kind = zone.get("kind", "custom")
        if kind not in ZONE_KINDS:
            ctx.err(f"{path}.kind", f"must be one of {ZONE_KINDS} (got {kind!r})")
        if "geometry" not in zone:
            ctx.err(path, "missing required 'geometry'")
        else:
            _validate_geometry(ctx, f"{path}.geometry", zone["geometry"])


def _validate_transects(ctx: _Ctx, transects: Any) -> None:
    if not ctx.check_type("transects", transects, list, "array"):
        return
    seen = set()
    for i, tr in enumerate(transects):
        path = f"transects[{i}]"
        if not isinstance(tr, dict):
            ctx.err(path, "must be an object")
            continue
        tid = tr.get("transect_id")
        _validate_id(ctx, f"{path}.transect_id", tid)
        if isinstance(tid, str):
            if tid in seen:
                ctx.err(f"{path}.transect_id", f"duplicate transect_id {tid!r}")
            seen.add(tid)
        geom = tr.get("geometry")
        if geom is None:
            ctx.err(path, "missing required 'geometry'")
        elif not isinstance(geom, dict) or geom.get("type") != "LineString":
            ctx.err(f"{path}.geometry", "must be a GeoJSON LineString")
        else:
            coords = geom.get("coordinates", [])
            if not isinstance(coords, list) or len(coords) < 2:
                ctx.err(f"{path}.geometry", "LineString needs at least 2 positions")


def _validate_engines(ctx: _Ctx, engines: Any, zones: List[Dict[str, Any]]) -> None:
    if not ctx.check_type("engines", engines, dict, "object"):
        return
    zone_ids = {z.get("zone_id") for z in zones if isinstance(z, dict)}
    dem3d = engines.get("dem3d")
    if dem3d is not None:
        if not isinstance(dem3d, dict):
            ctx.err("engines.dem3d", "must be an object")
        else:
            stable = dem3d.get("stable_zone")
            if stable is not None and stable not in zone_ids:
                ctx.err(
                    "engines.dem3d.stable_zone",
                    f"zone {stable!r} not found in zones[]",
                )
            for key in ("noise_floor_m", "dem_rmse_m"):
                val = dem3d.get(key)
                if val is not None and (
                    not isinstance(val, (int, float)) or val < 0
                ):
                    ctx.err(f"engines.dem3d.{key}", "must be a number >= 0")
    thermal = engines.get("thermal")
    if thermal is not None:
        if not isinstance(thermal, dict):
            ctx.err("engines.thermal", "must be an object")
        else:
            ref = thermal.get("reference_zone")
            if ref is not None and ref not in zone_ids:
                ctx.err(
                    "engines.thermal.reference_zone",
                    f"zone {ref!r} not found in zones[]",
                )
    for name in ("vegetation", "flood", "burn", "coast"):
        section = engines.get(name)
        if section is not None and not isinstance(section, dict):
            ctx.err(f"engines.{name}", "must be an object")


def _validate_alerts(ctx: _Ctx, alerts: Any) -> None:
    if not ctx.check_type("alerts", alerts, dict, "object"):
        return
    sev = alerts.get("severity_min", "warning")
    if sev not in SEVERITIES:
        ctx.err("alerts.severity_min", f"must be one of {SEVERITIES} (got {sev!r})")
    recipients = alerts.get("recipients", [])
    if not isinstance(recipients, list) or not all(
        isinstance(r, str) for r in recipients
    ):
        ctx.err("alerts.recipients", "must be an array of strings")


def _validate_license(ctx: _Ctx, lic: Any) -> None:
    if not ctx.check_type("license", lic, dict, "object"):
        return
    plan = lic.get("plan")
    if plan is not None and plan not in PLANS:
        ctx.err("license.plan", f"must be one of {PLANS} (got {plan!r})")
    key_id = lic.get("key_id")
    if key_id is not None and (
        not isinstance(key_id, str) or len(key_id) > 64
    ):
        ctx.err("license.key_id", "must be a short string (a key fingerprint, never a full key)")


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate a parsed site config. Returns a list of error strings (empty = valid)."""
    ctx = _Ctx()
    if not isinstance(config, dict):
        return ["config: expected an object at the top level"]

    _validate_id(ctx, "site_id", config.get("site_id"))
    name = config.get("name")
    if not isinstance(name, str) or not name:
        ctx.err("name", "required non-empty string")

    if "geometry" not in config:
        ctx.err("geometry", "required")
    else:
        _validate_geometry(ctx, "geometry", config["geometry"])

    indices = config.get("indices")
    if not isinstance(indices, list) or not indices:
        ctx.err("indices", "required non-empty array")
        indices = []
    else:
        unknown = [i for i in indices if i not in KNOWN_INDICES]
        if unknown:
            ctx.err("indices", f"unknown indices {unknown}; known: {list(KNOWN_INDICES)}")

    thresholds = config.get("thresholds")
    if not isinstance(thresholds, dict):
        ctx.err("thresholds", "required object")
    else:
        for index in indices:
            if index not in thresholds:
                ctx.err("thresholds", f"index {index!r} has no threshold entry")
            else:
                _validate_threshold(ctx, f"thresholds.{index}", thresholds[index])

    baseline = config.get("baseline") or {}
    if not isinstance(baseline, dict):
        ctx.err("baseline", "must be an object")
    else:
        strategy = baseline.get("strategy", "rolling")
        if strategy not in BASELINE_STRATEGIES:
            ctx.err("baseline.strategy", f"must be one of {BASELINE_STRATEGIES}")
        if strategy == "fixed" and not baseline.get("reference_pass"):
            ctx.err("baseline.reference_pass", "required when strategy is 'fixed'")

    max_cloud = config.get("max_cloud_cover", 30.0)
    if not isinstance(max_cloud, (int, float)) or not (0 <= max_cloud <= 100):
        ctx.err("max_cloud_cover", "must be a number between 0 and 100")

    cooldown = config.get("cooldown_passes", 0)
    if not isinstance(cooldown, int) or cooldown < 0:
        ctx.err("cooldown_passes", "must be an integer >= 0")

    zones = config.get("zones", [])
    _validate_zones(ctx, zones)
    _validate_transects(ctx, config.get("transects", []))
    if "engines" in config:
        _validate_engines(ctx, config["engines"], zones if isinstance(zones, list) else [])
    if "alerts" in config:
        _validate_alerts(ctx, config["alerts"])
    if "license" in config:
        _validate_license(ctx, config["license"])

    return ctx.errors


def _load_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith(".json"):
        return json.loads(text)
    if _HAVE_YAML:
        return yaml.safe_load(text)
    raise ValidationError(
        ["config: PyYAML is not installed; use a .json config or pip install pyyaml"]
    )


def validate_file(path: str) -> List[str]:
    """Load and validate a site config file (.yaml/.yml/.json)."""
    try:
        config = _load_file(path)
    except (json.JSONDecodeError, Exception) as exc:  # noqa: BLE001
        return [f"{path}: could not parse ({exc})"]
    errors = validate_config(config if isinstance(config, dict) else {})
    return [f"{path}: {e}" if not e.startswith(path) else e for e in errors]


def assert_valid_file(path: str) -> None:
    """Validate a file, raising ValidationError with every violation."""
    errors = validate_file(path)
    if errors:
        raise ValidationError(errors)
