"""Scaffold new site configs from templates."""

from __future__ import annotations

import os
from typing import Any, Dict

try:
    import yaml

    _HAVE_YAML = True
except ImportError:  # pragma: no cover
    _HAVE_YAML = False

TEMPLATES = ("minimal", "quarry", "farm", "coast", "forest")


def _base(site_id: str, name: str) -> Dict[str, Any]:
    return {
        "site_id": site_id,
        "name": name,
        "description": "",
        "crs": "EPSG:4326",
        "geometry": {"bbox": [-77.70, 43.05, -77.60, 43.12]},
        "zones": [],
        "indices": ["ndvi", "ndwi"],
        "thresholds": {
            "ndvi": {
                "direction": "down",
                "persistence": 2,
                "warning": {"absolute_delta": 0.10, "z_score": 2.0},
                "critical": {"absolute_delta": 0.20, "z_score": 3.0},
            },
            "ndwi": {
                "direction": "either",
                "persistence": 2,
                "warning": {"absolute_delta": 0.10, "z_score": 2.0},
            },
        },
        "baseline": {"strategy": "rolling", "window_passes": 5, "min_passes": 2},
        "collection": "sentinel-2-l2a",
        "api_url": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "max_cloud_cover": 30.0,
        "metric": "mean",
        "cooldown_passes": 0,
        "notifiers": [],
        "alerts": {"recipients": [], "severity_min": "warning", "digest": True},
        "license": {"plan": "trial"},
    }


def _zone(zone_id: str, name: str, kind: str, ring) -> Dict[str, Any]:
    return {
        "zone_id": zone_id,
        "name": name,
        "kind": kind,
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def scaffold(site_id: str, name: str, template: str = "minimal") -> Dict[str, Any]:
    """Build a site config dict from a named template."""
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}; choose from {TEMPLATES}")
    cfg = _base(site_id, name or site_id.replace("-", " ").title())

    if template == "quarry":
        cfg["description"] = "Active quarry: pit extraction and stockpile growth."
        cfg["indices"] = ["ndvi", "ndwi", "nbr"]
        cfg["thresholds"]["nbr"] = {
            "direction": "down",
            "persistence": 2,
            "warning": {"absolute_delta": 0.08, "z_score": 2.0},
        }
        cfg["zones"] = [
            _zone("pit-north", "North pit (active cut)", "pit",
                  [[-77.68, 43.08], [-77.64, 43.08], [-77.64, 43.11], [-77.68, 43.11], [-77.68, 43.08]]),
            _zone("stockpile-south", "South stockpile (fill)", "stockpile",
                  [[-77.68, 43.05], [-77.64, 43.05], [-77.64, 43.075], [-77.68, 43.075], [-77.68, 43.05]]),
            _zone("stable-road", "Haul road (stable reference)", "reference",
                  [[-77.635, 43.08], [-77.625, 43.08], [-77.625, 43.11], [-77.635, 43.11], [-77.635, 43.08]]),
        ]
        cfg["engines"] = {
            "dem3d": {"noise_floor_m": 0.15, "dem_rmse_m": 0.10,
                      "stable_zone": "stable-road"}
        }
    elif template == "farm":
        cfg["description"] = "Row-crop farm: vigor, stress anomalies, and heat."
        cfg["zones"] = [
            _zone("field-a", "Field A (corn)", "field",
                  [[-77.92, 42.93], [-77.88, 42.93], [-77.88, 42.96], [-77.92, 42.96], [-77.92, 42.93]]),
            _zone("field-b", "Field B (orchard)", "orchard",
                  [[-77.875, 42.93], [-77.845, 42.93], [-77.845, 42.96], [-77.875, 42.96], [-77.875, 42.93]]),
            _zone("field-c", "Field C (pasture, rural reference)", "pasture",
                  [[-77.92, 42.965], [-77.845, 42.965], [-77.845, 42.985], [-77.92, 42.985], [-77.92, 42.965]]),
        ]
        cfg["engines"] = {
            "vegetation": {"indices": ["ndvi", "evi", "savi"], "anomaly_z": 2.0,
                           "min_history_years": 1},
            "thermal": {"reference_zone": "field-c", "heatwave_threshold_c": 35.0,
                        "heatwave_passes": 2},
        }
    elif template == "coast":
        cfg["description"] = "Coastal site: shoreline change and flood extent."
        cfg["indices"] = ["ndwi", "ndvi"]
        cfg["thresholds"]["ndwi"]["direction"] = "either"
        cfg["zones"] = [
            _zone("marsh", "Tidal marsh (flood watch)", "wetland",
                  [[-72.88, 40.60], [-72.84, 40.60], [-72.84, 40.63], [-72.88, 40.63], [-72.88, 40.60]]),
            _zone("beach", "Beach face (erosion watch)", "custom",
                  [[-72.88, 40.635], [-72.84, 40.635], [-72.84, 40.65], [-72.88, 40.65], [-72.88, 40.635]]),
        ]
        cfg["transects"] = [
            {"transect_id": "shore-a", "name": "Transect A (west)",
             "geometry": {"type": "LineString",
                          "coordinates": [[-72.87, 40.655], [-72.87, 40.615]]}},
            {"transect_id": "shore-b", "name": "Transect B (east)",
             "geometry": {"type": "LineString",
                          "coordinates": [[-72.85, 40.655], [-72.85, 40.615]]}},
        ]
        cfg["engines"] = {
            "coast": {"index": "ndwi", "transect_ids": ["shore-a", "shore-b"]},
            "flood": {"index": "ndwi", "threshold": 0.0, "min_area_ha": 1.0,
                      "baseline_months": [6, 7, 8]},
        }
    elif template == "forest":
        cfg["description"] = "Forest site: burn scars and disturbance."
        cfg["indices"] = ["nbr", "ndvi"]
        cfg["thresholds"]["nbr"] = {
            "direction": "down",
            "persistence": 2,
            "warning": {"absolute_delta": 0.08, "z_score": 2.0},
            "critical": {"absolute_delta": 0.15, "z_score": 3.0},
        }
        cfg["zones"] = [
            _zone("stand-1", "Stand 1 (burn watch)", "forest",
                  [[-74.12, 44.03], [-74.08, 44.03], [-74.08, 44.06], [-74.12, 44.06], [-74.12, 44.03]]),
            _zone("stand-2", "Stand 2 (control)", "forest",
                  [[-74.075, 44.03], [-74.045, 44.03], [-74.045, 44.06], [-74.075, 44.06], [-74.075, 44.03]]),
        ]
        cfg["engines"] = {"burn": {"severity_bands": "usgs", "min_area_ha": 0.5}}
    return cfg


def write_config(config: Dict[str, Any], path: str) -> str:
    """Write a config dict to path as YAML (or JSON for .json)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        if path.endswith(".json"):
            import json

            json.dump(config, fh, indent=2)
        elif _HAVE_YAML:
            yaml.safe_dump(config, fh, sort_keys=False, default_flow_style=False)
        else:  # pragma: no cover
            raise RuntimeError("PyYAML is required to write .yaml configs")
    return path
