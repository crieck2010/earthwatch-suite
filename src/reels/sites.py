"""Build reel recipes from survey-sites site configs.

Reads a site config (YAML, schema ``site-config/v1``) and constructs a
:class:`ReelRecipe` from its ``currents:`` block plus the proposed
``animation:`` block (see docs/SITE_CONFIG.md for the block spec).
survey-sites itself is never modified — this is a read-only adapter.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from reels.recipe import ReelRecipe


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "reading site configs needs PyYAML: pip install pyyaml") from exc
    return yaml


def load_site_config(path: str) -> Dict[str, Any]:
    """Load a site.yaml into a dict."""
    yaml = _require_yaml()
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"site config {path} must be a mapping")
    return data


def find_site_configs(sites_dir: str) -> List[str]:
    """Return every ``*/site.yaml`` under a survey-sites checkout."""
    found = []
    if not os.path.isdir(sites_dir):
        return found
    for entry in sorted(os.listdir(sites_dir)):
        candidate = os.path.join(sites_dir, entry, "site.yaml")
        if os.path.isfile(candidate):
            found.append(candidate)
    return found


def recipe_from_site(site: Dict[str, Any],
                     overrides: Optional[Dict[str, Any]] = None) -> ReelRecipe:
    """Build a :class:`ReelRecipe` from a site-config dict.

    Uses ``site["currents"]`` (as documented by survey-currents'
    ``examples/site-currents.yaml``) and the proposed ``site["animation"]``
    block. ``overrides`` win over both.
    """
    currents = site.get("currents") or {}
    animation = site.get("animation") or {}
    overrides = overrides or {}

    geometry = site.get("geometry") or {}
    bbox = geometry.get("bbox")

    data: Dict[str, Any] = {
        "name": site.get("site_id", "reel"),
        "site_id": site.get("site_id", ""),
        "site_name": site.get("name", ""),
    }
    if bbox:
        data["bbox"] = list(bbox)

    source = str(currents.get("source", "noaa-ofs"))
    data["source"] = source
    if source == "noaa-ofs":
        data["ofs_code"] = str(currents.get("ofs_code", "LMHOFS"))
        data["cycle"] = str(currents.get("cycle", "00"))
        hours = currents.get("hours") or [0, 6, 12, 18]
        data["hours"] = [int(h) for h in hours]
    elif source == "cmems":
        data["cmems_preset"] = str(
            currents.get("preset", currents.get("cmems_preset",
                                                "global-physics-daily")))
        if currents.get("start"):
            data["start"] = str(currents["start"])
        if currents.get("end"):
            data["end"] = str(currents["end"])

    for key in ("n_frames", "flow_style", "flow_size", "n_particles", "seed",
                "animate_preset", "fps", "title", "subtitle", "output"):
        if key in animation:
            data[key] = animation[key]
    if not data.get("title"):
        data["title"] = f"{data.get('site_name') or data['name']} surface currents"

    data.update(overrides)
    return ReelRecipe.from_dict(data)


def recipe_from_site_file(path: str,
                          overrides: Optional[Dict[str, Any]] = None) -> ReelRecipe:
    """Load a site.yaml file and build a recipe from it."""
    return recipe_from_site(load_site_config(path), overrides=overrides)


def animation_block_spec() -> Dict[str, Any]:
    """Machine-readable spec of the proposed ``animation:`` site-config block."""
    return {
        "block": "animation",
        "description": "Proposed site-config block driving survey-reels recipes. "
                       "Read by survey-reels only; ignored by survey-sites v1.",
        "fields": {
            "n_frames": {"type": "integer", "default": 48,
                         "description": "frames to render from the model window"},
            "flow_style": {"type": "string", "default": "reel-dark",
                           "enum": ["reel-dark", "reel-light", "print"]},
            "flow_size": {"type": "string", "default": "reel",
                          "description": "survey-flow size preset"},
            "n_particles": {"type": "integer", "default": 1500},
            "seed": {"type": "integer", "default": 7},
            "animate_preset": {"type": "string", "default": "reel",
                               "enum": ["reel", "wide", "square", "preview"]},
            "fps": {"type": "number", "default": 30.0},
            "title": {"type": "string", "default": "<site name> surface currents"},
            "subtitle": {"type": "string", "default": ""},
            "output": {"type": "string", "default": "<site_id>-reel.mp4"},
        },
    }
