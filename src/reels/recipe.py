"""The :class:`ReelRecipe`: one config describing a whole reel.

A recipe is a plain-data dataclass (JSON/YAML round-trippable) with strict
validation. Its canonical hash keys the content-addressed cache.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence


RECIPE_SCHEMA = "survey-reels.recipe/1.0"

SOURCES = ("noaa-ofs", "cmems", "synthetic")

FLOW_STYLES = ("reel-dark", "reel-light", "print")
ANIMATE_PRESETS = ("reel", "wide", "square", "preview")


def _require_str(data: Dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"recipe field {key!r} must be a non-empty string")
    return value.strip()


def _require_bbox(data: Dict[str, Any]) -> List[float]:
    bbox = data.get("bbox")
    if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
            or not all(isinstance(x, (int, float)) for x in bbox)):
        raise ValueError("recipe field 'bbox' must be [min_lon, min_lat, max_lon, max_lat]")
    minx, miny, maxx, maxy = (float(x) for x in bbox)
    if not (minx < maxx and miny < maxy):
        raise ValueError("recipe bbox must satisfy min_lon < max_lon and min_lat < max_lat")
    if not (-180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0
            and -90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0):
        raise ValueError("recipe bbox coordinates out of geographic range")
    return [minx, miny, maxx, maxy]


@dataclass
class ReelRecipe:
    """One reel, fully described.

    ``source`` is one of ``"noaa-ofs"`` (needs network), ``"cmems"``
    (needs network + free CMEMS account), or ``"synthetic"`` (offline).
    """

    name: str
    source: str = "synthetic"
    # noaa-ofs
    ofs_code: str = "LMHOFS"
    model_date: str = ""          # YYYY-MM-DD; empty = latest available
    cycle: str = "00"             # model run cycle
    hours: List[int] = field(default_factory=lambda: [0, 6, 12, 18])
    # cmems
    cmems_preset: str = "global-physics-daily"
    start: str = ""               # ISO datetime
    end: str = ""                 # ISO datetime
    # shared
    bbox: List[float] = field(default_factory=lambda: [-87.9, 41.6, -86.7, 42.9])
    n_frames: int = 48
    flow_style: str = "reel-dark"
    flow_size: str = "reel"
    n_particles: int = 1500
    seed: int = 7
    animate_preset: str = "reel"
    fps: float = 30.0
    title: str = ""
    subtitle: str = ""
    output: str = "reel.mp4"
    site_id: str = ""
    site_name: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("recipe 'name' must be a non-empty string")
        self.name = self.name.strip()
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}; want one of {SOURCES}")
        if self.flow_style not in FLOW_STYLES:
            raise ValueError(f"unknown flow_style {self.flow_style!r}; want one of {FLOW_STYLES}")
        if self.animate_preset not in ANIMATE_PRESETS:
            raise ValueError(
                f"unknown animate_preset {self.animate_preset!r}; want one of {ANIMATE_PRESETS}")
        _require_bbox({"bbox": self.bbox})
        if self.n_frames <= 0:
            raise ValueError("n_frames must be positive")
        if self.n_particles <= 0:
            raise ValueError("n_particles must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.source == "noaa-ofs":
            if not self.hours:
                raise ValueError("noaa-ofs recipes need at least one forecast hour")
            if any(h < 0 for h in self.hours):
                raise ValueError("forecast hours must be non-negative")
        if self.source == "cmems" and not (self.start and self.end):
            raise ValueError("cmems recipes need 'start' and 'end' ISO datetimes")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["schema"] = RECIPE_SCHEMA
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReelRecipe":
        if not isinstance(data, dict):
            raise ValueError("recipe must be a mapping")
        schema = str(data.get("schema", ""))
        if schema and schema != RECIPE_SCHEMA:
            major = schema.rsplit("/", 1)[-1].split(".")[0]
            want = RECIPE_SCHEMA.rsplit("/", 1)[-1].split(".")[0]
            if major != want:
                raise ValueError(
                    f"unsupported recipe schema {schema!r}; want {RECIPE_SCHEMA}")
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known and k != "schema"}
        name = kwargs.get("name") or data.get("name")
        if not name:
            raise ValueError("recipe needs a 'name'")
        kwargs["name"] = name
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "ReelRecipe":
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "reading recipe YAML needs PyYAML: pip install pyyaml") from exc
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data)

    def to_yaml(self, path: str) -> str:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "writing recipe YAML needs PyYAML: pip install pyyaml") from exc
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)
        return path


def recipe_hash(recipe: ReelRecipe) -> str:
    """Canonical SHA-256 of the recipe (cache key)."""
    canonical = json.dumps(recipe.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def example_recipe() -> ReelRecipe:
    """A Lake Michigan reel recipe mirroring the mapped.earth reference."""
    return ReelRecipe(
        name="lake-michigan-week",
        source="noaa-ofs",
        ofs_code="LMHOFS",
        cycle="00",
        hours=[0, 6, 12, 18, 24, 30, 36, 42],
        bbox=[-87.9, 41.6, -86.7, 42.9],
        n_frames=48,
        flow_style="reel-dark",
        animate_preset="reel",
        title="Lake Michigan surface currents",
        subtitle="NOAA LMHOFS model · colored by water temperature",
        output="lake-michigan-reel.mp4",
    )
