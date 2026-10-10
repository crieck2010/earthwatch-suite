"""Frame manifest: the stable handoff from survey-flow to survey-animate.

A manifest is a JSON document listing every rendered PNG frame with the
metadata survey-animate needs to encode video: frame order, per-frame
timestamps, scalar range, colormap, style, and the render parameters.
The schema is versioned (``survey-flow.frame-manifest/1.0``); readers
must reject unknown major versions.

Schema (see docs/FRAME_MANIFEST.md for the full spec)::

    {
      "schema": "survey-flow.frame-manifest/1.0",
      "generator": "survey-flow 0.1.0",
      "created": "<ISO-8601>",
      "fps": 30.0,                       // intended playback rate (hint)
      "size_px": [1080, 1920],
      "field_source": "noaa-ofs/LMHOFS",
      "frames": [
        {
          "frame_index": 0,
          "timestamp_iso": "2026-09-16T00:00:00+00:00",
          "png_path": "frames/frame_0000.png",
          "scalar": "temperature",
          "scalar_range": [12.4, 21.8],
          "colormap": "cividis",
          "style": "reel-dark",
          "render_params": {"dpi": 100, "title": "..."}
        }, ...
      ]
    }

``png_path`` values are stored relative to the manifest file's directory
when possible, so the whole ``(manifest, frames/)`` directory is portable.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

MANIFEST_SCHEMA = "survey-flow.frame-manifest/1.0"


@dataclass
class FrameEntry:
    """Metadata for one rendered frame."""

    frame_index: int
    timestamp_iso: str
    png_path: str
    scalar: str = "temperature"
    scalar_range: Tuple[float, float] = (0.0, 1.0)
    colormap: str = "cividis"
    style: str = "reel-dark"
    render_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["scalar_range"] = [float(self.scalar_range[0]),
                             float(self.scalar_range[1])]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameEntry":
        required = ("frame_index", "timestamp_iso", "png_path")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"frame entry missing keys: {missing}")
        sr = data.get("scalar_range", (0.0, 1.0))
        return cls(
            frame_index=int(data["frame_index"]),
            timestamp_iso=str(data["timestamp_iso"]),
            png_path=str(data["png_path"]),
            scalar=str(data.get("scalar", "temperature")),
            scalar_range=(float(sr[0]), float(sr[1])),
            colormap=str(data.get("colormap", "cividis")),
            style=str(data.get("style", "reel-dark")),
            render_params=dict(data.get("render_params", {})),
        )


@dataclass
class FrameManifest:
    """An ordered set of frame entries plus run-level metadata."""

    entries: List[FrameEntry] = field(default_factory=list)
    fps: float = 30.0
    size_px: Tuple[int, int] = (1080, 1920)
    field_source: str = ""
    generator: str = ""

    def add(self, entry: FrameEntry) -> None:
        if self.entries and entry.frame_index != self.entries[-1].frame_index + 1:
            raise ValueError(
                f"frame indices must be sequential; got {entry.frame_index} "
                f"after {self.entries[-1].frame_index}")
        self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def png_paths(self, base_dir: Optional[str] = None) -> List[str]:
        """Absolute PNG paths, resolving relative entries against ``base_dir``."""
        paths = []
        for e in self.entries:
            p = e.png_path
            if not os.path.isabs(p) and base_dir:
                p = os.path.normpath(os.path.join(base_dir, p))
            paths.append(p)
        return paths

    def to_dict(self, generator: str = "") -> Dict[str, Any]:
        return {
            "schema": MANIFEST_SCHEMA,
            "generator": generator or self.generator or "survey-flow",
            "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "fps": float(self.fps),
            "size_px": [int(self.size_px[0]), int(self.size_px[1])],
            "field_source": self.field_source,
            "n_frames": len(self.entries),
            "frames": [e.to_dict() for e in self.entries],
        }

    def write(self, path: str, generator: str = "") -> str:
        """Write the manifest; frame paths are stored relative to ``path``."""
        base = os.path.dirname(os.path.abspath(path))
        entries = []
        for e in self.entries:
            p = e.png_path
            if os.path.isabs(p):
                try:
                    p = os.path.relpath(p, base).replace("\\", "/")
                except ValueError:
                    pass
            entries.append(FrameEntry(
                frame_index=e.frame_index, timestamp_iso=e.timestamp_iso,
                png_path=p, scalar=e.scalar, scalar_range=e.scalar_range,
                colormap=e.colormap, style=e.style,
                render_params=dict(e.render_params)))
        manifest = FrameManifest(entries=list(entries), fps=self.fps,
                                 size_px=self.size_px,
                                 field_source=self.field_source,
                                 generator=self.generator)
        os.makedirs(base, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest.to_dict(generator=generator), fh, indent=2)
        return path

    @classmethod
    def read(cls, path: str) -> "FrameManifest":
        """Read and validate a manifest file."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        schema = str(data.get("schema", ""))
        if not schema.startswith("survey-flow.frame-manifest/"):
            raise ValueError(f"not a survey-flow frame manifest: schema={schema!r}")
        major = schema.rsplit("/", 1)[-1].split(".")[0]
        if major != "1":
            raise ValueError(f"unsupported manifest major version: {schema!r}")
        frames = [FrameEntry.from_dict(d) for d in data.get("frames", [])]
        size = data.get("size_px", [1080, 1920])
        return cls(entries=frames, fps=float(data.get("fps", 30.0)),
                   size_px=(int(size[0]), int(size[1])),
                   field_source=str(data.get("field_source", "")),
                   generator=str(data.get("generator", "")))


def interpolate_times(times_iso: Sequence[str], n_frames: int) -> List[str]:
    """Spread ``n_frames`` ISO timestamps evenly across ``times_iso``.

    Used to label animation frames sampled at fractional model times.
    Falls back to repeating labels when timestamps are unparsable.
    """
    times_iso = list(times_iso)
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if len(times_iso) == 0:
        raise ValueError("need at least one timestamp")
    try:
        parsed = [_dt.datetime.fromisoformat(t) for t in times_iso]
    except ValueError:
        return [times_iso[min(i * len(times_iso) // n_frames, len(times_iso) - 1)]
                for i in range(n_frames)]
    if len(parsed) == 1:
        return [parsed[0].isoformat()] * n_frames
    t0, t1 = parsed[0], parsed[-1]
    span = (t1 - t0).total_seconds()
    return [(t0 + _dt.timedelta(seconds=span * i / (n_frames - 1))).isoformat()
            for i in range(n_frames)]
