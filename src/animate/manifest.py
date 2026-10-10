"""Frame manifest input: survey-flow handoff contract + plain PNG directories.

Primary input is a survey-flow frame manifest
(``survey-flow.frame-manifest/1.0`` — see survey-flow's
docs/FRAME_MANIFEST.md). Readers validate the schema id, reject unknown
major versions, enforce sequential frame indices, and resolve ``png_path``
entries relative to the manifest file so output directories stay portable.

A plain directory (or glob) of PNGs works too, for survey-monitor
per-pass rasters, survey-thermal series, or anything else.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

MANIFEST_SCHEMA_ID = "survey-flow.frame-manifest/1.0"
MANIFEST_SCHEMA_PREFIX = "survey-flow.frame-manifest/"


@dataclass
class ManifestFrame:
    """One ordered frame: PNG path plus its metadata."""

    index: int
    png_path: str  # absolute, resolved
    timestamp_iso: str = ""
    scalar: str = ""
    scalar_range: tuple = (0.0, 1.0)
    colormap: str = ""
    style: str = ""
    render_params: Dict = field(default_factory=dict)


@dataclass
class Manifest:
    """Ordered frames plus run-level metadata."""

    frames: List[ManifestFrame]
    fps: float = 30.0
    size_px: tuple = (1080, 1920)
    field_source: str = ""
    generator: str = ""
    source_path: str = ""  # manifest file, or directory for glob input

    def __len__(self) -> int:
        return len(self.frames)

    def png_paths(self) -> List[str]:
        return [f.png_path for f in self.frames]

    def timestamps(self) -> List[str]:
        return [f.timestamp_iso for f in self.frames]


def _check_schema(schema: str) -> None:
    if not isinstance(schema, str) or not schema.startswith(MANIFEST_SCHEMA_PREFIX):
        raise ValueError(
            f"not a survey-flow frame manifest: schema={schema!r} "
            f"(expected {MANIFEST_SCHEMA_ID!r})")
    major = schema[len(MANIFEST_SCHEMA_PREFIX):].split(".")[0]
    if major != "1":
        raise ValueError(
            f"unsupported manifest major version {major!r} "
            f"(schema={schema!r}); this reader supports 1.x")


def read_manifest(path: str) -> Manifest:
    """Read and validate a survey-flow frame manifest file."""
    with open(path, encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    _check_schema(data.get("schema", ""))
    raw_frames = data.get("frames", [])
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError(f"manifest has no frames: {path}")
    n_declared = data.get("n_frames")
    if n_declared is not None and int(n_declared) != len(raw_frames):
        raise ValueError(
            f"manifest n_frames={n_declared} but found {len(raw_frames)} "
            f"frame entries: {path}")
    base = os.path.dirname(os.path.abspath(path))
    frames: List[ManifestFrame] = []
    missing: List[str] = []
    for i, entry in enumerate(raw_frames):
        if not isinstance(entry, dict):
            raise ValueError(f"frame entry {i} is not an object: {path}")
        for key in ("frame_index", "timestamp_iso", "png_path"):
            if key not in entry:
                raise ValueError(
                    f"frame entry {i} missing required key {key!r}: {path}")
        idx = int(entry["frame_index"])
        if idx != i:
            raise ValueError(
                f"frame indices must be sequential 0..n-1; entry {i} has "
                f"frame_index={idx}: {path}")
        p = str(entry["png_path"])
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(base, p))
        if not os.path.isfile(p):
            missing.append(p)
        sr = entry.get("scalar_range", (0.0, 1.0))
        frames.append(ManifestFrame(
            index=idx,
            png_path=p,
            timestamp_iso=str(entry.get("timestamp_iso", "")),
            scalar=str(entry.get("scalar", "")),
            scalar_range=(float(sr[0]), float(sr[1])),
            colormap=str(entry.get("colormap", "")),
            style=str(entry.get("style", "")),
            render_params=dict(entry.get("render_params", {}) or {}),
        ))
    if missing:
        shown = "\n  ".join(missing[:5])
        extra = f"\n  ... and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"manifest references {len(missing)} missing PNG file(s):\n  "
            f"{shown}{extra}")
    size = data.get("size_px") or [1080, 1920]
    return Manifest(
        frames=frames,
        fps=float(data.get("fps", 30.0)),
        size_px=(int(size[0]), int(size[1])),
        field_source=str(data.get("field_source", "")),
        generator=str(data.get("generator", "")),
        source_path=os.path.abspath(path),
    )


def frames_from_dir(directory: str,
                    pattern: str = "*.png",
                    timestamps: Optional[Sequence[str]] = None,
                    fps: float = 30.0) -> Manifest:
    """Build a manifest from a plain directory of PNGs (sorted by name).

    ``timestamps`` optionally labels frames; anything else (scalar,
    colormap, ...) is left blank — the encoder only needs ordered PNGs.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"not a directory: {directory}")
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise ValueError(
            f"no PNG frames matching {pattern!r} in {directory}")
    stamps = list(timestamps) if timestamps else [""] * len(paths)
    if len(stamps) < len(paths):
        stamps = stamps + [stamps[-1] if stamps else ""] * (len(paths) - len(stamps))
    frames = [ManifestFrame(index=i, png_path=os.path.abspath(p),
                            timestamp_iso=str(stamps[i]))
              for i, p in enumerate(paths)]
    return Manifest(frames=frames, fps=float(fps),
                    source_path=os.path.abspath(directory))
