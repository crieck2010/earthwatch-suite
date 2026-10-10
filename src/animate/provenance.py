"""Provenance sidecars for encoded videos.

Every video written by the pipeline gets a ``<video>.provenance.json``
sidecar: source manifest identity + hash, preset, encode parameters,
encoder version, and the output's SHA-256. Six months later, when
someone asks which frames produced a reel, the answer is on disk.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import Any, Dict, Optional

PROVENANCE_SCHEMA = "survey-animate.provenance/1.0"


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_sidecar(video_path: str,
                  *,
                  source: str,
                  source_sha256: str = "",
                  preset: str = "",
                  fmt: str = "",
                  fps: float = 0.0,
                  n_frames: int = 0,
                  params: Optional[Dict[str, Any]] = None,
                  encoder: str = "",
                  generator: str = "") -> str:
    """Write ``<video_path>.provenance.json``; return its path."""
    sidecar = video_path + ".provenance.json"
    doc = {
        "schema": PROVENANCE_SCHEMA,
        "generator": generator or "survey-animate",
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "video": os.path.basename(video_path),
        "video_sha256": sha256_file(video_path) if os.path.isfile(video_path) else "",
        "source": source,
        "source_sha256": source_sha256,
        "preset": preset,
        "format": fmt,
        "fps": float(fps),
        "n_frames": int(n_frames),
        "params": dict(params or {}),
        "encoder": encoder,
    }
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return sidecar
