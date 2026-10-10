"""Finalize a reel run: publish artifacts and write the run summary.

``finalize`` copies the encoded video (+ its provenance sidecar when
present) into the output directory and writes ``run-summary.json`` whose
fields mirror survey-monitor's ``RunSummary`` (site_id, site_name,
timestamp, passes_processed, passes_skipped, alerts_fired, alerts,
errors) extended with reel-specific fields.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional

SUMMARY_SCHEMA = "survey-reels.run-summary/1.0"


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class FinalizeResult:
    """Paths produced by :func:`finalize`."""

    def __init__(self, summary_path: str, video_path: str) -> None:
        self.summary_path = summary_path
        self.video_path = video_path  # published copy under out_dir


def finalize(recipe, video_path: str, work_dir: str,
             *,
             stages_run: Optional[List[str]] = None,
             stages_skipped: Optional[List[str]] = None,
             out_dir: Optional[str] = None) -> FinalizeResult:
    """Publish the finished reel and write ``run-summary.json``.

    Returns a :class:`FinalizeResult`. Raises FileNotFoundError if the
    video is missing.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"encoded video not found: {video_path}")
    out_dir = out_dir or os.path.join(work_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    ext = os.path.splitext(video_path)[1].lower() or ".mp4"
    published = os.path.join(out_dir, f"{recipe.name}{ext}")
    shutil.copy2(video_path, published)

    published_sidecar = ""
    for candidate in (video_path + ".provenance.json",
                      os.path.splitext(video_path)[0] + ".provenance.json"):
        if os.path.isfile(candidate):
            published_sidecar = os.path.join(
                out_dir, os.path.basename(candidate))
            shutil.copy2(candidate, published_sidecar)
            break

    summary = {
        "schema": SUMMARY_SCHEMA,
        "recipe_name": recipe.name,
        "recipe_sha": _recipe_sha(recipe),
        "site_id": recipe.site_id,
        "site_name": recipe.site_name,
        "timestamp": _utcnow(),
        "passes_processed": recipe.n_frames,
        "passes_skipped": 0,
        "alerts_fired": 0,
        "alerts": {},
        "errors": [],
        "video_path": published,
        "video_sha256": _sha256(published),
        "provenance_sidecar": published_sidecar,
        "source": recipe.source,
        "ofs_code": recipe.ofs_code,
        "animate_preset": recipe.animate_preset,
        "flow_style": recipe.flow_style,
        "n_frames": recipe.n_frames,
        "stages_run": list(stages_run or []),
        "stages_skipped": list(stages_skipped or []),
    }
    summary_path = os.path.join(out_dir, "run-summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    return FinalizeResult(summary_path, published)


def _recipe_sha(recipe) -> str:
    from reels.recipe import recipe_hash

    return recipe_hash(recipe)


def _sha256(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
