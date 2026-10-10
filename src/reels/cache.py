"""Content-addressed cache for fetched fields and rendered frames.

Keys are derived from the recipe hash, so re-running an unchanged recipe
never re-downloads model data or re-renders frames. The cache is a plain
directory tree — no database, safe to delete at any time.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Any, Dict, Optional


class Cache:
    """File-backed cache rooted at ``root``."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    # -- fields -----------------------------------------------------------
    def field_key(self, recipe_hash: str) -> str:
        return os.path.join("fields", recipe_hash[:16], "field.json")

    def has_field(self, recipe_hash: str) -> bool:
        return os.path.isfile(self._path(self.field_key(recipe_hash)))

    def store_field(self, recipe_hash: str, field_json: str) -> str:
        dest = self._path(self.field_key(recipe_hash))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(field_json)
        return dest

    def field_path(self, recipe_hash: str) -> Optional[str]:
        path = self._path(self.field_key(recipe_hash))
        return path if os.path.isfile(path) else None

    # -- frames -----------------------------------------------------------
    def frames_key(self, recipe_hash: str) -> str:
        return os.path.join("frames", recipe_hash[:16])

    def has_frames(self, recipe_hash: str) -> bool:
        marker = self._path(self.frames_key(recipe_hash), "manifest.json")
        return os.path.isfile(marker)

    def frames_dir(self, recipe_hash: str) -> str:
        path = self._path(self.frames_key(recipe_hash))
        os.makedirs(path, exist_ok=True)
        return path

    # -- manifest sidecar --------------------------------------------------
    def write_entry(self, recipe_hash: str, entry: Dict[str, Any]) -> str:
        dest = self._path("entries", recipe_hash[:16] + ".json")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(entry, fh, indent=2)
            fh.write("\n")
        return dest

    def read_entry(self, recipe_hash: str) -> Optional[Dict[str, Any]]:
        dest = self._path("entries", recipe_hash[:16] + ".json")
        if not os.path.isfile(dest):
            return None
        with open(dest, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def clear(self, recipe_hash: Optional[str] = None) -> int:
        """Remove one entry (by hash) or the whole cache. Returns entries removed."""
        if recipe_hash is None:
            count = sum(len(files) for _, _, files in os.walk(self.root))
            shutil.rmtree(self.root, ignore_errors=True)
            os.makedirs(self.root, exist_ok=True)
            return count
        removed = 0
        for sub in ("fields", "frames", "entries"):
            target = self._path(sub, recipe_hash[:16])
            if sub == "entries":
                target += ".json"
            if os.path.isdir(target):
                removed += sum(len(files) for _, _, files in os.walk(target))
                shutil.rmtree(target, ignore_errors=True)
            elif os.path.isfile(target):
                os.remove(target)
                removed += 1
        return removed

    @staticmethod
    def content_hash(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
