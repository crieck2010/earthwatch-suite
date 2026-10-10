"""Demo site pack loading: the shipped sites/ directory."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .validate import _load_file, validate_config

PACK_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "sites")


@dataclass
class SitePack:
    """One demo site: config + GeoJSON sidecars + story docs."""

    site_id: str
    directory: str
    config: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def name(self) -> str:
        return self.config.get("name", self.site_id)

    @property
    def story(self) -> Optional[str]:
        readme = os.path.join(self.directory, "README.md")
        if os.path.exists(readme):
            with open(readme, encoding="utf-8") as fh:
                return fh.read()
        return None


def load_pack(directory: str) -> SitePack:
    """Load and validate one site directory (expects site.yaml or site.json)."""
    pack = SitePack(
        site_id=os.path.basename(os.path.abspath(directory)), directory=directory
    )
    config_path = None
    for fname in ("site.yaml", "site.yml", "site.json"):
        candidate = os.path.join(directory, fname)
        if os.path.exists(candidate):
            config_path = candidate
            break
    if config_path is None:
        pack.errors.append(f"{directory}: no site.yaml/site.json found")
        return pack
    try:
        config = _load_file(config_path)
    except Exception as exc:  # noqa: BLE001
        pack.errors.append(f"{config_path}: could not parse ({exc})")
        return pack
    if not isinstance(config, dict):
        pack.errors.append(f"{config_path}: top level must be an object")
        return pack
    pack.config = config
    pack.site_id = str(config.get("site_id", pack.site_id))
    pack.errors = [
        e if e.startswith(config_path) else f"{config_path}: {e}"
        for e in validate_config(config)
    ]
    return pack


def list_packs(pack_dir: Optional[str] = None) -> List[SitePack]:
    """Load every site directory in a pack dir (default: the shipped sites/)."""
    root = os.path.abspath(pack_dir or PACK_DIR)
    packs: List[SitePack] = []
    if not os.path.isdir(root):
        return packs
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if os.path.isdir(full):
            packs.append(load_pack(full))
    return packs
