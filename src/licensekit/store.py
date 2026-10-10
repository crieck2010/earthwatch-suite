"""Vendor-side key bookkeeping and revocation lists.

The store interface is deliberately small so a file-based JSON store (fine for
a small vendor) can be swapped for a database-backed store later without
touching validation logic — :func:`licensekit.validate.check_license` only
ever asks ``kid in revocation_list``.
"""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional


class KeyStore(ABC):
    """Vendor bookkeeping: which keys were issued, which are revoked."""

    @abstractmethod
    def record_issuance(self, key_id: str, record: Dict) -> None:
        """Persist an issuance record (email, plan, issued_at, ...)."""

    @abstractmethod
    def revoke(self, key_id: str, reason: str = "") -> bool:
        """Revoke a key id. Returns True if it was known, False otherwise."""

    @abstractmethod
    def is_revoked(self, key_id: str) -> bool:
        """Membership test used by validation (also via ``in``)."""

    def __contains__(self, key_id: str) -> bool:
        return self.is_revoked(key_id)

    @abstractmethod
    def list_keys(self) -> List[Dict]:
        """All issuance records, oldest first."""


class RevocationList(ABC):
    """Minimal revocation interface consumed by
    :func:`licensekit.validate.check_license`."""

    @abstractmethod
    def __contains__(self, key_id: str) -> bool: ...


def _atomic_write(path: str, data: Dict) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(os.path.abspath(path)) or ".", prefix=".license-tmp-"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class JsonFileKeyStore(KeyStore, RevocationList):
    """File-based key store: ``{"issued": {kid: record}, "revoked": {kid: record}}``.

    Good for a small vendor or a single-machine deployment. Writes are atomic
    (temp file + rename). For multi-process writers, prefer a DB-backed
    :class:`KeyStore`.
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self._data = self._load()

    def _load(self) -> Dict:
        if not os.path.exists(self.path):
            return {"issued": {}, "revoked": {}}
        with open(self.path, encoding="utf-8") as fh:
            raw = fh.read().strip()
        if not raw:
            return {"issued": {}, "revoked": {}}
        data = json.loads(raw)
        data.setdefault("issued", {})
        data.setdefault("revoked", {})
        return data

    def _save(self) -> None:
        _atomic_write(self.path, self._data)

    @staticmethod
    def _norm(kid: str) -> str:
        kid = kid.strip()
        return kid if kid.startswith("kid:") else "kid:" + kid

    def record_issuance(self, key_id: str, record: Dict) -> None:
        self._data["issued"][self._norm(key_id)] = dict(record)
        self._save()

    def revoke(self, key_id: str, reason: str = "") -> bool:
        kid = self._norm(key_id)
        known = kid in self._data["issued"]
        self._data["revoked"][kid] = {
            "revoked_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
        }
        self._save()
        return known

    def is_revoked(self, key_id: str) -> bool:
        return self._norm(key_id) in self._data["revoked"]

    def __contains__(self, key_id: str) -> bool:
        return self.is_revoked(key_id)

    def list_keys(self) -> List[Dict]:
        def _ts(rec: Dict) -> str:
            return str(rec.get("issued_at", ""))

        return [
            {"key_id": kid, **rec, "revoked": kid in self._data["revoked"]}
            for kid, rec in sorted(
                self._data["issued"].items(), key=lambda kv: _ts(kv[1])
            )
        ]


class FileRevocationList(RevocationList):
    """Read-only revocation list backed by a JSON file.

    Expected shape: ``{"revoked": ["kid:AAA", ...]}`` or
    ``{"revoked": {"kid:AAA": {"reason": ...}}}``. This is the file an engine
    operator drops next to their config; the vendor publishes it whenever keys
    are revoked. A remote (HTTP) revocation source can implement the same
    ``__contains__`` protocol — see docs/VENDOR_OPS.md.
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)

    def _kids(self) -> set:
        if not os.path.exists(self.path):
            return set()
        with open(self.path, encoding="utf-8") as fh:
            data = json.load(fh)
        revoked = data.get("revoked", [])
        if isinstance(revoked, dict):
            revoked = revoked.keys()
        return {str(k) for k in revoked}

    def __contains__(self, key_id: str) -> bool:
        kid = key_id if key_id.startswith("kid:") else "kid:" + key_id
        return kid in self._kids()
