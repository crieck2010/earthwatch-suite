"""Shared stdlib HTTP helper for the radar sources.

Every source module fetches through :func:`get_json`, which:

* uses only :mod:`urllib` (no third-party HTTP clients),
* sends an honest ``User-Agent`` identifying this project,
* caches responses in-memory per run (``_CACHE``) so a source is never
  re-polled inside one digest build,
* raises :class:`SourceUnreachableError` on any HTTP error (including
  429 rate-limit responses) or network failure — it never returns
  placeholder data,
* raises :class:`SourceUnparsableError` when a 200 response is not JSON.

Tests clear ``_CACHE`` in ``setUp`` and patch ``urllib.request.urlopen``;
no test performs network I/O.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional

from ...models import SourceUnparsableError, SourceUnreachableError

#: In-memory response cache, per run. Keyed by request URL.
_CACHE: dict[str, Any] = {}

REQUEST_TIMEOUT = 25  # seconds per request


def get_json(
    url: str,
    headers: Optional[Mapping[str, str]] = None,
    timeout: int = REQUEST_TIMEOUT,
    *,
    source_name: str = "source",
) -> Any:
    """GET ``url`` and return the decoded JSON body (cached per run)."""
    if url in _CACHE:
        return _CACHE[url]
    request = urllib.request.Request(url, headers=dict(headers or {}))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        # HTTPError is raised for non-2xx statuses, including 429.
        raise SourceUnreachableError(
            f"{source_name}: HTTP {exc.code} from {url} "
            f"({'rate limited' if exc.code == 429 else 'request failed'}); "
            "no data was fabricated."
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SourceUnreachableError(
            f"{source_name}: network error reaching {url}: {exc}; "
            "no data was fabricated."
        ) from exc
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise SourceUnparsableError(
            f"{source_name}: response from {url} was not decodable JSON."
        ) from exc
    _CACHE[url] = payload
    return payload


def clear_cache() -> None:
    """Drop the per-run cache (used by tests)."""
    _CACHE.clear()
