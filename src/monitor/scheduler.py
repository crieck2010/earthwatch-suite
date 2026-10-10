"""Pass scheduling: which satellite passes are new since the last run.

This is pure set/datetime logic. The engine asks a *provider* (real or
synthetic, see ``monitor.acquire``) for candidate passes and uses these
helpers to keep only the ones the site has not processed yet. Re-running
is therefore always incremental.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Set


@dataclass(frozen=True)
class PassInfo:
    """A candidate satellite pass, before any pixels are touched."""

    pass_id: str          # stable id, e.g. the STAC scene id
    date: str             # ISO date of acquisition, e.g. "2026-09-15"
    cloud_cover: Optional[float] = None
    source: str = ""      # where it was listed from (STAC url, "synthetic", ...)

    def sort_key(self):
        return (self.date, self.pass_id)


def new_passes(known_ids: Set[str], candidates: Iterable[PassInfo],
               since: Optional[str] = None,
               max_cloud_cover: Optional[float] = None) -> List[PassInfo]:
    """Return candidate passes not yet processed, oldest first.

    * ``known_ids`` — pass ids already recorded in the site's state.
    * ``since`` — optional ISO date cutoff; passes strictly *after* it are
      kept (ISO dates compare lexicographically).
    * ``max_cloud_cover`` — optional upper bound; passes with unknown cloud
      cover are kept.
    """
    out: List[PassInfo] = []
    for cand in candidates:
        if cand.pass_id in known_ids:
            continue
        if since is not None and cand.date <= since:
            continue
        if (max_cloud_cover is not None and cand.cloud_cover is not None
                and cand.cloud_cover > max_cloud_cover):
            continue
        out.append(cand)
    out.sort(key=lambda p: p.sort_key())
    return out
