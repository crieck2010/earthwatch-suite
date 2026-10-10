"""Bundled place records: loading and the :class:`Place` type.

The dataset lives at ``gazetteer/data/places.csv`` inside the package
and is read with :mod:`importlib.resources`, so the engine works
installed, from a source checkout, and offline. No network, ever.

Records are loaded once per process and cached in :func:`load_places`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from gazetteer.scoring import kind_of, priority_of

_DATA_PKG = "gazetteer.data"
_CSV_NAME = "places.csv"
_PROVENANCE_NAME = "PROVENANCE.json"


@dataclass(frozen=True)
class Place:
    """One populated place from the bundled dataset."""

    name: str        # NAMEASCII — the label text survey-aesthetics draws
    name_local: str  # NAME — diacritics kept, used by search()
    lon: float
    lat: float
    pop: int         # POP_MAX
    scalerank: int   # 0 (most important) .. 10 (least)
    featurecla: str  # Natural Earth FEATURECLA verbatim
    adm0: str        # country
    adm1: str        # region / state
    kind: str        # "city" | "town" | "other"
    priority: int    # higher = more label-worthy (see scoring.py)


def _open_data(name: str):
    return (resources.files(_DATA_PKG) / name).open("r", encoding="utf-8")


def _row_to_place(row: dict) -> Place:
    pop = int(row["pop"] or 0)
    scalerank = int(row["scalerank"] or 10)
    featurecla = row["featurecla"] or ""
    return Place(
        name=row["name"],
        name_local=row["name_local"],
        lon=float(row["lon"]),
        lat=float(row["lat"]),
        pop=pop,
        scalerank=scalerank,
        featurecla=featurecla,
        adm0=row["adm0"] or "",
        adm1=row["adm1"] or "",
        kind=kind_of(featurecla, pop),
        priority=priority_of(pop, scalerank, featurecla),
    )


@lru_cache(maxsize=1)
def load_places() -> tuple:
    """All bundled places as an immutable tuple (cached per process)."""
    with _open_data(_CSV_NAME) as fh:
        rows = csv.DictReader(fh)
        return tuple(_row_to_place(r) for r in rows)


def dataset_info() -> dict:
    """Provenance of the bundled dataset (record count, source, date)."""
    import json

    with _open_data(_PROVENANCE_NAME) as fh:
        info = json.load(fh)
    info["loaded_records"] = len(load_places())
    return info
