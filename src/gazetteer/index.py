"""Grid spatial index over the bundled places.

7,342 points fit comfortably in memory; a uniform lon/lat grid keeps
bbox queries O(candidates) instead of O(dataset). Cell size is 5° x 5°
(72 x 36 cells); the index is built once per process and cached.

Query cost: O(cells overlapped) dict lookups + O(candidates) point
tests. For a typical regional reel bbox (a few degrees) that is a
handful of cells and tens of candidates.
"""

from __future__ import annotations

from functools import lru_cache

CELL_DEG = 5.0
_N_LON_CELLS = int(360 / CELL_DEG)  # 72
_N_LAT_CELLS = int(180 / CELL_DEG)  # 36


def _cell(lon: float, lat: float) -> tuple:
    cx = int((lon + 180.0) // CELL_DEG) % _N_LON_CELLS
    cy = int(min(max(lat + 90.0, 0.0), 179.999) // CELL_DEG)
    return (cx, cy)


@lru_cache(maxsize=1)
def build_index() -> dict:
    """Map ``(cell_x, cell_y) -> [Place, ...]`` for the bundled dataset."""
    from gazetteer.places import load_places

    index: dict = {}
    for place in load_places():
        index.setdefault(_cell(place.lon, place.lat), []).append(place)
    return index


def _lon_cell_range(lon0: float, lon1: float) -> list:
    """Cell x-indices covering [lon0, lon1]; handles antimeridian wrap."""
    span = (lon1 - lon0) % 360.0
    if span == 0.0 and lon0 != lon1:
        return list(range(_N_LON_CELLS))  # full 360° span, e.g. (-180, 180)
    c0 = int((lon0 + 180.0) // CELL_DEG) % _N_LON_CELLS
    c1 = int((lon1 + 180.0) // CELL_DEG) % _N_LON_CELLS
    if lon0 <= lon1:
        if c1 >= c0:
            return list(range(c0, c1 + 1))
        return list(range(c0, _N_LON_CELLS)) + list(range(0, c1 + 1))
    # wrapped (antimeridian): [lon0, 180] U [-180, lon1]
    return list(range(c0, _N_LON_CELLS)) + list(range(0, c1 + 1))


def _lat_cell_range(lat0: float, lat1: float) -> list:
    c0 = int(min(max(lat0 + 90.0, 0.0), 179.999) // CELL_DEG)
    c1 = int(min(max(lat1 + 90.0, 0.0), 179.999) // CELL_DEG)
    return list(range(c0, c1 + 1))


def query_bbox(lon0: float, lon1: float, lat0: float, lat1: float) -> list:
    """Candidate places whose coordinates fall inside the bbox.

    ``lon0 > lon1`` means the bbox crosses the antimeridian. Bounds are
    inclusive. Latitude is clamped to [-90, 90].
    """
    lat0 = min(max(lat0, -90.0), 90.0)
    lat1 = min(max(lat1, -90.0), 90.0)
    if lat0 > lat1:
        lat0, lat1 = lat1, lat0

    index = build_index()
    out = []
    for cx in _lon_cell_range(lon0, lon1):
        for cy in _lat_cell_range(lat0, lat1):
            for place in index.get((cx, cy), ()):  # type: ignore[arg-type]
                if lat0 <= place.lat <= lat1 and (
                    (lon0 <= lon1 and lon0 <= place.lon <= lon1)
                    or (lon0 > lon1 and (place.lon >= lon0 or place.lon <= lon1))
                ):
                    out.append(place)
    return out
