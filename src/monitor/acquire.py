"""Pass providers: where per-pass index metrics come from.

A provider answers two questions the pipeline needs, and nothing else:

* ``list_passes(since=None)`` — candidate satellite passes, oldest first.
* ``metrics(pass_id)`` — ``{index: value}`` for the watched zonal statistic.

Two providers ship:

* :class:`SyntheticPassProvider` — seeded random-walk metrics with optional
  planted change. Stdlib-only; used by the test suite and the ``--synthetic``
  demo run.
* :class:`ImageryPassProvider` — real acquisition via ``survey-imagery``
  (STAC search + per-scene processing). ``survey-imagery`` is imported
  *lazily* and only when this provider is instantiated; a missing install
  raises a :class:`MonitorError` with a ``pip install`` hint.

New providers (Landsat, a local raster archive, a future streaming engine)
implement the same two methods — the pipeline never knows the difference.
"""

from __future__ import annotations

import datetime as _dt
import os
import random
from typing import Dict, List, Optional

from .config import MonitorError, SiteConfig
from .scheduler import PassInfo


class PassProvider:
    """Interface every pass provider implements."""

    def list_passes(self, since: Optional[str] = None) -> List[PassInfo]:
        raise NotImplementedError

    def metrics(self, pass_id: str) -> Dict[str, float]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Synthetic provider (tests / demo)
# ---------------------------------------------------------------------------


class SyntheticPassProvider(PassProvider):
    """Seeded synthetic index time series with optional planted change.

    ``plants`` maps ``pass_index -> {index: delta}``: at that pass the metric
    gets an extra additive shock, which is exactly what the demo uses to
    force a visible alert.
    """

    def __init__(self, config: SiteConfig, n_passes: int = 12,
                 start_date: str = "2026-06-01", step_days: int = 16,
                 seed: int = 7,
                 plants: Optional[Dict[int, Dict[str, float]]] = None,
                 base: Optional[Dict[str, float]] = None) -> None:
        self.config = config
        self.n_passes = n_passes
        self.start_date = start_date
        self.step_days = step_days
        self.seed = seed
        self.plants = plants or {}
        self.base = base or {}
        self._cache: Optional[Dict[str, Dict[str, float]]] = None

    def _build(self) -> Dict[str, Dict[str, float]]:
        rng = random.Random(self.seed)
        start = _dt.date.fromisoformat(self.start_date)
        out: Dict[str, Dict[str, float]] = {}
        series: Dict[str, float] = {}
        for i in range(self.n_passes):
            date = (start + _dt.timedelta(days=i * self.step_days)).isoformat()
            pass_id = f"synthetic-{self.config.site_id}-{date}"
            metrics: Dict[str, float] = {}
            for index in self.config.indices:
                v = series.get(index, self.base.get(index, 0.55))
                v += rng.gauss(0.0, 0.015)          # pass-to-pass noise
                v += self.plants.get(i, {}).get(index, 0.0)
                v = max(-1.0, min(1.0, v))
                series[index] = v
                metrics[index] = round(v, 4)
            out[pass_id] = metrics
        return out

    def _ensure(self) -> Dict[str, Dict[str, float]]:
        if self._cache is None:
            self._cache = self._build()
        return self._cache

    def list_passes(self, since: Optional[str] = None) -> List[PassInfo]:
        cache = self._ensure()
        passes: List[PassInfo] = []
        for pass_id, metrics in cache.items():
            date = pass_id.rsplit("-", 3)[-3:]  # synthetic-<site>-YYYY-MM-DD
            date = "-".join(date)
            if since is not None and date <= since:
                continue
            passes.append(PassInfo(pass_id=pass_id, date=date,
                                   cloud_cover=5.0, source="synthetic"))
        passes.sort(key=lambda p: p.sort_key())
        return passes

    def metrics(self, pass_id: str) -> Dict[str, float]:
        try:
            return dict(self._ensure()[pass_id])
        except KeyError as exc:
            raise MonitorError(f"unknown synthetic pass_id {pass_id!r}") from exc


# ---------------------------------------------------------------------------
# Real provider via survey-imagery (lazy optional dependency)
# ---------------------------------------------------------------------------


def _require_imagery():
    try:
        import imagery  # noqa: F401
        return imagery
    except ImportError as exc:
        raise MonitorError(
            "real acquisition needs survey-imagery: "
            "pip install 'survey-monitor[acquire]'"
        ) from exc


class ImageryPassProvider(PassProvider):
    """Real satellite acquisition through ``survey-imagery``.

    ``list_passes`` runs a STAC search (clearest scene per date);
    ``metrics`` processes one scene — windowed AOI read, cloud mask,
    spectral indices, zonal statistics — and returns the watched metric
    per index. Per-scene rasters are written to a throwaway temp dir; the
    engine keeps only the numbers (use ``survey-imagery monitor`` if you
    want the rasters). One bad scene raises :class:`MonitorError` — the
    pipeline treats it as a skipped pass and continues.
    """

    def __init__(self, config: SiteConfig) -> None:
        self.config = config
        self._scenes: Optional[Dict[str, object]] = None

    def _imagery(self):
        return _require_imagery()

    def _aoi(self):
        imagery = self._imagery()
        geometry = self.config.geometry
        if "bbox" in geometry:
            minx, miny, maxx, maxy = geometry["bbox"]
            return imagery.from_bbox(minx, miny, maxx, maxy,
                                     name=self.config.site_id)
        return imagery.from_geojson({"type": "Feature",
                                     "geometry": geometry,
                                     "properties": {}},
                                    name=self.config.site_id)

    def _monitor_config(self, out_dir: str):
        imagery = self._imagery()
        return imagery.MonitorConfig(
            name=self.config.site_id,
            aoi=self._aoi(),
            collections=[self.config.collection],
            indices=list(self.config.indices),
            output_dir=out_dir,
            api_url=self.config.api_url,
            max_cloud_cover=self.config.max_cloud_cover,
        )

    def list_passes(self, since: Optional[str] = None) -> List[PassInfo]:
        imagery = self._imagery()
        end = _dt.date.today().isoformat()
        scenes = imagery.search_scenes(
            self.config.api_url, [self.config.collection], self._aoi(),
            since or "2015-01-01", end,
            max_cloud_cover=self.config.max_cloud_cover)
        scenes = imagery.latest_per_date(scenes)
        self._scenes = {s.id: s for s in scenes}
        out = [PassInfo(pass_id=s.id, date=s.datetime[:10],
                        cloud_cover=getattr(s, "cloud_cover", None),
                        source=self.config.api_url)
               for s in scenes]
        out.sort(key=lambda p: p.sort_key())
        return out

    def metrics(self, pass_id: str) -> Dict[str, float]:
        import tempfile
        imagery = self._imagery()
        if self._scenes is None or pass_id not in self._scenes:
            self.list_passes()
        scene = self._scenes[pass_id]  # type: ignore[index]
        with tempfile.TemporaryDirectory(prefix="survey-monitor-") as tmp:
            os.makedirs(os.path.join(tmp, "rasters"), exist_ok=True)
            records = imagery.monitor.process_scene(
                scene, self._monitor_config(tmp), tmp)
        metric = self.config.metric
        out: Dict[str, float] = {}
        for record in records:
            if record.index in self.config.indices:
                value = getattr(record, metric, None)
                if value is not None:
                    out[record.index] = float(value)
        if not out:
            raise MonitorError(f"no metrics computed for scene {pass_id!r}")
        return out
