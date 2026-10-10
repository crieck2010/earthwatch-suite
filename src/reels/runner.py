"""The 4-stage pipeline: fetch → render → encode → deliver.

Resumable: ``run-state.json`` in the work dir records per-stage status and
artifact paths. Re-running with an unchanged recipe skips completed stages.
Changing the recipe (different hash) re-runs from the first affected stage.

Peer modules are injected as callables so the pipeline is fully testable
offline. Defaults resolve the real peers and raise :class:`PeerMissing`
with install guidance when one is absent.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

from reels import _peers
from reels.cache import Cache
from reels.recipe import ReelRecipe, recipe_hash

STAGES = ("fetch", "render", "encode", "deliver")
STATE_FILE = "run-state.json"


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class StageState:
    status: str = "pending"          # pending | running | done | failed
    artifact: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageState":
        return cls(
            status=str(data.get("status", "pending")),
            artifact=str(data.get("artifact", "")),
            detail=dict(data.get("detail") or {}),
            error=str(data.get("error", "")),
        )


@dataclass
class RunState:
    recipe_name: str
    recipe_sha: str
    stages: Dict[str, StageState] = field(default_factory=dict)
    updated: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "survey-reels.run-state/1.0",
            "recipe_name": self.recipe_name,
            "recipe_sha": self.recipe_sha,
            "updated": self.updated,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunState":
        stages = {k: StageState.from_dict(v)
                  for k, v in (data.get("stages") or {}).items()}
        return cls(
            recipe_name=str(data.get("recipe_name", "")),
            recipe_sha=str(data.get("recipe_sha", "")),
            stages=stages,
            updated=str(data.get("updated", "")),
        )


@dataclass
class RunResult:
    recipe: ReelRecipe
    work_dir: str
    video_path: str
    summary_path: str
    stages_run: List[str]
    stages_skipped: List[str]
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and os.path.isfile(self.video_path)


def _load_state(work_dir: str) -> Optional[RunState]:
    path = os.path.join(work_dir, STATE_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return RunState.from_dict(json.load(fh))
    except (ValueError, OSError):
        return None


def _save_state(work_dir: str, state: RunState) -> str:
    state.updated = _utcnow()
    path = os.path.join(work_dir, STATE_FILE)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state.to_dict(), fh, indent=2)
        fh.write("\n")
    return path


def _default_fetch(recipe: ReelRecipe, stage_dir: str, cache: Cache,
                   rhash: str):
    """Fetch a CurrentField via survey-currents (network unless synthetic)."""
    if recipe.source == "synthetic":
        cls = _peers.current_field_cls()
        fld = cls.synthetic(nt=max(len(recipe.hours), 2), ny=24, nx=32,
                            seed=recipe.seed)
        if recipe.bbox:
            fld = fld.select_bbox(recipe.bbox)
        dest = os.path.join(stage_dir, "field.json")
        fld.to_json(dest)
        with open(dest, "r", encoding="utf-8") as fh:
            cache.store_field(rhash, fh.read())
        return fld, dest
    if recipe.source == "noaa-ofs":
        fetch = _peers.fetch_currents_noaa()
        fld = fetch(recipe.ofs_code, recipe.model_date or _today(),
                    cycle=recipe.cycle, hours=tuple(recipe.hours),
                    work_dir=os.path.join(stage_dir, "noaa"),
                    bbox=list(recipe.bbox))
        dest = os.path.join(stage_dir, "field.json")
        fld.to_json(dest)
        with open(dest, "r", encoding="utf-8") as fh:
            cache.store_field(rhash, fh.read())
        return fld, dest
    if recipe.source == "cmems":
        subset = _peers.subset_cmems_fn()
        parse = _peers.parse_cmems_netcdf_fn()
        nc = subset(recipe.cmems_preset, list(recipe.bbox),
                    recipe.start, recipe.end,
                    os.path.join(stage_dir, "cmems"))
        fld = parse(nc)
        dest = os.path.join(stage_dir, "field.json")
        fld.to_json(dest)
        with open(dest, "r", encoding="utf-8") as fh:
            cache.store_field(rhash, fh.read())
        return fld, dest
    raise ValueError(f"unknown source {recipe.source!r}")


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _default_render(recipe: ReelRecipe, field_obj, stage_dir: str,
                    progress: Optional[Callable[[int, int], None]] = None):
    """Render flow frames via survey-flow. Returns the manifest path."""
    tvf_cls = _peers.time_varying_field_cls()
    render = _peers.render_animation_fn()
    tvf = tvf_cls.from_current_field(field_obj, scalar="temperature")
    manifest = render(tvf, stage_dir, style=recipe.flow_style,
                      size=recipe.flow_size, n_frames=recipe.n_frames,
                      n_particles=recipe.n_particles, seed=recipe.seed,
                      title=recipe.title or None, progress=progress)
    manifest_path = os.path.join(stage_dir, "manifest.json")
    manifest.write(manifest_path, generator="survey-reels/1.0")
    return manifest_path


def _default_encode(recipe: ReelRecipe, manifest_path: str, stage_dir: str,
                    progress: Optional[Callable[[int], None]] = None) -> str:
    """Encode the video via survey-animate. Returns the video path."""
    render_video = _peers.render_video_fn()
    ext = os.path.splitext(recipe.output)[1].lower() or ".mp4"
    out_name = f"{recipe.name}{ext}"
    out_path = os.path.join(stage_dir, out_name)
    result = render_video(manifest_path, out_path, preset=recipe.animate_preset,
                          fps=recipe.fps, title=recipe.title,
                          subtitle=recipe.subtitle, progress=progress)
    return result.path if hasattr(result, "path") else out_path


def run_recipe(recipe: ReelRecipe,
               work_dir: str,
               *,
               cache_dir: Optional[str] = None,
               fetch_fn: Optional[Callable] = None,
               render_fn: Optional[Callable] = None,
               encode_fn: Optional[Callable] = None,
               load_field_fn: Optional[Callable[[str], Any]] = None,
               progress: Optional[Callable[[str, int, int], None]] = None,
               ) -> RunResult:
    """Run the 4-stage pipeline. Completed stages are skipped on re-run.

    ``fetch_fn(recipe, stage_dir, cache, recipe_hash) -> (field, field_json_path)``,
    ``render_fn(recipe, field, stage_dir, progress) -> manifest_path``,
    ``encode_fn(recipe, manifest_path, stage_dir, progress) -> video_path``
    may be injected (tests, custom peers); defaults use the real peers.
    ``load_field_fn(field_json_path) -> field`` reloads a cached field when
    resuming mid-pipeline; defaults to ``CurrentField.from_json``.
    """
    os.makedirs(work_dir, exist_ok=True)
    rhash = recipe_hash(recipe)
    cache = Cache(cache_dir or os.path.join(work_dir, ".cache"))

    state = _load_state(work_dir)
    if state is None or state.recipe_sha != rhash or state.recipe_name != recipe.name:
        state = RunState(recipe_name=recipe.name, recipe_sha=rhash,
                         stages={s: StageState() for s in STAGES})

    stages_run: List[str] = []
    stages_skipped: List[str] = []
    errors: List[str] = []
    artifacts: Dict[str, str] = {}
    field_obj = None

    def _prog(stage: str, done: int, total: int) -> None:
        if progress:
            progress(stage, done, total)

    # -- fetch ----------------------------------------------------------
    st = state.stages["fetch"]
    fetch_dir = os.path.join(work_dir, "fetch")
    if st.status == "done" and os.path.isfile(st.artifact):
        stages_skipped.append("fetch")
        artifacts["field_json"] = st.artifact
    else:
        os.makedirs(fetch_dir, exist_ok=True)
        st.status = "running"
        _save_state(work_dir, state)
        try:
            fn = fetch_fn or _default_fetch
            field_obj, field_json = fn(recipe, fetch_dir, cache, rhash)
            artifacts["field_json"] = field_json
            st.status, st.artifact = "done", field_json
            st.detail = {"recipe_sha": rhash}
            stages_run.append("fetch")
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            st.status, st.error = "failed", f"{type(exc).__name__}: {exc}"
            errors.append(f"fetch: {exc}")
        _save_state(work_dir, state)
        if errors:
            return RunResult(recipe, work_dir, "", "", stages_run,
                             stages_skipped, errors)

    # -- render ---------------------------------------------------------
    st = state.stages["render"]
    render_dir = os.path.join(work_dir, "frames")
    manifest_guess = os.path.join(render_dir, "manifest.json")
    if st.status == "done" and os.path.isfile(st.artifact):
        stages_skipped.append("render")
        artifacts["manifest"] = st.artifact
    else:
        os.makedirs(render_dir, exist_ok=True)
        st.status = "running"
        _save_state(work_dir, state)
        try:
            if field_obj is None:
                loader = load_field_fn or _load_field_from_json
                field_obj = loader(artifacts["field_json"])
            fn = render_fn or _default_render
            manifest_path = fn(recipe, field_obj, render_dir,
                               lambda d, t: _prog("render", d, t))
            artifacts["manifest"] = manifest_path
            st.status, st.artifact = "done", manifest_path
            stages_run.append("render")
        except Exception as exc:  # noqa: BLE001
            st.status, st.error = "failed", f"{type(exc).__name__}: {exc}"
            errors.append(f"render: {exc}")
        _save_state(work_dir, state)
        if errors:
            return RunResult(recipe, work_dir, "", "", stages_run,
                             stages_skipped, errors)

    # -- encode ---------------------------------------------------------
    st = state.stages["encode"]
    encode_dir = os.path.join(work_dir, "video")
    if st.status == "done" and os.path.isfile(st.artifact):
        stages_skipped.append("encode")
        artifacts["video"] = st.artifact
    else:
        os.makedirs(encode_dir, exist_ok=True)
        st.status = "running"
        _save_state(work_dir, state)
        try:
            fn = encode_fn or _default_encode
            video_path = fn(recipe, artifacts["manifest"], encode_dir,
                            lambda d: _prog("encode", d, 100))
            artifacts["video"] = video_path
            st.status, st.artifact = "done", video_path
            stages_run.append("encode")
        except Exception as exc:  # noqa: BLE001
            st.status, st.error = "failed", f"{type(exc).__name__}: {exc}"
            errors.append(f"encode: {exc}")
        _save_state(work_dir, state)
        if errors:
            return RunResult(recipe, work_dir, "", "", stages_run,
                             stages_skipped, errors)

    # -- deliver --------------------------------------------------------
    from reels.deliver import finalize

    st = state.stages["deliver"]
    if st.status == "done" and os.path.isfile(st.artifact):
        stages_skipped.append("deliver")
        summary_path = st.artifact
    else:
        st.status = "running"
        _save_state(work_dir, state)
        try:
            done = finalize(recipe, artifacts["video"], work_dir,
                            stages_run=list(stages_run),
                            stages_skipped=list(stages_skipped))
            summary_path = done.summary_path
            artifacts["video"] = done.video_path  # published copy
            st.status, st.artifact = "done", summary_path
            stages_run.append("deliver")
        except Exception as exc:  # noqa: BLE001
            st.status, st.error = "failed", f"{type(exc).__name__}: {exc}"
            errors.append(f"deliver: {exc}")
            summary_path = ""
        _save_state(work_dir, state)

    return RunResult(recipe, work_dir, artifacts.get("video", ""),
                     summary_path, stages_run, stages_skipped, errors)


def _load_field_from_json(path: str):
    """Reload a CurrentField from a cached field.json (duck-typed)."""
    cls = _peers.current_field_cls()
    if hasattr(cls, "from_json"):
        return cls.from_json(path)
    raise _peers.PeerMissing(
        "survey-reels cannot resume here: the installed survey-currents "
        "lacks CurrentField.from_json (pip install -U survey-currents)")
