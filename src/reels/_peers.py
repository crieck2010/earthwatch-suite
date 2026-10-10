"""Lazy, duck-typed access to the three peer modules.

Every peer is OPTIONAL. Import errors are converted into :class:`PeerMissing`
with an actionable install message, never a bare ImportError. All call sites
accept injected callables so tests can run fully offline with fakes.
"""


class PeerMissing(RuntimeError):
    """Raised when an optional peer module is needed but not installed."""


def _load(dotted: str, pip_name: str):
    try:
        module_name, attr = dotted.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_name)
        return getattr(module, attr)
    except (ImportError, AttributeError) as exc:
        raise PeerMissing(
            f"survey-reels needs the peer package {pip_name!r} for this step, "
            f"but it is not installed ({dotted} could not be imported). "
            f"Install it with: pip install {pip_name}"
        ) from exc


def fetch_currents_noaa():
    """Return ``currents.noaa_ofs.fetch_currents`` or raise PeerMissing."""
    return _load("currents.noaa_ofs.fetch_currents", "survey-currents")


def subset_cmems_fn():
    """Return ``currents.cmems.subset_cmems`` or raise PeerMissing."""
    return _load("currents.cmems.subset_cmems", "survey-currents")


def parse_cmems_netcdf_fn():
    """Return ``currents.cmems.parse_cmems_netcdf`` or raise PeerMissing."""
    return _load("currents.cmems.parse_cmems_netcdf", "survey-currents")


def current_field_cls():
    """Return ``currents.models.CurrentField`` or raise PeerMissing."""
    return _load("currents.models.CurrentField", "survey-currents")


def time_varying_field_cls():
    """Return ``flow.fields.TimeVaryingField`` or raise PeerMissing."""
    return _load("flow.fields.TimeVaryingField", "survey-flow")


def render_animation_fn():
    """Return ``flow.pipeline.render_animation`` or raise PeerMissing."""
    return _load("flow.pipeline.render_animation", "survey-flow")


def render_video_fn():
    """Return ``animate.pipeline.render_video`` or raise PeerMissing."""
    return _load("animate.pipeline.render_video", "survey-animate")


def list_flow_styles():
    """Return available survey-flow style names, or [] when the peer is absent."""
    try:
        fn = _load("flow.styles.list_styles", "survey-flow")
    except PeerMissing:
        return []
    return [s.name for s in fn()]


def list_animate_presets():
    """Return available survey-animate preset names, or [] when absent."""
    try:
        fn = _load("animate.presets.list_presets", "survey-animate")
    except PeerMissing:
        return []
    return [p.name for p in fn()]
