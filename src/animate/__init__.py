"""survey-animate: frame-sequence to video engine.

Encode ordered PNG frames (from a survey-flow frame manifest, or any
directory of PNGs) into finished videos: MP4 (H.264), GIF, WebM.

The core is stdlib+numpy. Encoding shells out to an ffmpeg binary —
the ``imageio-ffmpeg`` wheel's bundled static binary when installed,
otherwise a system ``ffmpeg`` on PATH. Burned-in text overlays need
matplotlib (Agg); everything else works without it.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "Manifest",
    "ManifestFrame",
    "MotionSpec",
    "Preset",
    "VideoResult",
    "read_manifest",
    "frames_from_dir",
    "render_video",
    "list_presets",
]


def __getattr__(name: str):
    if name in ("Manifest", "ManifestFrame", "read_manifest", "frames_from_dir"):
        from animate import manifest as _m
        return getattr(_m, name)
    if name in ("Preset", "list_presets"):
        from animate import presets as _p
        return getattr(_p, name)
    if name in ("VideoResult", "render_video"):
        from animate import pipeline as _pl
        return getattr(_pl, name)
    if name == "MotionSpec":
        from animate import motion as _mo
        return _mo.MotionSpec
    raise AttributeError(f"module 'animate' has no attribute {name!r}")
