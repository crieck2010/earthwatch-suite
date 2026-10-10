"""Engine-registered colormaps.

One house colormap for now:

- ``teal_pink`` — the mapped.earth population-density ramp (the Italy
  reel look): deep teal in the sparse outskirts, through turquoise and
  a warm cream shoulder, to orange and finally hot pink at the densest
  cores. Sequential by construction: luminance climbs from the dark
  teal low end to the cream shoulder, then the hot, saturated top end
  carries the "core" read even where its luminance is a step below
  the cream.

The colormap is registered with matplotlib at import time under the
name ``teal_pink``, so ``matplotlib.colormaps["teal_pink"]`` and any
``cmap="teal_pink"`` argument anywhere in matplotlib work. Registration
is idempotent: importing this module twice (or after user code already
registered the same name) never raises.

Determinism: the ramp is built from fixed hex anchors via
:class:`matplotlib.colors.LinearSegmentedColormap` — no RNG, identical
on every import.
"""

import matplotlib

matplotlib.use("Agg")

from matplotlib.colors import LinearSegmentedColormap

# Anchor stops for teal_pink: (position in [0, 1], hex color).
# Deep teal -> turquoise -> cream -> orange -> pink. The cream shoulder
# sits just past the midpoint so mid-density reads warm before the
# orange/pink core takes over at the top end.
TEAL_PINK_STOPS = (
    (0.00, "#07304a"),
    (0.25, "#12808f"),
    (0.45, "#35b6a8"),
    (0.65, "#f5e7c1"),
    (0.82, "#f6a23c"),
    (1.00, "#ef5da8"),
)

_OWN_COLORMAPS = ("teal_pink",)


def _build_teal_pink() -> LinearSegmentedColormap:
    """Construct the (unregistered) ``teal_pink`` colormap object."""
    return LinearSegmentedColormap.from_list(
        "teal_pink", list(TEAL_PINK_STOPS), N=256
    )


def _register_teal_pink() -> None:
    """Register ``teal_pink`` with matplotlib once (idempotent)."""
    if "teal_pink" not in matplotlib.colormaps:
        matplotlib.colormaps.register(_build_teal_pink())


_register_teal_pink()


def get_colormap(name):
    """Resolve ``name`` to a :class:`matplotlib.colors.Colormap`.

    Accepts a colormap name (any matplotlib builtin, plus this engine's
    own registered names — see :func:`list_colormaps`) or a Colormap
    object, which is passed through unchanged. This mirrors how the
    other engine modules resolve ``cmap`` arguments.

    Raises ``ValueError`` for an unknown name.
    """
    if isinstance(name, matplotlib.colors.Colormap):
        return name
    if isinstance(name, str):
        try:
            return matplotlib.colormaps[name]
        except KeyError as exc:
            raise ValueError(f"unknown colormap {name!r}") from exc
    raise ValueError(
        f"cmap must be a colormap name or Colormap, got {type(name).__name__}"
    )


def list_colormaps() -> list:
    """Names of the colormaps this engine registers (not matplotlib's).

    ``get_colormap`` / ``render_surface`` accept any matplotlib colormap
    name; this list is only the engine's own additions.
    """
    return sorted(_OWN_COLORMAPS)
