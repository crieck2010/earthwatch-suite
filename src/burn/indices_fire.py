"""Burn/disturbance index computation.

The workhorse index (NBR) is **reused** from survey-imagery's
:mod:`imagery.indices` registry via a lazy import — it is not reimplemented
here. This module defines only what that registry does not carry and follows
the same NaN-honest conventions: any input pixel that is NaN, or whose
denominator is zero, yields NaN.

- **NBR** (Normalised Burn Ratio): (nir − swir2) / (nir + swir2).
  Healthy green vegetation is strongly positive; recently burned land drops
  sharply because NIR reflectance collapses and SWIR2 rises.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

#: Indices this engine works with. ``nbr`` resolves through survey-imagery's
#: registry; the dNBR/RdNBR arithmetic lives in :mod:`burn.dnbr`.
BURN_INDICES: Tuple[str, ...] = ("nbr", "dnbr", "rdnbr")


def _imagery_indices():
    try:
        from imagery import indices as _im
    except ImportError as exc:
        raise ImportError(
            "survey-burn needs the 'survey-imagery' package for "
            "NBR (pip install 'survey-burn[acquire]')"
        ) from exc
    return _im


def required_bands(index_name: str) -> Tuple[str, ...]:
    """Band aliases needed for a burn index (all resolve to NBR's bands)."""
    name = index_name.lower()
    if name in ("dnbr", "rdnbr"):
        name = "nbr"
    return tuple(_imagery_indices().required_bands(name))


def compute(index_name: str, bands: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
    """Compute a single-scene burn index from alias -> reflectance arrays."""
    name = index_name.lower()
    if name in ("dnbr", "rdnbr"):
        raise ValueError(
            f"{name} is a two-scene index — use burn.dnbr.dnbr() on pre/post NBR"
        )
    return _imagery_indices().compute(name, bands, **kwargs)
