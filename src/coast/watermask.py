"""Per-pass water masks from a water-index raster.

Pipeline: threshold → cloud-clear → despeckle.

- Pixels with index > threshold are *candidate* water.
- Pixels flagged by the cloud mask are **invalid** — never water, never
  land. Clouds over the coast hide the shoreline; they must not *become*
  the shoreline.
- Tiny water components (< min_pixels) are dropped as speckle.

Conventions shared with the sibling engines: ``cloud_cover`` is a percent
(0–100) and travels on every downstream record so consumers can gate on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .masking import remove_small_components


@dataclass
class WaterMask:
    """A per-pass water mask with provenance."""

    #: True where water was detected.
    water: np.ndarray
    #: True where the pixel is usable (not cloud / not NaN).
    valid: np.ndarray
    #: Index array the mask was derived from (kept for debugging, not required).
    index: Optional[np.ndarray] = None
    index_name: str = "ndwi"
    threshold: float = 0.0
    cloud_cover: float = 0.0  # percent 0-100

    @property
    def water_fraction(self) -> float:
        """Fraction of *valid* pixels classified as water."""
        v = self.valid
        if not v.any():
            return float("nan")
        return float(self.water[v].mean())


def water_mask(
    index: np.ndarray,
    threshold: float = 0.0,
    cloud: Optional[np.ndarray] = None,
    min_pixels: int = 4,
    index_name: str = "ndwi",
) -> WaterMask:
    """Build a water mask from an index raster.

    Parameters
    ----------
    index: 2D float array (NaN = missing).
    threshold: index value above which a pixel is candidate water.
    cloud: optional 2D bool array, True = cloud/cloud-shadow.
    min_pixels: water components smaller than this are dropped.
    """
    index = np.asarray(index, dtype=float)
    candidate = np.isfinite(index) & (index > threshold)
    valid = np.isfinite(index)
    cloud_cover = 0.0
    if cloud is not None:
        cloud = np.asarray(cloud, dtype=bool)
        if cloud.shape != index.shape:
            raise ValueError(
                f"cloud mask shape {cloud.shape} != index shape {index.shape}"
            )
        cloud_cover = float(cloud.mean() * 100.0)
        valid = valid & ~cloud
        candidate = candidate & ~cloud
    water = remove_small_components(candidate & valid, min_pixels)
    return WaterMask(
        water=water,
        valid=valid,
        index=index,
        index_name=index_name,
        threshold=threshold,
        cloud_cover=cloud_cover,
    )
