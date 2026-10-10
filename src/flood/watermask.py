"""Water masks from water-index rasters.

A water mask is a boolean array: True where the pixel is water. The pipeline
is deliberately small and explicit:

1. **Threshold** the water index (NDWI/MNDWI): ``index >= threshold``.
   NaN index pixels are never water.
2. **Cloud interaction**: cloud pixels are *invalid*, not water — a cloud
   mask (True = cloud) clears those pixels from the water mask and from the
   valid-pixel count. Without this step, bright cloud tops and especially
   cloud *shadows* (dark in NIR, bright in green) can masquerade as water.
3. **Despeckle**: drop connected water components smaller than
   ``min_pixels`` — single-pixel noise speckles, mixed shoreline pixels,
   and damp soil that barely clears the threshold.

Failure modes (documented, not hidden):

- **Cloud cover is the big one.** Optical water detection cannot see through
  clouds; heavy cloud on a flood pass means an underestimated extent. The
  ``cloud_cover`` fraction travels with every time-series record so
  downstream consumers (survey-monitor thresholds) can gate on it.
- **Mixed shoreline pixels.** A 10 m Sentinel-2 pixel straddling a bank is
  part water, part land; its index lands near the threshold and flickers
  pass to pass. Treat small area changes near shorelines as noise.
- **Threshold sensitivity.** The default 0.0 threshold is the literature
  convention for NDWI, but turbid floodwater and shadowed water read lower.
  Tune per catchment against a known event; never treat the default as truth.
- **Dark non-water.** Topographic/building shadows and asphalt can clear
  a low NDWI threshold. MNDWI suppresses most built-up leakage.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Tuple

import numpy as np

#: Literature-convention default NDWI/MNDWI water threshold.
DEFAULT_THRESHOLD = 0.0


def water_mask(index_array: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Boolean water mask: True where ``index >= threshold``.

    NaN index pixels are never water. Returns a fresh bool array.
    """
    arr = np.asarray(index_array, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        mask = np.isfinite(arr) & (arr >= threshold)
    return mask


def apply_cloud_mask(
    mask: np.ndarray, cloud_mask: np.ndarray | None
) -> Tuple[np.ndarray, np.ndarray]:
    """Clear clouded pixels from a water mask.

    Returns ``(clean_mask, valid)`` where ``valid`` is True for pixels that
    are not cloud (and not NaN-driven invalid). ``cloud_mask`` is a boolean
    array, True = cloud; ``None`` means no clouds.
    """
    mask = np.asarray(mask, dtype=bool)
    if cloud_mask is None:
        return mask.copy(), np.ones(mask.shape, dtype=bool)
    clouds = np.asarray(cloud_mask, dtype=bool)
    if clouds.shape != mask.shape:
        raise ValueError(
            f"cloud mask shape {clouds.shape} != water mask shape {mask.shape}"
        )
    valid = ~clouds
    return mask & valid, valid


def label_components(mask: np.ndarray, connectivity: int = 8) -> Tuple[np.ndarray, int]:
    """Label 4/8-connected water components (pure numpy, no scipy).

    Returns ``(labels, n_components)``: ``labels`` is an int32 array with 0
    for background and 1..n for components. Iterative flood fill, so it is
    O(pixels) time and fine for zone-sized rasters.
    """
    mask = np.asarray(mask, dtype=bool)
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    labels = np.zeros(mask.shape, dtype=np.int32)
    if not mask.any():
        return labels, 0
    rows, cols = mask.shape
    if connectivity == 8:
        neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        neigh = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    current = 0
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or labels[r, c]:
                continue
            current += 1
            queue = deque([(r, c)])
            labels[r, c] = current
            while queue:
                pr, pc = queue.popleft()
                for dr, dc in neigh:
                    nr, nc = pr + dr, pc + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and mask[nr, nc]
                        and not labels[nr, nc]
                    ):
                        labels[nr, nc] = current
                        queue.append((nr, nc))
    return labels, current


def component_sizes(labels: np.ndarray, n_components: int) -> Dict[int, int]:
    """Pixel count per component id (1-based)."""
    counts = np.bincount(labels.ravel(), minlength=n_components + 1)
    return {i: int(counts[i]) for i in range(1, n_components + 1)}


def despeckle(mask: np.ndarray, min_pixels: int = 4) -> np.ndarray:
    """Drop water components smaller than ``min_pixels``.

    ``min_pixels=1`` is a no-op (keeps everything). Returns a new bool array.
    """
    mask = np.asarray(mask, dtype=bool)
    if min_pixels <= 1:
        return mask.copy()
    labels, n = label_components(mask)
    if n == 0:
        return mask.copy()
    sizes = component_sizes(labels, n)
    keep = {i for i, s in sizes.items() if s >= min_pixels}
    out = np.zeros(mask.shape, dtype=bool)
    for i in keep:
        out |= labels == i
    return out


def build_water_mask(
    index_array: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    cloud_mask: np.ndarray | None = None,
    min_pixels: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full pipeline: threshold -> cloud-clear -> despeckle.

    Returns ``(water, valid)`` bool arrays.
    """
    raw = water_mask(index_array, threshold)
    clean, valid = apply_cloud_mask(raw, cloud_mask)
    return despeckle(clean, min_pixels), valid
