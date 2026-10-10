"""Pure-numpy connected-component labelling (4/8-connectivity).

Small shared utility: water-mask despeckling drops tiny water components
(noise speckles, boats, wet roofs) before shoreline extraction so the
shoreline is the *coast*, not clutter.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def label_components(mask: np.ndarray, connectivity: int = 8) -> Tuple[np.ndarray, int]:
    """Label connected True-components of a 2D bool array.

    Returns (labels, n) with labels 0 = background, 1..n = components.
    Iterative flood fill — no scipy needed.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, dtype=np.int32)
    if not mask.any():
        return labels, 0
    if connectivity == 4:
        neigh = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        neigh = (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        )
    n = 0
    rows, cols = mask.shape
    for sr in range(rows):
        for sc in range(cols):
            if not mask[sr, sc] or labels[sr, sc]:
                continue
            n += 1
            stack = [(sr, sc)]
            labels[sr, sc] = n
            while stack:
                r, c = stack.pop()
                for dr, dc in neigh:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        if mask[rr, cc] and not labels[rr, cc]:
                            labels[rr, cc] = n
                            stack.append((rr, cc))
    return labels, n


def component_sizes(labels: np.ndarray, n: int) -> Dict[int, int]:
    """Pixel count per component label (1..n)."""
    counts = np.bincount(labels.ravel(), minlength=n + 1)
    return {i: int(counts[i]) for i in range(1, n + 1)}


def remove_small_components(
    mask: np.ndarray, min_pixels: int, connectivity: int = 8
) -> np.ndarray:
    """Return mask with True-components smaller than min_pixels removed."""
    labels, n = label_components(mask, connectivity)
    if n == 0:
        return np.asarray(mask, dtype=bool).copy()
    sizes = component_sizes(labels, n)
    keep = np.zeros(n + 1, dtype=bool)
    for i, s in sizes.items():
        keep[i] = s >= min_pixels
    return keep[labels]
