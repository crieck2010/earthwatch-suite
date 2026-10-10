"""Pure-numpy binary morphology shared by the burn engines.

Connected-component labelling (no scipy) and small-speckle removal. The
same routines back the burn-mask despeckling in :mod:`burn.dnbr`-derived
products, the disturbance patches in :mod:`burn.disturbance`, and the
polygon vectoriser in :mod:`burn.polygons`.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def label_components(
    mask: np.ndarray, connectivity: int = 8
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Label connected components of a bool mask (two-pass, union-find).

    Returns ``(labels, sizes)``: int32 label image (0 = background, labels
    start at 1) and a dict label -> pixel count. ``connectivity`` is 4 or 8.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    m = np.asarray(mask, dtype=bool)
    rows, cols = m.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    next_label = 1
    for r in range(rows):
        for c in range(cols):
            if not m[r, c]:
                continue
            neighbours = []
            if r > 0 and labels[r - 1, c]:
                neighbours.append(labels[r - 1, c])
            if c > 0 and labels[r, c - 1]:
                neighbours.append(labels[r, c - 1])
            if connectivity == 8:
                if r > 0 and c > 0 and labels[r - 1, c - 1]:
                    neighbours.append(labels[r - 1, c - 1])
                if r > 0 and c < cols - 1 and labels[r - 1, c + 1]:
                    neighbours.append(labels[r - 1, c + 1])
            if not neighbours:
                labels[r, c] = next_label
                parent[next_label] = next_label
                next_label += 1
            else:
                lab = min(neighbours)
                labels[r, c] = lab
                for other in neighbours:
                    union(lab, int(other))
    # Second pass: resolve equivalences and compact labels.
    remap: Dict[int, int] = {}
    sizes: Dict[int, int] = {}
    for r in range(rows):
        for c in range(cols):
            if labels[r, c]:
                root = find(int(labels[r, c]))
                if root not in remap:
                    remap[root] = len(remap) + 1
                new = remap[root]
                labels[r, c] = new
                sizes[new] = sizes.get(new, 0) + 1
    return labels, sizes


def remove_small_components(
    mask: np.ndarray, min_pixels: int, connectivity: int = 8
) -> np.ndarray:
    """Bool mask with components smaller than ``min_pixels`` dropped."""
    labels, sizes = label_components(mask, connectivity)
    keep = {lab for lab, n in sizes.items() if n >= min_pixels}
    out = np.zeros_like(labels, dtype=bool)
    for lab in keep:
        out |= labels == lab
    return out
