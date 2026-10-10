"""dNBR: differenced Normalised Burn Ratio.

dNBR is the standard burn-severity metric: the pre-fire NBR minus the
post-fire NBR. A pixel that burned drops from a high NBR to a low (often
negative) NBR, so dNBR is large and positive over the scar and near zero
where nothing changed. Also provided: RdNBR (relative dNBR), which divides
by the square root of the pre-fire NBR magnitude so that the same absolute
change counts more in sparse vegetation — useful where pre-fire cover was
thin (Miller & Thode, 2007).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


def dnbr(pre_nbr: np.ndarray, post_nbr: np.ndarray) -> np.ndarray:
    """dNBR = pre-fire NBR − post-fire NBR, NaN-honest.

    Either input NaN → output NaN. Arrays must broadcast to the same shape.
    """
    pre = np.asarray(pre_nbr, dtype=np.float32)
    post = np.asarray(post_nbr, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        out = pre - post
    out[~np.isfinite(pre) | ~np.isfinite(post)] = np.nan
    return out.astype(np.float32)


def rdnbr(pre_nbr: np.ndarray, post_nbr: np.ndarray) -> np.ndarray:
    """Relative dNBR: dNBR / sqrt(|pre-fire NBR|), NaN-honest.

    Normalises the burn signal against how much vegetation was there to
    begin with. Guarded where |pre| is ~0 (returns NaN there).
    """
    pre = np.asarray(pre_nbr, dtype=np.float32)
    d = dnbr(pre_nbr, post_nbr)
    denom = np.sqrt(np.abs(pre))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom < 1e-6, np.nan, d / denom)
    out[~np.isfinite(d)] = np.nan
    return out.astype(np.float32)


@dataclass(frozen=True)
class PassInfo:
    """One candidate satellite pass for pre/post selection."""

    pass_id: str
    date: str  # ISO YYYY-MM-DD
    cloud_cover: float  # percent 0-100, suite convention
    nbr_path: Optional[str] = None  # path to the scene's NBR raster, if known


def select_pre_post(
    passes: Sequence[PassInfo],
    fire_date: str,
    pre_days: int = 60,
    post_days: int = 60,
    max_cloud: float = 20.0,
) -> Tuple[PassInfo, PassInfo]:
    """Pick the pre-fire and post-fire passes around a known fire date.

    Pre = the latest pass strictly before ``fire_date`` with
    ``cloud_cover <= max_cloud`` and within ``pre_days`` back.
    Post = the earliest pass strictly after ``fire_date`` within
    ``post_days`` forward. Raises ``ValueError`` when either side has no
    qualifying pass.
    """
    if pre_days <= 0 or post_days <= 0:
        raise ValueError("pre_days and post_days must be positive")
    pre_candidates: List[PassInfo] = []
    post_candidates: List[PassInfo] = []
    for p in passes:
        if p.date < fire_date and p.cloud_cover <= max_cloud:
            pre_candidates.append(p)
        elif p.date > fire_date and p.cloud_cover <= max_cloud:
            post_candidates.append(p)
    pre_candidates.sort(key=lambda p: p.date, reverse=True)
    post_candidates.sort(key=lambda p: p.date)
    if not pre_candidates:
        raise ValueError(
            f"no pre-fire pass within {pre_days} days before {fire_date} "
            f"under {max_cloud}% cloud"
        )
    if not post_candidates:
        raise ValueError(
            f"no post-fire pass within {post_days} days after {fire_date} "
            f"under {max_cloud}% cloud"
        )
    return pre_candidates[0], post_candidates[0]
