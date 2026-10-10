"""Erosion/accretion rates: ordinary least squares of position vs. time.

For each transect, fit position_m = a + b * t (t in decimal years):

- ``rate_m_per_yr`` (b): negative = erosion, positive = accretion —
  *given the landward→seaward transect convention* (see
  :mod:`coast.transects`).
- ``rate_se_m_per_yr``: standard error of the slope — the honest
  uncertainty bar. Wide SE = don't trust the sign.
- ``net_change_m``: last valid position − first valid position.
- ``endpoint_rate_m_per_yr``: net change / years spanned — a robustness
  cross-check against the regression (outliers move OLS more than endpoints).

Needs ≥ 3 valid positions; otherwise the rate fields are NaN.
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

#: Column order of the rates CSV. Append-only.
RATES_COLUMNS = [
    "transect_id",
    "transect_name",
    "site_id",
    "n_passes",
    "first_date",
    "last_date",
    "years_span",
    "rate_m_per_yr",
    "rate_se_m_per_yr",
    "intercept_m",
    "r_squared",
    "net_change_m",
    "endpoint_rate_m_per_yr",
]

#: These columns are the contract survey-monitor can threshold
#: (erosion watch: rate_m_per_yr < -1.0; net loss: net_change_m < -5.0).
MONITOR_CONTRACT_COLUMNS = ("rate_m_per_yr", "net_change_m")


@dataclass
class TransectRate:
    transect_id: str
    transect_name: str
    site_id: str
    n_passes: int
    first_date: str
    last_date: str
    years_span: float
    rate_m_per_yr: float
    rate_se_m_per_yr: float
    intercept_m: float
    r_squared: float
    net_change_m: float
    endpoint_rate_m_per_yr: float


def _decimal_years(dates: Sequence[_dt.date]) -> np.ndarray:
    base = dates[0]
    return np.array([(d - base).days for d in dates], dtype=float) / 365.25


def fit_rate(dates: Sequence[_dt.date], positions: Sequence[float]) -> Dict[str, float]:
    """OLS fit of position vs. decimal years. NaN fields when n < 3."""
    pairs = [
        (d, p)
        for d, p in zip(dates, positions)
        if d is not None and p == p  # drop NaN positions
    ]
    out = {
        "n": len(pairs),
        "rate": float("nan"),
        "se": float("nan"),
        "intercept": float("nan"),
        "r2": float("nan"),
        "net": float("nan"),
        "endpoint": float("nan"),
        "span": 0.0,
    }
    if len(pairs) < 2:
        return out
    ds = sorted(pairs, key=lambda t: t[0])
    t = _decimal_years([d for d, _ in ds])
    y = np.array([p for _, p in ds], dtype=float)
    out["span"] = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    out["net"] = float(y[-1] - y[0])
    out["endpoint"] = float(out["net"] / out["span"]) if out["span"] > 0 else float("nan")
    if len(pairs) < 3:
        return out
    tmean, ymean = t.mean(), y.mean()
    sxx = float(((t - tmean) ** 2).sum())
    if sxx == 0:
        return out
    sxy = float(((t - tmean) * (y - ymean)).sum())
    slope = sxy / sxx
    intercept = float(ymean - slope * tmean)
    yhat = intercept + slope * t
    sse = float(((y - yhat) ** 2).sum())
    sst = float(((y - ymean) ** 2).sum())
    dof = len(pairs) - 2
    se = float(np.sqrt(sse / dof / sxx)) if dof > 0 else float("nan")
    out.update(
        rate=float(slope),
        se=se,
        intercept=intercept,
        r2=float(1.0 - sse / sst) if sst > 0 else float("nan"),
    )
    return out


def compute_rates(
    records: Sequence[object],
) -> List[TransectRate]:
    """Group position records by transect and fit one rate each.

    ``records`` are :class:`coast.positions.TransectPassRecord` (duck-typed
    to avoid a hard import cycle in tests).
    """
    groups: Dict[Tuple[str, str, str], List[Tuple[_dt.date, float]]] = {}
    meta: Dict[Tuple[str, str, str], Tuple[str, str]] = {}
    for r in records:  # type: ignore[union-attr]
        if not r.valid:
            continue
        key = (r.transect_id, r.site_id, r.pass_id)
        gkey = (r.transect_id, r.site_id, r.transect_name)
        groups.setdefault(gkey, []).append(
            (_dt.date.fromisoformat(r.date), float(r.position_m))
        )
        meta[gkey] = (r.transect_name, r.site_id)
    rates: List[TransectRate] = []
    for (tid, sid, _tname), pairs in sorted(groups.items()):
        tname, site = meta[(tid, sid, _tname)]
        ds = sorted(pairs, key=lambda t: t[0])
        fit = fit_rate([d for d, _ in ds], [p for _, p in ds])
        rates.append(
            TransectRate(
                transect_id=tid,
                transect_name=tname,
                site_id=sid,
                n_passes=fit["n"],
                first_date=ds[0][0].isoformat(),
                last_date=ds[-1][0].isoformat(),
                years_span=fit["span"],
                rate_m_per_yr=fit["rate"],
                rate_se_m_per_yr=fit["se"],
                intercept_m=fit["intercept"],
                r_squared=fit["r2"],
                net_change_m=fit["net"],
                endpoint_rate_m_per_yr=fit["endpoint"],
            )
        )
    return rates


def write_rates_csv(path: str, rates: Sequence[TransectRate]) -> str:
    """Write rates (overwrite). Returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RATES_COLUMNS)
        writer.writeheader()
        for r in rates:
            writer.writerow(
                {
                    "transect_id": r.transect_id,
                    "transect_name": r.transect_name,
                    "site_id": r.site_id,
                    "n_passes": r.n_passes,
                    "first_date": r.first_date,
                    "last_date": r.last_date,
                    "years_span": f"{r.years_span:.4f}",
                    "rate_m_per_yr": "" if r.rate_m_per_yr != r.rate_m_per_yr else f"{r.rate_m_per_yr:.4f}",
                    "rate_se_m_per_yr": "" if r.rate_se_m_per_yr != r.rate_se_m_per_yr else f"{r.rate_se_m_per_yr:.4f}",
                    "intercept_m": "" if r.intercept_m != r.intercept_m else f"{r.intercept_m:.3f}",
                    "r_squared": "" if r.r_squared != r.r_squared else f"{r.r_squared:.4f}",
                    "net_change_m": "" if r.net_change_m != r.net_change_m else f"{r.net_change_m:.3f}",
                    "endpoint_rate_m_per_yr": ""
                    if r.endpoint_rate_m_per_yr != r.endpoint_rate_m_per_yr
                    else f"{r.endpoint_rate_m_per_yr:.4f}",
                }
            )
    return path
