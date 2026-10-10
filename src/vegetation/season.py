"""Growing-season summaries per zone.

From a zone's vigor time series for one year: green-up date (rising edge
crosses 50% of the seasonal amplitude), peak vigor and its date, senescence
onset (falling edge crosses 50%), season length, and the season integral
(trapezoidal area under the index curve — a proxy for total seasonal
productivity). The integral is also z-scored against prior years' integrals.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

from .timeseries import ZonePassRecord, select


@dataclass
class SeasonSummary:
    zone_id: str
    zone_name: str
    index: str
    year: int
    season_found: bool
    greenup_date: str | None
    peak_date: str | None
    peak_value: float | None
    senescence_date: str | None
    season_days: int | None
    season_integral: float | None  # index-days
    integral_baseline_mean: float | None
    integral_baseline_std: float | None
    integral_z_score: float | None
    n_passes: int
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ordinal(date_str: str) -> int:
    return _dt.date.fromisoformat(date_str[:10]).toordinal()


def _crossing(
    dates: List[int], values: List[float], threshold: float, rising: bool
) -> float | None:
    """Ordinal date where the series crosses threshold (linear interpolation)."""
    rng = range(len(values)) if rising else range(len(values))
    for i in range(1, len(values)):
        v0, v1 = values[i - 1], values[i]
        if rising and v0 < threshold <= v1:
            frac = 0.0 if v1 == v0 else (threshold - v0) / (v1 - v0)
            return dates[i - 1] + frac * (dates[i] - dates[i - 1])
        if not rising and v0 > threshold >= v1:
            frac = 0.0 if v0 == v1 else (v0 - threshold) / (v0 - v1)
            return dates[i - 1] + frac * (dates[i] - dates[i - 1])
    return None


def _trapz(
    dates: List[float], values: List[float], start: float, end: float
) -> float:
    """Trapezoidal integral of the polyline between start and end dates."""
    pts: List[tuple] = []
    n = len(dates)
    for i in range(n):
        if start <= dates[i] <= end:
            pts.append((dates[i], values[i]))
    # Interpolated endpoints.
    for edge in (start, end):
        if not any(abs(p[0] - edge) < 1e-9 for p in pts):
            for i in range(1, n):
                if dates[i - 1] <= edge <= dates[i]:
                    frac = (
                        0.0
                        if dates[i] == dates[i - 1]
                        else (edge - dates[i - 1]) / (dates[i] - dates[i - 1])
                    )
                    pts.append(
                        (edge, values[i - 1] + frac * (values[i] - values[i - 1]))
                    )
                    break
    pts.sort()
    area = 0.0
    for i in range(1, len(pts)):
        area += (pts[i][0] - pts[i - 1][0]) * (pts[i][1] + pts[i - 1][1]) / 2.0
    return area


def season_summary(
    records: Sequence[ZonePassRecord],
    zone_id: str,
    index: str,
    year: int,
    min_amplitude: float = 0.08,
    amplitude_fraction: float = 0.5,
    prior_integrals: Sequence[float] = (),
) -> SeasonSummary:
    """Summarise one zone's growing season for one year and index.

    Green-up (senescence) is the date the rising (falling) edge crosses
    ``vmin + amplitude_fraction * amplitude`` — with the default 0.5 this is
    mid-green-up / mid-senescence, the standard amplitude-fraction convention.
    """
    recs = [
        r
        for r in select(records, zone_id=zone_id, index=index, year=year)
        if math.isfinite(r.mean)
    ]
    name = recs[0].zone_name if recs else zone_id
    base = dict(
        zone_id=zone_id,
        zone_name=name,
        index=index,
        year=year,
        n_passes=len(recs),
    )
    _empty = dict(
        season_found=False,
        greenup_date=None,
        peak_date=None,
        peak_value=None,
        senescence_date=None,
        season_days=None,
        season_integral=None,
        integral_baseline_mean=None,
        integral_baseline_std=None,
        integral_z_score=None,
    )
    if len(recs) < 3:
        return SeasonSummary(**base, **_empty, note="fewer than 3 valid passes")
    dates = [_ordinal(r.date) for r in recs]
    values = [r.mean for r in recs]
    vmin, vmax = min(values), max(values)
    amplitude = vmax - vmin
    if amplitude < min_amplitude:
        return SeasonSummary(
            **base, **_empty,
            note=f"seasonal amplitude {amplitude:.3f} < {min_amplitude}",
        )
    threshold = vmin + amplitude_fraction * amplitude
    greenup = _crossing(dates, values, threshold, rising=True)
    peak_i = max(range(len(values)), key=lambda i: values[i])
    senescence = None
    for i in range(peak_i + 1, len(values)):
        v0, v1 = values[i - 1], values[i]
        if v0 > threshold >= v1:
            frac = 0.0 if v0 == v1 else (v0 - threshold) / (v0 - v1)
            senescence = dates[i - 1] + frac * (dates[i] - dates[i - 1])
            break
    if greenup is None or senescence is None:
        return SeasonSummary(
            **base, **_empty,
            note="green-up or senescence edge outside the pass window",
        )
    integral = _trapz([float(d) for d in dates], values, greenup, senescence)
    iz = None
    bmean = bstd = None
    priors = [p for p in prior_integrals if math.isfinite(p)]
    if len(priors) >= 2:
        bmean = sum(priors) / len(priors)
        bstd = math.sqrt(sum((p - bmean) ** 2 for p in priors) / (len(priors) - 1))
        iz = 0.0 if bstd == 0 else (integral - bmean) / bstd
    return SeasonSummary(
        **base,
        season_found=True,
        greenup_date=_dt.date.fromordinal(round(greenup)).isoformat(),
        peak_date=_dt.date.fromordinal(dates[peak_i]).isoformat(),
        peak_value=values[peak_i],
        senescence_date=_dt.date.fromordinal(round(senescence)).isoformat(),
        season_days=round(senescence) - round(greenup),
        season_integral=integral,
        integral_baseline_mean=bmean,
        integral_baseline_std=bstd,
        integral_z_score=iz,
    )


def summarize_all_years(
    records: Sequence[ZonePassRecord],
    index: str,
    zone_id: str | None = None,
    min_amplitude: float = 0.08,
    amplitude_fraction: float = 0.5,
) -> List[SeasonSummary]:
    """Season summaries for every zone/year, with integral z vs prior years."""
    recs = select(records, zone_id=zone_id, index=index)
    zones = sorted({r.zone_id for r in recs})
    out: List[SeasonSummary] = []
    for zid in zones:
        years = sorted({r.year for r in recs if r.zone_id == zid})
        priors: List[float] = []
        for y in years:
            s = season_summary(
                recs, zid, index, y, min_amplitude, amplitude_fraction,
                prior_integrals=priors,
            )
            out.append(s)
            if s.season_found and s.season_integral is not None:
                priors.append(s.season_integral)
    return out


def write_season_json(summaries: Sequence[SeasonSummary], path: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([s.to_dict() for s in summaries], fh, indent=2)
    return path
