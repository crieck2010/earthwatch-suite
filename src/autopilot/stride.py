"""Phenomenon-aware stride: how far apart in time should frames be?

Why this exists: an animation's stride is not a rendering detail, it is a
physics decision. Sampling tides daily aliases the ~12.4 h constituents into
nonsense; sampling the seasonal cycle 6-hourly wastes frames on mud. The right
stride follows the phenomenon's dominant timescale, so this module encodes the
timescale per phenomenon in one documented table instead of scattering magic
numbers across render scripts.

Honest limitations:
- The table is a starting point, not a law. A storm that stalls needs denser
  sampling than "synoptic" assumes; the caller owns deviations.
- "forecast_hours" is GFS-wind-specific (the GFS NOMADS feed publishes
  f000..f120); it is None for everything else and should be ignored for
  non-GFS sources.
- Unknown phenomena raise ValueError rather than returning a silent default —
  guessing a stride for an unknown phenomenon would be worse than failing.
"""

from __future__ import annotations

#: Documented stride table. stride_days/stride_hours are the recommended frame
#: spacing; forecast_hours lists the GFS forecast hours to sample (GFS only).
STRIDE_TABLE = {
    "tide": {
        "stride_days": 1,
        "stride_hours": 3,
        "forecast_hours": None,
        "reason": (
            "tidal constituents (~12.4h) need sub-daily sampling to stay legible; "
            "3-hourly resolves the semidiurnal cycle without aliasing"
        ),
    },
    "synoptic": {
        "stride_days": 1,
        "stride_hours": 6,
        "forecast_hours": (0, 6, 12, 18),
        "reason": (
            "synoptic-scale wind/storm evolution at 6-hourly; forecast_hours "
            "applies to GFS-wind sources only"
        ),
    },
    "seasonal": {
        "stride_days": 7,
        "stride_hours": 0,
        "forecast_hours": None,
        "reason": (
            "weekly sampling resolves the seasonal cycle without mud — denser "
            "sampling adds frames but no new information"
        ),
    },
    "climate": {
        "stride_days": 30,
        "stride_hours": 0,
        "forecast_hours": None,
        "reason": (
            "monthly sampling for multi-year climate context; sub-monthly "
            "variation is noise at this timescale"
        ),
    },
    "event": {
        "per_event": True,
        "stride_days": 1,
        "stride_hours": 0,
        "forecast_hours": None,
        "reason": (
            "event-driven: one frame per event, not a fixed cadence — the "
            "stride is informational, the event list drives the frames"
        ),
    },
}


def recommend_stride(phenomenon, source=None) -> dict:
    """Recommend a frame stride for a phenomenon.

    Args:
        phenomenon: one of "tide", "synoptic", "seasonal", "climate", "event".
        source: optional source name (e.g. "gfs-wind"). Currently
            informational — recorded in the output for future source-specific
            overrides, no behavior change yet.

    Returns:
        {"phenomenon", "stride_days", "stride_hours", "forecast_hours",
         "reason"} plus "per_event": True for the "event" phenomenon, plus
        "source" echoing the input.

    Raises:
        ValueError: for unknown phenomena, listing the valid keys.
    """
    if phenomenon not in STRIDE_TABLE:
        valid = ", ".join(sorted(STRIDE_TABLE))
        raise ValueError(
            f"unknown phenomenon {phenomenon!r}; valid keys: {valid}"
        )
    out = {"phenomenon": phenomenon, "source": source}
    out.update(STRIDE_TABLE[phenomenon])
    return out
