"""Human labels for variables and regions used in style templates.

Kept deliberately small and stdlib-only: a variable label lookup plus
a best-effort ``region_key -> "Title Case"`` prettifier. Consumers
that need authoritative labels (gazetteer names, unit strings) already
own them — survey-viz's gazetteer for regions, ``viz.render`` for
units — so this module only covers what templates need.
"""

from __future__ import annotations

#: Variable slug -> human label for ``{variable_label}`` in templates.
VARIABLE_LABELS = {
    # Ocean
    "sst": "Sea Surface Temperature",
    "currents": "Surface Currents",
    "ocean-color": "Chlorophyll-a",
    "chlorophyll": "Chlorophyll-a",
    "sea-ice": "Sea Ice Concentration",
    "night-lights": "Night Lights",
    "topography": "Topography",
    "water-storage": "Water Storage",
    # Atmosphere
    "wind": "Wind Speed",
    "msl": "Mean Sea Level Pressure",
    "t2m": "Air Temperature",
    "tp": "Precipitation",
    # Land / hazards
    "fires": "Active Fires",
    "burn-scar": "Burn Scar",
    "storm-tracks": "Storm Tracks",
    "earthquakes": "Earthquakes",
    "streamflow": "Streamflow",
}


def variable_label(variable: str) -> str:
    """Human label for ``variable``; falls back to a prettified slug."""
    if variable in VARIABLE_LABELS:
        return VARIABLE_LABELS[variable]
    return variable.replace("-", " ").replace("_", " ").title()


def region_label(region_key: str) -> str:
    """Best-effort human label for a gazetteer region slug."""
    if not region_key:
        return ""
    return region_key.replace("-", " ").replace("_", " ").title()
