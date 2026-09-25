"""earthwatch-suite: registry of the EarthWatch remote-sensing engines.

This meta-package wires the EarthWatch computation engines together and
documents the contracts between them. The engines themselves live in
their own public repos; install this package to pull them all in, or
install each engine standalone — every engine is usable on its own.
"""

__version__ = "0.1.0"

#: Registry of every engine in the EarthWatch remote-sensing project.
#: "status" is one of: shipped, planned.
MODULES = (
    {
        "package": "survey-imagery",
        "repo": "crieck2010/survey-imagery",
        "version": "0.1.1",
        "description": "Satellite imagery engine: STAC discovery, cloud masking, spectral indices, per-pass site monitor",
        "status": "shipped",
    },
    {
        "package": "survey-change",
        "repo": "crieck2010/survey-change",
        "version": "0.1.0",
        "description": "Change-detection engine: per-pass and per-pixel change statistics over imagery time series",
        "status": "shipped",
    },
    {
        "package": "survey-monitor",
        "repo": "crieck2010/survey-monitor",
        "version": "0.1.0",
        "description": "Scheduled site-monitoring engine: site configs, threshold alert evaluation, alert events",
        "status": "shipped",
    },
    {
        "package": "survey-alerts",
        "repo": "crieck2010/survey-alerts",
        "version": "planned",
        "description": "Alert reports: email and PDF delivery of monitoring events (per survey-monitor's notifier hook)",
        "status": "planned",
    },
    {
        "package": "survey-license",
        "repo": "crieck2010/survey-license",
        "version": "planned",
        "description": "License-key engine for gating the monitoring product to paying customers",
        "status": "planned",
    },
    {
        "package": "survey-vegetation",
        "repo": "crieck2010/survey-vegetation",
        "version": "planned",
        "description": "Vegetation/vigor analytics: NDVI/EVI/SAVI time series, anomalies vs historical baseline",
        "status": "planned",
    },
    {
        "package": "survey-flood",
        "repo": "crieck2010/survey-flood",
        "version": "planned",
        "description": "Flood/water-extent mapping: NDWI water masks, per-pass flood deltas vs dry-season baseline",
        "status": "planned",
    },
    {
        "package": "survey-burn",
        "repo": "crieck2010/survey-burn",
        "version": "planned",
        "description": "Burn-scar/disturbance mapping: NBR/dNBR fire severity, generalized clearing detection",
        "status": "planned",
    },
    {
        "package": "survey-coast",
        "repo": "crieck2010/survey-coast",
        "version": "planned",
        "description": "Coastal change: NDWI shoreline extraction per pass, transect erosion rates, storm deltas",
        "status": "planned",
    },
    {
        "package": "survey-thermal",
        "repo": "crieck2010/survey-thermal",
        "version": "planned",
        "description": "Land surface temperature: Landsat thermal → LST, urban heat-island maps, heat-anomaly trends",
        "status": "planned",
    },
    {
        "package": "survey-3d",
        "repo": "crieck2010/survey-3d",
        "version": "planned",
        "description": "DSM/DEM differencing from stereo or LiDAR pairs: cut/fill volumes for construction and mining",
        "status": "planned",
    },
)

CONTRACTS_DOC = "https://github.com/crieck2010/earthwatch-suite/blob/main/docs/CONTRACTS.md"
