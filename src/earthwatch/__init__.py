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
        "version": "0.1.0",
        "description": "Alert reports: email and PDF delivery of monitoring events (per survey-monitor's notifier hook)",
        "status": "shipped",
    },
    {
        "package": "survey-license",
        "repo": "crieck2010/survey-license",
        "version": "0.1.0",
        "description": "License-key engine for gating the monitoring product to paying customers",
        "status": "shipped",
    },
    {
        "package": "survey-vegetation",
        "repo": "crieck2010/survey-vegetation",
        "version": "0.1.0",
        "description": "Vegetation/vigor analytics: NDVI/EVI/SAVI time series, anomalies vs historical baseline",
        "status": "shipped",
    },
    {
        "package": "survey-flood",
        "repo": "crieck2010/survey-flood",
        "version": "0.1.0",
        "description": "Flood/water-extent mapping: NDWI water masks, per-pass flood deltas vs dry-season baseline",
        "status": "shipped",
    },
    {
        "package": "survey-burn",
        "repo": "crieck2010/survey-burn",
        "version": "0.1.0",
        "description": "Burn-scar/disturbance mapping: NBR/dNBR fire severity, generalized clearing detection",
        "status": "shipped",
    },
    {
        "package": "survey-coast",
        "repo": "crieck2010/survey-coast",
        "version": "0.1.0",
        "description": "Coastal change: NDWI shoreline extraction per pass, transect erosion rates, storm deltas",
        "status": "shipped",
    },
    {
        "package": "survey-thermal",
        "repo": "crieck2010/survey-thermal",
        "version": "0.1.0",
        "description": "Land surface temperature: Landsat thermal → LST, urban heat-island maps, heat-anomaly trends",
        "status": "shipped",
    },
    {
        "package": "survey-3d",
        "repo": "crieck2010/survey-3d",
        "version": "0.1.0",
        "description": "DSM/DEM differencing from stereo or LiDAR pairs: cut/fill volumes for construction and mining",
        "status": "shipped",
    },
    {
        "package": "survey-sites",
        "repo": "crieck2010/survey-sites",
        "version": "0.1.0",
        "description": "Canonical site-config schema and demo site pack: one site.yaml drives monitor, alerts, and all analytics engines",
        "status": "shipped",
    },
    {
        "package": "survey-qgis",
        "repo": "crieck2010/survey-qgis",
        "version": "0.2.0",
        "description": "Shared cartography bridge (lives in survey-suite): QGIS Processing algorithms for every earthwatch engine",
        "status": "shipped",
    },
)

CONTRACTS_DOC = "https://github.com/crieck2010/earthwatch-suite/blob/main/docs/CONTRACTS.md"
