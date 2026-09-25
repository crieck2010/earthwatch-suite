# EarthWatch roadmap

Eleven builds in order. One ships, gets checked in, and only then does
the next start. All consume the [cross-suite contracts](CONTRACTS.md).

## Shipped

- ✅ `survey-imagery` v0.1.1 — satellite imagery engine (STAC discovery, cloud masking, spectral indices, per-pass site monitor)
- ✅ `survey-change` v0.1.0 — change-detection engine (per-pass and per-pixel change statistics)
- ✅ `survey-monitor` v0.1.0 — scheduled site-monitoring engine (site configs, threshold alerts, alert events)

## Build 4 — `survey-alerts`

Alert reports: email and PDF delivery of monitoring events. Consumes
`alerts.jsonl` per the stable schema in survey-monitor's
`docs/ALERT_SCHEMA.md`; implements `register_notifier` hooks (email via
SMTP, one-page PDF with before/after thumbnails and plain-English
summaries). This is the deliverable customers actually see.

## Build 5 — `survey-license`

License-key engine stubbed from day one: key check hook the monitor
calls before running, wired to a merchant of record (Gumroad/Lemon
Squeezy) later. Follows the suite's engine-first convention and the
stubbed-hook pattern (non-blocking, never raises).

## Build 6 — `survey-vegetation`

Vegetation/vigor analytics: NDVI/EVI/SAVI time series per field and per
pixel, growing-degree-day overlays, anomalies against historical
baseline. Customers: ag, orchards, land managers.

## Build 7 — `survey-flood`

Flood/water-extent mapping: NDWI-based water detection, per-pass water
mask, flood delta vs dry-season baseline. Customers: municipalities
(stormwater/MS4), environmental consultants.

## Build 8 — `survey-burn`

Burn-scar/disturbance mapping: NBR/dNBR fire severity plus generalized
disturbance detection (logging, clearing). Customers: forestry,
construction, land-use compliance.

## Build 9 — `survey-coast`

Coastal/shoreline change: NDWI shoreline extraction per pass,
transect-based erosion rates, storm-event deltas. Customers: coastal
municipalities, insurers, coastal engineers.

## Build 10 — `survey-thermal`

Land surface temperature: Landsat thermal band → LST, urban
heat-island maps, per-site heat-anomaly trends. Customers: municipal
planning, real estate, public health.

## Build 11 — `survey-3d`

DSM/DEM differencing: elevation-model differencing from stereo or LiDAR
pairs → cut/fill volumes. Customers: construction, mining — quantifies
earthwork.

## After the eleven

- **QGIS wiring** — expose survey-imagery and survey-change as QGIS
  Processing algorithms (the `survey-qgis` bridge on the surveying side
  consumes them).
- **Site-config schema + demo site pack** — a killer-demo site with
  visible change; doubles as the landing-page hero and the first
  product video.

## Product direction (post-graduation, May 2027)

Phase 1 — productize the monitor (per-site configs, scheduled runs,
email/PDF alert reports, license key; `.exe` + installer). Phase 2 —
killer demo on a real site. Phase 3 — audience (YouTube/X). Phase 4 —
launch (per-site subscription). Until graduation, the focus is purely
on building.
