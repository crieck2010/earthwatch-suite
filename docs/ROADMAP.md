# EarthWatch roadmap

All eleven builds shipped. One shipped, got checked in, and only then
did the next start. All consume the [cross-suite contracts](CONTRACTS.md).

## Shipped — all eleven builds

- ✅ `survey-imagery` v0.1.1 — satellite imagery engine (STAC discovery, cloud masking, spectral indices, per-pass site monitor)
- ✅ `survey-change` v0.1.0 — change-detection engine (per-pass and per-pixel change statistics)
- ✅ `survey-monitor` v0.1.0 — scheduled site-monitoring engine (site configs, threshold alerts, alert events)
- ✅ `survey-alerts` v0.1.0 — alert reports: email and PDF delivery of monitoring events (49 tests)
- ✅ `survey-license` v0.1.0 — license-key engine: Ed25519-signed keys, offline verification, plan gating (40 tests)
- ✅ `survey-vegetation` v0.1.0 — vegetation/vigor analytics: NDVI/EVI/SAVI time series, phenology-aware anomalies (52 tests)
- ✅ `survey-flood` v0.1.0 — flood/water-extent mapping: NDWI water masks, flood deltas vs dry-season baseline (65 tests)
- ✅ `survey-burn` v0.1.0 — burn-scar/disturbance mapping: NBR/dNBR severity, USGS bands (67 tests)
- ✅ `survey-coast` v0.1.0 — coastal change: NDWI shoreline extraction, transect erosion rates, storm deltas (71 tests)
- ✅ `survey-thermal` v0.1.0 — land surface temperature: Landsat thermal → LST, heat-island maps, heatwave detection (57 tests)
- ✅ `survey-3d` v0.1.0 — DSM/DEM differencing: cut/fill volumetrics, DoD maps, coregistration (50 tests)

## Also shipped

- ✅ `survey-qgis` v0.2.0 — the shared cartography bridge (lives in
  survey-suite): QGIS Processing algorithms for every earthwatch engine
  (25 algorithms, 80 tests)
- ✅ `survey-sites` v0.1.0 — canonical site-config schema and demo site
  pack: one `site.yaml` drives monitor, alerts, and all analytics
  engines (four demo sites with offline `demo.sh` stories, 49 tests)

## Product direction (post-graduation, May 2027)

Phase 1 — productize the monitor (per-site configs, scheduled runs,
email/PDF alert reports, license key; `.exe` + installer). Phase 2 —
killer demo on a real site. Phase 3 — audience (YouTube/X). Phase 4 —
launch (per-site subscription). Until graduation, the focus is purely
on building.
