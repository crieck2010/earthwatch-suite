# earthwatch-suite

The remote-sensing half of the geospatial suite: pure-logic Python engines
that turn **free satellite imagery** (Sentinel-2, Landsat, via STAC) into
**marketable, per-pass monitoring products** — site watches, alert events,
and vertical analytics.

The terrestrial surveying half (COGO, levels, least-squares, geodesy, GNSS,
point clouds) lives in the sibling project
**[crieck2010/survey-suite](https://github.com/crieck2010/survey-suite)**.
The two projects are developed separately but stay compatible through a
small set of documented [cross-suite contracts](docs/CONTRACTS.md).

## The engines

| Package | Repo | Version | What it does | Status |
|---|---|---|---|---|
| `survey-imagery` | [crieck2010/survey-imagery](https://github.com/crieck2010/survey-imagery) | 0.1.1 | Satellite imagery engine: STAC discovery, cloud masking, spectral indices, per-pass site monitor | ✅ shipped |
| `survey-change` | [crieck2010/survey-change](https://github.com/crieck2010/survey-change) | 0.1.0 | Change-detection engine: per-pass and per-pixel change statistics over imagery time series | ✅ shipped |
| `survey-monitor` | [crieck2010/survey-monitor](https://github.com/crieck2010/survey-monitor) | 0.1.0 | Scheduled site-monitoring engine: site configs, threshold alert evaluation, alert events | ✅ shipped |
| `survey-alerts` | [crieck2010/survey-alerts](https://github.com/crieck2010/survey-alerts) | 0.1.0 | Alert reports: email + PDF delivery of monitoring events (consumes survey-monitor alerts.jsonl) | ✅ shipped |
| `survey-license` | [crieck2010/survey-license](https://github.com/crieck2010/survey-license) | 0.1.0 | License-key engine: Ed25519-signed keys, offline verification, plan gating for the monitoring product | ✅ shipped |
| `survey-vegetation` | [crieck2010/survey-vegetation](https://github.com/crieck2010/survey-vegetation) | 0.1.0 | Vegetation/vigor analytics: NDVI/EVI/SAVI time series, phenology-aware anomaly detection | ✅ shipped |
| `survey-flood` | [crieck2010/survey-flood](https://github.com/crieck2010/survey-flood) | 0.1.0 | Flood/water-extent mapping: NDWI water masks, per-pass flood deltas vs dry-season baseline | ✅ shipped |
| `survey-burn` | [crieck2010/survey-burn](https://github.com/crieck2010/survey-burn) | 0.1.0 | Burn-scar/disturbance mapping: NBR/dNBR severity, USGS bands, clearing detection | ✅ shipped |
| `survey-coast` | [crieck2010/survey-coast](https://github.com/crieck2010/survey-coast) | 0.1.0 | Coastal change: NDWI shoreline extraction per pass, transect erosion rates, storm deltas | ✅ shipped |
| `survey-thermal` | [crieck2010/survey-thermal](https://github.com/crieck2010/survey-thermal) | 0.1.0 | Land surface temperature: Landsat thermal to LST, urban heat-island maps, heatwave detection | ✅ shipped |
| `survey-3d` | [crieck2010/survey-3d](https://github.com/crieck2010/survey-3d) | 0.1.0 | DSM/DEM differencing: cut/fill volumetrics, DoD maps, coregistration vs stable reference zones | ✅ shipped |
| `survey-sites` | [crieck2010/survey-sites](https://github.com/crieck2010/survey-sites) | 0.1.0 | Canonical site-config schema + demo site pack: one `site.yaml` drives monitor, alerts, and all analytics engines | ✅ shipped |
| `survey-qgis` | [crieck2010/survey-qgis](https://github.com/crieck2010/survey-qgis) | 0.2.0 | Shared cartography bridge (lives in survey-suite): QGIS Processing algorithms for every earthwatch engine | ✅ shipped |

Each engine is pure Python, independently tested, and usable on its own.
This repo adds the registry (`earthwatch.MODULES`), the
[cross-suite contracts](docs/CONTRACTS.md), and the
[roadmap](docs/ROADMAP.md).

## How the pieces fit

```
┌────────────────┐   scenes     ┌────────────────┐  stats     ┌────────────────┐
│ survey-imagery │ ──────────── │ survey-change  │ ────────── │ survey-monitor │
│ STAC discovery │              │ change metrics │            │ site configs   │
│ cloud masking  │              │ per-pass /     │            │ threshold eval │
│ spectral idx   │              │ per-pixel      │            │ alert events   │
└────────────────┘              └────────────────┘            └───────┬────────┘
                                                                    │ alerts.jsonl
                                                                    │ (stable contract)
                                                        ┌───────────┴────────┐
                                                        │  survey-alerts     │  ✅ shipped
                                                        │  email + PDF       │
                                                        └────────────────────┘

        vertical analytics engines (all ✅ shipped): vegetation · flood · burn
        coast · thermal · 3d — all consume imagery/change outputs,
        all emit the same alert-event shape.

Site-config schema + demo pack: `survey-sites` v0.1.0 (canonical `site.yaml`,
four demo sites, `survey-sites` CLI). QGIS bridge: `survey-qgis` v0.2.0.
```

**GeoJSON in, alerts out.** Every engine takes footprints as GeoJSON
polygons and reports change through the stable alert-event schema
(`survey-monitor`'s `docs/ALERT_SCHEMA.md` is canonical).

## Quickstart

```bash
# Pull the shipped engines from their public repos
pip install -e .

# Watch a site with the monitor's seeded synthetic provider (no network)
survey-monitor add-site --template --config-dir ./sites
survey-monitor run --synthetic --config-dir ./sites
survey-monitor alerts --config-dir ./sites
```

Real satellite acquisition goes through `survey-imagery`'s STAC pipeline
(Copernicus Data Space / Microsoft Planetary Computer); see its README
for a live Sentinel-2 example.

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.9. Engine cores are stdlib-only; heavy deps
(`numpy`, `rasterio`) are lazy optionals inside the engines, never
required by this meta-package.

## The maths (orientation)

Every monitoring decision in the suite reduces to the same question:
**is this pass's metric unusual relative to the site's baseline?**

1. **Per-pass metric** — for a site footprint and a spectral index
   (e.g. NDVI), each scene yields one number: the zonal statistic
   (`mean` by default, `median`/`std`/`p10`/`p90` also available).
2. **Baseline** — `rolling`: trailing-N-pass mean and standard deviation
   (adapts to seasonal drift); `fixed`: anchored to a reference pass
   (auditable, reproducible).
3. **Two rules** — alert when `|value − baseline_mean| ≥ absolute_delta`
   **or** when `|z| ≥ z_score` where `z = (value − mean) / std`.
4. **Severity** — breaching a `critical` level is critical; breaching
   `warning` (and surviving the `persistence` gate of consecutive
   breaching passes) is a warning; sitting at ≥ 50% of the warning
   threshold is an `info` watch event.
5. **Noise defences** — cloud-cover filtering per pass, `persistence`
   against single-pass noise, `cooldown_passes` against repeat alerts on
   a standing condition.

The per-engine READMEs each carry a deeper "maths" section for their own
metrics.

## Honest limitations

- Free imagery means 5–16 day revisit and cloud gaps; thin baselines
  make z-scores noisy — this is stated in each engine's README.
- The monitor works site-wide per pass; per-pixel hotspots are
  `survey-change`'s job (the two are designed to be paired).
- No delivery yet: alerts are machine-readable events; `survey-alerts`
  (email/PDF) is next on the [roadmap](docs/ROADMAP.md).

## Shared conventions

Both `earthwatch-suite` and `survey-suite` follow the same rules —
see [docs/CONTRACTS.md](docs/CONTRACTS.md):

- **Engine-first** — pure-logic cores, zero UI imports; thin CLI/UI layers.
- **Stdlib-only cores** — heavy deps lazy and optional.
- **Semantic versioning** with changelogs; MIT licensed.
- **"The maths" README sections** — every engine explains its math.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) — 11 builds in order, from
`survey-alerts` through the vertical analytics engines to the demo
site pack.

## License

MIT — see [LICENSE](LICENSE).
