# Cross-suite contracts

`earthwatch-suite` (remote sensing) and
[`survey-suite`](https://github.com/crieck2010/survey-suite) (terrestrial
surveying) are separate projects with separate release trains. They stay
compatible through a small set of **contracts**: file formats and schemas
that are stable within a major version and documented in one canonical
place. Neither suite may break a contract without a major-version bump
and a migration note.

## Contract 1: Site-config schema

Canonical definition:
[`survey-monitor/docs/SITE_CONFIG.md`](https://github.com/crieck2010/survey-monitor/blob/main/docs/SITE_CONFIG.md).

Key fields (summary — the linked doc is authoritative):

| Field | Meaning |
|---|---|
| `site_id` | Stable id; doubles as config filename and state directory |
| `geometry` | GeoJSON `Polygon` **or** `bbox: [min_lon, min_lat, max_lon, max_lat]` |
| `indices` | Watched spectral indices, e.g. `ndvi`, `ndwi`, `nbr` |
| `thresholds.<index>` | `direction` (`up`/`down`/`either`), `persistence` (consecutive breaching passes), `warning`/`critical` levels with `absolute_delta` and/or `z_score` |
| `baseline` | `rolling` (trailing window) or `fixed` (reference pass) |
| `metric` | Watched zonal statistic (`mean`, `median`, `std`, `p10`, `p90`, ...) |
| `max_cloud_cover` | Passes cloudier than this are skipped |
| `notifiers` | Delivery hooks registered via `register_notifier` |

All future engines that watch sites (vegetation, flood, burn, coast,
thermal, 3d) accept the same site-config shape: they may add optional
fields, but must accept everything above with identical semantics.

## Contract 2: Alert-event schema

Canonical definition:
[`survey-monitor/docs/ALERT_SCHEMA.md`](https://github.com/crieck2010/survey-monitor/blob/main/docs/ALERT_SCHEMA.md).

Alert events are JSONL — one object per line — written per site to
`alerts.jsonl`, with a QGIS-draggable `alerts.geojson` alongside.
Core fields: `alert_id` (deterministic: `site_id:pass_id:index:severity:seq`,
safe to deduplicate on), `timestamp` (UTC), `site_id`, `index`, `metric`,
`value`, `baseline_mean`, `baseline_std`, `delta`, `z_score`,
`threshold_kind`, `threshold`, `severity` (`info` | `warning` | `critical`),
`direction`, `pass_id`, `pass_date`.

Stability rules: fields are only ever **added**, never renamed or removed,
within a major version. `survey-alerts` pins to this schema and needs no
changes as long as the rule holds.

## Contract 3: GeoJSON as the interchange format

Footprints, alert overlays, and any per-pass geometry cross module
boundaries as [GeoJSON](https://geojson.org/). Coordinates are
`[lon, lat]` in WGS84 unless the schema says otherwise. Both suites
speak it: `survey-qgis` (surveying side) drag-loads the monitor's
`alerts.geojson` with no conversion.

## Contract 4: Shared engineering conventions

Both projects follow the same build rules, so engines stay portable
across the fork:

- **Engine-first** — pure-logic cores with zero UI-framework imports;
  thin CLI/UI layers on top.
- **Stdlib-only cores** — `numpy`, `rasterio`, and other heavy deps are
  lazy, optional imports inside the engines, never required by cores or
  meta-packages.
- **Semantic versioning** — changelogs per release; breaking changes get
  a major-version bump.
- **"The maths" sections** — every engine README explains its math at an
  accessible level.
- **MIT license** — all engine and meta repos.
- **Tested** — pytest suites with synthetic/fixture data; no network in
  tests.

## Versioning the contracts

If a contract must change incompatibly, the owning engine releases a new
**major version**, documents the migration in its CHANGELOG, and the
meta-repo registry (`earthwatch.MODULES`) records the new version.
Consumers pin to major versions (`survey-monitor>=0,<1` style bounds)
so a contract break never arrives silently.
