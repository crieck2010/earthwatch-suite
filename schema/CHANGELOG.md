# Schema changelog

The canonical site-config schema lives in `site-config.schema.json`.
Its version is carried in the schema's top-level `version` field and in
its `$id` (`.../site-config/v1`, `/v2`, ...).

## v1 — 1.0.0 (2026-09-25)

Initial release. Strict superset of the survey-monitor site config
(`docs/SITE_CONFIG.md` in `crieck2010/survey-monitor` v0.1.0): every field
survey-monitor reads (`site_id`, `name`, `geometry`, `indices`,
`thresholds`, `baseline`, `collection`, `api_url`, `max_cloud_cover`,
`metric`, `cooldown_passes`, `notifiers`) keeps identical semantics, and
survey-monitor's `from_dict` ignores the new sections.

New sections owned by this schema:

- `crs`, `description`
- `zones[]` — named sub-polygons with a `kind` role enum
- `transects[]` — landward→seaward shoreline transects
- `engines.{vegetation,flood,burn,coast,thermal,dem3d}` — per-engine parameter overrides
- `alerts` — delivery routing for survey-alerts (`recipients`, `severity_min`, `digest`)
- `license` — plan gating reference for survey-license (`plan`, `key_id`, `max_sites`)
