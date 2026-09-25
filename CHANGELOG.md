# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- All eleven builds are shipped: the `earthwatch` registry (`MODULES`
  table) marks every engine shipped with its released version
  (survey-imagery 0.1.1, survey-change 0.1.0, survey-monitor 0.1.0,
  survey-alerts 0.1.0, survey-license 0.1.0, survey-vegetation 0.1.0,
  survey-flood 0.1.0, survey-burn 0.1.0, survey-coast 0.1.0,
  survey-thermal 0.1.0, survey-3d 0.1.0), plus the cartography bridge
  (survey-qgis 0.2.0) and the canonical site-config schema + demo site
  pack (survey-sites 0.1.0).
- `docs/ROADMAP.md` rewritten: all eleven builds marked complete, with
  the shipped QGIS bridge and site-config schema/demo pack.
- README registry table and suite diagram updated to the full shipped set.
- `requirements.txt` pins all shipped engine repos.

## [0.1.0] - 2026-09-24

### Added
- Initial release: the EarthWatch remote-sensing project is forked out of
  the surveying + remote sensing work into its own meta repo.
- `earthwatch` registry module (`MODULES` table: 3 shipped engines,
  8 planned).
- `docs/CONTRACTS.md`: cross-suite contracts (site-config schema,
  alert-event schema, GeoJSON interchange, shared engineering conventions).
- `docs/ROADMAP.md`: the 11-build roadmap (survey-alerts → survey-3d,
  then QGIS wiring and the demo site pack).
- README with suite-fit diagram, quickstart, "the maths" orientation,
  and honest limitations.
