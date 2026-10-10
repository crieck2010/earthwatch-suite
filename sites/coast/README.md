# Harbor Cove

**Story:** coastal erosion and flood monitoring.

A coastal site with an eroding west beach, an accreting east beach, and a
tidal marsh that floods seasonally. Two shoreline transects run
landward → seaward, perpendicular to the beach.

## What this site exercises

| Engine | What runs | Storyline assertion |
|---|---|---|
| `survey-sites` | `validate site.yaml` | config is schema-v1 valid |
| `survey-coast` | `demo` (synthetic) | erosion −2.0 m/yr planted → −2.43 recovered; storm bite −10.0 m detected |
| `survey-flood` | `demo` (synthetic) | planted flood +6.0 ha detected; 11 flooded passes flagged |

The `transects` array is the survey-coast contract: `transect_ids` in
`engines.coast` select which transects to analyze. `engines.flood` pins
the dry-season baseline months.

## Files

- `site.yaml` — canonical config (source of truth)
- `footprint.geojson` — site footprint (drag into QGIS)
- `zones.geojson` — marsh and beach zones
- `transects.geojson` — the two shoreline transects
- `demo.sh` — runs the full story offline; exits non-zero on any failed assertion

## Run it

```bash
./demo.sh
```
