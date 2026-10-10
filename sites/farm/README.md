# Green Acre Farm

**Story:** agricultural vigor and heat monitoring.

A row-crop farm with three management zones. Field A (corn) develops a
stress patch mid-season; Field B is orchard; Field C is pasture and
doubles as the rural reference for heat-island analysis.

## What this site exercises

| Engine | What runs | Storyline assertion |
|---|---|---|
| `survey-sites` | `validate site.yaml` | config is schema-v1 valid |
| `survey-vegetation` | `demo` (synthetic) | 19 stressed records flagged by phenology-aware anomalies |
| `survey-thermal` | `demo` (synthetic) | UHI +4.63 °C recovered (planted +4.5 °C); 1 heatwave detected |

The `engines.vegetation` section pins the vigor indices and anomaly
z-threshold; `engines.thermal` names Field C as the UHI reference zone.

## Files

- `site.yaml` — canonical config (source of truth)
- `footprint.geojson` — site footprint (drag into QGIS)
- `zones.geojson` — the three management zones
- `demo.sh` — runs the full story offline; exits non-zero on any failed assertion

## Run it

```bash
./demo.sh
```
