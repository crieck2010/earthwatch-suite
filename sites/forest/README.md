# Pine Ridge Forest

**Story:** forestry burn-scar and disturbance monitoring.

A managed forest. Stand 1 is the burn/disturbance watch area; Stand 2 is
an untouched control stand for comparison.

## What this site exercises

| Engine | What runs | Storyline assertion |
|---|---|---|
| `survey-sites` | `validate site.yaml` | config is schema-v1 valid |
| `survey-burn` | `demo` (synthetic) | planted 12.57 ha burn recovered exactly, per severity class |
| `survey-monitor` | `run --synthetic` on **this** `site.yaml` | alerts fire on NBR/NDVI thresholds |
| `survey-alerts` | `render` the monitor's `alerts.jsonl` | HTML site report produced |

The `engines.burn` section pins the USGS severity bands and the minimum
mapped burn area.

## Files

- `site.yaml` — canonical config (source of truth)
- `footprint.geojson` — site footprint (drag into QGIS)
- `zones.geojson` — the two forest stands
- `demo.sh` — runs the full story offline; exits non-zero on any failed assertion

## Run it

```bash
./demo.sh
```
