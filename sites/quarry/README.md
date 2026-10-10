# Black Rock Quarry

**Story:** construction / mining earthwork monitoring.

An active limestone quarry. The north pit is being cut deeper while the
south stockpile grows; a paved haul road crosses the site and never moves,
so it serves as the stable reference for DEM coregistration checks.

## What this site exercises

| Engine | What runs | Storyline assertion |
|---|---|---|
| `survey-sites` | `validate site.yaml` | config is schema-v1 valid |
| `survey-3d` | `demo` (synthetic) | net cut ≈ 18,732 m³, net fill ≈ 11,872 m³ (planted 18,750 / 12,300) |
| `survey-monitor` | `run --synthetic` on **this** `site.yaml` | alerts fire on NDVI/NDWI/NBR thresholds |
| `survey-alerts` | `render` the monitor's `alerts.jsonl` | HTML site report produced |

The `site.yaml` doubles as a survey-monitor config: the monitor reads the
shared fields (`indices`, `thresholds`, `baseline`, ...) and ignores the
`zones` / `engines` / `alerts` / `license` sections.

## Files

- `site.yaml` — canonical config (source of truth)
- `footprint.geojson` — site footprint (drag into QGIS)
- `zones.geojson` — pit, stockpile, and stable-reference zones
- `demo.sh` — runs the full story offline; exits non-zero on any failed assertion

## Run it

```bash
./demo.sh
```
