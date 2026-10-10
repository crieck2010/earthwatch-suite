#!/usr/bin/env bash
# Demo story: Black Rock Quarry (construction / mining).
#
# Runs fully offline on synthetic engine data:
#   1. validates this site config against the canonical schema
#   2. survey-3d demo  -> earthwork volumetrics (planted 18,750 m^3 cut / 12,300 m^3 fill)
#   3. survey-monitor   -> scheduled synthetic passes against THIS site.yaml (expects alerts)
#   4. survey-alerts    -> HTML site report from the monitor's alerts.jsonl
set -euo pipefail
# PYTHONHASHSEED=0 pins Python string hashing: survey-flood's synthetic
# data seeds from hash(zone_id), which otherwise varies per process.
export PYTHONHASHSEED=0
SITE_DIR="$(cd "$(dirname "$0")" && pwd)"
SITE_ID="black-rock-quarry"

need() { command -v "$1" >/dev/null 2>&1 || { echo "SKIP: $1 not installed"; exit 0; }; }
need survey-sites; need survey-3d; need survey-monitor; need survey-alerts

OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

echo "== [1/4] validate site config =="
survey-sites validate "$SITE_DIR/site.yaml"

echo "== [2/4] earthwork volumetrics (survey-3d, synthetic) =="
survey-3d demo > "$OUT/3d.log" 2>&1
grep -qF "cut=18732 m³" "$OUT/3d.log"  || { echo "FAIL: expected cut volume"; exit 1; }
grep -qF "fill=11872 m³" "$OUT/3d.log" || { echo "FAIL: expected fill volume"; exit 1; }
echo "story OK: net cut ~18,732 m3, net fill ~11,872 m3 (planted 18,750 / 12,300)"

echo "== [3/4] scheduled monitoring (survey-monitor, synthetic passes) =="
survey-monitor add-site --config "$SITE_DIR/site.yaml" \
  --config-dir "$OUT/sites" --store-dir "$OUT/state" >/dev/null
survey-monitor run --site "$SITE_ID" --config-dir "$OUT/sites" \
  --store-dir "$OUT/state" --synthetic > "$OUT/mon.log" 2>&1
grep -qE "[1-9][0-9]* alert\(s\)" "$OUT/mon.log" \
  || { echo "FAIL: expected monitor alerts"; cat "$OUT/mon.log"; exit 1; }
echo "story OK: monitor fired alerts on synthetic passes: $(grep -oE '[0-9]+ alert\(s\)' "$OUT/mon.log" | head -1)"

echo "== [4/4] alert report (survey-alerts) =="
survey-alerts render --alerts "$OUT/state/$SITE_ID/alerts.jsonl" \
  --format html --min-severity warning --out "$OUT/report.html"
test -s "$OUT/report.html" || { echo "FAIL: empty report"; exit 1; }
echo "story OK: HTML site report rendered ($(wc -c < "$OUT/report.html") bytes)"

echo "DEMO PASS: quarry"
