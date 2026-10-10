#!/usr/bin/env bash
# Demo story: Pine Ridge Forest (forestry / disturbance).
#
# Runs fully offline on synthetic engine data:
#   1. validates this site config against the canonical schema
#   2. survey-burn demo -> dNBR severity; expects the planted 12.57 ha burn
#      recovered exactly, per severity class
#   3. survey-monitor   -> scheduled synthetic passes against THIS site.yaml
#   4. survey-alerts    -> HTML site report from the monitor's alerts.jsonl
set -euo pipefail
# PYTHONHASHSEED=0 pins Python string hashing: survey-flood's synthetic
# data seeds from hash(zone_id), which otherwise varies per process.
export PYTHONHASHSEED=0
SITE_DIR="$(cd "$(dirname "$0")" && pwd)"
SITE_ID="pine-ridge-forest"

need() { command -v "$1" >/dev/null 2>&1 || { echo "SKIP: $1 not installed"; exit 0; }; }
need survey-sites; need survey-burn; need survey-monitor; need survey-alerts

OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

echo "== [1/4] validate site config =="
survey-sites validate "$SITE_DIR/site.yaml"

echo "== [2/4] burn severity (survey-burn, synthetic) =="
survey-burn demo --out-dir "$OUT/burn" > "$OUT/burn.log" 2>&1
grep -qF "burned_ha                  planted=  12.57 ha  measured=  12.57 ha" "$OUT/burn.log" \
  || { echo "FAIL: expected exact burn recovery"; cat "$OUT/burn.log"; exit 1; }
echo "story OK: planted 12.57 ha burn recovered exactly, per severity class"

echo "== [3/4] scheduled monitoring (survey-monitor, synthetic passes) =="
survey-monitor add-site --config "$SITE_DIR/site.yaml" \
  --config-dir "$OUT/sites" --store-dir "$OUT/state" >/dev/null
survey-monitor run --site "$SITE_ID" --config-dir "$OUT/sites" \
  --store-dir "$OUT/state" --synthetic > "$OUT/mon.log" 2>&1
grep -qE "[1-9][0-9]* alert\(s\)" "$OUT/mon.log" \
  || { echo "FAIL: expected monitor alerts"; cat "$OUT/mon.log"; exit 1; }
echo "story OK: monitor fired alerts on synthetic passes"

echo "== [4/4] alert report (survey-alerts) =="
survey-alerts render --alerts "$OUT/state/$SITE_ID/alerts.jsonl" \
  --format html --min-severity warning --out "$OUT/report.html"
test -s "$OUT/report.html" || { echo "FAIL: empty report"; exit 1; }
echo "story OK: HTML site report rendered"

echo "DEMO PASS: forest"
