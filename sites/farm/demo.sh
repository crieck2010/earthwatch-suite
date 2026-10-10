#!/usr/bin/env bash
# Demo story: Green Acre Farm (agriculture).
#
# Runs fully offline on synthetic engine data:
#   1. validates this site config against the canonical schema
#   2. survey-vegetation demo -> vigor time series; expects stressed zones flagged
#   3. survey-thermal demo    -> UHI intensity (planted +4.5 C) and a heatwave event
set -euo pipefail
# PYTHONHASHSEED=0 pins Python string hashing: survey-flood's synthetic
# data seeds from hash(zone_id), which otherwise varies per process.
export PYTHONHASHSEED=0
SITE_DIR="$(cd "$(dirname "$0")" && pwd)"

need() { command -v "$1" >/dev/null 2>&1 || { echo "SKIP: $1 not installed"; exit 0; }; }
need survey-sites; need survey-vegetation; need survey-thermal

OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

echo "== [1/3] validate site config =="
survey-sites validate "$SITE_DIR/site.yaml"

echo "== [2/3] vigor analytics (survey-vegetation, synthetic) =="
survey-vegetation demo --out-dir "$OUT/veg" > "$OUT/veg.log" 2>&1
grep -qF "19 stressed" "$OUT/veg.log" \
  || { echo "FAIL: expected stressed zones"; cat "$OUT/veg.log"; exit 1; }
echo "story OK: vegetation stress flagged ($(grep -oF '19 stressed' "$OUT/veg.log" | head -1) records)"

echo "== [3/3] heat analytics (survey-thermal, synthetic) =="
survey-thermal demo --out-dir "$OUT/th" > "$OUT/th.log" 2>&1
grep -qF "mean downtown UHI: +4.63 °C" "$OUT/th.log" \
  || { echo "FAIL: expected UHI value"; cat "$OUT/th.log"; exit 1; }
grep -qF "heatwave events : 1" "$OUT/th.log" \
  || { echo "FAIL: expected one heatwave"; exit 1; }
echo "story OK: UHI +4.63 C recovered (planted +4.5 C); 1 heatwave detected"

echo "DEMO PASS: farm"
