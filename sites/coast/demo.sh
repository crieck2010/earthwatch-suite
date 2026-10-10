#!/usr/bin/env bash
# Demo story: Harbor Cove (coastal).
#
# Runs fully offline on synthetic engine data:
#   1. validates this site config against the canonical schema
#   2. survey-coast demo -> shoreline transects; expects planted erosion/accretion
#      rates recovered and the storm bite detected
#   3. survey-flood demo -> water masks; expects the planted flood detected
set -euo pipefail
# PYTHONHASHSEED=0 pins Python string hashing: survey-flood's synthetic
# data seeds from hash(zone_id), which otherwise varies per process.
export PYTHONHASHSEED=0
SITE_DIR="$(cd "$(dirname "$0")" && pwd)"

need() { command -v "$1" >/dev/null 2>&1 || { echo "SKIP: $1 not installed"; exit 0; }; }
need survey-sites; need survey-coast; need survey-flood

OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

echo "== [1/3] validate site config =="
survey-sites validate "$SITE_DIR/site.yaml"

echo "== [2/3] shoreline change (survey-coast, synthetic) =="
survey-coast demo --out-dir "$OUT/coast" > "$OUT/coast.log" 2>&1
grep -qE "recovered=-2\.43.*0\.73" "$OUT/coast.log" \
  || { echo "FAIL: expected erosion rate"; cat "$OUT/coast.log"; exit 1; }
grep -qF "displacement=-10.0 m" "$OUT/coast.log" \
  || { echo "FAIL: expected storm displacement"; exit 1; }
echo "story OK: erosion -2.0 m/yr planted -> -2.43 recovered; storm bite -10.0 m detected"

echo "== [3/3] flood mapping (survey-flood, synthetic) =="
survey-flood demo --out-dir "$OUT/flood" > "$OUT/flood.log" 2>&1
grep -qF "planted flood detected: +6.0 ha" "$OUT/flood.log" \
  || { echo "FAIL: expected planted flood"; cat "$OUT/flood.log"; exit 1; }
grep -qF "11 flooded" "$OUT/flood.log" \
  || { echo "FAIL: expected flooded passes"; exit 1; }
echo "story OK: planted flood +6.0 ha detected; 11 flooded passes flagged"

echo "DEMO PASS: coast"
