#!/usr/bin/env bash
# Interactive ArUco table calibration helper for :8092
# Prefer: python3 /home/er/mycobot_humble/test_runs/aruco_calibrate.py
set -euo pipefail
BASE="${1:-http://127.0.0.1:8092}"

post() {
  local path="$1"
  local body="${2:-{}}"
  curl -sS -X POST "$BASE$path" -H 'Content-Type: application/json' -d "$body"
}

echo "ArUco calibration helper → $BASE"
echo "Print marker: $BASE/aruco  (DICT_4X4_50 id 0)"
echo "Tape it flat. Put OPEN pad centre on marker centre at z≈64 mm."
echo "Clear coloured caps off the table."
echo
echo "Or run the guided mover:"
echo "  PYTHONUNBUFFERED=1 python3 /home/er/mycobot_humble/test_runs/aruco_calibrate.py"
echo

post /api/calibrate/clearances '{"table_z_mm":64,"hover_mm":40}' >/dev/null || true

for label in "fit 1/5" "fit 2/5" "fit 3/5" "fit 4/5" "fit 5/5" "CHECK spot"; do
  read -r -p "Ready for ${label}? Pads on ArUco centre, then Enter (or skip): " ans
  if [[ "${ans}" == "skip" ]]; then
    echo "Skipped ${label}"
    continue
  fi
  post /api/calibrate/hold '{}' >/dev/null
  sleep 0.3
  resp=$(post /api/calibrate/aruco '{}')
  echo "$resp" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("message")); w=d.get("wizard") or {}; print("pairs", len(w.get("pairs") or []), "check", w.get("check") is not None); raise SystemExit(0 if d.get("success") else 1)' \
    || { echo "FAILED ${label}: marker must be fully in view (orange cross on /overview)."; exit 1; }
done

echo
echo "Capture DROPS (shared + R/W/B). Keep X under 275 mm."
read -r -p "Open gripper above SHARED drop, Enter: " _
post /api/calibrate/hold '{}' >/dev/null
post /api/calibrate/drop '{}' | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("message"), d.get("wizard",{}).get("drop_x_mm"), d.get("wizard",{}).get("drop_y_mm"))'

for colour in red_cap white_cap black_cap; do
  read -r -p "Above ${colour} sticky (X<=275), Enter: " _
  post /api/calibrate/hold '{}' >/dev/null
  post /api/calibrate/drop "{\"class_name\":\"${colour}\"}" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("message"))'
done

echo
echo "Accepting if residuals OK..."
post /api/calibrate/hold '{}' >/dev/null
post /api/calibrate/accept '{}' | python3 -m json.tool | head -80
echo "Done. Review http://127.0.0.1:8092/calibrate"
