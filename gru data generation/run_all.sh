#!/bin/bash
# Generate + verify gru annotations for all native nav datasets, sequentially.
# Designed to be launched detached (setsid nohup) so it survives SSH disconnect.
# Outputs + this log go to ./generated/ (git-ignored).

cd "$(dirname "$0")" || exit 1
export PYTHONUNBUFFERED=1

mkdir -p generated
LOG="generated/generate_all.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
section() { echo; echo "######## $(ts)  $* ########"; }

{
  echo "==================================================================="
  echo "  GRU data generation — full run started $(ts)"
  echo "  host=$(hostname) pid=$$"
  echo "==================================================================="

  rc_total=0
  # R2R is the anchor: regenerate a copy AND verify against the on-disk file.
  for ds in r2r rxr human; do
    section "GENERATE $ds"
    python3 add_gru_to_annotations.py --dataset "$ds"
    rc=$?; echo "[$(ts)] generate $ds exit=$rc"; [ $rc -ne 0 ] && rc_total=1

    section "VERIFY $ds"
    python3 verify_gru_annotations.py --dataset "$ds" \
        --generated "generated/${ds}_annotations_with_gru.json"
    rc=$?; echo "[$(ts)] verify $ds exit=$rc"; [ $rc -ne 0 ] && rc_total=1
  done

  section "FILE LISTING"
  ls -lh generated/*_annotations_with_gru.json

  echo
  echo "==================================================================="
  if [ $rc_total -eq 0 ]; then
    echo "  ALL DONE — every generate+verify step succeeded  $(ts)"
  else
    echo "  DONE WITH ERRORS — check the log above  $(ts)"
  fi
  echo "==================================================================="
} >> "$LOG" 2>&1
