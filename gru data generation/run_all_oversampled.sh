#!/bin/bash
# Generate + verify gru for the OVERSAMPLED nav annotations (what training loads),
# sequentially. The trajectory map is built from the union of plain + oversampled so
# exclusive prefixes stay complete. Designed to be launched detached (setsid nohup)
# so it survives SSH disconnect. Outputs + this log go to ./generated/ (git-ignored).

cd "$(dirname "$0")" || exit 1
export PYTHONUNBUFFERED=1
PY=/opt/conda-envs/navila-qwen/bin/python

mkdir -p generated
LOG="generated/generate_oversampled.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
section() { echo; echo "######## $(ts)  $* ########"; }

{
  echo "==================================================================="
  echo "  GRU OVERSAMPLED generation — full run started $(ts)"
  echo "  host=$(hostname) pid=$$  python=$PY"
  echo "==================================================================="

  rc_total=0
  for ds in r2r rxr human; do
    section "GENERATE $ds (oversampled)"
    $PY add_gru_to_annotations.py --dataset "$ds" --oversampled
    rc=$?; echo "[$(ts)] generate $ds exit=$rc"; [ $rc -ne 0 ] && rc_total=1

    section "VERIFY $ds (oversampled)"
    $PY verify_gru_annotations.py --dataset "$ds" --oversampled \
        --generated "generated/${ds}_oversampled_with_gru.json"
    rc=$?; echo "[$(ts)] verify $ds exit=$rc"; [ $rc -ne 0 ] && rc_total=1
  done

  section "FILE LISTING"
  ls -lh generated/*_oversampled_with_gru.json

  echo
  echo "==================================================================="
  if [ $rc_total -eq 0 ]; then
    echo "  ALL DONE — every generate+verify step succeeded  $(ts)"
  else
    echo "  DONE WITH ERRORS — check the log above  $(ts)"
  fi
  echo "==================================================================="
} >> "$LOG" 2>&1
