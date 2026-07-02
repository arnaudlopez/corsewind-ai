#!/usr/bin/env bash
set -euo pipefail

ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
REMOTE_ROOT="${REMOTE_ROOT:-/srv/data/corsewind/backfill_runner}"
PY="${PY:-/home/z2/corsewind-ml-smoke/.venv/bin/python}"
PREFIX="${PREFIX:-residual_windsup_sst_prev_phys_v3_dem_fetch}"
START_MONTH="${START_MONTH:-2024-01}"
END_MONTH="${END_MONTH:-2026-06}"
STATIC_FEATURES="${STATIC_FEATURES:-configs/ml_spot_static_features.fetch_v1.json}"
LOG_ROOT="${LOG_ROOT:-$ML_ROOT/run_logs}"
LOG="${LOG:-$LOG_ROOT/rebuild_phys_v3_dem_fetch_2024_2026.log}"
STATUS="${STATUS:-$LOG_ROOT/rebuild_phys_v3_dem_fetch_2024_2026.status}"
PID="${PID:-$LOG_ROOT/rebuild_phys_v3_dem_fetch_2024_2026.pid}"
REQUIRE_PHYS_V1_DONE="${REQUIRE_PHYS_V1_DONE:-1}"
PHYS_V1_DECISION_STATUS="${PHYS_V1_DECISION_STATUS:-$LOG_ROOT/phys_v1_decision_report_watcher.status}"

mkdir -p "$LOG_ROOT"
cd "$REMOTE_ROOT"

if [ "$REQUIRE_PHYS_V1_DONE" = "1" ]; then
  if ! grep -q '^complete ' "$PHYS_V1_DECISION_STATUS" 2>/dev/null; then
    echo "phys_v1 decision report is not complete yet: $PHYS_V1_DECISION_STATUS" >&2
    exit 70
  fi
fi

if [ ! -s "$STATIC_FEATURES" ]; then
  echo "missing static DEM+fetch features: $STATIC_FEATURES" >&2
  exit 71
fi

rm -f "$STATUS" "$PID"
(
  set -o pipefail
  echo "started $(date -Is)" > "$STATUS"
  ML_DATASET_ROOT="$ML_ROOT" "$PY" scripts/ml_dataset/run_monthly_training_shards.py \
    --ml-root "$ML_ROOT" \
    --registry configs/ml_spots.json \
    --context-registry configs/ml_context_stations.json \
    --spot-static-features "$STATIC_FEATURES" \
    --start-month "$START_MONTH" \
    --end-month "$END_MONTH" \
    --run-id-prefix "$PREFIX" \
    --collect-open-meteo \
    --collect-open-meteo-offsets \
    --cleanup-jsonl-after-parquet \
    --continue-on-error >> "$LOG" 2>&1
  code=$?
  echo "finished $(date -Is) code=$code" >> "$STATUS"
  exit "$code"
) &
echo $! > "$PID"
echo "pid=$(cat "$PID") log=$LOG status=$STATUS prefix=$PREFIX"
