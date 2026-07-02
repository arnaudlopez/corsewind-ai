#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/srv/data/corsewind/backfill_runner}"
ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
PYTHON="${PYTHON:-/home/z2/corsewind-ml-smoke/.venv/bin/python}"
LOG_ROOT="$ML_ROOT/backfill_logs"
RUN_ID="${RUN_ID:-residual_backfill_2025h2_2026_short_hpa_partial_signal_v1}"
BENCHMARK_ID="${BENCHMARK_ID:-tabular_lgbm_150k_short_hpa_partial_signal_2025h2_to_2026_v1}"
STATUS="$LOG_ROOT/partial_hpa_early_signal.status"
LOG="$LOG_ROOT/partial_hpa_early_signal.log"
AUDIT_JSON="$ML_ROOT/source_inventories/open_meteo_pressure_level_partial_signal_pre_audit.json"
BENCHMARK_ROOT="$ML_ROOT/benchmarks/$BENCHMARK_ID"
MEMORY_MIN_AVAILABLE_KB="${MEMORY_MIN_AVAILABLE_KB:-2200000}"
MEMORY_MAX_RSS_KB="${MEMORY_MAX_RSS_KB:-12000000}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CORSEWIND_SKLEARN_N_JOBS="${CORSEWIND_SKLEARN_N_JOBS:-1}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  echo "$(timestamp) $*" | tee -a "$LOG"
}

fail() {
  local code="$1"
  shift
  echo "failed:$code" > "$STATUS"
  log "failed code=$code $*"
  exit "$code"
}

run_guarded() {
  local name="$1"
  shift
  log "launch $name"
  "$@" &
  local pid=$!
  log "$name pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do
    local rss_kb="0"
    local available_kb="0"
    rss_kb="$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || echo 0)"
    available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
    rss_kb="${rss_kb:-0}"
    available_kb="${available_kb:-0}"
    log "$name pid=$pid rss_kb=$rss_kb mem_available_kb=$available_kb"
    if [[ "$available_kb" -gt 0 && "$available_kb" -lt "$MEMORY_MIN_AVAILABLE_KB" ]]; then
      log "$name killing pid=$pid: low available memory"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      fail 92 "$name memory guard low MemAvailable"
    fi
    if [[ "$rss_kb" -gt "$MEMORY_MAX_RSS_KB" ]]; then
      log "$name killing pid=$pid: rss guard"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      fail 93 "$name memory guard high RSS"
    fi
    sleep 20
  done
  set +e
  wait "$pid"
  local code=$?
  set -e
  log "$name finished code=$code"
  return "$code"
}

cd "$ROOT"
mkdir -p "$LOG_ROOT" "$ML_ROOT/source_inventories" "$BENCHMARK_ROOT"
echo "running" > "$STATUS"
log "partial hPa early-signal started run_id=$RUN_ID benchmark_id=$BENCHMARK_ID"

log "writing pre-run hPa coverage audit"
ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/audit_open_meteo_coverage.py \
  --input-root "$ML_ROOT/open_meteo/historical_forecast" \
  --start-date 2025-07-01 \
  --end-date 2026-06-23 \
  --model meteofrance_arome_france \
  --include-context-spots \
  --required-features temperature_1000hPa,relative_humidity_850hPa,geopotential_height_850hPa,wind_speed_850hPa,wind_direction_850hPa \
  > "$AUDIT_JSON"

log "pre-run hPa coverage: $("$PYTHON" - "$AUDIT_JSON" <<'PY'
import json
import sys

audit = json.load(open(sys.argv[1]))
observed = audit.get("observed_rows") or 0
complete = audit.get("required_feature_complete_rows") or 0
print(f"{complete}/{observed}={complete / observed if observed else 0.0:.8f}")
PY
)"

run_guarded "$RUN_ID.rebuild" \
env ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/run_training_backfill_pipeline.py \
  --ml-root "$ML_ROOT" \
  --run-id "$RUN_ID" \
  --start-date 2025-07-01 \
  --end-date 2026-06-23 \
  --start-hour-utc 10 \
  --end-hour-utc 18 \
  --chunk-days 31 \
  --lead-minutes 15,30,45,60 \
  --step-minutes 15 \
  --target-tolerance-minutes 8 \
  --forecast-valid-tolerance-minutes 31 \
  --no-collect-open-meteo \
  --include-context-spots \
  --export-parquet \
  --parquet-batch-size 5000 \
  --parquet-compression zstd \
  --command-timeout-sec 7200 \
  --continue-on-error \
  2>&1 | tee -a "$LOG" || fail 19 "$RUN_ID rebuild failed"

run_guarded "$BENCHMARK_ID.train" \
env ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/train_residual_correction_parquet.py \
  --training-table-root "$ML_ROOT/training_tables/$RUN_ID" \
  --output-root "$BENCHMARK_ROOT" \
  --run-id "$BENCHMARK_ID" \
  --split-time-utc 2026-01-01T00:00:00Z \
  --include-lead-minute 15 \
  --include-lead-minute 30 \
  --include-lead-minute 45 \
  --include-lead-minute 60 \
  --eval-lead-minute 15 \
  --eval-lead-minute 30 \
  --eval-lead-minute 45 \
  --eval-lead-minute 60 \
  --model-family lightgbm \
  --max-iter 350 \
  --learning-rate 0.04 \
  --max-leaf-nodes 63 \
  --lightgbm-max-bin 127 \
  --lightgbm-feature-fraction 0.85 \
  --lightgbm-bagging-fraction 0.85 \
  --lightgbm-bagging-freq 1 \
  --lightgbm-force-col-wise \
  --max-train-rows 150000 \
  --max-test-rows 100000 \
  --n-jobs 2 \
  --only-target residual_wind_mean_ms \
  --skip-classification \
  2>&1 | tee -a "$LOG" || fail 20 "$BENCHMARK_ID training failed"

ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/audit_tabular_rmse09_result.py \
  --training-results "$BENCHMARK_ROOT/training_results.json" \
  --previous-best-rmse 1.268019 \
  --output-json "$BENCHMARK_ROOT/tabular_rmse09_audit.json" \
  --output-md "$BENCHMARK_ROOT/tabular_rmse09_audit.md" \
  2>&1 | tee -a "$LOG"

if [[ -f "$BENCHMARK_ROOT/residual_wind_mean_ms.joblib" ]]; then
  ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/analyze_tabular_rmse09_errors.py \
    --training-results "$BENCHMARK_ROOT/training_results.json" \
    --feature-columns "$BENCHMARK_ROOT/feature_columns.json" \
    --model-path "$BENCHMARK_ROOT/residual_wind_mean_ms.joblib" \
    --include-lead-minute 15 \
    --include-lead-minute 30 \
    --include-lead-minute 45 \
    --include-lead-minute 60 \
    --metric-lead-minute 15 \
    --metric-lead-minute 30 \
    --metric-lead-minute 45 \
    --metric-lead-minute 60 \
    --output-predictions "$BENCHMARK_ROOT/tabular_holdout_predictions.parquet" \
    --output-json "$BENCHMARK_ROOT/tabular_error_analysis.json" \
    --output-md "$BENCHMARK_ROOT/tabular_error_analysis.md" \
    2>&1 | tee -a "$LOG"
fi

echo "complete" > "$STATUS"
log "partial hPa early-signal complete"
