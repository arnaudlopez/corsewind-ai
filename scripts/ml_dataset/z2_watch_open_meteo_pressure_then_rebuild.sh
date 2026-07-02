#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/srv/data/corsewind/backfill_runner}"
ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
PYTHON="${PYTHON:-/home/z2/corsewind-ml-smoke/.venv/bin/python}"
LOG_ROOT="$ML_ROOT/backfill_logs"
STATUS="$LOG_ROOT/open_meteo_pressure_rebuild_watcher.status"
LOG="$LOG_ROOT/open_meteo_pressure_rebuild_watcher.log"
AUDIT_JSON="$ML_ROOT/source_inventories/open_meteo_pressure_level_progress_final.json"
RUN_ID="${RUN_ID:-residual_backfill_2024_2026_short_hpa_v1}"
BENCHMARK_ID="${BENCHMARK_ID:-tabular_lgbm_300k_short_hpa_v1_2024_2025_to_2026_v1}"
MIN_OBSERVED_COVERAGE="${MIN_OBSERVED_COVERAGE:-0.995}"
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

pid_matches() {
  local pid="$1"
  local needle="$2"
  if [[ -z "$pid" || ! -r "/proc/$pid/cmdline" ]]; then
    return 1
  fi
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq "$needle"
}

running_count() {
  local count=0
  local pid
  shopt -s nullglob
  for pidfile in "$LOG_ROOT"/open_meteo_pressure_levels_seg*.pid; do
    case "$(basename "$pidfile")" in
      *rebuild*) continue ;;
    esac
    if [[ -f "$pidfile" ]]; then
      pid="$(cat "$pidfile")"
      if pid_matches "$pid" "collect_open_meteo_historical_forecast.py"; then
        count=$((count + 1))
      fi
    fi
  done
  shopt -u nullglob
  echo "$count"
}

cd "$ROOT"
mkdir -p "$LOG_ROOT" "$ML_ROOT/source_inventories"
echo "running" > "$STATUS"
log "watcher started for $RUN_ID"

while true; do
  count="$(running_count)"
  log "backfill_processes_running=$count"
  if [[ "$count" == "0" ]]; then
    break
  fi
  sleep 120
done

log "all pressure-level backfill partitions finished; running final audit"
ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/audit_open_meteo_coverage.py \
  --input-root "$ML_ROOT/open_meteo/historical_forecast" \
  --start-date 2024-01-02 \
  --end-date 2026-06-23 \
  --model meteofrance_arome_france \
  --include-context-spots \
  --required-features temperature_1000hPa,relative_humidity_850hPa,geopotential_height_850hPa,wind_speed_850hPa,wind_direction_850hPa \
  > "$AUDIT_JSON"

coverage="$("$PYTHON" - "$AUDIT_JSON" <<'PY'
import json
import sys

audit = json.load(open(sys.argv[1]))
observed = audit.get("observed_rows") or 0
complete = audit.get("required_feature_complete_rows") or 0
coverage = complete / observed if observed else 0.0
print(f"{coverage:.8f}")
PY
)"
log "observed_row_hpa_coverage=$coverage"

"$PYTHON" - "$AUDIT_JSON" "$MIN_OBSERVED_COVERAGE" <<'PY'
import json
import sys

audit = json.load(open(sys.argv[1]))
threshold = float(sys.argv[2])
observed = audit.get("observed_rows") or 0
complete = audit.get("required_feature_complete_rows") or 0
coverage = complete / observed if observed else 0.0
if coverage < threshold:
    raise SystemExit(
        f"coverage {coverage:.6f} below threshold {threshold:.6f}: "
        f"{complete}/{observed} observed rows complete"
    )
PY

log "coverage gate passed; launching feature-store/training-table rebuild"
run_guarded "$RUN_ID.rebuild" \
env ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/run_training_backfill_pipeline.py \
  --ml-root "$ML_ROOT" \
  --run-id "$RUN_ID" \
  --start-date 2024-01-02 \
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

log "rebuild finished for $RUN_ID"
log "launching first LightGBM short-horizon benchmark: $BENCHMARK_ID"
BENCHMARK_ROOT="$ML_ROOT/benchmarks/$BENCHMARK_ID"
mkdir -p "$BENCHMARK_ROOT"
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
  --max-iter 450 \
  --learning-rate 0.04 \
  --max-leaf-nodes 63 \
  --lightgbm-max-bin 127 \
  --lightgbm-feature-fraction 0.85 \
  --lightgbm-bagging-fraction 0.85 \
  --lightgbm-bagging-freq 1 \
  --lightgbm-force-col-wise \
  --max-train-rows 300000 \
  --max-test-rows 120000 \
  --n-jobs 2 \
  --only-target residual_wind_mean_ms \
  --skip-classification \
  2>&1 | tee -a "$LOG" || fail 20 "$BENCHMARK_ID training failed"

log "running tabular RMSE09 audit for $BENCHMARK_ID"
  ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/audit_tabular_rmse09_result.py \
  --training-results "$BENCHMARK_ROOT/training_results.json" \
  --previous-best-rmse 1.268019 \
  --output-json "$BENCHMARK_ROOT/tabular_rmse09_audit.json" \
  --output-md "$BENCHMARK_ROOT/tabular_rmse09_audit.md" \
  2>&1 | tee -a "$LOG"

if [[ -f "$BENCHMARK_ROOT/residual_wind_mean_ms.joblib" ]]; then
  log "writing prediction diagnostics for $BENCHMARK_ID"
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

log "benchmark finished for $BENCHMARK_ID"
echo "complete" > "$STATUS"
