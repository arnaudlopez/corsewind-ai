#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/srv/data/corsewind/backfill_runner}"
ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
PYTHON="${PYTHON:-/home/z2/corsewind-ml-smoke/.venv/bin/python}"
LOG_ROOT="$ML_ROOT/backfill_logs"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-residual_backfill_2025h2_2026_short_hpa_partial_signal_v1}"
SAMPLE_RUN_ID="${SAMPLE_RUN_ID:-residual_backfill_2025h2_2026_short_hpa_sampled_signal_v1}"
BENCHMARK_ID="${BENCHMARK_ID:-tabular_lgbm_150k_short_hpa_sampled_signal_2025h2_to_2026_v1}"
SOURCE_JSONL="$ML_ROOT/training_tables/$SOURCE_RUN_ID/training_rows.jsonl"
SAMPLE_ROOT="$ML_ROOT/training_tables/$SAMPLE_RUN_ID"
BENCHMARK_ROOT="$ML_ROOT/benchmarks/$BENCHMARK_ID"
STATUS="$LOG_ROOT/partial_hpa_sampled_signal.status"
LOG="$LOG_ROOT/partial_hpa_sampled_signal.log"
MEMORY_MIN_AVAILABLE_KB="${MEMORY_MIN_AVAILABLE_KB:-2200000}"
MEMORY_MAX_RSS_KB="${MEMORY_MAX_RSS_KB:-12000000}"
SAMPLE_MAX_TRAIN_ROWS="${SAMPLE_MAX_TRAIN_ROWS:-60000}"
SAMPLE_MAX_TEST_ROWS="${SAMPLE_MAX_TEST_ROWS:-40000}"

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

pid_matches() {
  local pid="$1"
  local needle="$2"
  if [[ -z "$pid" || ! -r "/proc/$pid/cmdline" ]]; then
    return 1
  fi
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq "$needle"
}

stop_leftover_full_export() {
  local pidfile="$LOG_ROOT/partial_hpa_manual_parquet_export.pid"
  local pids=()
  if [[ ! -f "$pidfile" ]]; then
    :
  else
    local pid
    pid="$(cat "$pidfile")"
    if pid_matches "$pid" "export_training_table_parquet.py"; then
      pids+=("$pid")
    fi
  fi
  while IFS= read -r pid; do
    [[ -n "$pid" ]] && pids+=("$pid")
  done < <(
    pgrep -f "export_training_table_parquet.py.*${SOURCE_JSONL}" 2>/dev/null || true
  )
  if [[ "${#pids[@]}" -eq 0 ]]; then
    return 0
  fi
  local seen=" "
  for pid in "${pids[@]}"; do
    if [[ "$seen" == *" $pid "* ]]; then
      continue
    fi
    seen="$seen$pid "
    log "stopping leftover full parquet export pid=$pid"
    kill "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${pids[@]}"; do
    if pid_matches "$pid" "export_training_table_parquet.py"; then
      log "force-stopping leftover full parquet export pid=$pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
}

stop_previous_failed_partial_run() {
  local pids=()
  while IFS= read -r pid; do
    [[ -n "$pid" ]] && pids+=("$pid")
  done < <(
    pgrep -f "run_training_backfill_pipeline.py.*${SOURCE_RUN_ID}" 2>/dev/null || true
  )
  if [[ "${#pids[@]}" -eq 0 ]]; then
    return 0
  fi
  for pid in "${pids[@]}"; do
    log "stopping previous failed partial rebuild pid=$pid"
    kill "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${pids[@]}"; do
    if pid_matches "$pid" "run_training_backfill_pipeline.py"; then
      log "force-stopping previous failed partial rebuild pid=$pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
}

stop_repair_temporarily() {
  if [[ "${STOP_REPAIR_DURING_SAMPLE:-1}" != "1" ]]; then
    return 0
  fi
  local pids=()
  while IFS= read -r pid; do
    [[ -n "$pid" ]] && pids+=("$pid")
  done < <(
    pgrep -f "z2_repair_open_meteo_pressure_after_rate_limit.sh" 2>/dev/null || true
  )
  for pid in "${pids[@]}"; do
    log "stopping hPa repair during sampled benchmark pid=$pid"
    kill "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${pids[@]}"; do
    if pid_matches "$pid" "z2_repair_open_meteo_pressure_after_rate_limit.sh"; then
      log "force-stopping hPa repair during sampled benchmark pid=$pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    log "stopping hPa repair collector child pid=$pid"
    kill "$pid" 2>/dev/null || true
  done < <(
    pgrep -f "collect_open_meteo_historical_forecast.py.*temperature_1000hPa" 2>/dev/null || true
  )
}

restart_repair_if_requested() {
  if [[ "${STOP_REPAIR_DURING_SAMPLE:-1}" != "1" || "${RESTART_REPAIR_AFTER_SAMPLE:-0}" != "1" ]]; then
    return 0
  fi
  log "restarting hPa repair after sampled benchmark"
  INITIAL_WAIT_SECONDS=0 WAIT_SECONDS="${REPAIR_WAIT_SECONDS:-3900}" \
    REQUEST_SLEEP_SEC="${REPAIR_REQUEST_SLEEP_SEC:-3}" \
    TIMEOUT_SEC="${REPAIR_TIMEOUT_SEC:-240}" \
    COLLECT_PROCESS_TIMEOUT_SEC="${REPAIR_COLLECT_PROCESS_TIMEOUT_SEC:-600}" \
    MAX_DAYS_PER_REQUEST="${REPAIR_MAX_DAYS_PER_REQUEST:-7}" \
    REPAIR_TASK_DAYS_PER_RANGE="${REPAIR_TASK_DAYS_PER_RANGE:-7}" \
    setsid bash scripts/ml_dataset/z2_repair_open_meteo_pressure_after_rate_limit.sh >/dev/null 2>&1 &
  echo $! > "$LOG_ROOT/open_meteo_pressure_repair_after_429.pid"
  log "hPa repair restarted pid=$(cat "$LOG_ROOT/open_meteo_pressure_repair_after_429.pid")"
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
mkdir -p "$LOG_ROOT" "$SAMPLE_ROOT" "$BENCHMARK_ROOT"
echo "running" > "$STATUS"
log "partial hPa sampled-signal resume started source_run_id=$SOURCE_RUN_ID sample_run_id=$SAMPLE_RUN_ID benchmark_id=$BENCHMARK_ID"

if [[ ! -f "$SOURCE_JSONL" ]]; then
  fail 10 "missing source JSONL $SOURCE_JSONL"
fi

stop_leftover_full_export
stop_previous_failed_partial_run
stop_repair_temporarily

run_guarded "$SAMPLE_RUN_ID.sample_jsonl" \
env ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/sample_residual_training_jsonl.py \
  --input-jsonl "$SOURCE_JSONL" \
  --output-root "$SAMPLE_ROOT" \
  --output-name training_rows.jsonl \
  --split-time-utc 2026-01-01T00:00:00Z \
  --include-lead-minute 15 \
  --include-lead-minute 30 \
  --include-lead-minute 45 \
  --include-lead-minute 60 \
  --max-train-rows "$SAMPLE_MAX_TRAIN_ROWS" \
  --max-test-rows "$SAMPLE_MAX_TEST_ROWS" \
  2>&1 | tee -a "$LOG" || fail 11 "$SAMPLE_RUN_ID sampling failed"

run_guarded "$SAMPLE_RUN_ID.export_parquet" \
env ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/export_training_table_parquet.py \
  --training-rows "$SAMPLE_ROOT/training_rows.jsonl" \
  --output-root "$SAMPLE_ROOT" \
  --batch-size 1000 \
  --compression zstd \
  2>&1 | tee -a "$LOG" || fail 12 "$SAMPLE_RUN_ID parquet export failed"

run_guarded "$BENCHMARK_ID.train" \
env ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/train_residual_correction_parquet.py \
  --training-parquet "$SAMPLE_ROOT/training_rows.parquet" \
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
  --max-train-rows "$SAMPLE_MAX_TRAIN_ROWS" \
  --max-test-rows "$SAMPLE_MAX_TEST_ROWS" \
  --n-jobs 2 \
  --only-target labels__residual_wind_mean_ms \
  --skip-classification \
  2>&1 | tee -a "$LOG" || fail 13 "$BENCHMARK_ID training failed"

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
log "partial hPa sampled-signal complete"
restart_repair_if_requested
