#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/srv/data/corsewind/backfill_runner}"
ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
PYTHON="${PYTHON:-/home/z2/corsewind-ml-smoke/.venv/bin/python}"
LOG_ROOT="$ML_ROOT/backfill_logs"
STATUS="$LOG_ROOT/hpa_extra_benchmarks_watcher.status"
LOG="$LOG_ROOT/hpa_extra_benchmarks_watcher.log"
RUN_ID="${RUN_ID:-residual_backfill_2024_2026_short_hpa_v1}"
PRIMARY_BENCHMARK_ID="${PRIMARY_BENCHMARK_ID:-tabular_lgbm_300k_short_hpa_v1_2024_2025_to_2026_v1}"
PREVIOUS_BEST_RMSE="${PREVIOUS_BEST_RMSE:-1.268019}"
SLEEP_SECONDS="${SLEEP_SECONDS:-180}"
SELECTION_ROOT="$ML_ROOT/benchmarks/hpa_tabular_rmse09_selection_v1"
PRIMARY_STATUS="${PRIMARY_STATUS:-$LOG_ROOT/open_meteo_pressure_rebuild_watcher.status}"
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

run_audit() {
  local benchmark_id="$1"
  local benchmark_root="$ML_ROOT/benchmarks/$benchmark_id"
  log "auditing $benchmark_id"
  ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/audit_tabular_rmse09_result.py \
    --training-results "$benchmark_root/training_results.json" \
    --previous-best-rmse "$PREVIOUS_BEST_RMSE" \
    --output-json "$benchmark_root/tabular_rmse09_audit.json" \
    --output-md "$benchmark_root/tabular_rmse09_audit.md" \
    2>&1 | tee -a "$LOG"
}

run_diagnostics_if_possible() {
  local benchmark_id="$1"
  local benchmark_root="$ML_ROOT/benchmarks/$benchmark_id"
  if [[ ! -f "$benchmark_root/residual_wind_mean_ms.joblib" ]]; then
    log "skipping diagnostics for $benchmark_id: no single global residual_wind_mean_ms.joblib"
    return 0
  fi
  log "writing diagnostics for $benchmark_id"
  ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/analyze_tabular_rmse09_errors.py \
    --training-results "$benchmark_root/training_results.json" \
    --feature-columns "$benchmark_root/feature_columns.json" \
    --model-path "$benchmark_root/residual_wind_mean_ms.joblib" \
    --include-lead-minute 15 \
    --include-lead-minute 30 \
    --include-lead-minute 45 \
    --include-lead-minute 60 \
    --metric-lead-minute 15 \
    --metric-lead-minute 30 \
    --metric-lead-minute 45 \
    --metric-lead-minute 60 \
    --output-predictions "$benchmark_root/tabular_holdout_predictions.parquet" \
    --output-json "$benchmark_root/tabular_error_analysis.json" \
    --output-md "$benchmark_root/tabular_error_analysis.md" \
    2>&1 | tee -a "$LOG"
}

select_hpa_best() {
  mkdir -p "$SELECTION_ROOT"
  log "selecting best audited hPa tabular RMSE09 result"
  ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/select_tabular_rmse09_result.py \
    --search-root "$ML_ROOT/benchmarks/$PRIMARY_BENCHMARK_ID" \
    --search-root "$ML_ROOT/benchmarks/tabular_extratrees_260k_short_hpa_v1_2024_2025_to_2026_v1" \
    --search-root "$ML_ROOT/benchmarks/tabular_lgbm_bylead_260k_short_hpa_v1_2024_2025_to_2026_v1" \
    --threshold-rmse 0.9 \
    --output-json "$SELECTION_ROOT/hpa_tabular_rmse09_selection.json" \
    --output-md "$SELECTION_ROOT/hpa_tabular_rmse09_selection.md" \
    2>&1 | tee -a "$LOG"

  log "asserting hPa tabular RMSE09 goal"
  set +e
  ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/assert_tabular_rmse09_goal.py \
    --selection-json "$SELECTION_ROOT/hpa_tabular_rmse09_selection.json" \
    --threshold-rmse 0.9 \
    --min-audit-count 3 \
    --min-source-parquets 25 \
    --min-train-rows 100000 \
    --min-test-rows 10000 \
    --min-metric-rows 10000 \
    --output-json "$SELECTION_ROOT/hpa_tabular_rmse09_assertion.json" \
    2>&1 | tee -a "$LOG"
  assertion_status="${PIPESTATUS[0]}"
  set -e
  if [[ "$assertion_status" == "0" ]]; then
    log "hPa tabular RMSE09 assertion passed"
  else
    log "hPa tabular RMSE09 assertion did not pass; continuing with diagnostics evidence"
  fi

  log "writing global wind-mean RMSE leaderboard"
  ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/select_wind_mean_rmse_leaderboard.py \
    --search-root "$ML_ROOT/benchmarks" \
    --threshold-rmse 0.9 \
    --output-json "$SELECTION_ROOT/wind_mean_rmse_leaderboard.json" \
    --output-md "$SELECTION_ROOT/wind_mean_rmse_leaderboard.md" \
    2>&1 | tee -a "$LOG"
}

train_extra_trees_global() {
  local benchmark_id="tabular_extratrees_260k_short_hpa_v1_2024_2025_to_2026_v1"
  local benchmark_root="$ML_ROOT/benchmarks/$benchmark_id"
  mkdir -p "$benchmark_root"
  log "training $benchmark_id"
  run_guarded "$benchmark_id.train" \
  env ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/train_residual_correction_parquet.py \
    --training-table-root "$ML_ROOT/training_tables/$RUN_ID" \
    --output-root "$benchmark_root" \
    --run-id "$benchmark_id" \
    --split-time-utc 2026-01-01T00:00:00Z \
    --include-lead-minute 15 \
    --include-lead-minute 30 \
    --include-lead-minute 45 \
    --include-lead-minute 60 \
    --eval-lead-minute 15 \
    --eval-lead-minute 30 \
    --eval-lead-minute 45 \
    --eval-lead-minute 60 \
    --model-family extra_trees \
    --max-iter 260 \
    --min-samples-leaf 4 \
    --max-train-rows 260000 \
    --max-test-rows 120000 \
    --n-jobs 2 \
    --only-target residual_wind_mean_ms \
    --skip-classification \
    2>&1 | tee -a "$LOG" || fail 20 "$benchmark_id training failed"
  run_audit "$benchmark_id"
  run_diagnostics_if_possible "$benchmark_id"
}

train_lightgbm_by_lead() {
  local benchmark_id="tabular_lgbm_bylead_260k_short_hpa_v1_2024_2025_to_2026_v1"
  local benchmark_root="$ML_ROOT/benchmarks/$benchmark_id"
  mkdir -p "$benchmark_root"
  log "training $benchmark_id"
  run_guarded "$benchmark_id.train" \
  env ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/train_residual_correction_parquet.py \
    --training-table-root "$ML_ROOT/training_tables/$RUN_ID" \
    --output-root "$benchmark_root" \
    --run-id "$benchmark_id" \
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
    --max-iter 360 \
    --learning-rate 0.04 \
    --max-leaf-nodes 63 \
    --lightgbm-max-bin 127 \
    --lightgbm-feature-fraction 0.85 \
    --lightgbm-bagging-fraction 0.85 \
    --lightgbm-bagging-freq 1 \
    --lightgbm-force-col-wise \
    --fit-group-column lead_time_minutes \
    --min-group-train-rows 20000 \
    --min-group-test-rows 5000 \
    --max-train-rows 260000 \
    --max-test-rows 120000 \
    --n-jobs 2 \
    --only-target residual_wind_mean_ms \
    --skip-classification \
    2>&1 | tee -a "$LOG" || fail 21 "$benchmark_id training failed"
  run_audit "$benchmark_id"
  run_diagnostics_if_possible "$benchmark_id"
}

cd "$ROOT"
mkdir -p "$LOG_ROOT"
echo "running" > "$STATUS"
log "extra benchmark watcher started for $RUN_ID"

primary_results="$ML_ROOT/benchmarks/$PRIMARY_BENCHMARK_ID/training_results.json"
while true; do
  primary_status="$(cat "$PRIMARY_STATUS" 2>/dev/null || echo missing)"
  if [[ "$primary_status" == failed:* ]]; then
    fail 10 "primary benchmark watcher failed: $primary_status"
  fi
  if [[ -f "$primary_results" && "$primary_status" == "complete" ]]; then
    break
  fi
  log "waiting_for_primary_benchmark=$PRIMARY_BENCHMARK_ID primary_status=$primary_status"
  sleep "$SLEEP_SECONDS"
done

log "primary benchmark complete; launching extra hPa benchmarks"
train_extra_trees_global
train_lightgbm_by_lead
select_hpa_best
log "extra hPa benchmarks finished"
echo "complete" > "$STATUS"
