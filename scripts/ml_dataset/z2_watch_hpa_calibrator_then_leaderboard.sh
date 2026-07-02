#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/srv/data/corsewind/backfill_runner}"
ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
PYTHON="${PYTHON:-/home/z2/corsewind-ml-smoke/.venv/bin/python}"
LOG_ROOT="$ML_ROOT/backfill_logs"
BENCH_ROOT="$ML_ROOT/benchmarks"
STATUS="$LOG_ROOT/hpa_calibrator_watcher.status"
LOG="$LOG_ROOT/hpa_calibrator_watcher.log"
RUN_ID="${RUN_ID:-residual_backfill_2024_2026_short_hpa_v1}"
PRIMARY_BENCHMARK_ID="${PRIMARY_BENCHMARK_ID:-tabular_lgbm_300k_short_hpa_v1_2024_2025_to_2026_v1}"
EXTRA_STATUS="${EXTRA_STATUS:-$LOG_ROOT/hpa_extra_benchmarks_watcher.status}"
PRIMARY_STATUS="${PRIMARY_STATUS:-$LOG_ROOT/open_meteo_pressure_rebuild_watcher.status}"
CALBASE_ID="${CALBASE_ID:-tabular_lgbm_calbase_180k_short_hpa_v1_2024_to_2025h2_v1}"
CALIBRATOR_ID="${CALIBRATOR_ID:-prediction_residual_calibrator_hpa_2025h2_to_2026_extratrees_scalegrid_v1}"
PREVIOUS_BEST_RMSE="${PREVIOUS_BEST_RMSE:-1.268019}"
SLEEP_SECONDS="${SLEEP_SECONDS:-180}"
MEMORY_MIN_AVAILABLE_KB="${MEMORY_MIN_AVAILABLE_KB:-2200000}"
MEMORY_MAX_RSS_KB="${MEMORY_MAX_RSS_KB:-12000000}"
WAIT_FOR_EXTRA="${WAIT_FOR_EXTRA:-1}"

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

audit_tabular() {
  local benchmark_id="$1"
  local root="$BENCH_ROOT/$benchmark_id"
  ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/audit_tabular_rmse09_result.py \
    --training-results "$root/training_results.json" \
    --previous-best-rmse "$PREVIOUS_BEST_RMSE" \
    --output-json "$root/tabular_rmse09_audit.json" \
    --output-md "$root/tabular_rmse09_audit.md" \
    2>&1 | tee -a "$LOG"
}

write_predictions() {
  local benchmark_id="$1"
  local root="$BENCH_ROOT/$benchmark_id"
  if [[ ! -f "$root/residual_wind_mean_ms.joblib" ]]; then
    fail 30 "missing model for $benchmark_id: $root/residual_wind_mean_ms.joblib"
  fi
  ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/analyze_tabular_rmse09_errors.py \
    --training-results "$root/training_results.json" \
    --feature-columns "$root/feature_columns.json" \
    --model-path "$root/residual_wind_mean_ms.joblib" \
    --include-lead-minute 15 \
    --include-lead-minute 30 \
    --include-lead-minute 45 \
    --include-lead-minute 60 \
    --metric-lead-minute 15 \
    --metric-lead-minute 30 \
    --metric-lead-minute 45 \
    --metric-lead-minute 60 \
    --output-predictions "$root/tabular_holdout_predictions.parquet" \
    --output-json "$root/tabular_error_analysis.json" \
    --output-md "$root/tabular_error_analysis.md" \
    2>&1 | tee -a "$LOG"
}

train_calbase() {
  local root="$BENCH_ROOT/$CALBASE_ID"
  rm -rf "$root"
  mkdir -p "$root"
  run_guarded "$CALBASE_ID.train" \
    env ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/train_residual_correction_parquet.py \
      --training-table-root "$ML_ROOT/training_tables/$RUN_ID" \
      --output-root "$root" \
      --run-id "$CALBASE_ID" \
      --split-time-utc 2025-07-01T00:00:00Z \
      --include-lead-minute 15 \
      --include-lead-minute 30 \
      --include-lead-minute 45 \
      --include-lead-minute 60 \
      --eval-lead-minute 15 \
      --eval-lead-minute 30 \
      --eval-lead-minute 45 \
      --eval-lead-minute 60 \
      --model-family lightgbm \
      --max-iter 260 \
      --learning-rate 0.04 \
      --max-leaf-nodes 63 \
      --lightgbm-max-bin 127 \
      --lightgbm-feature-fraction 0.85 \
      --lightgbm-bagging-fraction 0.85 \
      --lightgbm-bagging-freq 1 \
      --lightgbm-force-col-wise \
      --max-train-rows 180000 \
      --max-test-rows 80000 \
      --n-jobs 1 \
      --only-target residual_wind_mean_ms \
      --skip-classification
  audit_tabular "$CALBASE_ID" || true
  write_predictions "$CALBASE_ID"
  cp "$root/tabular_holdout_predictions.parquet" "$root/calibration_predictions_2025h2.parquet"
}

train_calibrator() {
  local root="$BENCH_ROOT/$CALIBRATOR_ID"
  local calbase_root="$BENCH_ROOT/$CALBASE_ID"
  local eval_root="$BENCH_ROOT/$PRIMARY_BENCHMARK_ID"
  mkdir -p "$root"
  if [[ ! -f "$eval_root/tabular_holdout_predictions.parquet" ]]; then
    write_predictions "$PRIMARY_BENCHMARK_ID"
  fi
  ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/train_prediction_residual_calibrator.py \
    --calibration-predictions "$calbase_root/calibration_predictions_2025h2.parquet" \
    --evaluation-predictions "$eval_root/tabular_holdout_predictions.parquet" \
    --calibration-start-utc "2025-07-01T00:00:00Z" \
    --calibration-end-utc "2026-01-01T00:00:00Z" \
    --evaluation-start-utc "2026-01-01T00:00:00Z" \
    --lead-minute 15 \
    --lead-minute 30 \
    --lead-minute 45 \
    --lead-minute 60 \
    --model-family extra_trees \
    --max-iter 160 \
    --min-samples-leaf 40 \
    --n-jobs 1 \
    --clip-correction-ms 2.0 \
    --scale-validation-start-utc "2025-10-01T00:00:00Z" \
    --scale-validation-end-utc "2026-01-01T00:00:00Z" \
    --scale-candidate 0.40 \
    --scale-candidate 0.50 \
    --scale-candidate 0.60 \
    --scale-candidate 0.70 \
    --scale-candidate 0.80 \
    --scale-candidate 0.90 \
    --scale-candidate 1.00 \
    --output-predictions "$root/calibrated_predictions_2026.parquet" \
    --output-model "$root/calibrator.joblib" \
    --output-json "$root/calibration_results.json" \
    --output-md "$root/calibration_results.md" \
    2>&1 | tee -a "$LOG"
}

write_leaderboard() {
  local root="$BENCH_ROOT/hpa_calibrator_selection_v1"
  mkdir -p "$root"
  ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/select_wind_mean_rmse_leaderboard.py \
    --search-root "$BENCH_ROOT" \
    --threshold-rmse 0.9 \
    --output-json "$root/wind_mean_rmse_leaderboard.json" \
    --output-md "$root/wind_mean_rmse_leaderboard.md" \
    2>&1 | tee -a "$LOG"
}

write_gap_audits() {
  local root="$BENCH_ROOT/$CALIBRATOR_ID"
  local eval_root="$BENCH_ROOT/$PRIMARY_BENCHMARK_ID"
  local old_best_root="$BENCH_ROOT/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1"
  local gap_json="$root/rmse09_gap_audit_hpa_calibrator_v1.json"
  local model_args=(
    --model "hpa_primary|$eval_root/tabular_holdout_predictions.parquet|corrected_wind_mean_ms"
  )
  if [[ -f "$old_best_root/calibrated_predictions_2026.parquet" ]]; then
    model_args+=(--model "old_best_scale070|$old_best_root/calibrated_predictions_2026.parquet|calibrated_wind_mean_ms")
  fi

  ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/analyze_rmse09_gap_oracles.py \
    --predictions "$root/calibrated_predictions_2026.parquet" \
    --prediction-column calibrated_wind_mean_ms \
    --target-column actual_wind_mean_ms \
    "${model_args[@]}" \
    --critical-spot la_tonnara \
    --critical-spot santa_manza \
    --critical-spot balistra \
    --critical-spot porticcio \
    --critical-spot porto_polo \
    --output-json "$gap_json" \
    --output-md "$root/rmse09_gap_audit_hpa_calibrator_v1.md" \
    2>&1 | tee -a "$LOG"

  ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/compute_rmse09_reduction_targets.py \
    --audit-json "$gap_json" \
    --threshold-rmse 0.9 \
    --output-json "$root/rmse09_reduction_targets_hpa_calibrator_v1.json" \
    --output-md "$root/rmse09_reduction_targets_hpa_calibrator_v1.md" \
    2>&1 | tee -a "$LOG"
}

write_iteration_summary() {
  local root="$BENCH_ROOT/$CALIBRATOR_ID"
  ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/summarize_hpa_rmse09_status.py \
    --ml-root "$ML_ROOT" \
    --output-json "$BENCH_ROOT/hpa_rmse09_status_current.json" \
    --output-md "$BENCH_ROOT/hpa_rmse09_status_current.md" \
    2>&1 | tee -a "$LOG"
  ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/summarize_hpa_rmse09_iteration.py \
    --status-json "$BENCH_ROOT/hpa_rmse09_status_current.json" \
    --leaderboard-json "$BENCH_ROOT/hpa_calibrator_selection_v1/wind_mean_rmse_leaderboard.json" \
    --hpa-selection-json "$BENCH_ROOT/hpa_tabular_rmse09_selection_v1/hpa_tabular_rmse09_selection.json" \
    --calibrator-results-json "$root/calibration_results.json" \
    --gap-audit-json "$root/rmse09_gap_audit_hpa_calibrator_v1.json" \
    --reduction-targets-json "$root/rmse09_reduction_targets_hpa_calibrator_v1.json" \
    --previous-best-rmse "$PREVIOUS_BEST_RMSE" \
    --threshold-rmse 0.9 \
    --output-json "$BENCH_ROOT/hpa_rmse09_iteration_summary.json" \
    --output-md "$BENCH_ROOT/hpa_rmse09_iteration_summary.md" \
    2>&1 | tee -a "$LOG"
}

cd "$ROOT"
mkdir -p "$LOG_ROOT" "$BENCH_ROOT"
echo "running" > "$STATUS"
log "hPa calibrator watcher started run_id=$RUN_ID primary=$PRIMARY_BENCHMARK_ID"

primary_results="$BENCH_ROOT/$PRIMARY_BENCHMARK_ID/training_results.json"
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

if [[ "$WAIT_FOR_EXTRA" == "1" ]]; then
  while true; do
    extra_status="$(cat "$EXTRA_STATUS" 2>/dev/null || echo missing)"
    if [[ "$extra_status" == failed:* ]]; then
      fail 11 "extra hPa benchmark watcher failed: $extra_status"
    fi
    if [[ "$extra_status" == "complete" ]]; then
      break
    fi
    log "waiting_for_extra_benchmarks status=$extra_status"
    sleep "$SLEEP_SECONDS"
  done
fi

log "training hPa 2025H2 calibration base"
train_calbase || fail 20 "calbase failed"
log "training hPa residual calibrator on primary 2026 predictions"
train_calibrator || fail 21 "calibrator failed"
log "refreshing global wind-mean leaderboard"
write_leaderboard || true
log "writing hPa calibrator RMSE09 gap audits"
write_gap_audits || true
log "writing hPa RMSE09 iteration summary"
write_iteration_summary || true

echo "complete" > "$STATUS"
log "hPa calibrator watcher finished"
