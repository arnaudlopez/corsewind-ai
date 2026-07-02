#!/usr/bin/env bash
set -euo pipefail

ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
PREFIX="${PREFIX:-residual_windsup_sst_prev_phys_v1}"
REMOTE_ROOT="${REMOTE_ROOT:-/srv/data/corsewind/backfill_runner}"
LOG_ROOT="${LOG_ROOT:-$ML_ROOT/run_logs}"
BENCH_ROOT="${BENCH_ROOT:-$ML_ROOT/benchmarks}"
PY="${PY:-/home/z2/corsewind-ml-smoke/.venv/bin/python}"
STATUS="${STATUS:-$LOG_ROOT/phys_v1_150k_bin63_benchmark.status}"
MEMORY_MIN_AVAILABLE_KB="${MEMORY_MIN_AVAILABLE_KB:-2500000}"
MEMORY_MAX_RSS_KB="${MEMORY_MAX_RSS_KB:-28000000}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CORSEWIND_SKLEARN_N_JOBS="${CORSEWIND_SKLEARN_N_JOBS:-1}"

BASE_RUN="tabular_lgbm_150k_bin63_prev_phys_v1_2024_2025_to_2026_v1"
CALBASE_RUN="tabular_lgbm_calbase_150k_bin63_phys_v1_2024_to_2025h2_v1"
CAL_RUN="prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_phys_v1_150k_bin63"

mkdir -p "$LOG_ROOT" "$BENCH_ROOT"

log() {
  echo "$(date -Is) $*"
}

fail() {
  local code="$1"
  shift
  echo "$code" > "$STATUS"
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
    if [ "$available_kb" -gt 0 ] && [ "$available_kb" -lt "$MEMORY_MIN_AVAILABLE_KB" ]; then
      log "$name killing pid=$pid: low available memory"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      fail 92 "$name memory guard low MemAvailable"
    fi
    if [ "$rss_kb" -gt "$MEMORY_MAX_RSS_KB" ]; then
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

train_lgbm_base() {
  local run_id="$1"
  local split="$2"
  local start_month="$3"
  local end_month="$4"
  local max_train_rows="$5"
  local max_test_rows="$6"
  local output="$BENCH_ROOT/$run_id"
  rm -rf "$output"
  run_guarded "$run_id.train" "$PY" scripts/ml_dataset/train_residual_correction_parquet.py \
    --training-table-root "$ML_ROOT/training_tables" \
    --run-id-prefix "$PREFIX" \
    --start-month "$start_month" \
    --end-month "$end_month" \
    --output-root "$output" \
    --run-id "$run_id" \
    --split-time-utc "$split" \
    --max-train-rows "$max_train_rows" \
    --max-test-rows "$max_test_rows" \
    --model-family lightgbm \
    --max-iter 180 \
    --learning-rate 0.04 \
    --max-leaf-nodes 31 \
    --min-samples-leaf 20 \
    --n-jobs 1 \
    --lightgbm-max-bin 63 \
    --lightgbm-feature-fraction 0.8 \
    --lightgbm-bagging-fraction 0.8 \
    --lightgbm-bagging-freq 1 \
    --lightgbm-force-col-wise \
    --only-target labels__residual_wind_mean_ms \
    --skip-classification \
    --eval-lead-minute 15 \
    --eval-lead-minute 30 \
    --eval-lead-minute 45 \
    --eval-lead-minute 60
}

analyze_base() {
  local run_id="$1"
  local output="$BENCH_ROOT/$run_id"
  "$PY" scripts/ml_dataset/analyze_tabular_rmse09_errors.py \
    --training-results "$output/training_results.json" \
    --feature-columns "$output/feature_columns.json" \
    --model-path "$output/labels__residual_wind_mean_ms.joblib" \
    --metric-lead-minute 15 \
    --metric-lead-minute 30 \
    --metric-lead-minute 45 \
    --metric-lead-minute 60 \
    --output-predictions "$output/tabular_holdout_predictions.parquet" \
    --output-json "$output/tabular_error_diagnosis.json" \
    --output-md "$output/tabular_error_diagnosis.md"
  "$PY" scripts/ml_dataset/audit_tabular_rmse09_result.py \
    --training-results "$output/training_results.json" \
    --output-json "$output/tabular_rmse09_audit.json" \
    --output-md "$output/tabular_rmse09_audit.md" || true
}

cd "$REMOTE_ROOT" || fail 10 "cannot cd remote root"
echo "started $(date -Is)" > "$STATUS"
log "phys_v1 150k bin63 benchmark started"

log "training 150k LightGBM base"
train_lgbm_base "$BASE_RUN" "2026-01-01T00:00:00Z" "2024-01" "2026-06" 150000 60000 || fail 20 "base training failed"
analyze_base "$BASE_RUN" || fail 21 "base analysis failed"

log "training 150k 2025-H2 calibration base"
train_lgbm_base "$CALBASE_RUN" "2025-07-01T00:00:00Z" "2024-01" "2025-12" 150000 60000 || fail 30 "calibration base training failed"
analyze_base "$CALBASE_RUN" || fail 31 "calibration base analysis failed"
cp "$BENCH_ROOT/$CALBASE_RUN/tabular_holdout_predictions.parquet" "$BENCH_ROOT/$CALBASE_RUN/calibration_predictions_2025h2.parquet"

log "training ExtraTrees second-stage calibrator"
mkdir -p "$BENCH_ROOT/$CAL_RUN"
"$PY" scripts/ml_dataset/train_prediction_residual_calibrator.py \
  --calibration-predictions "$BENCH_ROOT/$CALBASE_RUN/calibration_predictions_2025h2.parquet" \
  --evaluation-predictions "$BENCH_ROOT/$BASE_RUN/tabular_holdout_predictions.parquet" \
  --calibration-start-utc "2025-07-01T00:00:00Z" \
  --calibration-end-utc "2026-01-01T00:00:00Z" \
  --evaluation-start-utc "2026-01-01T00:00:00Z" \
  --lead-minute 15 \
  --lead-minute 30 \
  --lead-minute 45 \
  --lead-minute 60 \
  --model-family extra_trees \
  --max-iter 120 \
  --min-samples-leaf 40 \
  --n-jobs 1 \
  --clip-correction-ms 2.0 \
  --scale-validation-start-utc "2025-10-01T00:00:00Z" \
  --scale-validation-end-utc "2026-01-01T00:00:00Z" \
  --scale-candidate 0.70 \
  --scale-candidate 0.80 \
  --scale-candidate 0.90 \
  --scale-candidate 0.95 \
  --scale-candidate 1.00 \
  --output-predictions "$BENCH_ROOT/$CAL_RUN/calibrated_predictions_2026.parquet" \
  --output-model "$BENCH_ROOT/$CAL_RUN/calibrator.joblib" \
  --output-json "$BENCH_ROOT/$CAL_RUN/calibration_results.json" \
  --output-md "$BENCH_ROOT/$CAL_RUN/calibration_results.md" || fail 40 "calibrator training failed"

log "writing 150k phys_v1 decision report"
"$PY" scripts/ml_dataset/summarize_phys_v1_decision_report.py \
  --run-suffix "phys_v1_150k_bin63" \
  --base-audit "$BENCH_ROOT/$BASE_RUN/tabular_rmse09_audit.json" \
  --calibration-results "$BENCH_ROOT/$CAL_RUN/calibration_results.json" \
  --feature-audit "$ML_ROOT/training_tables/phys_v1_required_feature_audit.json" \
  --signal-coverage "$ML_ROOT/training_tables/phys_v1_signal_coverage.json" \
  --predictions "$BENCH_ROOT/$CAL_RUN/calibrated_predictions_2026.parquet" \
  --output-json "$BENCH_ROOT/phys_v1_150k_bin63_decision_report.json" \
  --output-md "$BENCH_ROOT/phys_v1_150k_bin63_decision_report.md" || fail 50 "decision report failed"

"$PY" scripts/ml_dataset/summarize_ml_pipeline_status.py \
  --ml-root "$ML_ROOT" \
  --disk-path /srv/data \
  --output-json "$LOG_ROOT/ml_pipeline_status.json" \
  --output-md "$LOG_ROOT/ml_pipeline_status.md" || true

echo "complete $(date -Is)" > "$STATUS"
log "phys_v1 150k bin63 benchmark complete"
