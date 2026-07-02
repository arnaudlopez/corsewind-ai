#!/usr/bin/env bash
set -euo pipefail

ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
PREFIX="${PREFIX:-residual_windsup_sst_prev_regime_v1}"
REMOTE_ROOT="${REMOTE_ROOT:-/srv/data/corsewind/backfill_runner}"
LOG_ROOT="${LOG_ROOT:-$ML_ROOT/run_logs}"
BENCH_ROOT="${BENCH_ROOT:-$ML_ROOT/benchmarks}"
START_MONTH="${START_MONTH:-2024-01}"
END_MONTH="${END_MONTH:-2026-06}"
PY="${PY:-/home/z2/corsewind-ml-smoke/.venv/bin/python}"
RUN_SUFFIX="${RUN_SUFFIX:-relief_active_v1}"
STATUS="${STATUS:-$LOG_ROOT/regime_v1_post_rebuild_lowmem.status}"
MEMORY_MIN_AVAILABLE_KB="${MEMORY_MIN_AVAILABLE_KB:-2200000}"
MEMORY_MAX_RSS_KB="${MEMORY_MAX_RSS_KB:-13000000}"
POLL_SECONDS="${POLL_SECONDS:-120}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CORSEWIND_SKLEARN_N_JOBS="${CORSEWIND_SKLEARN_N_JOBS:-1}"

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

is_rebuild_running() {
  "$PY" - <<'PY'
import os

needles = (
    "run_monthly_training_shards.py",
    "run_training_backfill_pipeline.py",
    "build_spot_feature_store.py",
    "build_residual_training_table.py",
    "export_training_table_parquet.py",
)
self_pid = os.getpid()
parent_pid = os.getppid()
for raw_name in os.listdir("/proc"):
    if not raw_name.isdigit():
        continue
    pid = int(raw_name)
    if pid in {self_pid, parent_pid}:
        continue
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "ignore")
    except OSError:
        continue
    if "z2_regime_v1_post_rebuild_lowmem_watcher.sh" in raw:
        continue
    if "setsid bash -lc" in raw and "rebuild_training_shards.log" in raw:
        continue
    if any(needle in raw for needle in needles):
        print(pid)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

month_args() {
  "$PY" - "$START_MONTH" "$END_MONTH" <<'PY'
import sys
start_year, start_month = map(int, sys.argv[1].split("-"))
end_year, end_month = map(int, sys.argv[2].split("-"))
year, month = start_year, start_month
out = []
while (year, month) <= (end_year, end_month):
    out.extend(["--month", f"{year:04d}-{month:02d}"])
    month += 1
    if month == 13:
        year += 1
        month = 1
print(" ".join(out))
PY
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
    --lightgbm-max-bin 127 \
    --lightgbm-feature-fraction 0.85 \
    --lightgbm-bagging-fraction 0.85 \
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
  local audit_name="$2"
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
    --output-json "$output/$audit_name.json" \
    --output-md "$output/$audit_name.md" || true
}

log "post-rebuild lowmem watcher started ml_root=$ML_ROOT prefix=$PREFIX run_suffix=$RUN_SUFFIX"
echo "started $(date -Is)" > "$STATUS"

while is_rebuild_running > /tmp/corsewind_regime_v1_rebuild_pid 2>/dev/null; do
  log "rebuild still running pid=$(cat /tmp/corsewind_regime_v1_rebuild_pid)"
  sleep "$POLL_SECONDS"
done

cd "$REMOTE_ROOT" || fail 10 "cannot cd remote root"

log "rebuild finished; auditing required feature families"
"$PY" scripts/ml_dataset/audit_training_table_features.py \
  --training-table-root "$ML_ROOT/training_tables" \
  --run-id-prefix "$PREFIX" \
  --start-month "$START_MONTH" \
  --end-month "$END_MONTH" \
  --required-pattern features__previous_run_open_meteo_best_match_day1_wind_speed_10m \
  --required-pattern features__previous_run_open_meteo_best_match_day2_wind_speed_10m \
  --required-pattern features__sst_c \
  --required-pattern features__context_global_relief_1_available \
  --required-pattern features__context_global_relief_1_wind_mean_ms \
  --required-pattern features__context_global_relief_1_temperature_c \
  --required-pattern features__thermal_land_minus_sst_c \
  --required-pattern features__thermal_inland_minus_coastal_temperature_c \
  --required-pattern features__eumetsat_land_surface_temperature_available \
  --output-json "$ML_ROOT/training_tables/regime_v1_post_rebuild_feature_audit.json" \
  --output-md "$ML_ROOT/training_tables/regime_v1_post_rebuild_feature_audit.md" \
  --fail-on-non-pass || fail 11 "feature audit failed"

log "auditing full relief coverage on critical spots"
# shellcheck disable=SC2046
"$PY" scripts/ml_dataset/audit_relief_context_coverage.py \
  --training-table-root "$ML_ROOT/training_tables" \
  --run-id-prefix "$PREFIX" \
  $(month_args) \
  --output-json "$BENCH_ROOT/relief_context_coverage_after_window_filter_v1/coverage_${START_MONTH}_${END_MONTH}.json" \
  --output-md "$BENCH_ROOT/relief_context_coverage_after_window_filter_v1/coverage_${START_MONTH}_${END_MONTH}.md" || fail 12 "relief coverage audit failed"

BASE_RUN="tabular_lgbm_225k_prev_${RUN_SUFFIX}_2024_2025_to_2026_v1"
CALBASE_RUN="tabular_lgbm_calbase_${RUN_SUFFIX}_2024_to_2025h2_v1"
CAL_RUN="prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_${RUN_SUFFIX}"

log "training comparable 225k LightGBM base on active-relief shards"
train_lgbm_base "$BASE_RUN" "2026-01-01T00:00:00Z" "$START_MONTH" "$END_MONTH" 225000 60000 || fail 20 "base training failed"
analyze_base "$BASE_RUN" "tabular_rmse09_audit"

log "training 2025-H2 calibration base"
train_lgbm_base "$CALBASE_RUN" "2025-07-01T00:00:00Z" "2024-01" "2025-12" 150000 60000 || fail 21 "calibration base training failed"
analyze_base "$CALBASE_RUN" "tabular_rmse09_audit"
cp "$BENCH_ROOT/$CALBASE_RUN/tabular_holdout_predictions.parquet" "$BENCH_ROOT/$CALBASE_RUN/calibration_predictions_2025h2.parquet"

log "training ExtraTrees temporal calibrator with scale selected on 2025-Q4"
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
  --output-md "$BENCH_ROOT/$CAL_RUN/calibration_results.md" || fail 30 "temporal calibrator failed"

log "selecting best tabular result and running gap audit"
"$PY" scripts/ml_dataset/select_tabular_rmse09_result.py \
  --search-root "$BENCH_ROOT" \
  --output-json "$BENCH_ROOT/tabular_rmse09_selection/tabular_selection_latest.json" \
  --output-md "$BENCH_ROOT/tabular_rmse09_selection/tabular_selection_latest.md" || true

"$PY" scripts/ml_dataset/analyze_rmse09_gap_oracles.py \
  --predictions "$BENCH_ROOT/$CAL_RUN/calibrated_predictions_2026.parquet" \
  --prediction-column calibrated_wind_mean_ms \
  --model "post_relief_base225|$BENCH_ROOT/$BASE_RUN/tabular_holdout_predictions.parquet|corrected_wind_mean_ms" \
  --model "post_relief_calibrated|$BENCH_ROOT/$CAL_RUN/calibrated_predictions_2026.parquet|calibrated_wind_mean_ms" \
  --critical-spot la_tonnara \
  --critical-spot santa_manza \
  --critical-spot balistra \
  --output-json "$BENCH_ROOT/rmse09_gap_audit_post_relief_${RUN_SUFFIX}/gap_audit.json" \
  --output-md "$BENCH_ROOT/rmse09_gap_audit_post_relief_${RUN_SUFFIX}/gap_audit.md" || true

"$PY" scripts/ml_dataset/summarize_post_relief_rmse09_iteration.py \
  --base-audit "$BENCH_ROOT/$BASE_RUN/tabular_rmse09_audit.json" \
  --calibration-results "$BENCH_ROOT/$CAL_RUN/calibration_results.json" \
  --relief-coverage "$BENCH_ROOT/relief_context_coverage_after_window_filter_v1/coverage_${START_MONTH}_${END_MONTH}.json" \
  --gap-audit "$BENCH_ROOT/rmse09_gap_audit_post_relief_${RUN_SUFFIX}/gap_audit.json" \
  --output-json "$BENCH_ROOT/post_relief_iteration_summary_${RUN_SUFFIX}.json" \
  --output-md "$BENCH_ROOT/post_relief_iteration_summary_${RUN_SUFFIX}.md" || true

echo "0" > "$STATUS"
log "post-rebuild lowmem watcher finished"
