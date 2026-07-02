#!/usr/bin/env bash
set -euo pipefail

ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
PREFIX="${PREFIX:-residual_windsup_sst_prev_phys_v3_dem_fetch}"
RUN_SUFFIX="${RUN_SUFFIX:-phys_v3_dem_fetch_150k_bin63}"
REMOTE_ROOT="${REMOTE_ROOT:-/srv/data/corsewind/backfill_runner}"
LOG_ROOT="${LOG_ROOT:-$ML_ROOT/run_logs}"
BENCH_ROOT="${BENCH_ROOT:-$ML_ROOT/benchmarks}"
PY="${PY:-/home/z2/corsewind-ml-smoke/.venv/bin/python}"
STATUS="${STATUS:-$LOG_ROOT/${RUN_SUFFIX}_wind_gust_benchmark.status}"
MEMORY_MIN_AVAILABLE_KB="${MEMORY_MIN_AVAILABLE_KB:-2500000}"
MEMORY_MAX_RSS_KB="${MEMORY_MAX_RSS_KB:-28000000}"
START_MONTH="${START_MONTH:-2024-01}"
END_MONTH="${END_MONTH:-2026-06}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-150000}"
MAX_TEST_ROWS="${MAX_TEST_ROWS:-60000}"
CAL_MAX_TRAIN_ROWS="${CAL_MAX_TRAIN_ROWS:-$MAX_TRAIN_ROWS}"
CAL_MAX_TEST_ROWS="${CAL_MAX_TEST_ROWS:-$MAX_TEST_ROWS}"
MAX_ITER="${MAX_ITER:-180}"
LEARNING_RATE="${LEARNING_RATE:-0.04}"
MAX_LEAF_NODES="${MAX_LEAF_NODES:-31}"
MIN_SAMPLES_LEAF="${MIN_SAMPLES_LEAF:-20}"
LIGHTGBM_MAX_BIN="${LIGHTGBM_MAX_BIN:-63}"
LIGHTGBM_FEATURE_FRACTION="${LIGHTGBM_FEATURE_FRACTION:-0.8}"
LIGHTGBM_BAGGING_FRACTION="${LIGHTGBM_BAGGING_FRACTION:-0.8}"
LIGHTGBM_BAGGING_FREQ="${LIGHTGBM_BAGGING_FREQ:-1}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CORSEWIND_SKLEARN_N_JOBS="${CORSEWIND_SKLEARN_N_JOBS:-1}"

BASE_RUN="tabular_lgbm_${RUN_SUFFIX}_2024_2025_to_2026_v1"
CALBASE_RUN="tabular_lgbm_calbase_${RUN_SUFFIX}_2024_to_2025h2_v1"
WIND_CAL_RUN="prediction_residual_calibrator_${RUN_SUFFIX}_wind_mean_2025h2_to_2026_v1"
GUST_CAL_RUN="prediction_residual_calibrator_${RUN_SUFFIX}_gust_2025h2_to_2026_v1"
SUMMARY_JSON="$BENCH_ROOT/${RUN_SUFFIX}_wind_gust_decision_report.json"
SUMMARY_MD="$BENCH_ROOT/${RUN_SUFFIX}_wind_gust_decision_report.md"

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

model_file_for_target() {
  case "$1" in
    wind_mean) echo "labels__residual_wind_mean_ms.joblib" ;;
    gust) echo "labels__residual_gust_ms.joblib" ;;
    *) fail 2 "unknown target $1" ;;
  esac
}

label_for_target() {
  case "$1" in
    wind_mean) echo "labels__residual_wind_mean_ms" ;;
    gust) echo "labels__residual_gust_ms" ;;
    *) fail 2 "unknown target $1" ;;
  esac
}

train_lgbm_base() {
  local run_id="$1"
  local split="$2"
  local start_month="$3"
  local end_month="$4"
  local max_train_rows="$5"
  local max_test_rows="$6"
  local output="$BENCH_ROOT/$run_id"
  local extra_args=()
  if [ -n "$TRAIN_EXTRA_ARGS" ]; then
    read -r -a extra_args <<< "$TRAIN_EXTRA_ARGS"
  fi
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
    --max-iter "$MAX_ITER" \
    --learning-rate "$LEARNING_RATE" \
    --max-leaf-nodes "$MAX_LEAF_NODES" \
    --min-samples-leaf "$MIN_SAMPLES_LEAF" \
    --n-jobs 1 \
    --lightgbm-max-bin "$LIGHTGBM_MAX_BIN" \
    --lightgbm-feature-fraction "$LIGHTGBM_FEATURE_FRACTION" \
    --lightgbm-bagging-fraction "$LIGHTGBM_BAGGING_FRACTION" \
    --lightgbm-bagging-freq "$LIGHTGBM_BAGGING_FREQ" \
    --lightgbm-force-col-wise \
    --skip-classification \
    --eval-lead-minute 15 \
    --eval-lead-minute 30 \
    --eval-lead-minute 45 \
    --eval-lead-minute 60 \
    "${extra_args[@]}"
}

analyze_target() {
  local run_id="$1"
  local target="$2"
  local output="$BENCH_ROOT/$run_id"
  local label
  local model_file
  label="$(label_for_target "$target")"
  model_file="$(model_file_for_target "$target")"
  "$PY" scripts/ml_dataset/analyze_tabular_rmse09_errors.py \
    --training-results "$output/training_results.json" \
    --feature-columns "$output/feature_columns.json" \
    --target "$label" \
    --model-path "$output/$model_file" \
    --metric-lead-minute 15 \
    --metric-lead-minute 30 \
    --metric-lead-minute 45 \
    --metric-lead-minute 60 \
    --output-predictions "$output/tabular_holdout_predictions_${target}.parquet" \
    --output-json "$output/tabular_error_diagnosis_${target}.json" \
    --output-md "$output/tabular_error_diagnosis_${target}.md"
}

train_calibrator() {
  local target="$1"
  local cal_run="$2"
  local cal_root="$BENCH_ROOT/$cal_run"
  mkdir -p "$cal_root"
  "$PY" scripts/ml_dataset/train_prediction_residual_calibrator.py \
    --target "$target" \
    --calibration-predictions "$BENCH_ROOT/$CALBASE_RUN/tabular_holdout_predictions_${target}.parquet" \
    --evaluation-predictions "$BENCH_ROOT/$BASE_RUN/tabular_holdout_predictions_${target}.parquet" \
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
    --scale-candidate 0.50 \
    --scale-candidate 0.60 \
    --scale-candidate 0.70 \
    --scale-candidate 0.80 \
    --scale-candidate 0.90 \
    --scale-candidate 1.00 \
    --output-predictions "$cal_root/calibrated_predictions_2026.parquet" \
    --output-model "$cal_root/calibrator.joblib" \
    --output-json "$cal_root/calibration_results.json" \
    --output-md "$cal_root/calibration_results.md"
}

write_summary() {
  "$PY" - "$BENCH_ROOT" "$RUN_SUFFIX" "$BASE_RUN" "$WIND_CAL_RUN" "$GUST_CAL_RUN" "$SUMMARY_JSON" "$SUMMARY_MD" <<'PY'
import json
import sys
from pathlib import Path

bench_root = Path(sys.argv[1])
run_suffix, base_run, wind_cal_run, gust_cal_run = sys.argv[2:6]
summary_json = Path(sys.argv[6])
summary_md = Path(sys.argv[7])

def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)

base = load(bench_root / base_run / "training_results.json")
wind_diag = load(bench_root / base_run / "tabular_error_diagnosis_wind_mean.json")
gust_diag = load(bench_root / base_run / "tabular_error_diagnosis_gust.json")
wind_cal = load(bench_root / wind_cal_run / "calibration_results.json")
gust_cal = load(bench_root / gust_cal_run / "calibration_results.json")

result = {
    "format": "corsewind.wind_gust_benchmark_decision.v1",
    "run_suffix": run_suffix,
    "base_run": base_run,
    "wind_calibrator_run": wind_cal_run,
    "gust_calibrator_run": gust_cal_run,
    "source_parquet_count": len(base.get("source_parquets") or []),
    "wind_mean": {
        "base_raw": wind_diag["overall"]["raw"],
        "base_corrected": wind_diag["overall"]["corrected"],
        "calibrated": wind_cal["calibrated_metrics"],
        "calibrator_gain_pct_vs_base": wind_cal.get("rmse_gain_pct_vs_base"),
        "selected_scale": (wind_cal.get("scale_selection") or {}).get("selected_scale"),
    },
    "gust": {
        "base_raw": gust_diag["overall"]["raw"],
        "base_corrected": gust_diag["overall"]["corrected"],
        "calibrated": gust_cal["calibrated_metrics"],
        "calibrator_gain_pct_vs_base": gust_cal.get("rmse_gain_pct_vs_base"),
        "selected_scale": (gust_cal.get("scale_selection") or {}).get("selected_scale"),
    },
}
summary_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# Wind + Gust Benchmark Decision Report",
    "",
    f"Run suffix: `{run_suffix}`",
    f"Base run: `{base_run}`",
    "",
    "| Target | Raw RMSE | Corrected RMSE | Calibrated RMSE | Calibrated MAE | Bias | Scale | Gain vs corrected |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for key, label in (("wind_mean", "wind mean"), ("gust", "gust")):
    item = result[key]
    lines.append(
        f"| `{label}` | {item['base_raw'].get('rmse')} | {item['base_corrected'].get('rmse')} | "
        f"{item['calibrated'].get('rmse')} | {item['calibrated'].get('mae')} | "
        f"{item['calibrated'].get('bias')} | {item.get('selected_scale')} | "
        f"{item.get('calibrator_gain_pct_vs_base')}% |"
    )
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
PY
}

cd "$REMOTE_ROOT" || fail 10 "cannot cd remote root"
echo "started $(date -Is)" > "$STATUS"
log "wind+gust benchmark started prefix=$PREFIX run_suffix=$RUN_SUFFIX max_train_rows=$MAX_TRAIN_ROWS max_test_rows=$MAX_TEST_ROWS train_extra_args=$TRAIN_EXTRA_ARGS"

log "training LightGBM base for wind mean and gust"
train_lgbm_base "$BASE_RUN" "2026-01-01T00:00:00Z" "$START_MONTH" "$END_MONTH" "$MAX_TRAIN_ROWS" "$MAX_TEST_ROWS" || fail 20 "base training failed"
analyze_target "$BASE_RUN" wind_mean || fail 21 "base wind analysis failed"
analyze_target "$BASE_RUN" gust || fail 22 "base gust analysis failed"

log "training 2025-H2 calibration base for wind mean and gust"
train_lgbm_base "$CALBASE_RUN" "2025-07-01T00:00:00Z" "$START_MONTH" "2025-12" "$CAL_MAX_TRAIN_ROWS" "$CAL_MAX_TEST_ROWS" || fail 30 "calibration base training failed"
analyze_target "$CALBASE_RUN" wind_mean || fail 31 "calibration wind analysis failed"
analyze_target "$CALBASE_RUN" gust || fail 32 "calibration gust analysis failed"

log "training wind mean second-stage calibrator"
train_calibrator wind_mean "$WIND_CAL_RUN" || fail 40 "wind calibrator failed"

log "training gust second-stage calibrator"
train_calibrator gust "$GUST_CAL_RUN" || fail 41 "gust calibrator failed"

log "writing wind+gust decision report"
write_summary || fail 50 "summary failed"

"$PY" scripts/ml_dataset/summarize_ml_pipeline_status.py \
  --ml-root "$ML_ROOT" \
  --disk-path /srv/data \
  --output-json "$LOG_ROOT/ml_pipeline_status.json" \
  --output-md "$LOG_ROOT/ml_pipeline_status.md" || true

echo "complete $(date -Is)" > "$STATUS"
log "wind+gust 150k bin63 benchmark complete"
