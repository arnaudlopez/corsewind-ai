#!/usr/bin/env bash
set -euo pipefail

ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
REMOTE_ROOT="${REMOTE_ROOT:-/srv/data/corsewind/backfill_runner}"
BENCH_ROOT="${BENCH_ROOT:-$ML_ROOT/benchmarks}"
LOG_ROOT="${LOG_ROOT:-$ML_ROOT/run_logs}"
PY="${PY:-/home/z2/corsewind-ml-smoke/.venv/bin/python}"
RUN_SUFFIX="${RUN_SUFFIX:-phys_v1}"
STATUS="${STATUS:-$LOG_ROOT/phys_v1_decision_report_watcher.status}"
LOG="${LOG:-$LOG_ROOT/phys_v1_decision_report_watcher.log}"
POLL_SECONDS="${POLL_SECONDS:-120}"

BASE_RUN="${BASE_RUN:-tabular_lgbm_225k_prev_${RUN_SUFFIX}_2024_2025_to_2026_v1}"
CAL_RUN="${CAL_RUN:-prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_${RUN_SUFFIX}}"

BASE_AUDIT="${BASE_AUDIT:-$BENCH_ROOT/$BASE_RUN/tabular_rmse09_audit.json}"
CALIBRATION_RESULTS="${CALIBRATION_RESULTS:-$BENCH_ROOT/$CAL_RUN/calibration_results.json}"
PREDICTIONS="${PREDICTIONS:-$BENCH_ROOT/$CAL_RUN/calibrated_predictions_2026.parquet}"
FEATURE_AUDIT="${FEATURE_AUDIT:-$ML_ROOT/training_tables/phys_v1_required_feature_audit.json}"
SIGNAL_COVERAGE="${SIGNAL_COVERAGE:-$ML_ROOT/training_tables/phys_v1_signal_coverage.json}"
OUTPUT_JSON="${OUTPUT_JSON:-$BENCH_ROOT/phys_v1_decision_report.json}"
OUTPUT_MD="${OUTPUT_MD:-$BENCH_ROOT/phys_v1_decision_report.md}"

mkdir -p "$LOG_ROOT" "$BENCH_ROOT"

log() {
  echo "$(date -Is) $*" | tee -a "$LOG"
}

echo "started $(date -Is)" > "$STATUS"
log "phys_v1 decision report watcher started"

while true; do
  missing=()
  for path in "$BASE_AUDIT" "$CALIBRATION_RESULTS" "$PREDICTIONS" "$FEATURE_AUDIT" "$SIGNAL_COVERAGE"; do
    if [ ! -s "$path" ]; then
      missing+=("$path")
    fi
  done
  if [ "${#missing[@]}" -eq 0 ]; then
    break
  fi
  log "waiting for artifacts missing_count=${#missing[@]} first_missing=${missing[0]}"
  sleep "$POLL_SECONDS"
done

cd "$REMOTE_ROOT"
log "all artifacts present; writing phys_v1 decision report"
"$PY" scripts/ml_dataset/summarize_phys_v1_decision_report.py \
  --run-suffix "$RUN_SUFFIX" \
  --base-audit "$BASE_AUDIT" \
  --calibration-results "$CALIBRATION_RESULTS" \
  --feature-audit "$FEATURE_AUDIT" \
  --signal-coverage "$SIGNAL_COVERAGE" \
  --predictions "$PREDICTIONS" \
  --output-json "$OUTPUT_JSON" \
  --output-md "$OUTPUT_MD" | tee -a "$LOG"

echo "complete $(date -Is)" > "$STATUS"
log "phys_v1 decision report watcher complete"
