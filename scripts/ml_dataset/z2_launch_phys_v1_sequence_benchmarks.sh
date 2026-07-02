#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/srv/data/corsewind/backfill_runner}"
ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
PY="${PY:-/home/z2/corsewind-ml-smoke/.venv/bin/python}"
PREFIX="${PREFIX:-residual_windsup_sst_prev_phys_v1}"
RUN_SUFFIX="${RUN_SUFFIX:-phys_v1_sequence_v1}"
MAX_CUTOFFS_PER_SPOT="${MAX_CUTOFFS_PER_SPOT:-120}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CALIBRATOR_N_JOBS="${CALIBRATOR_N_JOBS:-1}"
INCLUDE_MOIRAI="${INCLUDE_MOIRAI:-0}"
INCLUDE_LIGHTGBM="${INCLUDE_LIGHTGBM:-0}"
REQUIRE_PHYS_V1_DONE="${REQUIRE_PHYS_V1_DONE:-1}"
PHYS_V1_DECISION_STATUS="${PHYS_V1_DECISION_STATUS:-$ML_ROOT/run_logs/phys_v1_decision_report_watcher.status}"

cd "$REMOTE_ROOT"

if [ "$REQUIRE_PHYS_V1_DONE" = "1" ]; then
  if ! grep -q '^complete ' "$PHYS_V1_DECISION_STATUS" 2>/dev/null; then
    echo "phys_v1 decision report is not complete yet: $PHYS_V1_DECISION_STATUS" >&2
    exit 70
  fi
fi

cmd=(
  "$PY" scripts/ml_dataset/run_rmse09_sequence_experiment.py
  --repo-root "$REMOTE_ROOT"
  --ml-root "$ML_ROOT"
  --training-run-id-prefix "$PREFIX"
  --max-cutoffs-per-spot "$MAX_CUTOFFS_PER_SPOT"
  --batch-size "$BATCH_SIZE"
  --calibrator-n-jobs "$CALIBRATOR_N_JOBS"
  --sweep-suffix "$RUN_SUFFIX"
)

if [ "$INCLUDE_MOIRAI" = "1" ]; then
  cmd+=(--include-moirai)
fi
if [ "$INCLUDE_LIGHTGBM" = "1" ]; then
  cmd+=(--include-lightgbm)
fi

log_root="$ML_ROOT/run_logs"
mkdir -p "$log_root"
log="$log_root/phys_v1_sequence_benchmarks.log"
pid_file="$log_root/phys_v1_sequence_benchmarks.pid"
status="$log_root/phys_v1_sequence_benchmarks.status"
rm -f "$status" "$pid_file"

setsid bash -lc "$(printf '%q ' "${cmd[@]}") ; echo \$? > $(printf '%q' "$status")" > "$log" 2>&1 < /dev/null &
echo $! > "$pid_file"
echo "pid=$(cat "$pid_file") log=$log status=$status"
