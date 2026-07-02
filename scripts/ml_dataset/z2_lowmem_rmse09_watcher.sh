#!/usr/bin/env bash
set -u

ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset_z2_rebuild}"
PREFIX="${PREFIX:-residual_windsup_sst_prev_fresh}"
REMOTE_ROOT="${REMOTE_ROOT:-/srv/data/corsewind/backfill_runner}"
LOG_ROOT="${LOG_ROOT:-/srv/data/corsewind/ml_dataset/run_logs}"
STATUS="${STATUS:-$LOG_ROOT/rmse09_fresh_lowmem.status}"
START_MONTH="${START_MONTH:-2024-01}"
END_MONTH="${END_MONTH:-2026-06}"
SWEEP_SUFFIX="${SWEEP_SUFFIX:-context_fresh_lowmem_v1}"

MAX_CUTOFFS_PER_SPOT="${MAX_CUTOFFS_PER_SPOT:-60}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-120000}"
MAX_TRAINING_FEATURES="${MAX_TRAINING_FEATURES:-900}"
CALIBRATOR_N_JOBS="${CALIBRATOR_N_JOBS:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CORSEWIND_SKLEARN_N_JOBS="$CALIBRATOR_N_JOBS"

mkdir -p "$LOG_ROOT"

echo "$(date -Is) lowmem watcher started"

is_rebuild_running() {
  python3 - <<'PY'
import os

needles = [
    "run_monthly_training_shards.py",
    "run_training_backfill_pipeline.py",
    "build_spot_feature_store.py",
    "build_residual_training_table.py",
    "export_training_table_parquet.py",
]
self_pid = os.getpid()
parent_pid = os.getppid()
for name in os.listdir("/proc"):
    if not name.isdigit():
        continue
    pid = int(name)
    if pid in {self_pid, parent_pid}:
        continue
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "ignore")
    except OSError:
        continue
    if any(needle in raw for needle in needles):
        print(pid)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

while is_rebuild_running > /tmp/corsewind_rebuild_pid 2> /dev/null; do
  echo "$(date -Is) rebuild still running pid=$(cat /tmp/corsewind_rebuild_pid)"
  sleep 120
done

echo "$(date -Is) rebuild finished; auditing fresh shards"
cd "$REMOTE_ROOT" || exit 10
"/home/z2/corsewind-ml-smoke/.venv/bin/python" scripts/ml_dataset/audit_training_table_features.py \
  --training-table-root "$ML_ROOT/training_tables" \
  --run-id-prefix "$PREFIX" \
  --start-month "$START_MONTH" \
  --end-month "$END_MONTH" \
  --output-json "$ML_ROOT/training_tables/fresh_full_feature_audit.json" \
  --output-md "$ML_ROOT/training_tables/fresh_full_feature_audit.md" \
  --fail-on-non-pass
audit_code=$?
if [ "$audit_code" -ne 0 ]; then
  echo "$audit_code" > "$STATUS"
  echo "$(date -Is) fresh audit failed code=$audit_code"
  exit "$audit_code"
fi

echo "$(date -Is) fresh audit passed; auditing calibrator feature selection"
"/home/z2/corsewind-ml-smoke/.venv/bin/python" scripts/ml_dataset/audit_calibrator_feature_selection.py \
  --training-table-root "$ML_ROOT/training_tables" \
  --run-id-prefix "$PREFIX" \
  --start-month "$START_MONTH" \
  --end-month "$END_MONTH" \
  --max-training-features "$MAX_TRAINING_FEATURES" \
  --output-json "$ML_ROOT/training_tables/calibrator_feature_selection_audit.json" \
  --output-md "$ML_ROOT/training_tables/calibrator_feature_selection_audit.md" \
  --fail-on-missing-required
selection_code=$?
if [ "$selection_code" -ne 0 ]; then
  echo "$selection_code" > "$STATUS"
  echo "$(date -Is) calibrator feature selection audit failed code=$selection_code"
  exit "$selection_code"
fi

echo "$(date -Is) feature selection audit passed; launching RMSE09 lowmem"
set +e
"/home/z2/corsewind-ml-smoke/.venv/bin/python" scripts/ml_dataset/run_rmse09_sequence_experiment.py \
  --repo-root "$REMOTE_ROOT" \
  --ml-root "$ML_ROOT" \
  --training-run-id-prefix "$PREFIX" \
  --sweep-suffix "$SWEEP_SUFFIX" \
  --require-fresh-training-features \
  --assert-goal \
  --max-cutoffs-per-spot "$MAX_CUTOFFS_PER_SPOT" \
  --batch-size "$BATCH_SIZE" \
  --max-train-rows "$MAX_TRAIN_ROWS" \
  --max-training-features "$MAX_TRAINING_FEATURES" \
  --calibrator-n-jobs "$CALIBRATOR_N_JOBS" \
  --calibrator-model-family ridge \
  --calibrator-model-family hist_gradient_boosting \
  --force
code=$?
set -e
echo "$code" > "$STATUS"
echo "$(date -Is) RMSE09 lowmem finished code=$code"
exit "$code"
