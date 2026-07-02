#!/usr/bin/env bash
set -euo pipefail

ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
REMOTE_ROOT="${REMOTE_ROOT:-/srv/data/corsewind/backfill_runner}"
LOG_ROOT="${LOG_ROOT:-$ML_ROOT/run_logs}"
BENCH_ROOT="${BENCH_ROOT:-$ML_ROOT/benchmarks}"
PY="${PY:-/home/z2/corsewind-ml-smoke/.venv/bin/python}"
START_MONTH="${START_MONTH:-2024-01}"
END_MONTH="${END_MONTH:-2026-06}"
SOURCE_PREFIX="${SOURCE_PREFIX:-residual_windsup_sst_prev_phys_v3_dem_fetch}"
V4_PREFIX="${V4_PREFIX:-residual_windsup_sst_prev_phys_v4_directional}"
STATUS="${STATUS:-$LOG_ROOT/phys_v3_ablation_and_v4_directional.status}"
LOG="${LOG:-$LOG_ROOT/phys_v3_ablation_and_v4_directional.log}"
SUMMARY_JSON="${SUMMARY_JSON:-$BENCH_ROOT/phys_v3_ablation_and_v4_directional_summary.json}"
SUMMARY_MD="${SUMMARY_MD:-$BENCH_ROOT/phys_v3_ablation_and_v4_directional_summary.md}"

OLD_SIGNAL_SUFFIX="${OLD_SIGNAL_SUFFIX:-phys_v3_old_signal_225k_bin63}"
PRUNED_SUFFIX="${PRUNED_SUFFIX:-phys_v3_pruned_200k_bin63}"
V4_SUFFIX="${V4_SUFFIX:-phys_v4_directional_pruned_200k_bin63}"

OLD_SIGNAL_MAX_TRAIN_ROWS="${OLD_SIGNAL_MAX_TRAIN_ROWS:-225000}"
OLD_SIGNAL_MAX_TEST_ROWS="${OLD_SIGNAL_MAX_TEST_ROWS:-60000}"
PRUNED_MAX_TRAIN_ROWS="${PRUNED_MAX_TRAIN_ROWS:-200000}"
PRUNED_MAX_TEST_ROWS="${PRUNED_MAX_TEST_ROWS:-60000}"
V4_MAX_TRAIN_ROWS="${V4_MAX_TRAIN_ROWS:-200000}"
V4_MAX_TEST_ROWS="${V4_MAX_TEST_ROWS:-60000}"

OLD_SIGNAL_TRAIN_EXTRA_ARGS="${OLD_SIGNAL_TRAIN_EXTRA_ARGS:---exclude-feature-pattern features__spot_static_ --exclude-feature-pattern features__nwp_offset_ --exclude-feature-pattern features__previous_run_ --exclude-feature-pattern features__open_meteo_vertical_ --exclude-feature-pattern features__eumetsat_}"
PRUNED_TRAIN_EXTRA_ARGS="${PRUNED_TRAIN_EXTRA_ARGS:---exclude-feature-pattern features__spot_static_fetch_sector_ --exclude-feature-pattern features__spot_static_dem_sector_ --exclude-feature-pattern features__eumetsat_}"
V4_TRAIN_EXTRA_ARGS="${V4_TRAIN_EXTRA_ARGS:---exclude-feature-pattern features__spot_static_fetch_sector_ --exclude-feature-pattern features__spot_static_dem_sector_ --exclude-feature-pattern features__eumetsat_}"

CHAMPION_WIND_RMSE="${CHAMPION_WIND_RMSE:-1.268019}"
CHAMPION_WIND_MAE="${CHAMPION_WIND_MAE:-0.930465}"

mkdir -p "$LOG_ROOT" "$BENCH_ROOT"

log() {
  echo "$(date -Is) $*" | tee -a "$LOG"
}

fail() {
  local code="$1"
  shift
  echo "failed code=$code $(date -Is)" > "$STATUS"
  log "failed code=$code $*"
  exit "$code"
}

run_benchmark() {
  local label="$1"
  local prefix="$2"
  local suffix="$3"
  local max_train_rows="$4"
  local max_test_rows="$5"
  local extra_args="$6"
  log "benchmark $label prefix=$prefix suffix=$suffix max_train_rows=$max_train_rows extra_args=$extra_args"
  PREFIX="$prefix" \
    RUN_SUFFIX="$suffix" \
    MAX_TRAIN_ROWS="$max_train_rows" \
    MAX_TEST_ROWS="$max_test_rows" \
    TRAIN_EXTRA_ARGS="$extra_args" \
    bash scripts/ml_dataset/z2_run_wind_gust_150k_bin63_benchmark.sh
}

write_summary() {
  "$PY" - "$BENCH_ROOT" "$SUMMARY_JSON" "$SUMMARY_MD" "$CHAMPION_WIND_RMSE" "$CHAMPION_WIND_MAE" \
    "$OLD_SIGNAL_SUFFIX" "$PRUNED_SUFFIX" "$V4_SUFFIX" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

bench_root = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
summary_md = Path(sys.argv[3])
champion_rmse = float(sys.argv[4])
champion_mae = float(sys.argv[5])
suffixes = sys.argv[6:]

def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)

def metric(path, suffix):
    if not path.exists():
        return {"run_suffix": suffix, "status": "missing", "path": str(path)}
    payload = load(path)
    wind = payload.get("wind_mean") or {}
    gust = payload.get("gust") or {}
    wind_cal = wind.get("calibrated") or {}
    gust_cal = gust.get("calibrated") or {}
    return {
        "run_suffix": suffix,
        "status": "complete",
        "path": str(path),
        "wind_mean": {
            "raw_rmse": (wind.get("base_raw") or {}).get("rmse"),
            "corrected_rmse": (wind.get("base_corrected") or {}).get("rmse"),
            "calibrated_rmse": wind_cal.get("rmse"),
            "calibrated_mae": wind_cal.get("mae"),
            "calibrated_bias": wind_cal.get("bias"),
            "delta_vs_champion_rmse": None if wind_cal.get("rmse") is None else round(float(wind_cal["rmse"]) - champion_rmse, 6),
            "delta_vs_champion_mae": None if wind_cal.get("mae") is None else round(float(wind_cal["mae"]) - champion_mae, 6),
            "selected_scale": wind.get("selected_scale"),
        },
        "gust": {
            "raw_rmse": (gust.get("base_raw") or {}).get("rmse"),
            "corrected_rmse": (gust.get("base_corrected") or {}).get("rmse"),
            "calibrated_rmse": gust_cal.get("rmse"),
            "calibrated_mae": gust_cal.get("mae"),
            "calibrated_bias": gust_cal.get("bias"),
            "selected_scale": gust.get("selected_scale"),
        },
    }

runs = [metric(bench_root / f"{suffix}_wind_gust_decision_report.json", suffix) for suffix in suffixes]
valid_wind = [
    item for item in runs
    if item.get("status") == "complete" and item.get("wind_mean", {}).get("calibrated_rmse") is not None
]
best = min(valid_wind, key=lambda item: item["wind_mean"]["calibrated_rmse"], default=None)
payload = {
    "format": "corsewind.phys_v3_ablation_and_v4_directional_summary.v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "champion": {
        "wind_mean_rmse": champion_rmse,
        "wind_mean_mae": champion_mae,
    },
    "best_wind_mean_run": None if best is None else best["run_suffix"],
    "decision": (
        "missing_results"
        if best is None
        else "candidate_beats_champion"
        if best["wind_mean"]["calibrated_rmse"] < champion_rmse
        else "keep_champion"
    ),
    "runs": runs,
}
summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# phys_v3 Ablation and phys_v4 Directional Summary",
    "",
    f"Champion wind mean RMSE: `{champion_rmse}`",
    f"Champion wind mean MAE: `{champion_mae}`",
    f"Decision: `{payload['decision']}`",
    "",
    "| Run | Wind RMSE | Wind MAE | Delta RMSE | Gust RMSE | Gust MAE |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for item in runs:
    wind = item.get("wind_mean") or {}
    gust = item.get("gust") or {}
    lines.append(
        f"| `{item['run_suffix']}` | {wind.get('calibrated_rmse')} | {wind.get('calibrated_mae')} | "
        f"{wind.get('delta_vs_champion_rmse')} | {gust.get('calibrated_rmse')} | {gust.get('calibrated_mae')} |"
    )
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

cd "$REMOTE_ROOT" || fail 10 "cannot cd remote root"
: > "$LOG"
echo "started $(date -Is)" > "$STATUS"
log "phys_v3 ablation and phys_v4 directional campaign started"

log "augmenting phys_v4 directional shards from $SOURCE_PREFIX to $V4_PREFIX"
"$PY" scripts/ml_dataset/augment_directional_static_features.py \
  --training-table-root "$ML_ROOT/training_tables" \
  --source-run-id-prefix "$SOURCE_PREFIX" \
  --output-run-id-prefix "$V4_PREFIX" \
  --start-month "$START_MONTH" \
  --end-month "$END_MONTH" \
  --overwrite || fail 20 "directional augmentation failed"

run_benchmark "phys_v3_old_signal" "$SOURCE_PREFIX" "$OLD_SIGNAL_SUFFIX" "$OLD_SIGNAL_MAX_TRAIN_ROWS" "$OLD_SIGNAL_MAX_TEST_ROWS" "$OLD_SIGNAL_TRAIN_EXTRA_ARGS" \
  || fail 30 "old-signal benchmark failed"

run_benchmark "phys_v3_pruned" "$SOURCE_PREFIX" "$PRUNED_SUFFIX" "$PRUNED_MAX_TRAIN_ROWS" "$PRUNED_MAX_TEST_ROWS" "$PRUNED_TRAIN_EXTRA_ARGS" \
  || fail 40 "pruned benchmark failed"

run_benchmark "phys_v4_directional_pruned" "$V4_PREFIX" "$V4_SUFFIX" "$V4_MAX_TRAIN_ROWS" "$V4_MAX_TEST_ROWS" "$V4_TRAIN_EXTRA_ARGS" \
  || fail 50 "phys_v4 directional benchmark failed"

log "writing comparative summary"
write_summary || fail 60 "summary failed"

echo "complete $(date -Is)" > "$STATUS"
log "phys_v3 ablation and phys_v4 directional campaign complete"
