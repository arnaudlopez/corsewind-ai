#!/usr/bin/env bash
set -euo pipefail

ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
REMOTE_ROOT="${REMOTE_ROOT:-/srv/data/corsewind/backfill_runner}"
LOG_ROOT="${LOG_ROOT:-$ML_ROOT/run_logs}"
BENCH_ROOT="${BENCH_ROOT:-$ML_ROOT/benchmarks}"
PY="${PY:-/home/z2/corsewind-ml-smoke/.venv/bin/python}"
SOURCE_PREFIX="${SOURCE_PREFIX:-residual_windsup_sst_prev_phys_v3_dem_fetch}"
V4_PREFIX="${V4_PREFIX:-residual_windsup_sst_prev_phys_v4_directional}"
STATUS="${STATUS:-$LOG_ROOT/pruned_v4_directional_150k_resume.status}"
LOG="${LOG:-$LOG_ROOT/pruned_v4_directional_150k_resume.log}"
SUMMARY_JSON="${SUMMARY_JSON:-$BENCH_ROOT/pruned_v4_directional_150k_resume_summary.json}"
SUMMARY_MD="${SUMMARY_MD:-$BENCH_ROOT/pruned_v4_directional_150k_resume_summary.md}"

OLD_SIGNAL_SUFFIX="${OLD_SIGNAL_SUFFIX:-phys_v3_old_signal_225k_bin63}"
PRUNED_200K_SUFFIX="${PRUNED_200K_SUFFIX:-phys_v3_pruned_200k_bin63}"
PRUNED_SUFFIX="${PRUNED_SUFFIX:-phys_v3_pruned_150k_bin63}"
V4_SUFFIX="${V4_SUFFIX:-phys_v4_directional_pruned_150k_bin63}"

PRUNED_TRAIN_EXTRA_ARGS="${PRUNED_TRAIN_EXTRA_ARGS:---exclude-feature-pattern features__spot_static_fetch_sector_ --exclude-feature-pattern features__spot_static_dem_sector_ --exclude-feature-pattern features__eumetsat_}"
V4_TRAIN_EXTRA_ARGS="${V4_TRAIN_EXTRA_ARGS:---exclude-feature-pattern features__spot_static_fetch_sector_ --exclude-feature-pattern features__spot_static_dem_sector_ --exclude-feature-pattern features__eumetsat_}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-150000}"
MAX_TEST_ROWS="${MAX_TEST_ROWS:-60000}"
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
  local extra_args="$4"
  log "benchmark $label prefix=$prefix suffix=$suffix max_train_rows=$MAX_TRAIN_ROWS extra_args=$extra_args"
  PREFIX="$prefix" \
    RUN_SUFFIX="$suffix" \
    MAX_TRAIN_ROWS="$MAX_TRAIN_ROWS" \
    MAX_TEST_ROWS="$MAX_TEST_ROWS" \
    TRAIN_EXTRA_ARGS="$extra_args" \
    bash scripts/ml_dataset/z2_run_wind_gust_150k_bin63_benchmark.sh
}

write_summary() {
  "$PY" - "$BENCH_ROOT" "$SUMMARY_JSON" "$SUMMARY_MD" "$CHAMPION_WIND_RMSE" "$CHAMPION_WIND_MAE" \
    "$OLD_SIGNAL_SUFFIX" "$PRUNED_SUFFIX" "$V4_SUFFIX" "$PRUNED_200K_SUFFIX" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

bench_root = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
summary_md = Path(sys.argv[3])
champion_rmse = float(sys.argv[4])
champion_mae = float(sys.argv[5])
old_suffix, pruned_suffix, v4_suffix, pruned_200k_suffix = sys.argv[6:10]

def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)

def complete_run(suffix):
    path = bench_root / f"{suffix}_wind_gust_decision_report.json"
    if not path.exists():
        return {"run_suffix": suffix, "status": "missing", "path": str(path)}
    payload = load_json(path)
    out = {"run_suffix": suffix, "status": "complete", "path": str(path)}
    for target in ("wind_mean", "gust"):
        item = payload.get(target) or {}
        cal = item.get("calibrated") or {}
        out[target] = {
            "base_corrected": item.get("base_corrected"),
            "calibrated": cal,
            "delta_vs_champion_rmse": None if target != "wind_mean" or cal.get("rmse") is None else round(float(cal["rmse"]) - champion_rmse, 6),
            "delta_vs_champion_mae": None if target != "wind_mean" or cal.get("mae") is None else round(float(cal["mae"]) - champion_mae, 6),
            "selected_scale": item.get("selected_scale"),
            "calibrator_gain_pct_vs_base": item.get("calibrator_gain_pct_vs_base"),
        }
    return out

def partial_run(suffix):
    path = bench_root / f"tabular_lgbm_{suffix}_2024_2025_to_2026_v1" / "training_results.json"
    if not path.exists():
        return None
    payload = load_json(path)
    out = {
        "run_suffix": suffix,
        "status": "partial_base_only",
        "path": str(path),
        "feature_column_count": payload.get("feature_column_count"),
        "train_row_count": payload.get("train_row_count"),
        "test_row_count": payload.get("test_row_count"),
    }
    for target_key, target_name in (
        ("labels__residual_wind_mean_ms", "wind_mean"),
        ("labels__residual_gust_ms", "gust"),
    ):
        model = (payload.get("models") or {}).get(target_key) or {}
        out[target_name] = {"base_corrected": model.get("corrected_nwp_eval_leads")}
    return out

runs = [complete_run(old_suffix), complete_run(pruned_suffix), complete_run(v4_suffix)]
partial = partial_run(pruned_200k_suffix)
if partial:
    runs.append(partial)

complete_wind = [
    item for item in runs
    if item.get("status") == "complete"
    and (item.get("wind_mean") or {}).get("calibrated", {}).get("rmse") is not None
]
best = min(complete_wind, key=lambda item: item["wind_mean"]["calibrated"]["rmse"], default=None)
payload = {
    "format": "corsewind.pruned_v4_directional_150k_resume_summary.v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "champion": {"wind_mean_rmse": champion_rmse, "wind_mean_mae": champion_mae},
    "best_wind_mean_run": None if best is None else best["run_suffix"],
    "decision": (
        "missing_results"
        if best is None
        else "candidate_beats_champion"
        if best["wind_mean"]["calibrated"]["rmse"] < champion_rmse
        else "keep_champion"
    ),
    "runs": runs,
}
summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# Pruned + v4 Directional 150k Resume Summary",
    "",
    f"Champion wind mean RMSE: `{champion_rmse}`",
    f"Champion wind mean MAE: `{champion_mae}`",
    f"Decision: `{payload['decision']}`",
    "",
    "| Run | Status | Wind RMSE | Wind MAE | Delta RMSE | Gust RMSE | Gust MAE |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
]
for item in runs:
    wind = item.get("wind_mean") or {}
    gust = item.get("gust") or {}
    wind_cal = wind.get("calibrated") or wind.get("base_corrected") or {}
    gust_cal = gust.get("calibrated") or gust.get("base_corrected") or {}
    lines.append(
        f"| `{item['run_suffix']}` | `{item.get('status')}` | {wind_cal.get('rmse')} | "
        f"{wind_cal.get('mae')} | {wind.get('delta_vs_champion_rmse')} | "
        f"{gust_cal.get('rmse')} | {gust_cal.get('mae')} |"
    )
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

cd "$REMOTE_ROOT" || fail 10 "cannot cd remote root"
: > "$LOG"
echo "started $(date -Is)" > "$STATUS"
log "pruned/v4 directional 150k resume started"

run_benchmark "phys_v3_pruned_150k" "$SOURCE_PREFIX" "$PRUNED_SUFFIX" "$PRUNED_TRAIN_EXTRA_ARGS" \
  || fail 20 "phys_v3 pruned 150k benchmark failed"

run_benchmark "phys_v4_directional_pruned_150k" "$V4_PREFIX" "$V4_SUFFIX" "$V4_TRAIN_EXTRA_ARGS" \
  || fail 30 "phys_v4 directional 150k benchmark failed"

write_summary || fail 40 "summary failed"

echo "complete $(date -Is)" > "$STATUS"
log "pruned/v4 directional 150k resume complete"
