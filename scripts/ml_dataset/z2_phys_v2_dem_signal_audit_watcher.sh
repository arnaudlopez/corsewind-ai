#!/usr/bin/env bash
set -euo pipefail

ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
PREFIX="${PREFIX:-residual_windsup_sst_prev_phys_v2_dem}"
REMOTE_ROOT="${REMOTE_ROOT:-/srv/data/corsewind/backfill_runner}"
LOG_ROOT="${LOG_ROOT:-$ML_ROOT/run_logs}"
START_MONTH="${START_MONTH:-2024-01}"
END_MONTH="${END_MONTH:-2026-06}"
PY="${PY:-/home/z2/corsewind-ml-smoke/.venv/bin/python}"
STATUS="${STATUS:-$LOG_ROOT/phys_v2_dem_signal_audit_watcher.status}"
LOG="${LOG:-$LOG_ROOT/phys_v2_dem_signal_audit_watcher.log}"
POLL_SECONDS="${POLL_SECONDS:-120}"

mkdir -p "$LOG_ROOT" "$ML_ROOT/training_tables"

log() {
  echo "$(date -Is) $*" | tee -a "$LOG"
}

is_rebuild_running() {
  "$PY" - "$PREFIX" <<'PY'
import os
import sys

prefix = sys.argv[1]
needles = (
    "run_monthly_training_shards.py",
    "run_training_backfill_pipeline.py",
    "collect_open_meteo_historical_forecast.py",
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
    if "z2_phys_v2_dem_signal_audit_watcher.sh" in raw:
        continue
    if prefix in raw and any(needle in raw for needle in needles):
        print(pid)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

log "phys_v2_dem signal audit watcher started prefix=$PREFIX"
echo "started $(date -Is)" > "$STATUS"

while is_rebuild_running > /tmp/corsewind_phys_v2_dem_rebuild_pid 2>/dev/null; do
  log "rebuild still running pid=$(cat /tmp/corsewind_phys_v2_dem_rebuild_pid)"
  sleep "$POLL_SECONDS"
done

cd "$REMOTE_ROOT"

log "rebuild finished; checking required phys_v2_dem feature patterns"
"$PY" scripts/ml_dataset/audit_training_table_features.py \
  --training-table-root "$ML_ROOT/training_tables" \
  --run-id-prefix "$PREFIX" \
  --start-month "$START_MONTH" \
  --end-month "$END_MONTH" \
  --required-pattern features__open_meteo_vertical_geopotential_thickness_1000_850_m \
  --required-pattern features__open_meteo_vertical_temperature_lapse_rate_1000_850_c_per_km \
  --required-pattern features__open_meteo_vertical_wind_shear_speed_1000_850_ms \
  --required-pattern features__nwp_offset_gradient_east_west_pressure_msl_per_km \
  --required-pattern features__nwp_offset_gradient_north_south_pressure_msl_per_km \
  --required-pattern features__nwp_offset_gradient_pressure_msl_aligned_with_wind_hpa_per_km \
  --required-pattern features__thermal_air_minus_sst_c \
  --required-pattern features__context_agg_all_upwind_weighted_wind_mean_ms_mean \
  --required-pattern features__spot_static_dem_reference_elevation_m \
  --required-pattern features__spot_static_dem_radius_10p0km_relief_max \
  --required-pattern features__spot_static_dem_sector_n_20km_barrier_max_m \
  --required-pattern features__spot_static_dem_sector_s_20km_barrier_max_m \
  --required-pattern features__spot_static_dem_relief_gradient_e_minus_w_m \
  --output-json "$ML_ROOT/training_tables/phys_v2_dem_required_feature_audit.json" \
  --output-md "$ML_ROOT/training_tables/phys_v2_dem_required_feature_audit.md" \
  --fail-on-non-pass

echo "complete $(date -Is)" > "$STATUS"
log "phys_v2_dem signal audit watcher complete"
