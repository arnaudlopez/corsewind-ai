#!/usr/bin/env bash
set -euo pipefail

ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
PREFIX="${PREFIX:-residual_windsup_sst_prev_phys_v1}"
REMOTE_ROOT="${REMOTE_ROOT:-/srv/data/corsewind/backfill_runner}"
LOG_ROOT="${LOG_ROOT:-$ML_ROOT/run_logs}"
START_MONTH="${START_MONTH:-2024-01}"
END_MONTH="${END_MONTH:-2026-06}"
PY="${PY:-/home/z2/corsewind-ml-smoke/.venv/bin/python}"
STATUS="${STATUS:-$LOG_ROOT/phys_v1_signal_audit_watcher.status}"
LOG="${LOG:-$LOG_ROOT/phys_v1_signal_audit_watcher.log}"
POLL_SECONDS="${POLL_SECONDS:-120}"

mkdir -p "$LOG_ROOT" "$ML_ROOT/training_tables"

log() {
  echo "$(date -Is) $*" | tee -a "$LOG"
}

month_suffixes() {
  "$PY" - "$START_MONTH" "$END_MONTH" <<'PY'
import sys
start_year, start_month = map(int, sys.argv[1].split("-"))
end_year, end_month = map(int, sys.argv[2].split("-"))
year, month = start_year, start_month
while (year, month) <= (end_year, end_month):
    print(f"{year:04d}_{month:02d}")
    month += 1
    if month == 13:
        year += 1
        month = 1
PY
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
    if "z2_phys_v1_signal_audit_watcher.sh" in raw:
        continue
    if prefix in raw and any(needle in raw for needle in needles):
        print(pid)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

log "phys signal audit watcher started prefix=$PREFIX"
echo "started $(date -Is)" > "$STATUS"

while is_rebuild_running > /tmp/corsewind_phys_v1_rebuild_pid 2>/dev/null; do
  log "rebuild still running pid=$(cat /tmp/corsewind_phys_v1_rebuild_pid)"
  sleep "$POLL_SECONDS"
done

cd "$REMOTE_ROOT"

log "rebuild finished; checking required feature patterns"
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
  --output-json "$ML_ROOT/training_tables/phys_v1_required_feature_audit.json" \
  --output-md "$ML_ROOT/training_tables/phys_v1_required_feature_audit.md" \
  --fail-on-non-pass

log "computing non-null coverage for physical signals"
"$PY" - "$ML_ROOT" "$PREFIX" "$START_MONTH" "$END_MONTH" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

ml_root = Path(sys.argv[1])
prefix = sys.argv[2]
start_month = sys.argv[3]
end_month = sys.argv[4]

columns = [
    "features__open_meteo_vertical_geopotential_thickness_1000_850_m",
    "features__open_meteo_vertical_temperature_lapse_rate_1000_850_c_per_km",
    "features__open_meteo_vertical_relative_humidity_mean_1000_850_pct",
    "features__open_meteo_vertical_wind_shear_speed_1000_850_ms",
    "features__nwp_offset_gradient_east_west_pressure_msl_per_km",
    "features__nwp_offset_gradient_north_south_pressure_msl_per_km",
    "features__nwp_offset_gradient_pressure_msl_aligned_with_wind_hpa_per_km",
    "features__nwp_offset_gradient_east_west_temperature_2m_per_km",
    "features__nwp_offset_gradient_north_south_temperature_2m_per_km",
    "features__thermal_air_minus_sst_c",
    "features__thermal_inland_minus_coastal_temperature_c",
    "features__thermal_relief_minus_coastal_temperature_c",
    "features__context_agg_all_upwind_weighted_wind_mean_ms_mean",
    "features__context_agg_relief_upwind_weighted_wind_mean_ms_mean",
]

def months(start, end):
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield f"{y:04d}_{m:02d}"
        m += 1
        if m == 13:
            y += 1
            m = 1

rows = 0
present = {column: False for column in columns}
nonnull = {column: 0 for column in columns}
shards = []
for suffix in months(start_month, end_month):
    path = ml_root / "training_tables" / f"{prefix}_{suffix}" / "training_rows.parquet"
    item = {"suffix": suffix, "path": str(path), "exists": path.exists(), "rows": 0}
    if not path.exists():
        shards.append(item)
        continue
    pf = pq.ParquetFile(path)
    item["rows"] = pf.metadata.num_rows
    rows += pf.metadata.num_rows
    names = set(pf.schema_arrow.names)
    available = [column for column in columns if column in names]
    item["available_columns"] = len(available)
    if available:
        table = pq.read_table(path, columns=available)
        for column in available:
            present[column] = True
            nonnull[column] += len(table[column].combine_chunks().drop_null())
    shards.append(item)

result = {
    "format": "corsewind.phys_v1_signal_coverage.v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "ml_root": str(ml_root),
    "run_id_prefix": prefix,
    "start_month": start_month,
    "end_month": end_month,
    "row_count": rows,
    "columns": [
        {
            "column": column,
            "present": present[column],
            "nonnull": nonnull[column],
            "coverage_pct": round((nonnull[column] / rows * 100.0), 6) if rows else None,
        }
        for column in columns
    ],
    "shards": shards,
}

out_json = ml_root / "training_tables" / "phys_v1_signal_coverage.json"
out_md = ml_root / "training_tables" / "phys_v1_signal_coverage.md"
out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# Physical Signal Coverage",
    "",
    f"Generated: `{result['generated_at_utc']}`",
    f"Rows: `{rows}`",
    "",
    "| Column | Present | Non-null | Coverage |",
    "| --- | ---: | ---: | ---: |",
]
for item in result["columns"]:
    lines.append(
        f"| `{item['column']}` | `{item['present']}` | {item['nonnull']} | {item['coverage_pct']}% |"
    )
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"output_json": str(out_json), "output_md": str(out_md), "rows": rows}, indent=2, sort_keys=True))
PY

echo "complete $(date -Is)" > "$STATUS"
log "phys signal audit watcher complete"
