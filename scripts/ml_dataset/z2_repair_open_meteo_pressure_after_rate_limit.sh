#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/srv/data/corsewind/backfill_runner}"
ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
PYTHON="${PYTHON:-/home/z2/corsewind-ml-smoke/.venv/bin/python}"
LOG_ROOT="$ML_ROOT/backfill_logs"
LOG="$LOG_ROOT/open_meteo_pressure_repair_after_429.log"
STATUS="$LOG_ROOT/open_meteo_pressure_repair_after_429.status"
TASKS="$LOG_ROOT/open_meteo_pressure_repair_after_429_tasks.tsv"
AUDIT_JSON="$ML_ROOT/source_inventories/open_meteo_pressure_level_repair_audit.json"
MODEL="${MODEL:-meteofrance_arome_france}"
WAIT_SECONDS="${WAIT_SECONDS:-3900}"
INITIAL_WAIT_SECONDS="${INITIAL_WAIT_SECONDS:-$WAIT_SECONDS}"
MIN_OBSERVED_COVERAGE="${MIN_OBSERVED_COVERAGE:-0.995}"
REQUEST_SLEEP_SEC="${REQUEST_SLEEP_SEC:-8}"
MAX_DAYS_PER_REQUEST="${MAX_DAYS_PER_REQUEST:-7}"
REPAIR_TASK_DAYS_PER_RANGE="${REPAIR_TASK_DAYS_PER_RANGE:-$MAX_DAYS_PER_REQUEST}"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"
COLLECT_PROCESS_TIMEOUT_SEC="${COLLECT_PROCESS_TIMEOUT_SEC:-600}"
MAX_TASK_ATTEMPTS="${MAX_TASK_ATTEMPTS:-3}"
NON_429_RETRY_WAIT_SECONDS="${NON_429_RETRY_WAIT_SECONDS:-120}"

HOURLY="wind_speed_10m,wind_direction_10m,wind_gusts_10m,temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,surface_pressure,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,shortwave_radiation,direct_radiation,diffuse_radiation,cape,lifted_index,boundary_layer_height,precipitation,rain,showers,temperature_1000hPa,temperature_950hPa,temperature_925hPa,temperature_900hPa,temperature_850hPa,relative_humidity_1000hPa,relative_humidity_950hPa,relative_humidity_925hPa,relative_humidity_900hPa,relative_humidity_850hPa,geopotential_height_1000hPa,geopotential_height_950hPa,geopotential_height_925hPa,geopotential_height_900hPa,geopotential_height_850hPa,wind_speed_1000hPa,wind_speed_950hPa,wind_speed_925hPa,wind_speed_900hPa,wind_speed_850hPa,wind_direction_1000hPa,wind_direction_950hPa,wind_direction_925hPa,wind_direction_900hPa,wind_direction_850hPa"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  echo "$(timestamp) $*" | tee -a "$LOG"
}

on_error() {
  local code=$?
  local line=${BASH_LINENO[0]:-unknown}
  echo "failed:$code" > "$STATUS"
  log "repair watcher failed code=$code line=$line"
  exit "$code"
}

trap on_error ERR

run_audit() {
  ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/audit_open_meteo_coverage.py \
    --input-root "$ML_ROOT/open_meteo/historical_forecast" \
    --start-date 2024-01-02 \
    --end-date 2026-06-23 \
    --model "$MODEL" \
    --include-context-spots \
    --required-features temperature_1000hPa,relative_humidity_850hPa,geopotential_height_850hPa,wind_speed_850hPa,wind_direction_850hPa \
    > "$AUDIT_JSON"
}

coverage_from_audit() {
  "$PYTHON" - "$AUDIT_JSON" <<'PY'
import json
import sys

audit = json.load(open(sys.argv[1]))
observed = audit.get("observed_rows") or 0
complete = audit.get("required_feature_complete_rows") or 0
print(f"{(complete / observed if observed else 0.0):.8f}")
PY
}

write_tasks() {
  "$PYTHON" - "$AUDIT_JSON" "$TASKS" "$REPAIR_TASK_DAYS_PER_RANGE" <<'PY'
import json
import sys
from datetime import datetime, timedelta


def parse_day(value):
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_ranges(start, end, days_per_range):
    current = parse_day(start)
    end_day = parse_day(end)
    while current <= end_day:
        chunk_end = min(end_day, current + timedelta(days=days_per_range - 1))
        yield current.isoformat(), chunk_end.isoformat()
        current = chunk_end + timedelta(days=1)

audit = json.load(open(sys.argv[1]))
days_per_range = max(1, int(sys.argv[3]))
tasks = []
for spot in audit.get("by_spot") or []:
    spot_id = spot.get("spot_id")
    if not spot_id:
        continue
    for item in spot.get("required_feature_missing_day_ranges") or []:
        for start, end in iter_ranges(item["start"], item["end"], days_per_range):
            tasks.append((spot_id, start, end, "missing_feature_range_chunk"))
    partial_days = spot.get("required_feature_partial_days")
    if partial_days is None:
        partial_days = spot.get("required_feature_partial_days_sample") or []
    for item in partial_days:
        day = item.get("date")
        if day:
            tasks.append((spot_id, day, day, "partial_feature_day"))

seen = set()
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    for task in tasks:
        if task in seen:
            continue
        seen.add(task)
        handle.write("\t".join(task) + "\n")
print(len(seen))
PY
}

pid_matches() {
  local pid="$1"
  local needle="$2"
  if [[ -z "$pid" || ! -r "/proc/$pid/cmdline" ]]; then
    return 1
  fi
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq "$needle"
}

launch_primary_watcher() {
  local pidfile="$LOG_ROOT/open_meteo_pressure_rebuild_watcher.pid"
  local pid=""
  if [[ -f "$pidfile" ]]; then
    pid="$(cat "$pidfile")"
  fi
  if pid_matches "$pid" "z2_watch_open_meteo_pressure_then_rebuild.sh"; then
    log "primary rebuild watcher already running pid=$pid"
    return 0
  fi
  log "launching primary rebuild watcher after repair"
  cd "$ROOT"
  setsid bash scripts/ml_dataset/z2_watch_open_meteo_pressure_then_rebuild.sh >/dev/null 2>&1 &
  echo $! > "$pidfile"
  log "primary rebuild watcher pid=$(cat "$pidfile")"
}

cd "$ROOT"
mkdir -p "$LOG_ROOT" "$ML_ROOT/source_inventories"
echo "running" > "$STATUS"
log "repair watcher started after Open-Meteo 429; waiting ${INITIAL_WAIT_SECONDS}s for quota reset"
sleep "$INITIAL_WAIT_SECONDS"

log "running pre-repair hPa audit"
run_audit
coverage="$(coverage_from_audit)"
task_count="$(write_tasks)"
log "pre_repair_coverage=$coverage task_count=$task_count"

if [[ "$task_count" == "0" ]]; then
  log "no repair tasks generated"
else
  index=0
  while IFS=$'\t' read -r spot_id start_date end_date reason; do
    index=$((index + 1))
    attempt=1
    while true; do
      task_log="$LOG_ROOT/open_meteo_pressure_repair_task_${index}_attempt_${attempt}.log"
      log "repair_task index=$index/$task_count attempt=$attempt/$MAX_TASK_ATTEMPTS spot=$spot_id start=$start_date end=$end_date reason=$reason"
      trap - ERR
      set +e
      timeout --kill-after=30s "${COLLECT_PROCESS_TIMEOUT_SEC}s" \
        env ML_DATASET_ROOT="$ML_ROOT" "$PYTHON" scripts/ml_dataset/collect_open_meteo_historical_forecast.py \
        --output-root "$ML_ROOT/open_meteo/historical_forecast" \
        --start-date "$start_date" \
        --end-date "$end_date" \
        --model "$MODEL" \
        --include-context-spots \
        --spot-id "$spot_id" \
        --max-days-per-request "$MAX_DAYS_PER_REQUEST" \
        --request-sleep-sec "$REQUEST_SLEEP_SEC" \
        --timeout-sec "$TIMEOUT_SEC" \
        --no-skip-existing-complete \
        --hourly "$HOURLY" \
        2>&1 | tee "$task_log" | tee -a "$LOG"
      status="${PIPESTATUS[0]}"
      set -e
      trap on_error ERR
      if [[ "$status" == "0" ]]; then
        break
      fi
      if [[ "$attempt" -ge "$MAX_TASK_ATTEMPTS" ]]; then
        log "repair_task_failed_permanently index=$index spot=$spot_id status=$status attempts=$attempt"
        break
      fi
      if grep -q "Hourly API request limit exceeded\\|Open-Meteo HTTP 429" "$task_log"; then
        log "repair_task_hit_429 index=$index spot=$spot_id; waiting ${WAIT_SECONDS}s before retry"
        sleep "$WAIT_SECONDS"
      else
        log "repair_task_failed index=$index spot=$spot_id status=$status; waiting ${NON_429_RETRY_WAIT_SECONDS}s before retry"
        sleep "$NON_429_RETRY_WAIT_SECONDS"
      fi
      attempt=$((attempt + 1))
    done
  done < "$TASKS"
fi

log "running post-repair hPa audit"
run_audit
coverage="$(coverage_from_audit)"
log "post_repair_coverage=$coverage"

"$PYTHON" - "$coverage" "$MIN_OBSERVED_COVERAGE" <<'PY'
import sys

coverage = float(sys.argv[1])
threshold = float(sys.argv[2])
if coverage < threshold:
    raise SystemExit(f"coverage {coverage:.6f} below threshold {threshold:.6f}")
PY

echo "complete" > "$STATUS"
log "repair coverage gate passed"
launch_primary_watcher
