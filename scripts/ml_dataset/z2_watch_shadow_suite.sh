#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/srv/data/corsewind/backfill_runner}"
ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
PY_ML="${PY_ML:-/srv/data/corsewind/pyenv/bin/python}"
PY_SYS="${PY_SYS:-python3}"
TARGET_DATE="${TARGET_DATE:-$(date -u +'%Y-%m-%d')}"
TARGET_DATE_COMPACT="${TARGET_DATE//-/}"
TARGET_END_UTC="${TARGET_END_UTC:-${TARGET_DATE}T17:00:00Z}"
SLEEP_SECONDS="${SLEEP_SECONDS:-900}"
MAX_LOOPS="${MAX_LOOPS:-80}"
SCORE_SPOTS="${SCORE_SPOTS:-cap_corse,la_parata,lfkf,lfkj,lfks,lfvf,lfvh}"
SUITE_VERSION="${SUITE_VERSION:-v1}"
RUN_LABEL="${RUN_LABEL:-full_day_${SUITE_VERSION}}"
LOG_DIR="${LOG_DIR:-$ML_ROOT/live_inference/watch_logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${TARGET_DATE_COMPACT}_${RUN_LABEL}_shadow.log}"
PID_FILE="${PID_FILE:-$LOG_DIR/${TARGET_DATE_COMPACT}_${RUN_LABEL}_shadow.pid}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ML_ROOT/live_inference/collector_hindcast_shadow_unseen_${TARGET_DATE_COMPACT}_${RUN_LABEL}}"
CASES_JSON="${CASES_JSON:-configs/ml_collector_shadow_cases_${TARGET_DATE_COMPACT}_full_day_${SUITE_VERSION}.json}"
SHADOW_ARTIFACT="${SHADOW_ARTIFACT:-$ML_ROOT/live_inference/collector_hindcast_replay_20260702_v1/router_v1_shadow_artifact_all_cases/router_v1_final_models.joblib}"
RUN_POSTPROCESS="${RUN_POSTPROCESS:-1}"

mkdir -p "$LOG_DIR"
printf '%s\n' "$$" >"$PID_FILE"
cd "$REPO_ROOT"

log() {
  printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG_FILE"
}

refresh_observations() {
  log "refreshing Meteo-France observations"
  "$PY_SYS" scripts/ml_dataset/collect_meteo_france_observations.py \
    --env-file .env \
    --registry configs/ml_spots.json \
    --output-root "$ML_ROOT/observations/meteo_france" \
    --mode station-6m \
    --mode station-hourly \
    --mode synop \
    --mode bouees >>"$LOG_FILE" 2>&1 || return 1
}

coverage_status() {
  TARGET_DATE="$TARGET_DATE" TARGET_END_UTC="$TARGET_END_UTC" SCORE_SPOTS="$SCORE_SPOTS" ML_ROOT="$ML_ROOT" "$PY_SYS" - <<'PY'
import glob
import json
import os
from datetime import datetime, timezone

target_end = datetime.fromisoformat(os.environ["TARGET_END_UTC"].replace("Z", "+00:00")).astimezone(timezone.utc)
spots = [item.strip() for item in os.environ["SCORE_SPOTS"].split(",") if item.strip()]
ml_root = os.environ["ML_ROOT"]
target_date = os.environ["TARGET_DATE"]
paths = [
    f"{ml_root}/observations/meteo_france/source_dataset=dpobs_station_infrahoraire_6m/date={target_date}/observations.jsonl",
    f"{ml_root}/observations/meteo_france/source_dataset=dpobs_station_horaire/date={target_date}/observations.jsonl",
]
latest = {spot: None for spot in spots}
rows = 0
for path in paths:
    for filename in glob.glob(path):
        with open(filename, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                spot = row.get("spot_id")
                timestamp = row.get("timestamp_utc")
                if spot not in latest or not timestamp:
                    continue
                try:
                    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(timezone.utc)
                except ValueError:
                    continue
                rows += 1
                if latest[spot] is None or parsed > latest[spot]:
                    latest[spot] = parsed
missing = [spot for spot, value in latest.items() if value is None or value < target_end]
payload = {
    "rows": rows,
    "target_end_utc": os.environ["TARGET_END_UTC"],
    "complete": not missing,
    "missing": missing,
    "latest_by_spot": {spot: (value.isoformat().replace("+00:00", "Z") if value else None) for spot, value in latest.items()},
}
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["complete"] else 2)
PY
}

run_suite() {
  log "coverage complete; running shadow suite"
  "$PY_ML" scripts/ml_dataset/run_collector_hindcast_suite.py \
    --output-root "$OUTPUT_ROOT" \
    --cases-json "$CASES_JSON" \
    --source aromepi \
    --shadow-artifact "$SHADOW_ARTIFACT" \
    --shadow-allow-missing-features \
    --continue-on-error \
    --python "$PY_ML" >>"$LOG_FILE" 2>&1
  log "shadow suite finished: $OUTPUT_ROOT"
  if [[ "$RUN_POSTPROCESS" == "1" ]]; then
    log "running shadow suite postprocess"
    OUTPUT_ROOT="$OUTPUT_ROOT" TARGET_DATE="$TARGET_DATE" SUITE_VERSION="$SUITE_VERSION" RUN_LABEL="$RUN_LABEL" \
      "$REPO_ROOT/scripts/ml_dataset/run_shadow_suite_postprocess.sh" >>"$LOG_FILE" 2>&1
    log "shadow suite postprocess finished: $OUTPUT_ROOT"
  fi
}

if [[ ! -f "$CASES_JSON" ]]; then
  log "cases file missing: $CASES_JSON"
  exit 2
fi

log "watcher started target=$TARGET_END_UTC output=$OUTPUT_ROOT cases=$CASES_JSON"
for loop in $(seq 1 "$MAX_LOOPS"); do
  log "loop $loop/$MAX_LOOPS"
  refresh_observations || log "observation refresh failed; will retry"
  set +e
  status="$(coverage_status 2>>"$LOG_FILE")"
  code=$?
  set -e
  log "coverage $status"
  if [[ "$code" == "0" ]]; then
    run_suite
    exit 0
  fi
  sleep "$SLEEP_SECONDS"
done

log "watcher ended without complete coverage"
exit 2
