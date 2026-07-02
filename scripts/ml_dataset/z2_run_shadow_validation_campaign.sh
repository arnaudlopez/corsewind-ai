#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/srv/data/corsewind/backfill_runner}"
ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
PY_SYS="${PY_SYS:-python3}"
START_DATE="${START_DATE:-$(date -u +'%Y-%m-%d')}"
DAY_COUNT="${DAY_COUNT:-3}"
LAUNCH_TIME_UTC="${LAUNCH_TIME_UTC:-05:30}"
SUITE_VERSION="${SUITE_VERSION:-v1}"
MAX_LOOPS="${MAX_LOOPS:-160}"
POST_MAX_LOOPS="${POST_MAX_LOOPS:-320}"
FORCE_CASES="${FORCE_CASES:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
LOG_DIR="${LOG_DIR:-$ML_ROOT/live_inference/watch_logs}"
CAMPAIGN_ID="${CAMPAIGN_ID:-shadow_campaign_${START_DATE//-/}_${DAY_COUNT}d_${SUITE_VERSION}}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${CAMPAIGN_ID}.log}"
PID_FILE="${PID_FILE:-$LOG_DIR/${CAMPAIGN_ID}.pid}"

mkdir -p "$LOG_DIR"
printf '%s\n' "$$" >"$PID_FILE"
cd "$REPO_ROOT"

log() {
  printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG_FILE"
}

campaign_dates() {
  "$PY_SYS" - "$START_DATE" "$DAY_COUNT" <<'PY'
from __future__ import annotations

import sys
from datetime import date, timedelta

start = date.fromisoformat(sys.argv[1])
count = int(sys.argv[2])
if count < 1:
    raise SystemExit("DAY_COUNT must be >= 1")
for offset in range(count):
    print((start + timedelta(days=offset)).isoformat())
PY
}

sleep_seconds_until_launch() {
  "$PY_SYS" - "$1" "$LAUNCH_TIME_UTC" <<'PY'
from __future__ import annotations

import sys
from datetime import datetime, timezone

target_date = sys.argv[1]
launch_time = sys.argv[2]
hour, minute = [int(part) for part in launch_time.split(":", 1)]
year, month, day = [int(part) for part in target_date.split("-", 2)]
launch = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
now = datetime.now(timezone.utc)
print(max(0, int((launch - now).total_seconds())))
PY
}

log "campaign started start=$START_DATE days=$DAY_COUNT launch_time_utc=$LAUNCH_TIME_UTC suite=$SUITE_VERSION"
log "repo=$REPO_ROOT ml_root=$ML_ROOT"

if [[ "$PREPARE_ONLY" == "1" ]]; then
  while IFS= read -r target_date; do
    seconds="$(sleep_seconds_until_launch "$target_date")"
    log "prepared target_date=$target_date wait_seconds=$seconds"
  done < <(campaign_dates)
  exit 0
fi

while IFS= read -r target_date; do
  seconds="$(sleep_seconds_until_launch "$target_date")"
  if [[ "$seconds" -gt 0 ]]; then
    log "waiting ${seconds}s before launching target_date=$target_date"
    sleep "$seconds"
  else
    log "launch time already reached; launching target_date=$target_date now"
  fi

  log "launching daily shadow validation target_date=$target_date"
  TARGET_DATE="$target_date" \
    SUITE_VERSION="$SUITE_VERSION" \
    MAX_LOOPS="$MAX_LOOPS" \
    POST_MAX_LOOPS="$POST_MAX_LOOPS" \
    FORCE_CASES="$FORCE_CASES" \
    scripts/ml_dataset/z2_launch_shadow_validation_day.sh >>"$LOG_FILE" 2>&1
  log "daily launcher returned target_date=$target_date"
done < <(campaign_dates)

log "campaign finished"
