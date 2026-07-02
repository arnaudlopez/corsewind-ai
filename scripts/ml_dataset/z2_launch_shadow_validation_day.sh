#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/srv/data/corsewind/backfill_runner}"
ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
PY_ML="${PY_ML:-/srv/data/corsewind/pyenv/bin/python}"
TARGET_DATE="${TARGET_DATE:-$(date -u +'%Y-%m-%d')}"
TARGET_DATE_COMPACT="${TARGET_DATE//-/}"
SUITE_VERSION="${SUITE_VERSION:-v1}"
RUN_LABEL="${RUN_LABEL:-full_day_${SUITE_VERSION}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ML_ROOT/live_inference/collector_hindcast_shadow_unseen_${TARGET_DATE_COMPACT}_${RUN_LABEL}}"
CASES_JSON="${CASES_JSON:-configs/ml_collector_shadow_cases_${TARGET_DATE_COMPACT}_full_day_${SUITE_VERSION}.json}"
LOG_DIR="${LOG_DIR:-$ML_ROOT/live_inference/watch_logs}"
MAIN_LOG="${MAIN_LOG:-$LOG_DIR/${TARGET_DATE_COMPACT}_${RUN_LABEL}_shadow.log}"
MAIN_PID="${MAIN_PID:-$LOG_DIR/${TARGET_DATE_COMPACT}_${RUN_LABEL}_shadow.pid}"
MAIN_LAUNCH_LOG="${MAIN_LAUNCH_LOG:-$LOG_DIR/${TARGET_DATE_COMPACT}_${RUN_LABEL}_shadow.launch.log}"
POST_LOG="${POST_LOG:-$LOG_DIR/${TARGET_DATE_COMPACT}_${RUN_LABEL}_postprocess.log}"
POST_PID="${POST_PID:-$LOG_DIR/${TARGET_DATE_COMPACT}_${RUN_LABEL}_postprocess.pid}"
POST_LAUNCH_LOG="${POST_LAUNCH_LOG:-$LOG_DIR/${TARGET_DATE_COMPACT}_${RUN_LABEL}_postprocess.launch.log}"
SLEEP_SECONDS="${SLEEP_SECONDS:-900}"
POST_SLEEP_SECONDS="${POST_SLEEP_SECONDS:-300}"
MAX_LOOPS="${MAX_LOOPS:-160}"
POST_MAX_LOOPS="${POST_MAX_LOOPS:-320}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
FORCE_CASES="${FORCE_CASES:-0}"

cd "$REPO_ROOT"

is_running() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] && ps -p "$(cat "$pid_file")" >/dev/null 2>&1
}

if [[ ! -f "$CASES_JSON" || "$FORCE_CASES" == "1" ]]; then
  "$PY_ML" scripts/ml_dataset/generate_collector_shadow_cases.py \
    --date "$TARGET_DATE" \
    --output "$CASES_JSON" \
    --overwrite
fi

if [[ "$PREPARE_ONLY" == "1" ]]; then
  printf 'prepared cases=%s output=%s\n' "$CASES_JSON" "$OUTPUT_ROOT"
  exit 0
fi

mkdir -p "$LOG_DIR"

if is_running "$MAIN_PID"; then
  printf 'main watcher already running pid=%s\n' "$(cat "$MAIN_PID")"
else
  rm -f "$MAIN_PID" "$MAIN_LOG" "$MAIN_LAUNCH_LOG"
  setsid env \
    TARGET_DATE="$TARGET_DATE" \
    TARGET_END_UTC="${TARGET_DATE}T17:00:00Z" \
    SUITE_VERSION="$SUITE_VERSION" \
    RUN_LABEL="$RUN_LABEL" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    CASES_JSON="$CASES_JSON" \
    LOG_FILE="$MAIN_LOG" \
    PID_FILE="$MAIN_PID" \
    SLEEP_SECONDS="$SLEEP_SECONDS" \
    MAX_LOOPS="$MAX_LOOPS" \
    scripts/ml_dataset/z2_watch_shadow_suite.sh >"$MAIN_LAUNCH_LOG" 2>&1 &
  printf 'main watcher launched pid=%s\n' "$!"
fi

if is_running "$POST_PID"; then
  printf 'postprocess watcher already running pid=%s\n' "$(cat "$POST_PID")"
else
  rm -f "$POST_PID" "$POST_LOG" "$POST_LAUNCH_LOG"
  setsid env \
    TARGET_DATE="$TARGET_DATE" \
    SUITE_VERSION="$SUITE_VERSION" \
    RUN_LABEL="$RUN_LABEL" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    LOG_FILE="$POST_LOG" \
    PID_FILE="$POST_PID" \
    SLEEP_SECONDS="$POST_SLEEP_SECONDS" \
    MAX_LOOPS="$POST_MAX_LOOPS" \
    scripts/ml_dataset/z2_watch_shadow_postprocess.sh >"$POST_LAUNCH_LOG" 2>&1 &
  printf 'postprocess watcher launched pid=%s\n' "$!"
fi

printf 'cases=%s\n' "$CASES_JSON"
printf 'output=%s\n' "$OUTPUT_ROOT"
printf 'main_pid=%s\n' "$MAIN_PID"
printf 'main_log=%s\n' "$MAIN_LOG"
printf 'post_pid=%s\n' "$POST_PID"
printf 'post_log=%s\n' "$POST_LOG"
