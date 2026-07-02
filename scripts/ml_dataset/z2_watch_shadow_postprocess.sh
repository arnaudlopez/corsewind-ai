#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/srv/data/corsewind/backfill_runner}"
ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
PY_SYS="${PY_SYS:-python3}"
TARGET_DATE="${TARGET_DATE:-$(date -u +'%Y-%m-%d')}"
TARGET_DATE_COMPACT="${TARGET_DATE//-/}"
SUITE_VERSION="${SUITE_VERSION:-v1}"
RUN_LABEL="${RUN_LABEL:-full_day_${SUITE_VERSION}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ML_ROOT/live_inference/collector_hindcast_shadow_unseen_${TARGET_DATE_COMPACT}_${RUN_LABEL}}"
LOG_DIR="${LOG_DIR:-$ML_ROOT/live_inference/watch_logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${TARGET_DATE_COMPACT}_${RUN_LABEL}_postprocess.log}"
PID_FILE="${PID_FILE:-$LOG_DIR/${TARGET_DATE_COMPACT}_${RUN_LABEL}_postprocess.pid}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
MAX_LOOPS="${MAX_LOOPS:-240}"

mkdir -p "$LOG_DIR"
printf '%s\n' "$$" >"$PID_FILE"
cd "$REPO_ROOT"

log() {
  printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG_FILE"
}

suite_status() {
  OUTPUT_ROOT="$OUTPUT_ROOT" "$PY_SYS" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
summary_path = root / "suite_summary.json"
payload = {
    "output_root": str(root),
    "suite_summary": str(summary_path),
    "exists": summary_path.exists(),
    "complete": False,
    "case_count": 0,
    "scored_cases": 0,
    "shadow_cases": 0,
}
if summary_path.exists():
    try:
        summary = json.loads(summary_path.read_text())
    except Exception as exc:
        payload["error"] = str(exc)
    else:
        cases = summary.get("cases") or []
        payload["case_count"] = len(cases)
        payload["failures"] = summary.get("failures") or []
        payload["scored_cases"] = sum(1 for case in cases if case.get("score_json") or case.get("joined_rows"))
        payload["shadow_cases"] = sum(1 for case in cases if case.get("shadow_router_v1"))
        payload["complete"] = bool(cases) and payload["scored_cases"] == len(cases) and payload["shadow_cases"] == len(cases)
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["complete"] else 2)
PY
}

postprocess_done() {
  [[ -s "$OUTPUT_ROOT/shadow_aggregate_v1.json" && -s "$OUTPUT_ROOT/wind_router_promotion_gate_v1.json" && -s "$OUTPUT_ROOT/gust_guarded_stacker_promotion_gate_v1.json" ]]
}

log "postprocess watcher started output=$OUTPUT_ROOT"
for loop in $(seq 1 "$MAX_LOOPS"); do
  if postprocess_done; then
    log "postprocess artifacts already present; exiting"
    exit 0
  fi
  log "loop $loop/$MAX_LOOPS"
  set +e
  status="$(suite_status 2>>"$LOG_FILE")"
  code=$?
  set -e
  log "suite $status"
  if [[ "$code" == "0" ]]; then
    log "suite complete; running postprocess"
    OUTPUT_ROOT="$OUTPUT_ROOT" TARGET_DATE="$TARGET_DATE" SUITE_VERSION="$SUITE_VERSION" RUN_LABEL="$RUN_LABEL" \
      "$REPO_ROOT/scripts/ml_dataset/run_shadow_suite_postprocess.sh" >>"$LOG_FILE" 2>&1
    log "postprocess finished"
    exit 0
  fi
  sleep "$SLEEP_SECONDS"
done

log "postprocess watcher ended without complete suite"
exit 2
