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
AGGREGATE_JSON="${AGGREGATE_JSON:-$OUTPUT_ROOT/shadow_aggregate_v1.json}"
AGGREGATE_MARKDOWN="${AGGREGATE_MARKDOWN:-$OUTPUT_ROOT/shadow_aggregate_v1.md}"
WIND_GATE_JSON="${WIND_GATE_JSON:-$OUTPUT_ROOT/wind_router_promotion_gate_v1.json}"
WIND_GATE_MARKDOWN="${WIND_GATE_MARKDOWN:-$OUTPUT_ROOT/wind_router_promotion_gate_v1.md}"
GUST_GATE_JSON="${GUST_GATE_JSON:-$OUTPUT_ROOT/gust_guarded_stacker_promotion_gate_v1.json}"
GUST_GATE_MARKDOWN="${GUST_GATE_MARKDOWN:-$OUTPUT_ROOT/gust_guarded_stacker_promotion_gate_v1.md}"
PROMOTION_REVIEW_JSON="${PROMOTION_REVIEW_JSON:-$OUTPUT_ROOT/promotion_candidate_review_v1.json}"
PROMOTION_REVIEW_MARKDOWN="${PROMOTION_REVIEW_MARKDOWN:-$OUTPUT_ROOT/promotion_candidate_review_v1.md}"
PROMOTION_DECISION_JSON="${PROMOTION_DECISION_JSON:-$OUTPUT_ROOT/promotion_decision_v1.json}"
PROMOTION_DECISION_MARKDOWN="${PROMOTION_DECISION_MARKDOWN:-$OUTPUT_ROOT/promotion_decision_v1.md}"
THRESHOLD_GUARD_AUDIT_JSON="${THRESHOLD_GUARD_AUDIT_JSON:-$OUTPUT_ROOT/threshold_guard_impact_audit_v1.json}"
THRESHOLD_GUARD_AUDIT_MARKDOWN="${THRESHOLD_GUARD_AUDIT_MARKDOWN:-$OUTPUT_ROOT/threshold_guard_impact_audit_v1.md}"
SHADOW_CANDIDATE_IMPACT_AUDIT_JSON="${SHADOW_CANDIDATE_IMPACT_AUDIT_JSON:-$OUTPUT_ROOT/shadow_candidate_impact_audit_v1.json}"
SHADOW_CANDIDATE_IMPACT_AUDIT_MARKDOWN="${SHADOW_CANDIDATE_IMPACT_AUDIT_MARKDOWN:-$OUTPUT_ROOT/shadow_candidate_impact_audit_v1.md}"
WIND_EVENT_HEAD_AUDIT_JSON="${WIND_EVENT_HEAD_AUDIT_JSON:-$OUTPUT_ROOT/wind_threshold_event_head_audit_v1.json}"
WIND_EVENT_HEAD_AUDIT_MARKDOWN="${WIND_EVENT_HEAD_AUDIT_MARKDOWN:-$OUTPUT_ROOT/wind_threshold_event_head_audit_v1.md}"
GUST_EVENT_HEAD_AUDIT_JSON="${GUST_EVENT_HEAD_AUDIT_JSON:-$OUTPUT_ROOT/gust_threshold_event_head_audit_v1.json}"
GUST_EVENT_HEAD_AUDIT_MARKDOWN="${GUST_EVENT_HEAD_AUDIT_MARKDOWN:-$OUTPUT_ROOT/gust_threshold_event_head_audit_v1.md}"
NEXT_SPECIALIST_PLAN_JSON="${NEXT_SPECIALIST_PLAN_JSON:-$OUTPUT_ROOT/next_nowcasting_specialist_plan_v1.json}"
NEXT_SPECIALIST_PLAN_MARKDOWN="${NEXT_SPECIALIST_PLAN_MARKDOWN:-$OUTPUT_ROOT/next_nowcasting_specialist_plan_v1.md}"
RUN_ROLLUP="${RUN_ROLLUP:-1}"
ROLLUP_ID="${ROLLUP_ID:-shadow_rollup_latest}"
ROLLUP_MIN_COMPLETE_SUITES="${ROLLUP_MIN_COMPLETE_SUITES:-1}"

cd "$REPO_ROOT"

if [[ -z "${SUITE_SUMMARIES:-}" ]]; then
  SUITE_SUMMARIES="$OUTPUT_ROOT/suite_summary.json"
fi

summary_args=()
for path in $SUITE_SUMMARIES; do
  summary_args+=(--suite-summary "$path")
done

"$PY_ML" scripts/ml_dataset/summarize_shadow_suites.py \
  "${summary_args[@]}" \
  --output-json "$AGGREGATE_JSON" \
  --output-markdown "$AGGREGATE_MARKDOWN"

"$PY_ML" scripts/ml_dataset/assert_shadow_promotion_gate.py \
  --aggregate-json "$AGGREGATE_JSON" \
  --target wind \
  --candidate router \
  --output-json "$WIND_GATE_JSON" \
  --output-markdown "$WIND_GATE_MARKDOWN" \
  --no-fail-on-reject

"$PY_ML" scripts/ml_dataset/assert_shadow_promotion_gate.py \
  --aggregate-json "$AGGREGATE_JSON" \
  --target gust \
  --candidate guarded_stacker \
  --output-json "$GUST_GATE_JSON" \
  --output-markdown "$GUST_GATE_MARKDOWN" \
  --no-fail-on-reject

"$PY_ML" scripts/ml_dataset/review_shadow_promotion_candidates.py \
  --aggregate-json "$AGGREGATE_JSON" \
  --output-json "$PROMOTION_REVIEW_JSON" \
  --output-markdown "$PROMOTION_REVIEW_MARKDOWN"

mapfile -t threshold_guard_scored < <(
  "$PY_ML" - $SUITE_SUMMARIES <<'PY'
import json
import sys
from pathlib import Path

for summary_path in sys.argv[1:]:
    payload = json.loads(Path(summary_path).read_text())
    for case in payload.get("cases") or []:
        output_root = case.get("output_root")
        if not output_root:
            continue
        path = Path(output_root) / "hindcast_scored_rows_with_threshold_guard_v1.parquet"
        if path.exists():
            print(path)
PY
)

if (( ${#threshold_guard_scored[@]} > 0 )); then
  audit_args=()
  for path in "${threshold_guard_scored[@]}"; do
    audit_args+=(--scored-parquet "$path")
  done
  "$PY_ML" scripts/ml_dataset/audit_threshold_guard_impact.py \
    "${audit_args[@]}" \
    --output-json "$THRESHOLD_GUARD_AUDIT_JSON" \
    --output-markdown "$THRESHOLD_GUARD_AUDIT_MARKDOWN"
fi

mapfile -t event_head_scored < <(
  "$PY_ML" - $SUITE_SUMMARIES <<'PY'
import json
import sys
from pathlib import Path

for summary_path in sys.argv[1:]:
    payload = json.loads(Path(summary_path).read_text())
    for case in payload.get("cases") or []:
        output_root = case.get("output_root")
        if not output_root:
            continue
        case_root = Path(output_root)
        for name in (
            "hindcast_scored_rows_with_local_fallback_guard_v1.parquet",
            "hindcast_scored_rows_with_threshold_guard_v1.parquet",
            "hindcast_scored_rows_with_shadow_router_v1.parquet",
            "hindcast_scored_rows.parquet",
        ):
            path = case_root / name
            if path.exists():
                print(path)
                break
PY
)

if (( ${#event_head_scored[@]} > 0 )); then
  event_head_args=()
  for path in "${event_head_scored[@]}"; do
    event_head_args+=(--scored-parquet "$path")
  done
  "$PY_ML" scripts/ml_dataset/audit_shadow_candidate_impact.py \
    "${event_head_args[@]}" \
    --output-json "$SHADOW_CANDIDATE_IMPACT_AUDIT_JSON" \
    --output-markdown "$SHADOW_CANDIDATE_IMPACT_AUDIT_MARKDOWN"
  "$PY_ML" scripts/ml_dataset/audit_wind_threshold_event_heads.py \
    "${event_head_args[@]}" \
    --output-json "$WIND_EVENT_HEAD_AUDIT_JSON" \
    --output-markdown "$WIND_EVENT_HEAD_AUDIT_MARKDOWN"
  "$PY_ML" scripts/ml_dataset/audit_gust_threshold_event_heads.py \
    "${event_head_args[@]}" \
    --output-json "$GUST_EVENT_HEAD_AUDIT_JSON" \
    --output-markdown "$GUST_EVENT_HEAD_AUDIT_MARKDOWN"
fi

decision_args=(--promotion-review-json "$PROMOTION_REVIEW_JSON")
if [[ -s "$THRESHOLD_GUARD_AUDIT_JSON" ]]; then
  decision_args+=(--threshold-guard-audit-json "$THRESHOLD_GUARD_AUDIT_JSON")
fi
if [[ -s "$SHADOW_CANDIDATE_IMPACT_AUDIT_JSON" ]]; then
  decision_args+=(--candidate-impact-audit-json "$SHADOW_CANDIDATE_IMPACT_AUDIT_JSON")
fi
"$PY_ML" scripts/ml_dataset/decide_shadow_promotion.py \
  "${decision_args[@]}" \
  --output-json "$PROMOTION_DECISION_JSON" \
  --output-markdown "$PROMOTION_DECISION_MARKDOWN"

specialist_plan_args=(
  --promotion-review-json "$PROMOTION_REVIEW_JSON"
  --promotion-decision-json "$PROMOTION_DECISION_JSON"
)
if [[ -s "$THRESHOLD_GUARD_AUDIT_JSON" ]]; then
  specialist_plan_args+=(--threshold-guard-audit-json "$THRESHOLD_GUARD_AUDIT_JSON")
fi
if [[ -s "$SHADOW_CANDIDATE_IMPACT_AUDIT_JSON" ]]; then
  specialist_plan_args+=(--candidate-impact-audit-json "$SHADOW_CANDIDATE_IMPACT_AUDIT_JSON")
fi
if [[ -s "$GUST_EVENT_HEAD_AUDIT_JSON" ]]; then
  specialist_plan_args+=(--gust-event-head-audit-json "$GUST_EVENT_HEAD_AUDIT_JSON")
fi
if [[ -s "$WIND_EVENT_HEAD_AUDIT_JSON" ]]; then
  specialist_plan_args+=(--wind-event-head-audit-json "$WIND_EVENT_HEAD_AUDIT_JSON")
fi
"$PY_ML" scripts/ml_dataset/plan_next_nowcasting_specialists.py \
  "${specialist_plan_args[@]}" \
  --output-json "$NEXT_SPECIALIST_PLAN_JSON" \
  --output-markdown "$NEXT_SPECIALIST_PLAN_MARKDOWN"

if [[ "$RUN_ROLLUP" == "1" ]]; then
  (
    unset OUTPUT_ROOT
    ROLLUP_ID="$ROLLUP_ID" MIN_COMPLETE_SUITES="$ROLLUP_MIN_COMPLETE_SUITES" \
      "$REPO_ROOT/scripts/ml_dataset/run_shadow_multi_day_rollup.sh"
  )
fi

printf 'postprocess complete\n'
printf 'aggregate=%s\n' "$AGGREGATE_JSON"
printf 'wind_gate=%s\n' "$WIND_GATE_JSON"
printf 'gust_gate=%s\n' "$GUST_GATE_JSON"
printf 'promotion_review=%s\n' "$PROMOTION_REVIEW_JSON"
printf 'promotion_decision=%s\n' "$PROMOTION_DECISION_JSON"
printf 'next_specialist_plan=%s\n' "$NEXT_SPECIALIST_PLAN_JSON"
if (( ${#event_head_scored[@]} > 0 )); then
  printf 'shadow_candidate_impact_audit=%s\n' "$SHADOW_CANDIDATE_IMPACT_AUDIT_JSON"
  printf 'wind_threshold_event_head_audit=%s\n' "$WIND_EVENT_HEAD_AUDIT_JSON"
  printf 'gust_threshold_event_head_audit=%s\n' "$GUST_EVENT_HEAD_AUDIT_JSON"
fi
if (( ${#threshold_guard_scored[@]} > 0 )); then
  printf 'threshold_guard_audit=%s\n' "$THRESHOLD_GUARD_AUDIT_JSON"
fi
