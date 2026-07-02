#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/srv/data/corsewind/backfill_runner}"
ML_ROOT="${ML_ROOT:-/srv/data/corsewind/ml_dataset}"
PY_ML="${PY_ML:-/srv/data/corsewind/pyenv/bin/python}"
ROLLUP_ID="${ROLLUP_ID:-shadow_rollup_latest}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ML_ROOT/live_inference/shadow_rollups/$ROLLUP_ID}"
DISCOVERY_GLOB="${DISCOVERY_GLOB:-$ML_ROOT/live_inference/collector_hindcast_shadow_unseen*/suite_summary.json}"
MIN_COMPLETE_SUITES="${MIN_COMPLETE_SUITES:-1}"
MIN_SHADOW_CASES="${MIN_SHADOW_CASES:-1}"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT"

mapfile -t summaries < <(
  DISCOVERY_GLOB="$DISCOVERY_GLOB" MIN_SHADOW_CASES="$MIN_SHADOW_CASES" "$PY_ML" - <<'PY'
import glob
import json
import os
from pathlib import Path

min_shadow_cases = int(os.environ["MIN_SHADOW_CASES"])
for raw in sorted(glob.glob(os.environ["DISCOVERY_GLOB"])):
    path = Path(raw)
    try:
        payload = json.loads(path.read_text())
    except Exception:
        continue
    cases = payload.get("cases") or []
    if not cases:
        continue
    scored = sum(1 for case in cases if case.get("score_json") or case.get("joined_rows"))
    shadow = sum(1 for case in cases if case.get("shadow_router_v1"))
    if scored == len(cases) and shadow == len(cases) and shadow >= min_shadow_cases:
        print(path)
PY
)

if (( ${#summaries[@]} < MIN_COMPLETE_SUITES )); then
  printf 'not enough complete shadow suites: found=%s required=%s\n' "${#summaries[@]}" "$MIN_COMPLETE_SUITES" >&2
  printf '%s\n' "${summaries[@]}" > "$OUTPUT_ROOT/discovered_suite_summaries.txt"
  exit 2
fi

printf '%s\n' "${summaries[@]}" > "$OUTPUT_ROOT/discovered_suite_summaries.txt"
{
  printf '# Shadow Multi-Day Rollup\n\n'
  printf -- '- generated: `%s`\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  printf -- '- output: `%s`\n' "$OUTPUT_ROOT"
  printf -- '- complete suite count: `%s`\n\n' "${#summaries[@]}"
  printf '## Discovered Suites\n\n'
  for path in "${summaries[@]}"; do
    printf -- '- `%s`\n' "$path"
  done
  printf '\n'
} > "$OUTPUT_ROOT/rollup_index.md"

summary_args=()
for path in "${summaries[@]}"; do
  summary_args+=(--suite-summary "$path")
done

"$PY_ML" scripts/ml_dataset/summarize_shadow_suites.py \
  "${summary_args[@]}" \
  --output-json "$OUTPUT_ROOT/shadow_multi_day_aggregate.json" \
  --output-markdown "$OUTPUT_ROOT/shadow_multi_day_aggregate.md"

"$PY_ML" scripts/ml_dataset/assert_shadow_promotion_gate.py \
  --aggregate-json "$OUTPUT_ROOT/shadow_multi_day_aggregate.json" \
  --target wind \
  --candidate router \
  --output-json "$OUTPUT_ROOT/wind_router_promotion_gate.json" \
  --output-markdown "$OUTPUT_ROOT/wind_router_promotion_gate.md" \
  --no-fail-on-reject

"$PY_ML" scripts/ml_dataset/assert_shadow_promotion_gate.py \
  --aggregate-json "$OUTPUT_ROOT/shadow_multi_day_aggregate.json" \
  --target gust \
  --candidate guarded_stacker \
  --output-json "$OUTPUT_ROOT/gust_guarded_stacker_promotion_gate.json" \
  --output-markdown "$OUTPUT_ROOT/gust_guarded_stacker_promotion_gate.md" \
  --no-fail-on-reject

"$PY_ML" scripts/ml_dataset/review_shadow_promotion_candidates.py \
  --aggregate-json "$OUTPUT_ROOT/shadow_multi_day_aggregate.json" \
  --output-json "$OUTPUT_ROOT/promotion_candidate_review.json" \
  --output-markdown "$OUTPUT_ROOT/promotion_candidate_review.md"

mapfile -t threshold_guard_scored < <(
  "${PY_ML}" - "${summaries[@]}" <<'PY'
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

printf '%s\n' "${threshold_guard_scored[@]}" > "$OUTPUT_ROOT/threshold_guard_scored_parquets.txt"
if (( ${#threshold_guard_scored[@]} > 0 )); then
  audit_args=()
  for path in "${threshold_guard_scored[@]}"; do
    audit_args+=(--scored-parquet "$path")
  done
  "$PY_ML" scripts/ml_dataset/audit_threshold_guard_impact.py \
    "${audit_args[@]}" \
    --output-json "$OUTPUT_ROOT/threshold_guard_impact_audit.json" \
    --output-markdown "$OUTPUT_ROOT/threshold_guard_impact_audit.md"
fi

mapfile -t event_head_scored < <(
  "${PY_ML}" - "${summaries[@]}" <<'PY'
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

printf '%s\n' "${event_head_scored[@]}" > "$OUTPUT_ROOT/event_head_scored_parquets.txt"
if (( ${#event_head_scored[@]} > 0 )); then
  event_head_args=()
  for path in "${event_head_scored[@]}"; do
    event_head_args+=(--scored-parquet "$path")
  done
  "$PY_ML" scripts/ml_dataset/audit_shadow_candidate_impact.py \
    "${event_head_args[@]}" \
    --output-json "$OUTPUT_ROOT/shadow_candidate_impact_audit.json" \
    --output-markdown "$OUTPUT_ROOT/shadow_candidate_impact_audit.md"
  "$PY_ML" scripts/ml_dataset/audit_wind_threshold_event_heads.py \
    "${event_head_args[@]}" \
    --output-json "$OUTPUT_ROOT/wind_threshold_event_head_audit.json" \
    --output-markdown "$OUTPUT_ROOT/wind_threshold_event_head_audit.md"
  "$PY_ML" scripts/ml_dataset/audit_gust_threshold_event_heads.py \
    "${event_head_args[@]}" \
    --output-json "$OUTPUT_ROOT/gust_threshold_event_head_audit.json" \
    --output-markdown "$OUTPUT_ROOT/gust_threshold_event_head_audit.md"
fi

decision_args=(--promotion-review-json "$OUTPUT_ROOT/promotion_candidate_review.json")
if [[ -s "$OUTPUT_ROOT/threshold_guard_impact_audit.json" ]]; then
  decision_args+=(--threshold-guard-audit-json "$OUTPUT_ROOT/threshold_guard_impact_audit.json")
fi
if [[ -s "$OUTPUT_ROOT/shadow_candidate_impact_audit.json" ]]; then
  decision_args+=(--candidate-impact-audit-json "$OUTPUT_ROOT/shadow_candidate_impact_audit.json")
fi
"$PY_ML" scripts/ml_dataset/decide_shadow_promotion.py \
  "${decision_args[@]}" \
  --output-json "$OUTPUT_ROOT/promotion_decision.json" \
  --output-markdown "$OUTPUT_ROOT/promotion_decision.md"

specialist_plan_args=(
  --promotion-review-json "$OUTPUT_ROOT/promotion_candidate_review.json"
  --promotion-decision-json "$OUTPUT_ROOT/promotion_decision.json"
)
if [[ -s "$OUTPUT_ROOT/threshold_guard_impact_audit.json" ]]; then
  specialist_plan_args+=(--threshold-guard-audit-json "$OUTPUT_ROOT/threshold_guard_impact_audit.json")
fi
if [[ -s "$OUTPUT_ROOT/shadow_candidate_impact_audit.json" ]]; then
  specialist_plan_args+=(--candidate-impact-audit-json "$OUTPUT_ROOT/shadow_candidate_impact_audit.json")
fi
if [[ -s "$OUTPUT_ROOT/gust_threshold_event_head_audit.json" ]]; then
  specialist_plan_args+=(--gust-event-head-audit-json "$OUTPUT_ROOT/gust_threshold_event_head_audit.json")
fi
if [[ -s "$OUTPUT_ROOT/wind_threshold_event_head_audit.json" ]]; then
  specialist_plan_args+=(--wind-event-head-audit-json "$OUTPUT_ROOT/wind_threshold_event_head_audit.json")
fi
"$PY_ML" scripts/ml_dataset/plan_next_nowcasting_specialists.py \
  "${specialist_plan_args[@]}" \
  --output-json "$OUTPUT_ROOT/next_nowcasting_specialist_plan.json" \
  --output-markdown "$OUTPUT_ROOT/next_nowcasting_specialist_plan.md"

printf 'rollup complete\n'
printf 'output=%s\n' "$OUTPUT_ROOT"
printf 'suite_count=%s\n' "${#summaries[@]}"
printf 'aggregate=%s\n' "$OUTPUT_ROOT/shadow_multi_day_aggregate.json"
printf 'wind_gate=%s\n' "$OUTPUT_ROOT/wind_router_promotion_gate.json"
printf 'gust_gate=%s\n' "$OUTPUT_ROOT/gust_guarded_stacker_promotion_gate.json"
printf 'promotion_review=%s\n' "$OUTPUT_ROOT/promotion_candidate_review.json"
printf 'promotion_decision=%s\n' "$OUTPUT_ROOT/promotion_decision.json"
printf 'next_specialist_plan=%s\n' "$OUTPUT_ROOT/next_nowcasting_specialist_plan.json"
if (( ${#event_head_scored[@]} > 0 )); then
  printf 'shadow_candidate_impact_audit=%s\n' "$OUTPUT_ROOT/shadow_candidate_impact_audit.json"
  printf 'wind_threshold_event_head_audit=%s\n' "$OUTPUT_ROOT/wind_threshold_event_head_audit.json"
  printf 'gust_threshold_event_head_audit=%s\n' "$OUTPUT_ROOT/gust_threshold_event_head_audit.json"
fi
if (( ${#threshold_guard_scored[@]} > 0 )); then
  printf 'threshold_guard_audit=%s\n' "$OUTPUT_ROOT/threshold_guard_impact_audit.json"
fi
