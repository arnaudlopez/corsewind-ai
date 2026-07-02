#!/usr/bin/env python3
"""Package shadow rollup evidence into a human promotion decision brief."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def missing_evidence(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for actual_key, required_key, label in (
        ("actual_days", "required_days", "issue_days"),
        ("case_count", "required_cases", "cases"),
        ("shadow_case_count", "required_shadow_cases", "shadow_cases"),
        ("joined_rows", "required_rows", "joined_rows"),
    ):
        actual = int(evidence.get(actual_key) or 0)
        required = int(evidence.get(required_key) or 0)
        if required and actual < required:
            rows.append({"metric": label, "actual": actual, "required": required, "missing": required - actual})
    return rows


def slim_candidate(item: dict[str, Any] | None) -> dict[str, Any]:
    if not item:
        return {}
    overall = item.get("overall_ms") or {}
    return {
        "candidate": item.get("candidate"),
        "decision": item.get("decision"),
        "passed": item.get("passed"),
        "rmse_ms": overall.get("rmse_ms"),
        "mae_ms": overall.get("mae_ms"),
        "bias_ms": overall.get("bias_ms"),
        "n": overall.get("n"),
        "rmse_gain_vs_raw_ms": item.get("rmse_gain_vs_raw_ms"),
        "rmse_gain_vs_champion_ms": item.get("rmse_gain_vs_champion_ms"),
        "failed_check_count": item.get("failed_check_count"),
        "global_failed_check_count": item.get("global_failed_check_count"),
        "performance_failed_check_count": item.get("performance_failed_check_count"),
        "performance_gap_summary": item.get("performance_gap_summary") or {},
    }


def top_candidate_table(review: dict[str, Any], target: str, limit: int = 6) -> list[dict[str, Any]]:
    rows = []
    for item in (review.get("by_target") or {}).get(target) or []:
        candidate = slim_candidate(item)
        if candidate:
            rows.append(candidate)
    return rows[:limit]


def local_risk_for(decision: dict[str, Any], target: str) -> dict[str, Any]:
    target_decision = (decision.get("targets") or {}).get(target) or {}
    risk = target_decision.get("local_risk") or {}
    return {
        "flag_count": int(risk.get("flag_count") or 0),
        "top_flags": (risk.get("flags") or [])[:5],
    }


def target_readiness(item: dict[str, Any]) -> dict[str, Any]:
    best = item.get("best_gate_candidate") or {}
    local_risk = item.get("local_risk") or {}
    tolerant_review = item.get("tolerant_threshold_review") or {}
    perf_fails = int(best.get("performance_failed_check_count") or 0)
    local_flags = int(local_risk.get("flag_count") or 0)
    strict_failures = int(tolerant_review.get("strict_failure_count") or 0)
    unresolved_tolerant = int(tolerant_review.get("unresolved_count") or 0)
    blocker = item.get("blocker_type")
    if item.get("decision") == "promote_candidate":
        state = "ready_to_promote"
        reason = "candidate passed evidence, performance, and local-risk gates"
    elif perf_fails > 0 and strict_failures > 0 and unresolved_tolerant == 0 and local_flags == 0:
        state = "strict_threshold_failures_resolved_by_tolerance"
        reason = "strict threshold failures are resolved by the configured near-threshold tolerance"
    elif blocker in {"evidence_only", "global_gate"} and perf_fails == 0 and local_flags == 0:
        state = "waiting_for_evidence_only"
        reason = "candidate has no current performance or local-risk blocker; evidence gate is still incomplete"
    elif blocker in {"evidence_only", "global_gate"} and perf_fails == 0 and local_flags > 0:
        state = "waiting_for_evidence_with_local_risk"
        reason = "candidate performance is clean, but local regressions must disappear or be guarded on more days"
    elif perf_fails > 0 and local_flags == 0:
        state = "needs_performance_specialist"
        reason = "candidate still misses at least one windsurf performance gate"
    elif perf_fails > 0 and local_flags > 0:
        state = "needs_performance_and_local_risk_work"
        reason = "candidate has both performance misses and local regressions"
    elif local_flags > 0:
        state = "needs_local_risk_work"
        reason = "candidate has local regressions versus safer baselines"
    else:
        state = "needs_review"
        reason = f"candidate is not promotable; blocker={blocker}"
    return {
        "state": state,
        "reason": reason,
        "performance_failed_check_count": perf_fails,
        "local_risk_flag_count": local_flags,
    }


def specialist_actions(plan: dict[str, Any], target: str) -> list[dict[str, Any]]:
    target_plan = (plan.get("targets") or {}).get(target) or {}
    actions = target_plan.get("recommended_actions") or target_plan.get("actions") or []
    return list(actions)[:5]


def threshold_margin_notes(event_audit: dict[str, Any], candidate: str | None) -> list[dict[str, Any]]:
    if not event_audit or not candidate:
        return []
    notes = []
    for threshold_name, item in (event_audit.get("thresholds") or {}).items():
        deterministic = item.get("deterministic") or {}
        candidate_metric = deterministic.get(candidate) or {}
        raw_metric = deterministic.get("raw") or {}
        if not candidate_metric or not raw_metric:
            continue
        candidate_csi = candidate_metric.get("csi")
        raw_csi = raw_metric.get("csi")
        if candidate_csi is None or raw_csi is None or candidate_csi >= raw_csi:
            continue
        fp_margin = candidate_metric.get("false_positive_actual_margin_kt") or {}
        fn_margin = candidate_metric.get("false_negative_actual_margin_kt") or {}
        candidate_tolerant = candidate_metric.get("tolerant_by_actual_margin_kt") or {}
        raw_tolerant = raw_metric.get("tolerant_by_actual_margin_kt") or {}
        candidate_tol1 = candidate_tolerant.get("1kt") or {}
        candidate_tol2 = candidate_tolerant.get("2kt") or {}
        raw_tol2 = raw_tolerant.get("2kt") or {}
        fp_count = int(candidate_metric.get("fp") or 0)
        fn_count = int(candidate_metric.get("fn") or 0)
        fp_within_2kt = int(fp_margin.get("within_2kt") or 0)
        fn_within_2kt = int(fn_margin.get("within_2kt") or 0)
        near_threshold_count = fp_within_2kt + fn_within_2kt
        error_count = fp_count + fn_count
        notes.append(
            {
                "threshold": threshold_name,
                "candidate": candidate,
                "candidate_csi": candidate_csi,
                "raw_csi": raw_csi,
                "csi_gap_vs_raw": raw_csi - candidate_csi,
                "false_positive_count": fp_count,
                "false_negative_count": fn_count,
                "near_threshold_error_count_2kt": near_threshold_count,
                "error_count": error_count,
                "all_errors_within_2kt": bool(error_count and near_threshold_count == error_count),
                "candidate_tolerant_csi_1kt": candidate_tol1.get("csi"),
                "candidate_tolerant_csi_2kt": candidate_tol2.get("csi"),
                "raw_tolerant_csi_2kt": raw_tol2.get("csi"),
                "candidate_neutralized_disagreements_2kt": candidate_tol2.get("neutralized_disagreements"),
                "false_positive_margin_kt": fp_margin,
                "false_negative_margin_kt": fn_margin,
            }
        )
    return notes


def refine_readiness_with_threshold_notes(packaged: dict[str, Any]) -> None:
    readiness = packaged.get("readiness") or {}
    notes = packaged.get("threshold_margin_notes") or []
    if readiness.get("state") != "needs_performance_specialist":
        return
    if notes and all(item.get("all_errors_within_2kt") for item in notes):
        readiness["state"] = "needs_threshold_margin_review"
        readiness["reason"] = (
            "candidate misses a threshold CSI gate, but current errors are near the threshold; "
            "review tolerance/noise before training a heavier specialist"
        )


def target_package(
    review: dict[str, Any],
    decision: dict[str, Any],
    plan: dict[str, Any],
    event_audits: dict[str, dict[str, Any]],
    tolerant_gate_review: dict[str, Any],
    target: str,
) -> dict[str, Any]:
    target_decision = (decision.get("targets") or {}).get(target) or {}
    best_name = ((target_decision.get("best") or ((review.get("best") or {}).get(target))) or {}).get("candidate")
    packaged = {
        "target": target,
        "decision": target_decision.get("decision"),
        "blocker_type": target_decision.get("blocker_type"),
        "promoted_candidate": slim_candidate(target_decision.get("candidate")),
        "best_gate_candidate": slim_candidate(target_decision.get("best") or ((review.get("best") or {}).get(target))),
        "best_rmse_candidate": slim_candidate(
            target_decision.get("best_by_rmse") or ((review.get("best_by_rmse") or {}).get(target))
        ),
        "local_risk": local_risk_for(decision, target),
        "top_candidates": top_candidate_table(review, target),
        "recommended_actions": specialist_actions(plan, target),
        "threshold_margin_notes": threshold_margin_notes(event_audits.get(target) or {}, best_name),
        "tolerant_threshold_review": ((tolerant_gate_review.get("targets") or {}).get(target)) or {},
    }
    packaged["readiness"] = target_readiness(packaged)
    refine_readiness_with_threshold_notes(packaged)
    return packaged


def global_recommendation(
    *,
    evidence: dict[str, Any],
    decision: dict[str, Any],
    status: dict[str, Any],
    targets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    status_action = (status.get("recommended_next_action") or {}).get("action")
    status_reason = (status.get("recommended_next_action") or {}).get("reason")
    if status_action == "wait_for_observations":
        return {
            "action": "wait_for_observations",
            "reason": status_reason or "current full-day shadow suite is still waiting for observations",
        }
    if decision.get("decision") == "promote_candidate":
        promoted = [
            {"target": target, "candidate": item.get("promoted_candidate", {}).get("candidate")}
            for target, item in targets.items()
            if item.get("decision") == "promote_candidate"
        ]
        return {
            "action": "review_and_package_promotion_candidate",
            "reason": "at least one candidate passed all current promotion gates",
            "promoted": promoted,
        }
    if not evidence.get("ready"):
        return {
            "action": "continue_shadow_collection",
            "reason": "promotion evidence gate is incomplete",
            "missing_evidence": missing_evidence(evidence),
        }
    blocking = [
        {"target": target, "blocker_type": item.get("blocker_type")}
        for target, item in targets.items()
        if item.get("decision") != "promote_candidate"
    ]
    return {
        "action": "train_or_adjust_next_specialist",
        "reason": "evidence is ready but no candidate passed all gates",
        "blocking_targets": blocking,
    }


def package(args: argparse.Namespace) -> dict[str, Any]:
    review = read_json(args.promotion_review_json)
    decision = read_json(args.promotion_decision_json)
    aggregate = read_json(args.aggregate_json)
    plan = read_json(args.next_specialist_plan_json)
    status = read_json(args.shadow_status_json)
    tolerant_gate_review = read_json(args.tolerant_threshold_gate_review_json)
    event_audits = {
        "wind": read_json(args.wind_event_head_audit_json),
        "gust": read_json(args.gust_event_head_audit_json),
    }
    evidence = decision.get("evidence_progress") or review.get("evidence_progress") or {}
    targets = {
        target: target_package(review, decision, plan, event_audits, tolerant_gate_review, target)
        for target in args.target
    }
    recommendation = global_recommendation(
        evidence=evidence,
        decision=decision,
        status=status,
        targets=targets,
    )
    return {
        "format": "corsewind.shadow_promotion_package.v1",
        "generated_at_utc": utc_now(),
        "rollup_root": str(args.rollup_root) if args.rollup_root else None,
        "aggregate_json": str(args.aggregate_json) if args.aggregate_json else None,
        "promotion_review_json": str(args.promotion_review_json),
        "promotion_decision_json": str(args.promotion_decision_json),
        "next_specialist_plan_json": None if args.next_specialist_plan_json is None else str(args.next_specialist_plan_json),
        "shadow_status_json": None if args.shadow_status_json is None else str(args.shadow_status_json),
        "wind_event_head_audit_json": None if args.wind_event_head_audit_json is None else str(args.wind_event_head_audit_json),
        "gust_event_head_audit_json": None if args.gust_event_head_audit_json is None else str(args.gust_event_head_audit_json),
        "tolerant_threshold_gate_review_json": None
        if args.tolerant_threshold_gate_review_json is None
        else str(args.tolerant_threshold_gate_review_json),
        "evidence": evidence,
        "missing_evidence": missing_evidence(evidence),
        "aggregate": {
            "case_count": aggregate.get("case_count"),
            "shadow_case_count": aggregate.get("shadow_case_count"),
            "joined_rows": aggregate.get("joined_rows"),
            "generated_at_utc": aggregate.get("generated_at_utc"),
        },
        "status": {
            "health": status.get("health") or {},
            "coverage_wait": status.get("coverage_wait") or {},
            "recommended_next_action": status.get("recommended_next_action") or {},
        },
        "decision": decision.get("decision"),
        "recommendation": recommendation,
        "targets": targets,
    }


def render_markdown(result: dict[str, Any]) -> str:
    recommendation = result.get("recommendation") or {}
    evidence = result.get("evidence") or {}
    lines = [
        "# Shadow Promotion Package",
        "",
        f"- generated: `{result.get('generated_at_utc')}`",
        f"- rollup: `{result.get('rollup_root')}`",
        f"- decision: `{result.get('decision')}`",
        f"- recommended action: `{recommendation.get('action')}`",
        f"- reason: `{recommendation.get('reason')}`",
        "",
        "## Evidence",
        "",
        f"- ready: `{evidence.get('ready')}`",
        f"- days: `{evidence.get('actual_days')}/{evidence.get('required_days')}`",
        f"- cases: `{evidence.get('case_count')}/{evidence.get('required_cases')}`",
        f"- shadow cases: `{evidence.get('shadow_case_count')}/{evidence.get('required_shadow_cases')}`",
        f"- rows: `{evidence.get('joined_rows')}/{evidence.get('required_rows')}`",
        "",
    ]
    missing = result.get("missing_evidence") or []
    if missing:
        lines.extend(["### Missing Evidence", "", "| Metric | Actual | Required | Missing |", "| --- | ---: | ---: | ---: |"])
        for item in missing:
            lines.append(f"| `{item['metric']}` | {item['actual']} | {item['required']} | {item['missing']} |")
        lines.append("")

    status = result.get("status") or {}
    if status.get("coverage_wait"):
        coverage = status["coverage_wait"]
        lines.extend(
            [
                "## Live Status",
                "",
                f"- coverage status: `{coverage.get('status')}`",
                f"- coverage reason: `{coverage.get('reason')}`",
                f"- oldest observation: `{coverage.get('min_latest_observation_utc')}`",
                f"- target end: `{coverage.get('target_end_utc')}`",
                "",
            ]
        )

    for target, item in (result.get("targets") or {}).items():
        best = item.get("best_gate_candidate") or {}
        best_rmse = item.get("best_rmse_candidate") or {}
        risk = item.get("local_risk") or {}
        lines.extend(
            [
                f"## {target.title()}",
                "",
                f"- decision: `{item.get('decision')}`",
                f"- blocker: `{item.get('blocker_type')}`",
                f"- best gate candidate: `{best.get('candidate')}` RMSE `{fmt(best.get('rmse_ms'))}` m/s",
                f"- best RMSE candidate: `{best_rmse.get('candidate')}` RMSE `{fmt(best_rmse.get('rmse_ms'))}` m/s",
                f"- gain vs raw: `{fmt(best_rmse.get('rmse_gain_vs_raw_ms'))}` m/s",
                f"- gain vs champion: `{fmt(best_rmse.get('rmse_gain_vs_champion_ms'))}` m/s",
                f"- local risk flags: `{risk.get('flag_count')}`",
                f"- readiness: `{(item.get('readiness') or {}).get('state')}`",
                f"- readiness reason: `{(item.get('readiness') or {}).get('reason')}`",
                "",
                "| Candidate | RMSE | MAE | Bias | Gain Raw | Gain Champion | Perf Fails |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for candidate in item.get("top_candidates") or []:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{candidate.get('candidate')}`",
                        fmt(candidate.get("rmse_ms")),
                        fmt(candidate.get("mae_ms")),
                        fmt(candidate.get("bias_ms")),
                        fmt(candidate.get("rmse_gain_vs_raw_ms")),
                        fmt(candidate.get("rmse_gain_vs_champion_ms")),
                        str(candidate.get("performance_failed_check_count")),
                    ]
                )
                + " |"
            )
        lines.append("")
        if risk.get("top_flags"):
            lines.extend(["Top local risks:", ""])
            for flag in risk.get("top_flags") or []:
                lines.append(
                    f"- `{flag.get('group')}={flag.get('value')}`: candidate RMSE "
                    f"`{fmt(flag.get('candidate_rmse'))}` kt vs `{flag.get('best_baseline')}` "
                    f"`{fmt(flag.get('best_baseline_rmse'))}` kt"
                )
            lines.append("")
        if item.get("threshold_margin_notes"):
            lines.extend(["Threshold margin notes:", ""])
            for note in item.get("threshold_margin_notes") or []:
                lines.append(
                    f"- `{note.get('threshold')}`: CSI gap vs raw `{fmt(note.get('csi_gap_vs_raw'))}`, "
                    f"errors within 2kt `{note.get('near_threshold_error_count_2kt')}/{note.get('error_count')}`, "
                    f"tolerant CSI 2kt `{fmt(note.get('candidate_tolerant_csi_2kt'))}`"
                )
            lines.append("")
        tolerant_review = item.get("tolerant_threshold_review") or {}
        if tolerant_review:
            lines.extend(
                [
                    "Tolerant threshold gate review:",
                    "",
                    f"- strict failures: `{tolerant_review.get('strict_failure_count')}`",
                    f"- resolved by tolerance: `{tolerant_review.get('tolerant_resolved_count')}`",
                    f"- unresolved: `{tolerant_review.get('unresolved_count')}`",
                    "",
                ]
            )
        if item.get("recommended_actions"):
            lines.extend(["Recommended target actions:", ""])
            for action in item.get("recommended_actions") or []:
                lines.append(f"- `{action.get('action')}`: {action.get('why')}")
            lines.append("")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    result = package(args)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollup-root", type=Path)
    parser.add_argument("--aggregate-json", type=Path)
    parser.add_argument("--promotion-review-json", type=Path, required=True)
    parser.add_argument("--promotion-decision-json", type=Path, required=True)
    parser.add_argument("--next-specialist-plan-json", type=Path)
    parser.add_argument("--shadow-status-json", type=Path)
    parser.add_argument("--wind-event-head-audit-json", type=Path)
    parser.add_argument("--gust-event-head-audit-json", type=Path)
    parser.add_argument("--tolerant-threshold-gate-review-json", type=Path)
    parser.add_argument("--target", action="append", choices=("wind", "gust"), default=[])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    if not args.target:
        args.target = ["wind", "gust"]
    if args.rollup_root:
        if args.aggregate_json is None:
            args.aggregate_json = args.rollup_root / "shadow_multi_day_aggregate.json"
        if args.next_specialist_plan_json is None:
            args.next_specialist_plan_json = args.rollup_root / "next_nowcasting_specialist_plan.json"
        if args.wind_event_head_audit_json is None:
            args.wind_event_head_audit_json = args.rollup_root / "wind_threshold_event_head_audit.json"
        if args.gust_event_head_audit_json is None:
            args.gust_event_head_audit_json = args.rollup_root / "gust_threshold_event_head_audit.json"
        if args.tolerant_threshold_gate_review_json is None:
            args.tolerant_threshold_gate_review_json = args.rollup_root / "tolerant_threshold_gate_review.json"
    if args.aggregate_json is None:
        raise SystemExit("--aggregate-json is required when --rollup-root is not provided")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
