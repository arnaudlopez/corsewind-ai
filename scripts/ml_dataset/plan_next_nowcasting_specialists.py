#!/usr/bin/env python3
"""Plan the next CorseWind nowcasting specialist work from shadow evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GLOBAL_CHECK_PREFIXES = (
    "coverage_days >=",
    "case_count >=",
    "shadow_case_count >=",
    "joined_rows >=",
)

TARGET_THRESHOLDS = {
    "wind": ("12kt", "15kt", "20kt", "25kt"),
    "gust": ("15kt", "20kt", "25kt", "30kt"),
}

HARD_REGIMES = {
    "wind": (
        "actual_wind_regime_kt=20-25kt",
        "actual_wind_regime_kt=25+kt",
        "spot_id=la_tonnara",
        "spot_id=santa_manza",
        "lead_bucket=45-60m",
        "lead_bucket=60m+",
    ),
    "gust": (
        "actual_gust_regime_kt=20-25kt",
        "actual_gust_regime_kt=25+kt",
        "spot_id=la_tonnara",
        "spot_id=santa_manza",
        "lead_bucket=45-60m",
        "lead_bucket=60m+",
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def is_global_failure(check: dict[str, Any]) -> bool:
    reason = str(check.get("reason") or "")
    return reason.startswith(GLOBAL_CHECK_PREFIXES)


def evidence_status(review: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    evidence = decision.get("evidence_progress") or review.get("evidence_progress") or {}
    missing = []
    for actual_key, required_key, label in (
        ("actual_days", "required_days", "issue_days"),
        ("case_count", "required_cases", "cases"),
        ("shadow_case_count", "required_shadow_cases", "shadow_cases"),
        ("joined_rows", "required_rows", "joined_rows"),
    ):
        actual = int(evidence.get(actual_key) or 0)
        required = int(evidence.get(required_key) or 0)
        if required and actual < required:
            missing.append({"metric": label, "actual": actual, "required": required, "missing": required - actual})
    return {
        "ready": bool(evidence.get("ready")),
        "issue_days": evidence.get("issue_days") or [],
        "actual_days": int(evidence.get("actual_days") or 0),
        "case_count": int(evidence.get("case_count") or 0),
        "shadow_case_count": int(evidence.get("shadow_case_count") or 0),
        "joined_rows": int(evidence.get("joined_rows") or 0),
        "missing": missing,
    }


def performance_failures(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    checks = candidate.get("performance_failed_checks")
    if checks is not None:
        return list(checks)
    return [item for item in candidate.get("failed_checks") or [] if not is_global_failure(item)]


def local_risk_flags(audit: dict[str, Any], target: str) -> list[dict[str, Any]]:
    return list((((audit.get("risk_flags") or {}).get("by_target") or {}).get(target) or {}).get("flags") or [])


def candidate_local_risk_flags(candidate_audit: dict[str, Any], target: str, candidate: str | None) -> list[dict[str, Any]]:
    if not candidate:
        return []
    by_target = (candidate_audit.get("risk_flags") or {}).get("by_target_candidate") or {}
    item = ((by_target.get(target) or {}).get(candidate)) or {}
    return list(item.get("flags") or [])


def selected_candidate_name(best: dict[str, Any], best_by_rmse: dict[str, Any]) -> str | None:
    return (best_by_rmse or best or {}).get("candidate")


def risk_priority(flag: dict[str, Any]) -> float:
    rows = float(flag.get("rows") or 0)
    regression = float(flag.get("regression") or flag.get("rmse_regression") or 0)
    return regression * max(rows, 1.0)


def failure_priority(check: dict[str, Any]) -> float:
    margin = check.get("margin") or {}
    miss = float(margin.get("miss_by") or 0)
    evidence = check.get("evidence") or {}
    if margin.get("type") == "csi_min":
        return miss * 100.0
    if evidence.get("candidate_rmse_ms") is not None:
        return miss * 10.0
    return miss


def classify_failure(check: dict[str, Any]) -> str:
    margin = check.get("margin") or {}
    reason = str(check.get("reason") or "")
    evidence = check.get("evidence") or {}
    if margin.get("type") == "csi_min" or "CSI" in reason:
        return "threshold_recall_precision"
    if margin.get("type") == "calm_rmse_max" or evidence.get("regime") in ("<12kt", "<15kt"):
        return "calm_regime_guard"
    if margin.get("type") == "rmse_max":
        return "global_residual_skill"
    return "other_performance"


def top_failed_checks(candidate: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    failures = performance_failures(candidate)
    return sorted(failures, key=failure_priority, reverse=True)[:limit]


def candidate_summary(candidate: dict[str, Any] | None) -> dict[str, Any]:
    if not candidate:
        return {}
    return {
        "candidate": candidate.get("candidate"),
        "decision": candidate.get("decision"),
        "rmse_ms": (candidate.get("overall_ms") or {}).get("rmse_ms"),
        "mae_ms": (candidate.get("overall_ms") or {}).get("mae_ms"),
        "bias_ms": (candidate.get("overall_ms") or {}).get("bias_ms"),
        "rmse_gain_vs_raw_ms": candidate.get("rmse_gain_vs_raw_ms"),
        "rmse_gain_vs_champion_ms": candidate.get("rmse_gain_vs_champion_ms"),
        "failed_check_count": candidate.get("failed_check_count"),
        "performance_failed_check_count": candidate.get("performance_failed_check_count"),
        "performance_gap_summary": candidate.get("performance_gap_summary") or {},
    }


def event_head_summary(event_audits: dict[str, dict[str, Any]], target: str) -> dict[str, Any]:
    event_audit = event_audits.get(target) or {}
    if not event_audit:
        return {}
    thresholds = {}
    for threshold_name, item in (event_audit.get("thresholds") or {}).items():
        thresholds[threshold_name] = {
            "best_deterministic": item.get("best_deterministic") or {},
            "best_probability": item.get("best_probability") or {},
        }
    return {
        "rows": event_audit.get("rows"),
        "thresholds": thresholds,
    }


def work_order_for_target(
    target: str,
    review: dict[str, Any],
    decision: dict[str, Any],
    audit: dict[str, Any],
    candidate_audit: dict[str, Any],
    event_audits: dict[str, dict[str, Any]],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    target_decision = (decision.get("targets") or {}).get(target) or {}
    best = ((review.get("best") or {}).get(target)) or target_decision.get("best") or {}
    best_by_rmse = ((review.get("best_by_rmse") or {}).get(target)) or target_decision.get("best_by_rmse") or {}
    risk_candidate = selected_candidate_name(best, best_by_rmse)
    flags = candidate_local_risk_flags(candidate_audit, target, risk_candidate)
    if not flags and risk_candidate == "threshold_guard":
        flags = local_risk_flags(audit, target)
    flags = sorted(flags, key=risk_priority, reverse=True)
    top_failures = top_failed_checks(best_by_rmse or best)
    failure_classes = {}
    for check in top_failures:
        klass = classify_failure(check)
        failure_classes[klass] = failure_classes.get(klass, 0) + 1

    actions: list[dict[str, Any]] = []
    if not evidence.get("ready"):
        actions.append(
            {
                "priority": 1,
                "action": "collect_more_fresh_shadow_evidence",
                "why": "Promotion evidence gate is not ready; do not promote or overfit small samples.",
                "details": evidence.get("missing") or [],
            }
        )
    if flags:
        actions.append(
            {
                "priority": 2,
                "action": "build_local_risk_fallback_gate",
                "why": "Best RMSE candidate has local regressions versus safer baselines.",
                "details": flags[:8],
            }
        )
    if failure_classes.get("threshold_recall_precision"):
        actions.append(
            {
                "priority": 3,
                "action": "train_threshold_probability_or_event_head",
                "why": "Candidate misses windsurf threshold CSI gates.",
                "details": [item for item in top_failures if classify_failure(item) == "threshold_recall_precision"],
            }
        )
    if failure_classes.get("calm_regime_guard"):
        actions.append(
            {
                "priority": 4,
                "action": "add_calm_regime_guard",
                "why": "Candidate improves some regimes but damages calm reliability.",
                "details": [item for item in top_failures if classify_failure(item) == "calm_regime_guard"],
            }
        )
    if failure_classes.get("global_residual_skill") and evidence.get("ready"):
        actions.append(
            {
                "priority": 5,
                "action": "train_constrained_residual_specialist",
                "why": "Evidence is sufficient but residual RMSE skill is still not enough.",
                "details": [item for item in top_failures if classify_failure(item) == "global_residual_skill"],
            }
        )
    if not actions:
        actions.append(
            {
                "priority": 9,
                "action": "hold_current_candidate_in_shadow",
                "why": "No stronger model action is justified by the current evidence.",
                "details": [],
            }
        )

    actions = sorted(actions, key=lambda item: int(item["priority"]))
    return {
        "target": target,
        "decision": target_decision.get("decision"),
        "blocker_type": target_decision.get("blocker_type"),
        "best_gate_candidate": candidate_summary(best),
        "best_rmse_candidate": candidate_summary(best_by_rmse),
        "local_risk_candidate": risk_candidate,
        "local_risk_flag_count": len(flags),
        "top_local_risk_flags": flags[:8],
        "top_performance_failures": top_failures,
        "event_head_audit": event_head_summary(event_audits, target),
        "hard_regimes_to_keep_in_eval": HARD_REGIMES[target],
        "thresholds_to_keep_in_eval": TARGET_THRESHOLDS[target],
        "recommended_actions": actions,
    }


def global_recommendation(result: dict[str, Any]) -> str:
    evidence = result["evidence"]
    if not evidence["ready"]:
        return "wait_for_fresh_shadow_evidence_and_keep_preparing_specialists"
    targets = result["targets"]
    if any(item.get("local_risk_flag_count") for item in targets.values()):
        return "fix_local_risk_before_promotion"
    if any(item.get("decision") != "promote_candidate" for item in targets.values()):
        return "train_targeted_specialists_from_failure_cells"
    return "package_promotable_candidates"


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    review = read_json(args.promotion_review_json)
    decision = read_json(args.promotion_decision_json)
    audit = read_json(args.threshold_guard_audit_json)
    candidate_audit = read_json(args.candidate_impact_audit_json)
    event_audits = {
        "wind": read_json(args.wind_event_head_audit_json),
        "gust": read_json(args.gust_event_head_audit_json),
    }
    evidence = evidence_status(review, decision)
    targets = args.target or ["wind", "gust"]
    result = {
        "format": "corsewind.next_nowcasting_specialist_plan.v1",
        "generated_at_utc": utc_now(),
        "promotion_review_json": str(args.promotion_review_json) if args.promotion_review_json else None,
        "promotion_decision_json": str(args.promotion_decision_json) if args.promotion_decision_json else None,
        "threshold_guard_audit_json": str(args.threshold_guard_audit_json) if args.threshold_guard_audit_json else None,
        "candidate_impact_audit_json": str(args.candidate_impact_audit_json) if args.candidate_impact_audit_json else None,
        "wind_event_head_audit_json": str(args.wind_event_head_audit_json) if args.wind_event_head_audit_json else None,
        "gust_event_head_audit_json": str(args.gust_event_head_audit_json) if args.gust_event_head_audit_json else None,
        "evidence": evidence,
        "targets": {
            target: work_order_for_target(target, review, decision, audit, candidate_audit, event_audits, evidence)
            for target in targets
        },
    }
    result["global_recommendation"] = global_recommendation(result)
    return result


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(result: dict[str, Any]) -> str:
    evidence = result["evidence"]
    lines = [
        "# Next Nowcasting Specialist Plan",
        "",
        f"Recommendation: `{result['global_recommendation']}`",
        "",
        f"- generated: `{result['generated_at_utc']}`",
        f"- evidence ready: `{evidence.get('ready')}`",
        f"- days/cases/shadow/rows: `{evidence.get('actual_days')}` "
        f"`{evidence.get('case_count')}` `{evidence.get('shadow_case_count')}` `{evidence.get('joined_rows')}`",
    ]
    if evidence.get("missing"):
        lines.extend(["", "## Missing Evidence", "", "| Metric | Actual | Required | Missing |", "| --- | ---: | ---: | ---: |"])
        for item in evidence["missing"]:
            lines.append(f"| `{item['metric']}` | {item['actual']} | {item['required']} | {item['missing']} |")

    for target, item in result["targets"].items():
        best = item.get("best_gate_candidate") or {}
        best_rmse = item.get("best_rmse_candidate") or {}
        lines.extend(
            [
                "",
                f"## {target.title()}",
                "",
                f"- decision: `{item.get('decision')}`",
                f"- blocker: `{item.get('blocker_type')}`",
                f"- best gate candidate: `{best.get('candidate')}` RMSE `{fmt(best.get('rmse_ms'))}`",
                f"- best RMSE candidate: `{best_rmse.get('candidate')}` RMSE `{fmt(best_rmse.get('rmse_ms'))}`",
                f"- local risk candidate: `{item.get('local_risk_candidate')}`",
                f"- local risk flags: `{item.get('local_risk_flag_count')}`",
                "",
                "### Recommended Actions",
                "",
                "| Priority | Action | Why |",
                "| ---: | --- | --- |",
            ]
        )
        for action in item.get("recommended_actions") or []:
            lines.append(f"| {action['priority']} | `{action['action']}` | {action['why']} |")

        risks = item.get("top_local_risk_flags") or []
        if risks:
            lines.extend(
                [
                    "",
                    "### Top Local Risks",
                    "",
                    "| Group | Value | Rows | Candidate RMSE | Best Baseline | Baseline RMSE | Regression |",
                    "| --- | --- | ---: | ---: | --- | ---: | ---: |",
                ]
            )
            for flag in risks:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{flag.get('group')}`",
                            f"`{flag.get('value')}`",
                            str(flag.get("rows")),
                            fmt(flag.get("candidate_rmse")),
                            f"`{flag.get('best_baseline')}`",
                            fmt(flag.get("best_baseline_rmse")),
                            fmt(flag.get("regression") or flag.get("rmse_regression")),
                        ]
                    )
                    + " |"
                )

        failures = item.get("top_performance_failures") or []
        if failures:
            lines.extend(["", "### Top Performance Failures", "", "| Type | Reason | Miss |", "| --- | --- | ---: |"])
            for check in failures:
                margin = check.get("margin") or {}
                lines.append(
                    f"| `{classify_failure(check)}` | {check.get('reason')} | "
                    f"{fmt(margin.get('miss_by'))} {margin.get('unit') or ''} |"
                )
        event_audit = item.get("event_head_audit") or {}
        if event_audit.get("thresholds"):
            lines.extend(
                [
                    "",
                    "### Event Head Audit",
                    "",
                    "| Threshold | Best deterministic | CSI | Best probability | CSI | Cutoff |",
                    "| --- | --- | ---: | --- | ---: | ---: |",
                ]
            )
            for threshold_name, threshold_item in event_audit["thresholds"].items():
                best_det = threshold_item.get("best_deterministic") or {}
                best_prob = threshold_item.get("best_probability") or {}
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{threshold_name}`",
                            f"`{best_det.get('name')}`",
                            fmt(best_det.get("csi")),
                            f"`{best_prob.get('name')}`" if best_prob else "",
                            fmt(best_prob.get("csi")),
                            fmt(best_prob.get("probability_cutoff")),
                        ]
                    )
                    + " |"
                )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    result = build_plan(args)
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
    parser.add_argument("--promotion-review-json", type=Path, required=True)
    parser.add_argument("--promotion-decision-json", type=Path, required=True)
    parser.add_argument("--threshold-guard-audit-json", type=Path)
    parser.add_argument("--candidate-impact-audit-json", type=Path)
    parser.add_argument("--wind-event-head-audit-json", type=Path)
    parser.add_argument("--gust-event-head-audit-json", type=Path)
    parser.add_argument("--target", choices=("wind", "gust"), action="append", default=[])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
