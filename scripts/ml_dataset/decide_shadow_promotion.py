#!/usr/bin/env python3
"""Produce the final promotion decision from a multi-candidate review."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def slim_candidate(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if not item:
        return None
    return {
        "candidate": item.get("candidate"),
        "decision": item.get("decision"),
        "passed": item.get("passed"),
        "failed_check_count": item.get("failed_check_count"),
        "global_failed_check_count": item.get("global_failed_check_count"),
        "performance_failed_check_count": item.get("performance_failed_check_count"),
        "performance_gap_summary": item.get("performance_gap_summary") or {},
        "overall_ms": item.get("overall_ms") or {},
        "rmse_gain_vs_raw_ms": item.get("rmse_gain_vs_raw_ms"),
        "rmse_gain_vs_champion_ms": item.get("rmse_gain_vs_champion_ms"),
    }


def first_promotable(review: dict[str, Any], target: str) -> dict[str, Any] | None:
    items = review.get("promotable", {}).get(target) or []
    return items[0] if items else None


def threshold_guard_risks(audit: dict[str, Any] | None, target: str) -> dict[str, Any]:
    if not audit:
        return {"flag_count": 0, "flags": []}
    return ((audit.get("risk_flags") or {}).get("by_target") or {}).get(target) or {"flag_count": 0, "flags": []}


def candidate_risks(
    *,
    threshold_audit: dict[str, Any] | None,
    candidate_audit: dict[str, Any] | None,
    target: str,
    candidate: str | None,
) -> dict[str, Any]:
    if not candidate:
        return {"flag_count": 0, "flags": []}
    if candidate_audit:
        by_target = (candidate_audit.get("risk_flags") or {}).get("by_target_candidate") or {}
        item = ((by_target.get(target) or {}).get(candidate)) or {}
        if item:
            return {"flag_count": int(item.get("flag_count") or 0), "flags": item.get("flags") or []}
    if candidate == "threshold_guard":
        return threshold_guard_risks(threshold_audit, target)
    return {"flag_count": 0, "flags": []}


def blocker_type(best: dict[str, Any] | None, evidence_ready: bool) -> str:
    if not best:
        return "no_candidate"
    if best.get("passed"):
        return "none"
    performance_failures = int(best.get("performance_failed_check_count") or 0)
    global_failures = int(best.get("global_failed_check_count") or 0)
    if not evidence_ready and performance_failures == 0:
        return "evidence_only"
    if not evidence_ready and performance_failures > 0:
        return "evidence_and_performance"
    if evidence_ready and performance_failures > 0:
        return "performance"
    if global_failures > 0:
        return "global_gate"
    return "unknown"


def target_decision(
    review: dict[str, Any],
    target: str,
    threshold_audit: dict[str, Any] | None,
    candidate_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    evidence = review.get("evidence_progress") or {}
    evidence_ready = bool(evidence.get("ready"))
    promotable = first_promotable(review, target)
    best = (review.get("best") or {}).get(target)
    best_by_rmse = (review.get("best_by_rmse") or {}).get(target)
    best_risks = candidate_risks(
        threshold_audit=threshold_audit,
        candidate_audit=candidate_audit,
        target=target,
        candidate=(best or {}).get("candidate"),
    )
    if promotable:
        promotable_risks = candidate_risks(
            threshold_audit=threshold_audit,
            candidate_audit=candidate_audit,
            target=target,
            candidate=promotable.get("candidate"),
        )
        if int(promotable_risks.get("flag_count") or 0) > 0:
            return {
                "target": target,
                "decision": "do_not_promote",
                "candidate": None,
                "rejected_candidate": slim_candidate(promotable),
                "blocker_type": "local_risk",
                "local_risk": promotable_risks,
                "best": slim_candidate(best),
                "best_by_rmse": slim_candidate(best_by_rmse),
            }
        return {
            "target": target,
            "decision": "promote_candidate",
            "candidate": slim_candidate(promotable),
            "blocker_type": "none",
            "local_risk": promotable_risks,
            "best": slim_candidate(best),
            "best_by_rmse": slim_candidate(best_by_rmse),
        }
    return {
        "target": target,
        "decision": "do_not_promote",
        "candidate": None,
        "blocker_type": blocker_type(best, evidence_ready),
        "local_risk": best_risks,
        "best": slim_candidate(best),
        "best_by_rmse": slim_candidate(best_by_rmse),
    }


def decide(args: argparse.Namespace) -> dict[str, Any]:
    review = read_json(args.promotion_review_json)
    threshold_audit = read_json(args.threshold_guard_audit_json) if args.threshold_guard_audit_json and args.threshold_guard_audit_json.exists() else None
    candidate_audit = read_json(args.candidate_impact_audit_json) if args.candidate_impact_audit_json and args.candidate_impact_audit_json.exists() else None
    targets = args.target or ["wind", "gust"]
    decisions = {
        target: target_decision(review, target, threshold_audit, candidate_audit)
        for target in targets
    }
    promote_any = any(item["decision"] == "promote_candidate" for item in decisions.values())
    return {
        "format": "corsewind.shadow_promotion_decision.v1",
        "generated_at_utc": utc_now(),
        "promotion_review_json": str(args.promotion_review_json),
        "threshold_guard_audit_json": None if args.threshold_guard_audit_json is None else str(args.threshold_guard_audit_json),
        "candidate_impact_audit_json": None if args.candidate_impact_audit_json is None else str(args.candidate_impact_audit_json),
        "evidence_progress": review.get("evidence_progress") or {},
        "decision": "promote_candidate" if promote_any else "do_not_promote",
        "targets": decisions,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(result: dict[str, Any]) -> str:
    evidence = result.get("evidence_progress") or {}
    lines = [
        "# Shadow Promotion Decision",
        "",
        f"Decision: `{result['decision']}`",
        "",
        f"- generated: `{result['generated_at_utc']}`",
        f"- review: `{result['promotion_review_json']}`",
        f"- evidence ready: `{evidence.get('ready')}`",
        f"- days/cases/rows: `{evidence.get('actual_days')}/{evidence.get('required_days')}` "
        f"`{evidence.get('case_count')}/{evidence.get('required_cases')}` "
        f"`{evidence.get('joined_rows')}/{evidence.get('required_rows')}`",
        "",
        "| Target | Decision | Blocker | Local Risk Flags | Best Gate Candidate | Best RMSE Candidate | RMSE m/s | Perf Fails |",
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: |",
    ]
    for target, item in result.get("targets", {}).items():
        best = item.get("best") or {}
        best_rmse = item.get("best_by_rmse") or {}
        best_rmse_metric = best_rmse.get("overall_ms") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{target}`",
                    f"`{item.get('decision')}`",
                    f"`{item.get('blocker_type')}`",
                    str((item.get("local_risk") or {}).get("flag_count") or 0),
                    f"`{best.get('candidate')}`",
                    f"`{best_rmse.get('candidate')}`",
                    fmt(best_rmse_metric.get("rmse_ms")),
                    str(best.get("performance_failed_check_count")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    result = decide(args)
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
    parser.add_argument("--threshold-guard-audit-json", type=Path)
    parser.add_argument("--candidate-impact-audit-json", type=Path)
    parser.add_argument("--target", choices=("wind", "gust"), action="append", default=[])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
