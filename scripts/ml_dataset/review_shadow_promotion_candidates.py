#!/usr/bin/env python3
"""Review all available shadow/current candidates against promotion gates."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assert_shadow_promotion_gate as gate  # noqa: E402


DEFAULT_CANDIDATES = {
    "wind": ("strong_gated", "router", "stacker", "guarded_stacker", "threshold_guard", "high_event_guard"),
    "gust": ("high", "strong_gated", "router", "stacker", "guarded_stacker", "threshold_guard", "local_fallback_guard"),
}

GLOBAL_CHECK_PREFIXES = (
    "coverage_days >=",
    "case_count >=",
    "shadow_case_count >=",
    "joined_rows >=",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary: dict[str, Any], target: str, rail: str) -> dict[str, Any]:
    return (summary.get("overall_ms") or {}).get(f"{target}_{rail}") or {}


def threshold(summary: dict[str, Any], prefix: str, rail: str) -> dict[str, Any]:
    return (summary.get("thresholds") or {}).get(f"{prefix}_{rail}") or {}


def failed_checks(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in result.get("checks") or [] if not item.get("pass")]


def is_global_evidence_check(check: dict[str, Any]) -> bool:
    reason = str(check.get("reason") or "")
    return reason.startswith(GLOBAL_CHECK_PREFIXES)


def split_failed_checks(result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    failures = failed_checks(result)
    global_failures = [item for item in failures if is_global_evidence_check(item)]
    performance_failures = [item for item in failures if not is_global_evidence_check(item)]
    return global_failures, performance_failures


def failure_margin(check: dict[str, Any]) -> dict[str, Any]:
    evidence = check.get("evidence") or {}
    if evidence.get("candidate_rmse_ms") is not None and evidence.get("required_max_rmse_ms") is not None:
        candidate = float(evidence["candidate_rmse_ms"])
        required = float(evidence["required_max_rmse_ms"])
        return {
            "type": "rmse_max",
            "candidate": candidate,
            "required": required,
            "miss_by": candidate - required,
            "unit": "m/s",
        }
    if evidence.get("candidate_csi") is not None and evidence.get("required_min_csi") is not None:
        candidate = float(evidence["candidate_csi"])
        required = float(evidence["required_min_csi"])
        return {
            "type": "csi_min",
            "candidate": candidate,
            "required": required,
            "miss_by": required - candidate,
            "unit": "CSI",
        }
    if evidence.get("candidate_rmse_ms") is not None and evidence.get("allowed_max_rmse_ms") is not None:
        candidate = float(evidence["candidate_rmse_ms"])
        allowed = float(evidence["allowed_max_rmse_ms"])
        return {
            "type": "calm_rmse_max",
            "candidate": candidate,
            "required": allowed,
            "miss_by": candidate - allowed,
            "unit": "m/s",
        }
    return {}


def enrich_failures(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for check in checks:
        enriched = dict(check)
        margin = failure_margin(check)
        if margin:
            enriched["margin"] = margin
        out.append(enriched)
    return out


def total_positive_margin(checks: list[dict[str, Any]], margin_type: str) -> float:
    total = 0.0
    for check in checks:
        margin = check.get("margin") or {}
        if margin.get("type") == margin_type and margin.get("miss_by") is not None:
            total += max(0.0, float(margin["miss_by"]))
    return total


def evidence_progress(args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    days = gate.issue_days(summary)
    return {
        "issue_days": sorted(days),
        "actual_days": len(days),
        "required_days": args.min_days,
        "case_count": int(summary.get("case_count") or 0),
        "required_cases": args.min_cases,
        "shadow_case_count": int(summary.get("shadow_case_count") or 0),
        "required_shadow_cases": args.min_shadow_cases,
        "joined_rows": int(summary.get("joined_rows") or 0),
        "required_rows": args.min_rows,
        "ready": (
            len(days) >= args.min_days
            and int(summary.get("case_count") or 0) >= args.min_cases
            and int(summary.get("shadow_case_count") or 0) >= args.min_shadow_cases
            and int(summary.get("joined_rows") or 0) >= args.min_rows
        ),
    }


def build_gate_args(args: argparse.Namespace, target: str, candidate: str) -> Namespace:
    return Namespace(
        aggregate_json=args.aggregate_json,
        target=target,
        candidate=candidate,
        baseline=args.baseline,
        threshold_baseline=args.threshold_baseline,
        calm_baseline=args.calm_baseline,
        threshold=args.wind_threshold if target == "wind" else args.gust_threshold,
        min_days=args.min_days,
        min_cases=args.min_cases,
        min_shadow_cases=args.min_shadow_cases,
        min_rows=args.min_rows,
        min_rmse_gain_ms=args.min_rmse_gain_ms,
        max_csi_regression=args.max_csi_regression,
        require_calm_regime=args.require_calm_regime,
        calm_regime=args.wind_calm_regime if target == "wind" else args.gust_calm_regime,
        max_calm_rmse_regression_ms=args.max_calm_rmse_regression_ms,
        output_json=None,
        output_markdown=None,
        fail_on_reject=False,
    )


def candidate_record(args: argparse.Namespace, summary: dict[str, Any], target: str, candidate: str) -> dict[str, Any]:
    result = gate.evaluate(build_gate_args(args, target, candidate))
    candidate_metric = metric(summary, target, candidate)
    raw_metric = metric(summary, target, "raw")
    champion_metric = metric(summary, target, "champion")
    candidate_rmse = candidate_metric.get("rmse_ms")
    raw_rmse = raw_metric.get("rmse_ms")
    champion_rmse = champion_metric.get("rmse_ms")
    prefixes = args.wind_threshold if target == "wind" else args.gust_threshold
    prefixes = prefixes or gate.DEFAULT_THRESHOLDS[target]
    thresholds = {
        prefix: {
            "candidate": threshold(summary, prefix, candidate),
            "raw": threshold(summary, prefix, "raw"),
            "champion": threshold(summary, prefix, "champion"),
        }
        for prefix in prefixes
    }
    failures = failed_checks(result)
    global_failures, performance_failures = split_failed_checks(result)
    enriched_global_failures = enrich_failures(global_failures)
    enriched_performance_failures = enrich_failures(performance_failures)
    return {
        "target": target,
        "candidate": candidate,
        "decision": result["decision"],
        "passed": result["passed"],
        "failed_check_count": len(failures),
        "global_failed_check_count": len(global_failures),
        "performance_failed_check_count": len(performance_failures),
        "failed_checks": failures,
        "global_failed_checks": enriched_global_failures,
        "performance_failed_checks": enriched_performance_failures,
        "performance_gap_summary": {
            "rmse_miss_total_ms": total_positive_margin(enriched_performance_failures, "rmse_max"),
            "csi_miss_total": total_positive_margin(enriched_performance_failures, "csi_min"),
            "calm_rmse_miss_total_ms": total_positive_margin(enriched_performance_failures, "calm_rmse_max"),
        },
        "overall_ms": candidate_metric,
        "rmse_gain_vs_raw_ms": None
        if candidate_rmse is None or raw_rmse is None
        else float(raw_rmse) - float(candidate_rmse),
        "rmse_gain_vs_champion_ms": None
        if candidate_rmse is None or champion_rmse is None
        else float(champion_rmse) - float(candidate_rmse),
        "thresholds": thresholds,
        "gate": result,
    }


def candidates_for(summary: dict[str, Any], target: str, requested: list[str]) -> list[str]:
    candidates = requested or list(DEFAULT_CANDIDATES[target])
    available = []
    for candidate in candidates:
        if metric(summary, target, candidate).get("rmse_ms") is not None:
            available.append(candidate)
    return available


def sort_key(item: dict[str, Any]) -> tuple[int, int, float, float, int, float]:
    rmse = item.get("overall_ms", {}).get("rmse_ms")
    gaps = item.get("performance_gap_summary") or {}
    return (
        0 if item.get("passed") else 1,
        int(item.get("performance_failed_check_count") or 0),
        float(gaps.get("csi_miss_total") or 0.0),
        float(gaps.get("rmse_miss_total_ms") or 0.0),
        int(item.get("global_failed_check_count") or 0),
        float("inf") if rmse is None else float(rmse),
    )


def rmse_sort_key(item: dict[str, Any]) -> float:
    rmse = item.get("overall_ms", {}).get("rmse_ms")
    return float("inf") if rmse is None else float(rmse)


def review(args: argparse.Namespace) -> dict[str, Any]:
    summary = read_json(args.aggregate_json)
    wind_candidates = candidates_for(summary, "wind", args.wind_candidate)
    gust_candidates = candidates_for(summary, "gust", args.gust_candidate)
    records = [
        candidate_record(args, summary, "wind", candidate) for candidate in wind_candidates
    ] + [
        candidate_record(args, summary, "gust", candidate) for candidate in gust_candidates
    ]
    by_target: dict[str, list[dict[str, Any]]] = {"wind": [], "gust": []}
    for item in records:
        by_target[item["target"]].append(item)
    for target in by_target:
        by_target[target].sort(key=sort_key)
    by_rmse = {
        target: (min(items, key=rmse_sort_key) if items else None)
        for target, items in by_target.items()
    }
    return {
        "format": "corsewind.shadow_promotion_candidate_review.v1",
        "generated_at_utc": utc_now(),
        "aggregate_json": str(args.aggregate_json),
        "evidence_progress": evidence_progress(args, summary),
        "wind_candidates": wind_candidates,
        "gust_candidates": gust_candidates,
        "by_target": by_target,
        "best": {
            target: (items[0] if items else None)
            for target, items in by_target.items()
        },
        "best_by_rmse": by_rmse,
        "promotable": {
            target: [item for item in items if item.get("passed")]
            for target, items in by_target.items()
        },
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Shadow Promotion Candidate Review",
        "",
        f"- generated: `{result['generated_at_utc']}`",
        f"- aggregate: `{result['aggregate_json']}`",
        "",
    ]
    evidence = result.get("evidence_progress") or {}
    lines.extend(
        [
            "## Evidence Progress",
            "",
            f"- ready: `{evidence.get('ready')}`",
            f"- days: `{evidence.get('actual_days')}/{evidence.get('required_days')}`",
            f"- cases: `{evidence.get('case_count')}/{evidence.get('required_cases')}`",
            f"- shadow cases: `{evidence.get('shadow_case_count')}/{evidence.get('required_shadow_cases')}`",
            f"- rows: `{evidence.get('joined_rows')}/{evidence.get('required_rows')}`",
            "",
        ]
    )
    for target in ("wind", "gust"):
        lines.extend(
            [
                f"## {target.title()} Candidates",
                "",
                "| Candidate | Decision | Global Fails | Performance Fails | RMSE m/s | Gain vs Raw | Gain vs Champion |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in result["by_target"].get(target) or []:
            metric = item.get("overall_ms") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{item['candidate']}`",
                        f"`{item['decision']}`",
                        str(item.get("global_failed_check_count")),
                        str(item.get("performance_failed_check_count")),
                        fmt(metric.get("rmse_ms")),
                        fmt(item.get("rmse_gain_vs_raw_ms")),
                        fmt(item.get("rmse_gain_vs_champion_ms")),
                    ]
                )
                + " |"
            )
        lines.append("")
        best = result.get("best", {}).get(target)
        if best:
            lines.extend([f"Best current `{target}` candidate by gate sort: `{best['candidate']}`.", ""])
            by_rmse = (result.get("best_by_rmse") or {}).get(target)
            if by_rmse and by_rmse.get("candidate") != best.get("candidate"):
                lines.extend([f"Best current `{target}` candidate by RMSE: `{by_rmse['candidate']}`.", ""])
            failures = best.get("performance_failed_checks") or []
            if failures:
                lines.extend(["Top performance failures:", ""])
                for failure in failures[:8]:
                    margin = failure.get("margin") or {}
                    suffix = ""
                    if margin.get("miss_by") is not None:
                        suffix = f" (miss by {fmt(margin.get('miss_by'))} {margin.get('unit')})"
                    lines.append(f"- {failure.get('reason')}{suffix}")
                lines.append("")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    result = review(args)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return result


def parse_calm_regime(value: str) -> tuple[str, str]:
    return gate.parse_calm_regime(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-json", type=Path, required=True)
    parser.add_argument("--wind-candidate", action="append", default=[])
    parser.add_argument("--gust-candidate", action="append", default=[])
    parser.add_argument("--baseline", action="append", default=["raw", "champion"])
    parser.add_argument("--threshold-baseline", action="append", default=["raw", "champion"])
    parser.add_argument("--calm-baseline", action="append", default=["raw", "champion"])
    parser.add_argument("--wind-threshold", action="append", default=[])
    parser.add_argument("--gust-threshold", action="append", default=[])
    parser.add_argument("--min-days", type=int, default=2)
    parser.add_argument("--min-cases", type=int, default=6)
    parser.add_argument("--min-shadow-cases", type=int, default=6)
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--min-rmse-gain-ms", type=float, default=0.02)
    parser.add_argument("--max-csi-regression", type=float, default=0.02)
    parser.add_argument("--require-calm-regime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wind-calm-regime", type=parse_calm_regime)
    parser.add_argument("--gust-calm-regime", type=parse_calm_regime)
    parser.add_argument("--max-calm-rmse-regression-ms", type=float, default=0.02)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
