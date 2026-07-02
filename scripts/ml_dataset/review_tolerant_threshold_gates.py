#!/usr/bin/env python3
"""Review near-threshold tolerant CSI for current shadow promotion candidates."""

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


def threshold_audit_for_target(event_audits: dict[str, dict[str, Any]], target: str) -> dict[str, Any]:
    return event_audits.get(target) or {}


def selected_candidate(review: dict[str, Any], target: str) -> dict[str, Any]:
    return ((review.get("best") or {}).get(target)) or {}


def deterministic_metric(event_audit: dict[str, Any], threshold_name: str, rail: str) -> dict[str, Any]:
    return (((event_audit.get("thresholds") or {}).get(threshold_name) or {}).get("deterministic") or {}).get(rail) or {}


def tolerant_metric(metric: dict[str, Any], tolerance: float) -> dict[str, Any]:
    key = f"{tolerance:g}kt"
    return (metric.get("tolerant_by_actual_margin_kt") or {}).get(key) or {}


def strict_csi_check(candidate_metric: dict[str, Any], baseline_metric: dict[str, Any], max_regression: float) -> dict[str, Any]:
    candidate_csi = candidate_metric.get("csi")
    baseline_csi = baseline_metric.get("csi")
    candidate_events = threshold_event_count(candidate_metric)
    baseline_events = threshold_event_count(baseline_metric)
    if candidate_csi is None and baseline_csi is None and candidate_events == 0 and baseline_events == 0:
        return {
            "candidate_csi": candidate_csi,
            "baseline_csi": baseline_csi,
            "required_min_csi": None,
            "passes": True,
            "skipped": True,
            "skip_reason": "candidate and baseline have no threshold events",
        }
    if candidate_csi is None or baseline_csi is None:
        return {
            "candidate_csi": candidate_csi,
            "baseline_csi": baseline_csi,
            "required_min_csi": None,
            "passes": False,
        }
    required = float(baseline_csi) - max_regression
    return {
        "candidate_csi": candidate_csi,
        "baseline_csi": baseline_csi,
        "required_min_csi": required,
        "passes": float(candidate_csi) >= required,
        "miss_by": max(0.0, required - float(candidate_csi)),
    }


def threshold_event_count(metric: dict[str, Any]) -> int:
    return int(metric.get("tp") or 0) + int(metric.get("fp") or 0) + int(metric.get("fn") or 0)


def tolerant_csi_check(
    candidate_metric: dict[str, Any],
    baseline_metric: dict[str, Any],
    tolerance: float,
    max_regression: float,
) -> dict[str, Any]:
    candidate_tol = tolerant_metric(candidate_metric, tolerance)
    baseline_tol = tolerant_metric(baseline_metric, tolerance)
    candidate_csi = candidate_tol.get("csi")
    baseline_csi = baseline_tol.get("csi")
    if candidate_csi is None or baseline_csi is None:
        return {
            "tolerance_kt": tolerance,
            "candidate_csi": candidate_csi,
            "baseline_csi": baseline_csi,
            "required_min_csi": None,
            "passes": False,
            "candidate_neutralized_disagreements": candidate_tol.get("neutralized_disagreements"),
            "baseline_neutralized_disagreements": baseline_tol.get("neutralized_disagreements"),
        }
    required = float(baseline_csi) - max_regression
    return {
        "tolerance_kt": tolerance,
        "candidate_csi": candidate_csi,
        "baseline_csi": baseline_csi,
        "required_min_csi": required,
        "passes": float(candidate_csi) >= required,
        "miss_by": max(0.0, required - float(candidate_csi)),
        "candidate_neutralized_disagreements": candidate_tol.get("neutralized_disagreements"),
        "baseline_neutralized_disagreements": baseline_tol.get("neutralized_disagreements"),
    }


def review_target(args: argparse.Namespace, review: dict[str, Any], event_audits: dict[str, dict[str, Any]], target: str) -> dict[str, Any]:
    event_audit = threshold_audit_for_target(event_audits, target)
    candidate_record = selected_candidate(review, target)
    candidate = candidate_record.get("candidate")
    thresholds = []
    strict_failures = []
    tolerant_resolved = []
    unresolved = []
    if not candidate:
        return {
            "target": target,
            "candidate": None,
            "thresholds": [],
            "strict_failure_count": 0,
            "tolerant_resolved_count": 0,
            "unresolved_count": 0,
        }

    for threshold_name, threshold_payload in (event_audit.get("thresholds") or {}).items():
        candidate_metric = deterministic_metric(event_audit, threshold_name, candidate)
        if not candidate_metric:
            continue
        baseline_reviews = []
        threshold_has_strict_failure = False
        threshold_resolved_by_tolerance = False
        for baseline in args.baseline:
            baseline_metric = deterministic_metric(event_audit, threshold_name, baseline)
            if not baseline_metric:
                continue
            strict = strict_csi_check(candidate_metric, baseline_metric, args.max_csi_regression)
            tolerant = [
                tolerant_csi_check(candidate_metric, baseline_metric, tolerance, args.max_csi_regression)
                for tolerance in args.tolerance_kt
            ]
            best_tolerant = next((item for item in tolerant if item.get("passes")), None)
            item = {
                "baseline": baseline,
                "strict": strict,
                "tolerant": tolerant,
                "resolved_by_tolerance": bool((not strict.get("passes")) and best_tolerant),
                "best_passing_tolerance_kt": None if best_tolerant is None else best_tolerant.get("tolerance_kt"),
            }
            baseline_reviews.append(item)
            if strict.get("skipped"):
                continue
            if not strict.get("passes"):
                threshold_has_strict_failure = True
                strict_failures.append(
                    {
                        "threshold": threshold_name,
                        "baseline": baseline,
                        "strict": strict,
                        "best_passing_tolerance_kt": item["best_passing_tolerance_kt"],
                    }
                )
                if best_tolerant:
                    threshold_resolved_by_tolerance = True
                else:
                    unresolved.append({"threshold": threshold_name, "baseline": baseline, "strict": strict})
        if threshold_has_strict_failure and threshold_resolved_by_tolerance:
            tolerant_resolved.append(threshold_name)
        thresholds.append(
            {
                "threshold": threshold_name,
                "candidate": candidate,
                "candidate_strict": {
                    "csi": candidate_metric.get("csi"),
                    "precision": candidate_metric.get("precision"),
                    "recall": candidate_metric.get("recall"),
                    "tp": candidate_metric.get("tp"),
                    "fp": candidate_metric.get("fp"),
                    "fn": candidate_metric.get("fn"),
                },
                "baselines": baseline_reviews,
            }
        )
    return {
        "target": target,
        "candidate": candidate,
        "thresholds": thresholds,
        "strict_failure_count": len(strict_failures),
        "tolerant_resolved_count": len(tolerant_resolved),
        "unresolved_count": len(unresolved),
        "strict_failures": strict_failures,
        "tolerant_resolved_thresholds": sorted(set(tolerant_resolved)),
        "unresolved": unresolved,
    }


def review(args: argparse.Namespace) -> dict[str, Any]:
    promotion_review = read_json(args.promotion_review_json)
    event_audits = {
        "wind": read_json(args.wind_event_head_audit_json),
        "gust": read_json(args.gust_event_head_audit_json),
    }
    targets = {
        target: review_target(args, promotion_review, event_audits, target)
        for target in args.target
    }
    return {
        "format": "corsewind.tolerant_threshold_gate_review.v1",
        "generated_at_utc": utc_now(),
        "promotion_review_json": str(args.promotion_review_json),
        "wind_event_head_audit_json": None if args.wind_event_head_audit_json is None else str(args.wind_event_head_audit_json),
        "gust_event_head_audit_json": None if args.gust_event_head_audit_json is None else str(args.gust_event_head_audit_json),
        "max_csi_regression": args.max_csi_regression,
        "tolerance_kt": args.tolerance_kt,
        "targets": targets,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Tolerant Threshold Gate Review",
        "",
        f"- generated: `{result['generated_at_utc']}`",
        f"- max CSI regression: `{result['max_csi_regression']}`",
        f"- tolerances kt: `{', '.join(str(x) for x in result['tolerance_kt'])}`",
        "",
    ]
    for target, item in (result.get("targets") or {}).items():
        lines.extend(
            [
                f"## {target.title()}",
                "",
                f"- candidate: `{item.get('candidate')}`",
                f"- strict failures: `{item.get('strict_failure_count')}`",
                f"- tolerant resolved thresholds: `{', '.join(item.get('tolerant_resolved_thresholds') or [])}`",
                f"- unresolved failures: `{item.get('unresolved_count')}`",
                "",
                "| Threshold | Baseline | Strict CSI | Baseline CSI | Strict Pass | Tol1 CSI | Tol2 CSI | Tol2 Pass |",
                "| --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
            ]
        )
        for threshold in item.get("thresholds") or []:
            for baseline_review in threshold.get("baselines") or []:
                strict = baseline_review.get("strict") or {}
                tolerant = {f"{entry.get('tolerance_kt'):g}kt": entry for entry in baseline_review.get("tolerant") or []}
                tol1 = tolerant.get("1kt") or {}
                tol2 = tolerant.get("2kt") or {}
                strict_pass = "skipped" if strict.get("skipped") else str(strict.get("passes"))
                tol2_pass = "skipped" if strict.get("skipped") else str(tol2.get("passes"))
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{threshold.get('threshold')}`",
                            f"`{baseline_review.get('baseline')}`",
                            fmt(strict.get("candidate_csi")),
                            fmt(strict.get("baseline_csi")),
                            f"`{strict_pass}`",
                            fmt(tol1.get("candidate_csi")),
                            fmt(tol2.get("candidate_csi")),
                            f"`{tol2_pass}`",
                        ]
                    )
                    + " |"
                )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion-review-json", type=Path, required=True)
    parser.add_argument("--wind-event-head-audit-json", type=Path)
    parser.add_argument("--gust-event-head-audit-json", type=Path)
    parser.add_argument("--target", choices=("wind", "gust"), action="append", default=[])
    parser.add_argument("--baseline", action="append", default=["raw", "champion"])
    parser.add_argument("--tolerance-kt", type=float, action="append", default=[1.0, 2.0])
    parser.add_argument("--max-csi-regression", type=float, default=0.02)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    if not args.target:
        args.target = ["wind", "gust"]
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
