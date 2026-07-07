#!/usr/bin/env python3
"""Assert whether a shadow aggregate is strong enough for promotion."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = {
    "wind": ["wind_12kt", "wind_15kt", "wind_20kt", "wind_25kt"],
    "gust": ["gust_12kt", "gust_15kt", "gust_20kt", "gust_25kt"],
}

DEFAULT_CALM_REGIMES = {
    "wind": ("actual_wind", "<12kt"),
    "gust": ("actual_gust", "<15kt"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def issue_days(summary: dict[str, Any]) -> set[str]:
    days: set[str] = set()
    for case in summary.get("cases") or []:
        issue = str(case.get("issue_time_utc") or "")
        if len(issue) >= 10:
            days.add(issue[:10])
    return days


def metric(summary: dict[str, Any], target: str, rail: str) -> dict[str, Any]:
    return (summary.get("overall_ms") or {}).get(f"{target}_{rail}") or {}


def baseline_coverage(summary: dict[str, Any], target: str, candidate: str) -> dict[str, Any]:
    return (
        ((summary.get("baseline_coverage") or {}).get(target) or {})
        .get("candidates", {})
        .get(candidate, {})
    )


def threshold(summary: dict[str, Any], prefix: str, rail: str) -> dict[str, Any]:
    return (summary.get("thresholds") or {}).get(f"{prefix}_{rail}") or {}


def threshold_event_count(item: dict[str, Any]) -> int:
    return int(item.get("tp") or 0) + int(item.get("fp") or 0) + int(item.get("fn") or 0)


def regime_metric(summary: dict[str, Any], target: str, rail: str, group: str, regime: str) -> dict[str, Any]:
    return (((summary.get("regimes_ms") or {}).get(group) or {}).get(regime) or {}).get(f"{target}_{rail}") or {}


def variance_ratio(item: dict[str, Any]) -> float | None:
    value = item.get("variance_ratio")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pass_fail(condition: bool, reason: str, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"pass": bool(condition), "reason": reason, "evidence": evidence or {}}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    summary = read_json(args.aggregate_json)
    target = args.target
    candidate = args.candidate
    candidate_metric = metric(summary, target, candidate)
    checks: list[dict[str, Any]] = []

    days = issue_days(summary)
    checks.append(
        pass_fail(
            len(days) >= args.min_days,
            f"coverage_days >= {args.min_days}",
            {"issue_days": sorted(days), "actual_days": len(days)},
        )
    )
    checks.append(
        pass_fail(
            int(summary.get("case_count") or 0) >= args.min_cases,
            f"case_count >= {args.min_cases}",
            {"case_count": summary.get("case_count")},
        )
    )
    checks.append(
        pass_fail(
            int(summary.get("shadow_case_count") or 0) >= args.min_shadow_cases,
            f"shadow_case_count >= {args.min_shadow_cases}",
            {"shadow_case_count": summary.get("shadow_case_count")},
        )
    )
    checks.append(
        pass_fail(
            int(summary.get("joined_rows") or 0) >= args.min_rows,
            f"joined_rows >= {args.min_rows}",
            {"joined_rows": summary.get("joined_rows")},
        )
    )
    checks.append(
        pass_fail(
            bool(candidate_metric) and candidate_metric.get("rmse_ms") is not None,
            "candidate overall metric exists",
            {"candidate": f"{target}_{candidate}", "metric": candidate_metric},
        )
    )
    min_target_baseline_coverage_ratio = float(getattr(args, "min_target_baseline_coverage_ratio", 1.0))
    joined_rows = int(summary.get("joined_rows") or 0)
    target_coverage_failures = []
    for baseline in args.baseline:
        baseline_metric = metric(summary, target, baseline)
        baseline_n = int(baseline_metric.get("n") or 0)
        ratio = None if joined_rows <= 0 else baseline_n / joined_rows
        if joined_rows > 0 and (ratio is None or ratio < min_target_baseline_coverage_ratio):
            target_coverage_failures.append(
                {
                    "baseline": f"{target}_{baseline}",
                    "baseline_n": baseline_n,
                    "joined_rows": joined_rows,
                    "coverage_ratio": ratio,
                    "required_min_coverage_ratio": min_target_baseline_coverage_ratio,
                    "missing_rows_vs_joined_rows": max(0, joined_rows - baseline_n),
                }
            )
    checks.append(
        pass_fail(
            not target_coverage_failures,
            f"target_baseline_coverage >= {min_target_baseline_coverage_ratio:.3f}",
            {
                "target": target,
                "joined_rows": joined_rows,
                "failures": target_coverage_failures,
            },
        )
    )

    min_baseline_coverage_ratio = float(getattr(args, "min_baseline_coverage_ratio", 1.0))
    coverage = baseline_coverage(summary, target, candidate)
    coverage_baselines = coverage.get("baselines") or {}
    coverage_failures = []
    for baseline in args.baseline:
        item = coverage_baselines.get(baseline) or {}
        candidate_n = int(item.get("candidate_n") or coverage.get("candidate_n") or 0)
        baseline_n = int(item.get("baseline_n") or 0)
        ratio = None if candidate_n <= 0 else baseline_n / candidate_n
        if candidate_n > 0 and (ratio is None or ratio < min_baseline_coverage_ratio):
            coverage_failures.append(
                {
                    "baseline": f"{target}_{baseline}",
                    "candidate": f"{target}_{candidate}",
                    "candidate_n": candidate_n,
                    "baseline_n": baseline_n,
                    "coverage_ratio": ratio,
                    "required_min_coverage_ratio": min_baseline_coverage_ratio,
                    "missing_rows_vs_candidate": max(0, candidate_n - baseline_n),
                }
            )
    checks.append(
        pass_fail(
            not coverage_failures,
            f"same_key_baseline_coverage >= {min_baseline_coverage_ratio:.3f}",
            {
                "candidate": f"{target}_{candidate}",
                "coverage": coverage,
                "failures": coverage_failures,
            },
        )
    )

    if candidate_metric.get("rmse_ms") is not None:
        candidate_rmse = float(candidate_metric["rmse_ms"])
        for baseline in args.baseline:
            baseline_metric = metric(summary, target, baseline)
            baseline_rmse = baseline_metric.get("rmse_ms")
            required = None if baseline_rmse is None else float(baseline_rmse) - args.min_rmse_gain_ms
            checks.append(
                pass_fail(
                    baseline_rmse is not None and candidate_rmse <= float(required),
                    f"{target}_{candidate} RMSE beats {target}_{baseline} by >= {args.min_rmse_gain_ms:.3f} m/s",
                    {
                        "candidate_rmse_ms": candidate_rmse,
                        "baseline": f"{target}_{baseline}",
                        "baseline_rmse_ms": baseline_rmse,
                        "required_max_rmse_ms": required,
                    },
                )
            )

    min_variance_ratio = float(getattr(args, "min_variance_ratio", 0.0) or 0.0)
    if min_variance_ratio > 0.0:
        candidate_variance_ratio = variance_ratio(candidate_metric)
        checks.append(
            pass_fail(
                candidate_variance_ratio is not None and candidate_variance_ratio >= min_variance_ratio,
                f"{target}_{candidate} variance_ratio >= {min_variance_ratio:.3f}",
                {
                    "candidate": f"{target}_{candidate}",
                    "candidate_variance_ratio": candidate_variance_ratio,
                    "required_min_variance_ratio": min_variance_ratio,
                    "metric": candidate_metric,
                },
            )
        )

    threshold_prefixes = args.threshold or DEFAULT_THRESHOLDS[target]
    max_false_alarm_ratio_regression = float(getattr(args, "max_false_alarm_ratio_regression", 0.03))
    for prefix in threshold_prefixes:
        candidate_threshold = threshold(summary, prefix, candidate)
        candidate_csi = candidate_threshold.get("csi")
        candidate_events = threshold_event_count(candidate_threshold)
        checks.append(
            pass_fail(
                bool(candidate_threshold) and (candidate_csi is not None or candidate_events == 0),
                f"{prefix}_{candidate} CSI exists or threshold has no events",
                {
                    "threshold": f"{prefix}_{candidate}",
                    "metric": candidate_threshold,
                    "event_count": candidate_events,
                },
            )
        )
        if candidate_csi is None and candidate_events == 0:
            continue
        if candidate_csi is None:
            continue
        for baseline in args.threshold_baseline:
            baseline_threshold = threshold(summary, prefix, baseline)
            baseline_csi = baseline_threshold.get("csi")
            baseline_events = threshold_event_count(baseline_threshold)
            if baseline_csi is None and baseline_events == 0:
                checks.append(
                    pass_fail(
                        True,
                        f"{prefix}_{baseline} has no events; CSI comparison skipped",
                        {
                            "candidate_csi": candidate_csi,
                            "baseline": f"{prefix}_{baseline}",
                            "baseline_metric": baseline_threshold,
                            "baseline_event_count": baseline_events,
                        },
                    )
                )
                continue
            required_csi = None if baseline_csi is None else float(baseline_csi) - args.max_csi_regression
            checks.append(
                pass_fail(
                    baseline_csi is not None and float(candidate_csi) >= float(required_csi),
                    f"{prefix}_{candidate} CSI does not regress vs {prefix}_{baseline} by more than {args.max_csi_regression:.3f}",
                    {
                        "candidate_csi": candidate_csi,
                        "baseline": f"{prefix}_{baseline}",
                        "baseline_csi": baseline_csi,
                        "required_min_csi": required_csi,
                    },
                )
            )
            candidate_far = candidate_threshold.get("false_alarm_ratio")
            baseline_far = baseline_threshold.get("false_alarm_ratio")
            if candidate_far is not None or baseline_far is not None:
                allowed_far = None if baseline_far is None else float(baseline_far) + max_false_alarm_ratio_regression
                checks.append(
                    pass_fail(
                        candidate_far is not None and baseline_far is not None and float(candidate_far) <= float(allowed_far),
                        (
                            f"{prefix}_{candidate} false_alarm_ratio does not regress vs {prefix}_{baseline} "
                            f"by more than {max_false_alarm_ratio_regression:.3f}"
                        ),
                        {
                            "candidate_false_alarm_ratio": candidate_far,
                            "baseline": f"{prefix}_{baseline}",
                            "baseline_false_alarm_ratio": baseline_far,
                            "allowed_max_false_alarm_ratio": allowed_far,
                        },
                    )
                )

    if args.require_calm_regime:
        group, calm_regime = args.calm_regime or DEFAULT_CALM_REGIMES[target]
        candidate_calm = regime_metric(summary, target, candidate, group, calm_regime)
        candidate_calm_rmse = candidate_calm.get("rmse_ms")
        checks.append(
            pass_fail(
                candidate_calm_rmse is not None,
                f"calm regime metric exists: {group}/{calm_regime}/{target}_{candidate}",
                {"metric": candidate_calm},
            )
        )
        if candidate_calm_rmse is not None:
            for baseline in args.calm_baseline:
                baseline_calm = regime_metric(summary, target, baseline, group, calm_regime)
                baseline_calm_rmse = baseline_calm.get("rmse_ms")
                allowed = None if baseline_calm_rmse is None else float(baseline_calm_rmse) + args.max_calm_rmse_regression_ms
                checks.append(
                    pass_fail(
                        baseline_calm_rmse is not None and float(candidate_calm_rmse) <= float(allowed),
                        f"calm RMSE does not regress vs {target}_{baseline} by more than {args.max_calm_rmse_regression_ms:.3f} m/s",
                        {
                            "group": group,
                            "regime": calm_regime,
                            "candidate_rmse_ms": candidate_calm_rmse,
                            "baseline": f"{target}_{baseline}",
                            "baseline_rmse_ms": baseline_calm_rmse,
                            "allowed_max_rmse_ms": allowed,
                        },
                    )
                )

    require_variance_regime = bool(getattr(args, "require_variance_regime", False))
    variance_regime = getattr(args, "variance_regime", None)
    min_regime_variance_ratio = float(getattr(args, "min_regime_variance_ratio", 0.0) or 0.0)
    if require_variance_regime:
        if not variance_regime:
            checks.append(pass_fail(False, "--require-variance-regime needs --variance-regime", {}))
        else:
            group, variance_regime_name = variance_regime
            candidate_regime = regime_metric(summary, target, candidate, group, variance_regime_name)
            candidate_regime_variance_ratio = variance_ratio(candidate_regime)
            checks.append(
                pass_fail(
                    candidate_regime_variance_ratio is not None
                    and candidate_regime_variance_ratio >= min_regime_variance_ratio,
                    (
                        f"{group}/{variance_regime_name}/{target}_{candidate} variance_ratio "
                        f">= {min_regime_variance_ratio:.3f}"
                    ),
                    {
                        "group": group,
                        "regime": variance_regime_name,
                        "candidate": f"{target}_{candidate}",
                        "candidate_variance_ratio": candidate_regime_variance_ratio,
                        "required_min_variance_ratio": min_regime_variance_ratio,
                        "metric": candidate_regime,
                    },
                )
            )

    passed = all(item["pass"] for item in checks)
    return {
        "format": "corsewind.shadow_promotion_gate.v1",
        "generated_at_utc": utc_now(),
        "aggregate_json": str(args.aggregate_json),
        "target": target,
        "candidate": candidate,
        "decision": "promote_candidate" if passed else "do_not_promote",
        "passed": passed,
        "checks": checks,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Shadow Promotion Gate",
        "",
        f"Decision: `{result['decision']}`",
        "",
        f"- target: `{result['target']}`",
        f"- candidate: `{result['candidate']}`",
        f"- aggregate: `{result['aggregate_json']}`",
        "",
        "| Pass | Check | Evidence |",
        "| --- | --- | --- |",
    ]
    for check in result["checks"]:
        evidence = json.dumps(check.get("evidence") or {}, sort_keys=True, default=str)
        lines.append(f"| `{check['pass']}` | {check['reason']} | `{evidence}` |")
    return "\n".join(lines) + "\n"


def parse_calm_regime(value: str) -> tuple[str, str]:
    if "/" not in value:
        raise argparse.ArgumentTypeError("--calm-regime must use group/regime, e.g. actual_wind/<12kt")
    group, regime = value.split("/", 1)
    return group, regime


def run(args: argparse.Namespace) -> dict[str, Any]:
    result = evaluate(args)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if not result["passed"] and args.fail_on_reject:
        raise SystemExit(1)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-json", type=Path, required=True)
    parser.add_argument("--target", choices=("wind", "gust"), required=True)
    parser.add_argument("--candidate", required=True, help="Rail suffix, e.g. router or guarded_stacker.")
    parser.add_argument("--baseline", action="append", default=["raw", "champion"])
    parser.add_argument("--threshold-baseline", action="append", default=["raw", "champion"])
    parser.add_argument("--calm-baseline", action="append", default=["raw", "champion"])
    parser.add_argument("--threshold", action="append", default=[])
    parser.add_argument("--min-days", type=int, default=2)
    parser.add_argument("--min-cases", type=int, default=6)
    parser.add_argument("--min-shadow-cases", type=int, default=6)
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--min-rmse-gain-ms", type=float, default=0.02)
    parser.add_argument("--min-target-baseline-coverage-ratio", type=float, default=1.0)
    parser.add_argument("--min-baseline-coverage-ratio", type=float, default=1.0)
    parser.add_argument("--max-csi-regression", type=float, default=0.02)
    parser.add_argument("--max-false-alarm-ratio-regression", type=float, default=0.03)
    parser.add_argument("--min-variance-ratio", type=float, default=0.0)
    parser.add_argument("--require-calm-regime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calm-regime", type=parse_calm_regime)
    parser.add_argument("--max-calm-rmse-regression-ms", type=float, default=0.02)
    parser.add_argument("--require-variance-regime", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--variance-regime", type=parse_calm_regime)
    parser.add_argument("--min-regime-variance-ratio", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--fail-on-reject", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
