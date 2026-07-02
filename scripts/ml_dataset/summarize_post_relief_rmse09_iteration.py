#!/usr/bin/env python3
"""Summarize a post-relief RMSE09 iteration against established baselines."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def delta(reference: float | None, value: float | None) -> float | None:
    if reference is None or value is None:
        return None
    return round(value - reference, 6)


def gain_pct(reference: float | None, value: float | None) -> float | None:
    if reference is None or value is None or reference == 0:
        return None
    return round((reference - value) / reference * 100.0, 3)


def pick_calibrated_rmse(calibration: dict[str, Any] | None) -> float | None:
    if not calibration:
        return None
    return as_float((calibration.get("calibrated_metrics") or {}).get("rmse"))


def pick_base_rmse(base_audit: dict[str, Any] | None) -> float | None:
    if not base_audit:
        return None
    return as_float(base_audit.get("corrected_rmse"))


def relief_summary(coverage: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not coverage:
        return []
    rows = []
    for item in coverage.get("by_spot") or []:
        rows.append({
            "spot_id": item.get("spot_id"),
            "rows": item.get("rows"),
            "months": item.get("month_count"),
            "global_relief_1_available_rate_pct": item.get("global_relief_1_available_rate_pct"),
            "global_relief_1_station_ids": item.get("global_relief_1_station_ids") or [],
        })
    return rows


def decision(args: argparse.Namespace, base_rmse: float | None, calibrated_rmse: float | None) -> dict[str, Any]:
    best_available = calibrated_rmse if calibrated_rmse is not None else base_rmse
    if best_available is None:
        return {
            "status": "incomplete",
            "reasons": ["Missing post-relief RMSE metrics."],
            "next_action": "Inspect watcher logs and missing artifacts.",
        }
    if best_available < args.threshold_rmse:
        return {
            "status": "candidate_achieved",
            "reasons": [f"Best available RMSE {best_available:.6f} is below threshold {args.threshold_rmse:.6f}."],
            "next_action": "Run the formal RMSE09 assertion gate before marking the goal complete.",
        }
    reasons = [f"Best available RMSE {best_available:.6f} is above threshold {args.threshold_rmse:.6f}."]
    if calibrated_rmse is not None and calibrated_rmse >= args.current_best_calibrated_rmse:
        reasons.append("Post-relief calibrated model did not beat the current calibrated baseline.")
    elif calibrated_rmse is not None:
        reasons.append("Post-relief calibrated model improved the current calibrated baseline but not enough.")
    if base_rmse is not None and base_rmse >= args.current_best_base_rmse:
        reasons.append("Post-relief base model did not beat the current base baseline.")
    elif base_rmse is not None:
        reasons.append("Post-relief base model improved the current base baseline but not enough.")
    return {
        "status": "not_achieved",
        "reasons": reasons,
        "next_action": (
            "If relief coverage is high but RMSE is still far from 0.9, prioritize missing "
            "land-heating/LST proxy coverage, spot-specific high-wind error features, and "
            "45-60 minute lead diagnostics."
        ),
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    base_audit = load_json(args.base_audit)
    calibration = load_json(args.calibration_results)
    coverage = load_json(args.relief_coverage)
    gap_audit = load_json(args.gap_audit)

    base_rmse = pick_base_rmse(base_audit)
    calibrated_rmse = pick_calibrated_rmse(calibration)
    best_rmse = min(value for value in (base_rmse, calibrated_rmse) if value is not None) if any(
        value is not None for value in (base_rmse, calibrated_rmse)
    ) else None
    return {
        "format": "corsewind.post_relief_rmse09_iteration_summary.v1",
        "generated_at_utc": utc_now(),
        "threshold_rmse": args.threshold_rmse,
        "current_best_base_rmse": args.current_best_base_rmse,
        "current_best_calibrated_rmse": args.current_best_calibrated_rmse,
        "artifacts": {
            "base_audit": str(args.base_audit) if args.base_audit else None,
            "calibration_results": str(args.calibration_results) if args.calibration_results else None,
            "relief_coverage": str(args.relief_coverage) if args.relief_coverage else None,
            "gap_audit": str(args.gap_audit) if args.gap_audit else None,
        },
        "metrics": {
            "base_rmse": base_rmse,
            "base_delta_vs_current_best": delta(args.current_best_base_rmse, base_rmse),
            "base_gain_pct_vs_current_best": gain_pct(args.current_best_base_rmse, base_rmse),
            "calibrated_rmse": calibrated_rmse,
            "calibrated_delta_vs_current_best": delta(args.current_best_calibrated_rmse, calibrated_rmse),
            "calibrated_gain_pct_vs_current_best": gain_pct(args.current_best_calibrated_rmse, calibrated_rmse),
            "best_post_relief_rmse": best_rmse,
            "gap_to_threshold": delta(args.threshold_rmse, best_rmse),
        },
        "relief_coverage_by_spot": relief_summary(coverage),
        "gap_audit_overall": None if not gap_audit else {
            "rmse": (gap_audit.get("overall") or {}).get("rmse"),
            "mae": (gap_audit.get("overall") or {}).get("mae"),
            "mse_reduction_needed_pct": (gap_audit.get("tail") or {}).get("mse_reduction_needed_pct"),
            "rowwise_existing_model_oracle_rmse": (
                (gap_audit.get("model_oracle") or {}).get("rowwise_best_existing_model") or {}
            ).get("rmse"),
        },
        "decision": decision(args, base_rmse, calibrated_rmse),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    decision_block = result["decision"]
    lines = [
        "# Post-Relief RMSE09 Iteration Summary",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Decision: `{decision_block['status']}`",
        "",
        "## Metrics",
        "",
        f"- Base RMSE: `{metrics['base_rmse']}`",
        f"- Base delta vs previous best `{result['current_best_base_rmse']}`: `{metrics['base_delta_vs_current_best']}`",
        f"- Calibrated RMSE: `{metrics['calibrated_rmse']}`",
        f"- Calibrated delta vs previous best `{result['current_best_calibrated_rmse']}`: `{metrics['calibrated_delta_vs_current_best']}`",
        f"- Best post-relief RMSE: `{metrics['best_post_relief_rmse']}`",
        f"- Gap to `{result['threshold_rmse']}`: `{metrics['gap_to_threshold']}`",
        "",
        "## Relief Coverage",
        "",
        "| Spot | Rows | Months | Available | Stations |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for item in result.get("relief_coverage_by_spot") or []:
        lines.append(
            f"| `{item.get('spot_id')}` | {item.get('rows')} | {item.get('months')} | "
            f"{item.get('global_relief_1_available_rate_pct')}% | "
            f"`{','.join(item.get('global_relief_1_station_ids') or [])}` |"
        )
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend(f"- {reason}" for reason in decision_block.get("reasons") or [])
    lines.extend(["", "## Next Action", "", decision_block.get("next_action") or "None."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-audit", type=Path)
    parser.add_argument("--calibration-results", type=Path)
    parser.add_argument("--relief-coverage", type=Path)
    parser.add_argument("--gap-audit", type=Path)
    parser.add_argument("--threshold-rmse", type=float, default=0.9)
    parser.add_argument("--current-best-base-rmse", type=float, default=1.276846)
    parser.add_argument("--current-best-calibrated-rmse", type=float, default=1.269403)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize(args)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, result)
    print(json.dumps({
        "decision": result["decision"]["status"],
        "base_rmse": result["metrics"]["base_rmse"],
        "calibrated_rmse": result["metrics"]["calibrated_rmse"],
        "best_post_relief_rmse": result["metrics"]["best_post_relief_rmse"],
        "gap_to_threshold": result["metrics"]["gap_to_threshold"],
        "next_action": result["decision"]["next_action"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
