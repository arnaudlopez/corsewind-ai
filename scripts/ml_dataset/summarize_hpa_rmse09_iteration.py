#!/usr/bin/env python3
"""Summarize the hPa RMSE09 iteration and next decision."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def metric_from_calibrator(payload: dict[str, Any] | None) -> dict[str, Any]:
    metrics = (payload or {}).get("calibrated_metrics") or {}
    return {
        "rmse": as_float(metrics.get("rmse")),
        "mae": as_float(metrics.get("mae")),
        "count": metrics.get("count"),
        "run_id": None if not payload else Path(str(payload.get("output_predictions", ""))).parent.name or None,
        "verdict": (payload or {}).get("verdict"),
        "path": None,
    }


def metric_from_tabular_selection(payload: dict[str, Any] | None) -> dict[str, Any]:
    best = (payload or {}).get("best") or {}
    return {
        "rmse": as_float(best.get("corrected_rmse") or best.get("rmse")),
        "mae": as_float(best.get("corrected_mae") or best.get("mae")),
        "count": best.get("metric_count"),
        "run_id": best.get("run_id"),
        "verdict": (payload or {}).get("decision"),
        "path": best.get("path"),
    }


def metric_from_leaderboard(payload: dict[str, Any] | None) -> dict[str, Any]:
    best = (payload or {}).get("best") or {}
    return {
        "rmse": as_float(best.get("rmse")),
        "mae": as_float(best.get("mae")),
        "count": best.get("metric_count"),
        "run_id": best.get("run_id"),
        "verdict": (payload or {}).get("decision"),
        "path": best.get("path"),
    }


def delta(reference: float | None, value: float | None) -> float | None:
    if reference is None or value is None:
        return None
    return round(value - reference, 6)


def gain_pct(reference: float | None, value: float | None) -> float | None:
    if reference is None or value is None or reference == 0:
        return None
    return round((reference - value) / reference * 100.0, 3)


def top_groups(gap_audit: dict[str, Any] | None, group_name: str, limit: int) -> list[dict[str, Any]]:
    return list(((gap_audit or {}).get("groups") or {}).get(group_name) or [])[:limit]


def composite_targets(targets: dict[str, Any] | None) -> dict[str, Any]:
    selected = {}
    for key in ("critical_spots_or_lead_45_60", "lead_45_60", "actual_8plus", "critical_spots"):
        value = ((targets or {}).get("composite_targets") or {}).get(key)
        if value:
            selected[key] = value
    return selected


def decide(args: argparse.Namespace, metrics: dict[str, Any], gap_audit: dict[str, Any] | None) -> dict[str, Any]:
    hpa_best = metrics["best_hpa_rmse"]
    if hpa_best is None:
        return {
            "status": "waiting_for_hpa_artifacts",
            "reasons": ["No hPa benchmark or calibrator RMSE artifact exists yet."],
            "next_action": "Wait for repair -> primary rebuild -> hPa benchmark chain.",
        }
    if hpa_best < args.threshold_rmse:
        return {
            "status": "candidate_achieved",
            "reasons": [f"hPa best RMSE {hpa_best:.6f} is below threshold {args.threshold_rmse:.6f}."],
            "next_action": "Run the formal leakage/coverage assertion gate before marking the goal complete.",
        }
    reasons = [f"hPa best RMSE {hpa_best:.6f} remains above threshold {args.threshold_rmse:.6f}."]
    if hpa_best < args.previous_best_rmse:
        reasons.append("hPa improved the previous best but not enough.")
    elif hpa_best == args.previous_best_rmse:
        reasons.append("hPa did not produce a new best score yet.")
    else:
        reasons.append("hPa worsened or failed to beat the previous best.")
    tail = (gap_audit or {}).get("tail") or {}
    mse_reduction = tail.get("mse_reduction_needed_pct")
    if mse_reduction is not None:
        reasons.append(f"Remaining MSE reduction needed is {mse_reduction}%.")
    return {
        "status": "not_achieved",
        "reasons": reasons,
        "next_action": (
            "Use the hPa gap audit to target the largest remaining groups; if the row-wise oracle "
            "is still far above 0.9, prioritize new input data rather than another calibrator."
        ),
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    status = load_json(args.status_json)
    leaderboard = load_json(args.leaderboard_json)
    hpa_selection = load_json(args.hpa_selection_json)
    calibrator = load_json(args.calibrator_results_json)
    gap_audit = load_json(args.gap_audit_json)
    reduction_targets = load_json(args.reduction_targets_json)

    leaderboard_metric = metric_from_leaderboard(leaderboard)
    hpa_tabular_metric = metric_from_tabular_selection(hpa_selection)
    hpa_calibrator_metric = metric_from_calibrator(calibrator)
    candidates = [
        ("hpa_tabular", hpa_tabular_metric),
        ("hpa_calibrator", hpa_calibrator_metric),
    ]
    available = [(name, item) for name, item in candidates if item.get("rmse") is not None]
    best_name, best_item = min(available, key=lambda pair: pair[1]["rmse"]) if available else (None, {})
    metrics = {
        "previous_best_rmse": args.previous_best_rmse,
        "current_leaderboard_rmse": leaderboard_metric.get("rmse"),
        "hpa_tabular_rmse": hpa_tabular_metric.get("rmse"),
        "hpa_calibrator_rmse": hpa_calibrator_metric.get("rmse"),
        "best_hpa_source": best_name,
        "best_hpa_rmse": best_item.get("rmse"),
        "best_hpa_delta_vs_previous_best": delta(args.previous_best_rmse, best_item.get("rmse")),
        "best_hpa_gain_pct_vs_previous_best": gain_pct(args.previous_best_rmse, best_item.get("rmse")),
        "gap_to_threshold": delta(args.threshold_rmse, best_item.get("rmse")),
    }
    result = {
        "format": "corsewind.hpa_rmse09_iteration_summary.v1",
        "generated_at_utc": utc_now(),
        "threshold_rmse": args.threshold_rmse,
        "status_next_action": (status or {}).get("next_action"),
        "status_warnings": (status or {}).get("warnings") or [],
        "artifacts": {
            "status_json": str(args.status_json) if args.status_json else None,
            "leaderboard_json": str(args.leaderboard_json) if args.leaderboard_json else None,
            "hpa_selection_json": str(args.hpa_selection_json) if args.hpa_selection_json else None,
            "calibrator_results_json": str(args.calibrator_results_json) if args.calibrator_results_json else None,
            "gap_audit_json": str(args.gap_audit_json) if args.gap_audit_json else None,
            "reduction_targets_json": str(args.reduction_targets_json) if args.reduction_targets_json else None,
        },
        "metrics": metrics,
        "gap_audit_overall": None if not gap_audit else {
            "rmse": (gap_audit.get("overall") or {}).get("rmse"),
            "mae": (gap_audit.get("overall") or {}).get("mae"),
            "mse_reduction_needed_pct": (gap_audit.get("tail") or {}).get("mse_reduction_needed_pct"),
            "rowwise_existing_model_oracle_rmse": (
                (gap_audit.get("model_oracle") or {}).get("rowwise_best_existing_model") or {}
            ).get("rmse"),
        },
        "top_spot_lead_groups": top_groups(gap_audit, "spot_id+lead_time_minutes", args.limit),
        "top_lead_groups": top_groups(gap_audit, "lead_time_minutes", args.limit),
        "key_composite_targets": composite_targets(reduction_targets),
    }
    result["decision"] = decide(args, metrics, gap_audit)
    return result


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    decision = result["decision"]
    lines = [
        "# hPa RMSE09 Iteration Summary",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Decision: `{decision['status']}`",
        f"Status next action: `{result.get('status_next_action')}`",
        "",
        "## Metrics",
        "",
        f"- Previous best RMSE: `{metrics['previous_best_rmse']}`",
        f"- hPa tabular RMSE: `{metrics['hpa_tabular_rmse']}`",
        f"- hPa calibrator RMSE: `{metrics['hpa_calibrator_rmse']}`",
        f"- Best hPa source: `{metrics['best_hpa_source']}`",
        f"- Best hPa RMSE: `{metrics['best_hpa_rmse']}`",
        f"- Delta vs previous best: `{metrics['best_hpa_delta_vs_previous_best']}`",
        f"- Gain vs previous best: `{metrics['best_hpa_gain_pct_vs_previous_best']}%`",
        f"- Gap to threshold: `{metrics['gap_to_threshold']}`",
        "",
        "## Decision Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in decision.get("reasons") or [])
    lines.extend(["", "## Next Action", "", decision.get("next_action") or "None."])
    if result.get("top_spot_lead_groups"):
        lines.extend(["", "## Top Spot/Lead Groups", "", "| Group | Count | RMSE | SSE share |", "| --- | ---: | ---: | ---: |"])
        for item in result["top_spot_lead_groups"]:
            lines.append(f"| `{item.get('group')}` | {item.get('count')} | {item.get('rmse')} | {item.get('sse_share_pct')}% |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-json", type=Path)
    parser.add_argument("--leaderboard-json", type=Path)
    parser.add_argument("--hpa-selection-json", type=Path)
    parser.add_argument("--calibrator-results-json", type=Path)
    parser.add_argument("--gap-audit-json", type=Path)
    parser.add_argument("--reduction-targets-json", type=Path)
    parser.add_argument("--previous-best-rmse", type=float, default=1.268019)
    parser.add_argument("--threshold-rmse", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=8)
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
        "best_hpa_source": result["metrics"]["best_hpa_source"],
        "best_hpa_rmse": result["metrics"]["best_hpa_rmse"],
        "delta_vs_previous_best": result["metrics"]["best_hpa_delta_vs_previous_best"],
        "gap_to_threshold": result["metrics"]["gap_to_threshold"],
        "next_action": result["decision"]["next_action"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
