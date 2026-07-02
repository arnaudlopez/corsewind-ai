#!/usr/bin/env python3
"""Summarize whether the RMSE-0.9 objective is achieved or what blocks it."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def decide(audit: dict[str, Any], analysis: dict[str, Any], threshold: float) -> dict[str, Any]:
    audit_verdict = audit.get("verdict")
    effective_rmse = as_float(audit.get("effective_rmse"))
    reasons = list(audit.get("reasons") or [])
    warnings = list(audit.get("warnings") or [])
    oracle = analysis.get("oracle_summary") or {}
    spot_lead_oracle = as_float(oracle.get("oracle_by_spot_lead_rmse"))
    row_oracle = as_float(oracle.get("oracle_by_row_rmse"))
    global_best = as_float(oracle.get("global_best_rmse"))
    calibrated_rmse = as_float((analysis.get("overall") or {}).get("rmse"))
    raw_delta = ((analysis.get("prediction_comparison") or {}).get("calibrated_vs_raw") or {}).get("rmse_delta")
    raw_delta = as_float(raw_delta)

    if audit_verdict == "pass" and effective_rmse is not None and effective_rmse < threshold:
        decision = "achieved"
        summary = "RMSE objective is achieved under the leakage-safe audit gate."
        next_actions = [
            "Freeze the exact dataset roots, model artifact, audit JSON, and error analysis as the validated reference.",
            "Promote only after reproducing the same command on z2 from a clean synced repo.",
        ]
    elif audit_verdict == "inconclusive":
        decision = "inconclusive"
        summary = "The run cannot prove or disprove RMSE < threshold because the audit gate is incomplete."
        next_actions = [
            "Resolve audit reasons first; do not tune models against this run.",
            "If reasons mention coverage, rebuild dense RMSE09 sequence roots and training shards.",
            "If reasons mention leakage or missing diagnostics, fix the feature/prediction artifact before rerunning.",
        ]
    elif spot_lead_oracle is not None and spot_lead_oracle < threshold and calibrated_rmse is not None and calibrated_rmse >= threshold:
        decision = "calibration_gap"
        summary = "Available predictors contain enough spot/lead-specific signal, but the trainable calibrator has not learned it yet."
        next_actions = [
            "Increase training cutoffs and keep the 2026 holdout untouched.",
            "Try stronger supervised calibration: LightGBM/CatBoost or per-spot fine-tuning on top of the global residual model.",
            "Inspect oracle spot/lead choices to identify which predictors should be weighted by spot and horizon.",
        ]
    elif row_oracle is not None and row_oracle < threshold and (spot_lead_oracle is None or spot_lead_oracle >= threshold):
        decision = "routing_or_feature_gap"
        summary = "Some row-level signal exists, but it is not captured by stable spot/lead routing."
        next_actions = [
            "Add features that explain when to trust each predictor: recent model error, observation freshness, upwind station state, thermal regime.",
            "Evaluate lead-specific or regime-specific calibrators before adding more foundation models.",
        ]
    elif row_oracle is not None and row_oracle >= threshold:
        decision = "input_signal_gap"
        summary = "Even the optimistic row oracle over available predictors does not reach the threshold."
        next_actions = [
            "Do not expect another calibrator on the same inputs to reach RMSE < threshold.",
            "Add independent inputs: denser observations, true upwind station sequences, AROME vertical/profile gradients, land-sea thermal features.",
            "Rebuild training shards and rerun the dense RMSE09 benchmark after those inputs are present.",
        ]
    else:
        decision = "needs_more_evidence"
        summary = "The audit did not pass and oracle evidence is missing or insufficient to classify the failure."
        next_actions = [
            "Ensure rmse09_error_analysis.json was produced from the best model predictions.",
            "Rerun the experiment with prediction diagnostics and oracle analysis enabled.",
        ]

    if raw_delta is not None and raw_delta > 0:
        warnings.append(f"Calibrated model is worse than raw NWP by RMSE delta {raw_delta}.")

    return {
        "format": "corsewind.rmse09_decision.v1",
        "generated_at_utc": utc_now(),
        "decision": decision,
        "summary": summary,
        "threshold_rmse": threshold,
        "audit_verdict": audit_verdict,
        "audit_effective_rmse": effective_rmse,
        "analysis_overall_rmse": calibrated_rmse,
        "raw_rmse_delta": raw_delta,
        "oracle_summary": oracle,
        "global_best_oracle_rmse": global_best,
        "spot_lead_oracle_rmse": spot_lead_oracle,
        "row_oracle_rmse": row_oracle,
        "audit_reasons": reasons,
        "warnings": warnings,
        "recommended_next_actions": next_actions,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# RMSE09 Decision",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Decision: `{result['decision']}`",
        f"Summary: {result['summary']}",
        "",
        "## Evidence",
        "",
        f"- Threshold RMSE: `{result['threshold_rmse']}`",
        f"- Audit verdict: `{result['audit_verdict']}`",
        f"- Audit effective RMSE: `{result['audit_effective_rmse']}`",
        f"- Analysis overall RMSE: `{result['analysis_overall_rmse']}`",
        f"- Raw RMSE delta: `{result['raw_rmse_delta']}`",
        f"- Global best oracle RMSE: `{result['global_best_oracle_rmse']}`",
        f"- Spot/lead oracle RMSE: `{result['spot_lead_oracle_rmse']}`",
        f"- Row oracle RMSE: `{result['row_oracle_rmse']}`",
        "",
        "## Audit Reasons",
        "",
    ]
    if result["audit_reasons"]:
        lines.extend(f"- {item}" for item in result["audit_reasons"])
    else:
        lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    if result["warnings"]:
        lines.extend(f"- {item}" for item in result["warnings"])
    else:
        lines.append("- None.")
    lines.extend(["", "## Recommended Next Actions", ""])
    lines.extend(f"- {item}" for item in result["recommended_next_actions"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--threshold-rmse", type=float, default=0.9)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = decide(load_json(args.audit_json), load_json(args.analysis_json), args.threshold_rmse)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
