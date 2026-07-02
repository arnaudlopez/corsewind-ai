#!/usr/bin/env python3
"""Audit a tabular residual-correction result against the RMSE-0.9 objective."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TARGET = "labels__residual_wind_mean_ms"


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


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def metric_block(model: dict[str, Any], prefer_eval_leads: bool) -> tuple[str, dict[str, Any]]:
    if prefer_eval_leads and model.get("corrected_nwp_eval_leads"):
        return "eval_leads", model["corrected_nwp_eval_leads"]
    return "all_test_leads", model.get("corrected_nwp_test") or {}


def raw_metric_block(model: dict[str, Any], source: str) -> dict[str, Any]:
    if source == "eval_leads":
        return model.get("raw_nwp_eval_leads") or {}
    return model.get("raw_nwp_test") or {}


def compare_gain(reference: float | None, value: float | None) -> float | None:
    if reference is None or value is None or reference == 0:
        return None
    return round((reference - value) / reference * 100.0, 3)


def worst_groups(groups: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = []
    for key, item in groups.items():
        rmse = as_float((item or {}).get("rmse"))
        if rmse is None:
            continue
        rows.append({
            "key": key,
            "group": (item or {}).get("group"),
            "count": (item or {}).get("count"),
            "rmse": rmse,
            "mae": (item or {}).get("mae"),
            "bias": (item or {}).get("bias"),
        })
    return sorted(rows, key=lambda item: item["rmse"], reverse=True)[:limit]


def group_value_matches_eval_leads(value: Any, eval_leads: set[int]) -> bool:
    if isinstance(value, list):
        return any(group_value_matches_eval_leads(item, eval_leads) for item in value)
    try:
        return int(float(value)) in eval_leads
    except (TypeError, ValueError):
        return False


def skipped_eval_lead_groups(model: dict[str, Any], eval_leads: list[int]) -> list[dict[str, Any]]:
    fit_group_columns = model.get("fit_group_columns") or []
    if "lead_time_minutes" not in fit_group_columns:
        return []
    eval_lead_set = set(int(lead) for lead in eval_leads)
    skipped = []
    for item in model.get("skipped_fit_groups") or []:
        group = item.get("group") or {}
        if group_value_matches_eval_leads(group.get("lead_time_minutes"), eval_lead_set):
            skipped.append(item)
    return skipped


def audit(args: argparse.Namespace) -> dict[str, Any]:
    result = load_json(args.training_results)
    reasons: list[str] = []
    warnings: list[str] = []

    if result.get("format") != "corsewind.residual_correction_parquet_training.v1":
        warnings.append(f"Unexpected result format: {result.get('format')!r}.")

    model = (result.get("models") or {}).get(args.target)
    if not model:
        reasons.append(f"Target {args.target!r} is missing from training results.")
        model = {}

    split_time = result.get("temporal_split_issue_time_utc")
    if args.require_split_time and split_time != args.require_split_time:
        reasons.append(f"Temporal split is {split_time!r}, expected {args.require_split_time!r}.")

    source_parquet_count = as_int(result.get("source_parquet_count"))
    if source_parquet_count is None or source_parquet_count < args.min_source_parquets:
        reasons.append(f"source_parquet_count {source_parquet_count} is below required {args.min_source_parquets}.")

    train_rows = as_int(result.get("train_row_count"))
    if train_rows is None or train_rows < args.min_train_rows:
        reasons.append(f"train_row_count {train_rows} is below required {args.min_train_rows}.")

    test_rows = as_int(result.get("test_row_count"))
    if test_rows is None or test_rows < args.min_test_rows:
        reasons.append(f"test_row_count {test_rows} is below required {args.min_test_rows}.")

    metric_source, corrected = metric_block(model, args.prefer_eval_leads)
    raw = raw_metric_block(model, metric_source)
    corrected_rmse = as_float(corrected.get("rmse"))
    corrected_mae = as_float(corrected.get("mae"))
    raw_rmse = as_float(raw.get("rmse"))
    raw_mae = as_float(raw.get("mae"))
    count = as_int(corrected.get("count"))
    if corrected_rmse is None:
        reasons.append(f"Corrected RMSE is missing for metric source {metric_source}.")
    if count is None or count < args.min_metric_rows:
        reasons.append(f"Metric row count {count} is below required {args.min_metric_rows}.")
    fit_group_count = len(model.get("fit_groups") or [])
    skipped_group_count = len(model.get("skipped_fit_groups") or [])
    skipped_eval_groups = skipped_eval_lead_groups(model, args.eval_lead_minute)
    if skipped_group_count:
        warnings.append(f"{skipped_group_count} fit groups were skipped.")
    if skipped_eval_groups and args.fail_on_skipped_eval_groups:
        reasons.append(
            f"{len(skipped_eval_groups)} skipped fit groups overlap requested eval leads {args.eval_lead_minute}."
        )

    by_lead = {}
    for lead, item in sorted((model.get("corrected_nwp_by_lead") or {}).items(), key=lambda pair: int(pair[0])):
        if args.eval_lead_minute and int(lead) not in set(args.eval_lead_minute):
            continue
        by_lead[str(lead)] = {
            "count": item.get("count"),
            "rmse": item.get("rmse"),
            "mae": item.get("mae"),
            "raw_rmse": (model.get("raw_nwp_by_lead") or {}).get(str(lead), {}).get("rmse"),
        }

    corrected_by_spot = model.get("corrected_nwp_by_spot") or {}
    corrected_by_spot_lead = model.get("corrected_nwp_by_spot_lead") or {}

    verdict = "invalid"
    if not reasons:
        verdict = "achieved" if corrected_rmse is not None and corrected_rmse < args.threshold_rmse else "not_achieved"

    return {
        "format": "corsewind.tabular_rmse09_audit.v1",
        "generated_at_utc": utc_now(),
        "training_results": str(args.training_results),
        "run_id": result.get("run_id"),
        "target": args.target,
        "verdict": verdict,
        "threshold_rmse": args.threshold_rmse,
        "metric_source": metric_source,
        "eval_lead_minutes": args.eval_lead_minute,
        "corrected_rmse": corrected_rmse,
        "corrected_mae": corrected_mae,
        "raw_rmse": raw_rmse,
        "raw_mae": raw_mae,
        "metric_count": count,
        "rmse_gain_pct_vs_raw": compare_gain(raw_rmse, corrected_rmse),
        "previous_best_rmse": args.previous_best_rmse,
        "rmse_gain_pct_vs_previous_best": compare_gain(args.previous_best_rmse, corrected_rmse),
        "rmse_gap_to_threshold": round(corrected_rmse - args.threshold_rmse, 6) if corrected_rmse is not None else None,
        "model_family": result.get("model_family"),
        "temporal_split_issue_time_utc": split_time,
        "source_parquet_count": source_parquet_count,
        "train_row_count": train_rows,
        "test_row_count": test_rows,
        "model_test_row_count": model.get("test_rows"),
        "feature_column_count": result.get("feature_column_count"),
        "fit_group_columns": model.get("fit_group_columns") or [],
        "fit_group_count": fit_group_count,
        "skipped_fit_group_count": skipped_group_count,
        "skipped_eval_lead_group_count": len(skipped_eval_groups),
        "skipped_eval_lead_groups_preview": skipped_eval_groups[: args.worst_group_limit],
        "by_lead": by_lead,
        "worst_spots": worst_groups(corrected_by_spot, args.worst_group_limit),
        "worst_spot_leads": worst_groups(corrected_by_spot_lead, args.worst_group_limit),
        "reasons": reasons,
        "warnings": warnings,
    }


def write_markdown(path: Path, audit_result: dict[str, Any]) -> None:
    lines = [
        "# Tabular RMSE09 Audit",
        "",
        f"Generated: `{audit_result['generated_at_utc']}`",
        f"Run: `{audit_result.get('run_id')}`",
        f"Verdict: `{audit_result['verdict']}`",
        "",
        "## Metric",
        "",
        f"- Source: `{audit_result['metric_source']}`",
        f"- Corrected RMSE: `{audit_result['corrected_rmse']}`",
        f"- Corrected MAE: `{audit_result['corrected_mae']}`",
        f"- Raw NWP RMSE: `{audit_result['raw_rmse']}`",
        f"- Gain vs raw NWP: `{audit_result['rmse_gain_pct_vs_raw']}%`",
        f"- Previous best RMSE: `{audit_result['previous_best_rmse']}`",
        f"- Gain vs previous best: `{audit_result['rmse_gain_pct_vs_previous_best']}%`",
        f"- Gap to 0.9 threshold: `{audit_result['rmse_gap_to_threshold']}`",
        "",
        "## Dataset Gate",
        "",
        f"- Split: `{audit_result['temporal_split_issue_time_utc']}`",
        f"- Source Parquets: `{audit_result['source_parquet_count']}`",
        f"- Train rows: `{audit_result['train_row_count']}`",
        f"- Test rows: `{audit_result['test_row_count']}`",
        f"- Feature columns: `{audit_result['feature_column_count']}`",
        "",
        "## By Lead",
        "",
        "| Lead min | Count | Corrected RMSE | Corrected MAE | Raw RMSE |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for lead, item in audit_result["by_lead"].items():
        lines.append(f"| `{lead}` | `{item.get('count')}` | `{item.get('rmse')}` | `{item.get('mae')}` | `{item.get('raw_rmse')}` |")
    lines.extend(["", "## Worst Spots", "", "| Group | Count | RMSE | MAE | Bias |", "| --- | ---: | ---: | ---: | ---: |"])
    for item in audit_result.get("worst_spots") or []:
        lines.append(f"| `{item.get('key')}` | `{item.get('count')}` | `{item.get('rmse')}` | `{item.get('mae')}` | `{item.get('bias')}` |")
    lines.extend(["", "## Worst Spot Leads", "", "| Group | Count | RMSE | MAE | Bias |", "| --- | ---: | ---: | ---: | ---: |"])
    for item in audit_result.get("worst_spot_leads") or []:
        lines.append(f"| `{item.get('key')}` | `{item.get('count')}` | `{item.get('rmse')}` | `{item.get('mae')}` | `{item.get('bias')}` |")
    lines.extend(["", "## Reasons", ""])
    lines.extend(f"- {item}" for item in audit_result["reasons"]) if audit_result["reasons"] else lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in audit_result["warnings"]) if audit_result["warnings"] else lines.append("- None.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-results", type=Path, required=True)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--threshold-rmse", type=float, default=0.9)
    parser.add_argument("--previous-best-rmse", type=float, default=1.278997)
    parser.add_argument("--require-split-time", default="2026-01-01T00:00:00Z")
    parser.add_argument("--min-source-parquets", type=int, default=30)
    parser.add_argument("--min-train-rows", type=int, default=100000)
    parser.add_argument("--min-test-rows", type=int, default=10000)
    parser.add_argument("--min-metric-rows", type=int, default=10000)
    parser.add_argument("--worst-group-limit", type=int, default=12)
    parser.add_argument("--eval-lead-minute", type=int, action="append", default=[15, 30, 45, 60])
    parser.add_argument("--prefer-eval-leads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-on-skipped-eval-groups", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--fail-unless-achieved", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.fail_unless_achieved and result["verdict"] != "achieved":
        sys.exit(1)


if __name__ == "__main__":
    main()
