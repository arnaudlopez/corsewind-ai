#!/usr/bin/env python3
"""Analyze the RMSE-0.9 gap with error contribution and diagnostic oracles."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KEY_COLUMNS = ("issue_time_utc", "spot_id", "lead_time_minutes")
DEFAULT_GROUP_COLUMNS = (
    "spot_id",
    "lead_time_minutes",
    "issue_hour_utc",
    "issue_month",
    "actual_wind_bin_ms",
    "predicted_wind_bin_ms",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def metric(prediction: Any, observation: Any, np: Any) -> dict[str, Any]:
    valid = ~(np.isnan(prediction) | np.isnan(observation))
    prediction = prediction[valid]
    observation = observation[valid]
    if len(prediction) == 0:
        return {"count": 0}
    errors = prediction - observation
    abs_errors = np.abs(errors)
    return {
        "count": int(len(errors)),
        "mae": round(float(np.mean(abs_errors)), 6),
        "rmse": round(float(math.sqrt(float(np.mean(errors * errors)))), 6),
        "bias": round(float(np.mean(errors)), 6),
        "p50_abs_error": round(float(np.quantile(abs_errors, 0.50)), 6),
        "p90_abs_error": round(float(np.quantile(abs_errors, 0.90)), 6),
        "p95_abs_error": round(float(np.quantile(abs_errors, 0.95)), 6),
        "p99_abs_error": round(float(np.quantile(abs_errors, 0.99)), 6),
    }


def metric_frame(frame: Any, prediction_column: str, target_column: str, np: Any) -> dict[str, Any]:
    valid = frame[[prediction_column, target_column]].dropna()
    return metric(
        valid[prediction_column].astype(float).to_numpy(),
        valid[target_column].astype(float).to_numpy(),
        np,
    )


def json_scalar(pd: Any, value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def add_bins(frame: Any, prediction_column: str, target_column: str, pd: Any) -> Any:
    out = frame.copy()
    bins = [-0.001, 2.0, 4.0, 6.0, 8.0, 999.0]
    labels = ["0-2", "2-4", "4-6", "6-8", "8+"]
    out["actual_wind_bin_ms"] = pd.cut(out[target_column].astype(float), bins=bins, labels=labels).astype(str)
    out["predicted_wind_bin_ms"] = pd.cut(out[prediction_column].astype(float), bins=bins, labels=labels).astype(str)
    return out


def load_model_spec(value: str) -> tuple[str, Path, str]:
    parts = value.split("|")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("model spec must be name|parquet_path|prediction_column")
    name, path, column = parts
    if not name:
        raise argparse.ArgumentTypeError("model name cannot be empty")
    return name, Path(path), column


def load_primary(args: argparse.Namespace, pd: Any) -> Any:
    frame = pd.read_parquet(args.predictions)
    frame["issue_time_utc"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    if args.lead_minute:
        frame = frame[frame["lead_time_minutes"].astype("Int64").isin([int(lead) for lead in args.lead_minute])]
    if args.start_utc:
        frame = frame[frame["issue_time_utc"] >= pd.Timestamp(args.start_utc, tz="UTC")]
    if args.end_utc:
        frame = frame[frame["issue_time_utc"] < pd.Timestamp(args.end_utc, tz="UTC")]
    frame = frame.copy()
    if "issue_hour_utc" not in frame.columns:
        frame["issue_hour_utc"] = frame["issue_time_utc"].dt.hour
    if "issue_month" not in frame.columns:
        frame["issue_month"] = frame["issue_time_utc"].dt.strftime("%Y-%m")
    return frame


def merge_model_predictions(primary: Any, specs: list[tuple[str, Path, str]], target_column: str, pd: Any) -> tuple[Any, list[str]]:
    out = primary.copy()
    prediction_columns: list[str] = []
    for name, path, column in specs:
        model_frame = pd.read_parquet(path)
        model_frame["issue_time_utc"] = pd.to_datetime(model_frame["issue_time_utc"], utc=True, errors="coerce")
        keep = [*KEY_COLUMNS, column]
        missing = [item for item in keep if item not in model_frame.columns]
        if missing:
            raise SystemExit(f"{path} is missing required columns for {name}: {missing}")
        out_column = f"model__{name}"
        prediction_columns.append(out_column)
        model_frame = model_frame[keep].rename(columns={column: out_column})
        out = out.merge(model_frame, on=list(KEY_COLUMNS), how="inner", validate="one_to_one")
    if target_column not in out.columns:
        raise SystemExit(f"Primary predictions are missing target column: {target_column}")
    return out, prediction_columns


def grouped_metrics(
    frame: Any,
    group_column: str,
    prediction_column: str,
    target_column: str,
    np: Any,
    pd: Any,
    limit: int,
) -> list[dict[str, Any]]:
    if group_column not in frame.columns:
        return []
    valid = frame.dropna(subset=[group_column, prediction_column, target_column])
    total_sse = float(((valid[prediction_column].astype(float) - valid[target_column].astype(float)) ** 2).sum())
    total_count = int(len(valid))
    current_rmse = math.sqrt(total_sse / total_count) if total_count else 0.0
    rows = []
    for raw_key, group in valid.groupby(group_column, dropna=False):
        errors = group[prediction_column].astype(float) - group[target_column].astype(float)
        sse = float((errors * errors).sum())
        rmse_if_perfect = math.sqrt(max(0.0, total_sse - sse) / total_count) if total_count else 0.0
        item = {
            "group": json_scalar(pd, raw_key),
            "sse_share_pct": round((sse / total_sse * 100.0) if total_sse else 0.0, 3),
            "global_rmse_if_group_perfect": round(float(rmse_if_perfect), 6),
            "global_rmse_gain_if_group_perfect": round(float(current_rmse - rmse_if_perfect), 6),
            **metric_frame(group, prediction_column, target_column, np),
        }
        rows.append(item)
    rows.sort(key=lambda item: (item.get("sse_share_pct", 0), item.get("rmse", 0)), reverse=True)
    return rows[:limit]


def grouped_pair_metrics(
    frame: Any,
    group_columns: list[str],
    prediction_column: str,
    target_column: str,
    np: Any,
    pd: Any,
    limit: int,
) -> list[dict[str, Any]]:
    missing = [column for column in group_columns if column not in frame.columns]
    if missing:
        return []
    valid = frame.dropna(subset=[*group_columns, prediction_column, target_column])
    total_sse = float(((valid[prediction_column].astype(float) - valid[target_column].astype(float)) ** 2).sum())
    total_count = int(len(valid))
    current_rmse = math.sqrt(total_sse / total_count) if total_count else 0.0
    rows = []
    for raw_key, group in valid.groupby(group_columns, dropna=False):
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        errors = group[prediction_column].astype(float) - group[target_column].astype(float)
        sse = float((errors * errors).sum())
        rmse_if_perfect = math.sqrt(max(0.0, total_sse - sse) / total_count) if total_count else 0.0
        rows.append({
            "group": dict(zip(group_columns, [json_scalar(pd, value) for value in values], strict=True)),
            "sse_share_pct": round((sse / total_sse * 100.0) if total_sse else 0.0, 3),
            "global_rmse_if_group_perfect": round(float(rmse_if_perfect), 6),
            "global_rmse_gain_if_group_perfect": round(float(current_rmse - rmse_if_perfect), 6),
            **metric_frame(group, prediction_column, target_column, np),
        })
    rows.sort(key=lambda item: (item.get("sse_share_pct", 0), item.get("rmse", 0)), reverse=True)
    return rows[:limit]


def error_tail(frame: Any, prediction_column: str, target_column: str, threshold_rmse: float, np: Any) -> dict[str, Any]:
    valid = frame[[prediction_column, target_column]].dropna()
    squared = ((valid[prediction_column].astype(float) - valid[target_column].astype(float)) ** 2).to_numpy()
    squared = np.sort(squared)[::-1]
    n_rows = len(squared)
    current_sse = float(squared.sum())
    target_sse = float((threshold_rmse * threshold_rmse) * n_rows)
    excess_sse = max(0.0, current_sse - target_sse)
    cumulative = np.cumsum(squared)
    rows_to_perfect = int(np.searchsorted(cumulative, excess_sse, side="left") + 1) if excess_sse > 0 else 0
    rows_to_perfect = min(rows_to_perfect, n_rows)
    shares = {}
    for pct in (1, 2, 5, 10, 20):
        count = max(1, int(math.ceil(n_rows * pct / 100.0))) if n_rows else 0
        shares[f"top_{pct}_pct_sse_share_pct"] = round(float(squared[:count].sum() / current_sse * 100.0), 3) if current_sse else 0.0
    return {
        "row_count": int(n_rows),
        "current_sse": round(current_sse, 6),
        "target_sse_for_threshold": round(target_sse, 6),
        "excess_sse_above_threshold": round(excess_sse, 6),
        "mse_reduction_needed_pct": round((excess_sse / current_sse * 100.0) if current_sse else 0.0, 3),
        "rows_that_would_need_perfect_fix": rows_to_perfect,
        "rows_that_would_need_perfect_fix_pct": round((rows_to_perfect / n_rows * 100.0) if n_rows else 0.0, 3),
        **shares,
    }


def model_oracles(frame: Any, model_columns: list[str], target_column: str, np: Any) -> dict[str, Any]:
    if not model_columns:
        return {"available": False}
    valid = frame.dropna(subset=[*model_columns, target_column]).copy()
    if valid.empty:
        return {"available": False, "reason": "no common rows"}
    observations = valid[target_column].astype(float).to_numpy()
    metrics = {}
    for column in model_columns:
        metrics[column] = metric(valid[column].astype(float).to_numpy(), observations, np)
    prediction_matrix = valid[model_columns].astype(float).to_numpy()
    abs_errors = np.abs(prediction_matrix - observations[:, None])
    best_index = np.argmin(abs_errors, axis=1)
    rowwise_prediction = prediction_matrix[np.arange(len(valid)), best_index]
    counts = {model_columns[index]: int((best_index == index).sum()) for index in range(len(model_columns))}
    return {
        "available": True,
        "note": "diagnostic_oracle_uses_observed_target_not_valid_for_model_selection",
        "common_rows": int(len(valid)),
        "model_metrics": metrics,
        "rowwise_best_existing_model": {
            **metric(rowwise_prediction, observations, np),
            "selection_counts": counts,
        },
    }


def composite_counterfactuals(
    frame: Any,
    prediction_column: str,
    target_column: str,
    critical_spots: list[str],
    np: Any,
) -> dict[str, Any]:
    valid = frame.dropna(subset=[prediction_column, target_column]).copy()
    if valid.empty:
        return {}
    squared = (valid[prediction_column].astype(float) - valid[target_column].astype(float)) ** 2
    total_sse = float(squared.sum())
    total_count = int(len(valid))
    current_rmse = math.sqrt(total_sse / total_count) if total_count else 0.0

    masks: dict[str, Any] = {
        "actual_8plus": valid[target_column].astype(float) >= 8.0,
        "lead_60": valid["lead_time_minutes"].astype(float) == 60.0,
        "lead_45_60": valid["lead_time_minutes"].astype(float).isin([45.0, 60.0]),
        "actual_8plus_or_lead_45_60": (valid[target_column].astype(float) >= 8.0)
        | valid["lead_time_minutes"].astype(float).isin([45.0, 60.0]),
    }
    if critical_spots:
        critical_mask = valid["spot_id"].astype(str).isin(critical_spots)
        masks["critical_spots"] = critical_mask
        masks["critical_spots_or_actual_8plus"] = critical_mask | (valid[target_column].astype(float) >= 8.0)
        masks["critical_spots_or_lead_45_60"] = critical_mask | valid["lead_time_minutes"].astype(float).isin([45.0, 60.0])

    out: dict[str, Any] = {}
    for name, mask in masks.items():
        group_sse = float(squared[mask].sum())
        rmse_if_perfect = math.sqrt(max(0.0, total_sse - group_sse) / total_count) if total_count else 0.0
        out[name] = {
            "rows": int(mask.sum()),
            "row_share_pct": round(float(mask.mean()) * 100.0, 3),
            "sse_share_pct": round((group_sse / total_sse * 100.0) if total_sse else 0.0, 3),
            "global_rmse_if_perfect": round(float(rmse_if_perfect), 6),
            "global_rmse_gain_if_perfect": round(float(current_rmse - rmse_if_perfect), 6),
        }
    return out


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# RMSE 0.9 Gap Audit",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Verdict: `{result['verdict']}`",
        f"Prediction column: `{result['prediction_column']}`",
        f"Rows: `{result['overall']['count']}`",
        f"RMSE: `{result['overall']['rmse']}`",
        f"Gap to 0.9: `{result['rmse_gap_to_threshold']}`",
        f"MSE reduction needed: `{result['tail']['mse_reduction_needed_pct']}%`",
        f"Rows needing perfect correction: `{result['tail']['rows_that_would_need_perfect_fix']}` "
        f"(`{result['tail']['rows_that_would_need_perfect_fix_pct']}%`)",
        "",
        "## Error Tail",
        "",
    ]
    for key, value in result["tail"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Highest SSE Groups", ""])
    for group_name, rows in result["groups"].items():
        lines.extend([
            f"### {group_name}",
            "",
            "| Group | Count | RMSE | Bias | SSE share | Global RMSE if perfect |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for item in rows:
            lines.append(
                f"| `{item.get('group')}` | {item.get('count')} | {item.get('rmse')} | "
                f"{item.get('bias')} | {item.get('sse_share_pct')}% | "
                f"{item.get('global_rmse_if_group_perfect')} |"
            )
        lines.append("")
    if result.get("composite_counterfactuals"):
        lines.extend([
            "## Composite Perfect-Correction Counterfactuals",
            "",
            "| Mask | Rows | Row share | SSE share | Global RMSE if perfect |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for name, item in result["composite_counterfactuals"].items():
            lines.append(
                f"| `{name}` | {item.get('rows')} | {item.get('row_share_pct')}% | "
                f"{item.get('sse_share_pct')}% | {item.get('global_rmse_if_perfect')} |"
            )
        lines.append("")
    if result["model_oracle"].get("available"):
        oracle = result["model_oracle"]["rowwise_best_existing_model"]
        lines.extend([
            "## Diagnostic Existing-Model Oracle",
            "",
            "This oracle uses the observed target to choose the best model per row. It is not a deployable score.",
            "",
            f"Row-wise oracle RMSE: `{oracle.get('rmse')}`",
            f"Row-wise oracle gap to 0.9: `{round(float(oracle.get('rmse')) - 0.9, 6)}`",
            "",
            "| Model | RMSE | MAE | Bias | Count |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for name, metrics in result["model_oracle"]["model_metrics"].items():
            lines.append(f"| `{name}` | {metrics.get('rmse')} | {metrics.get('mae')} | {metrics.get('bias')} | {metrics.get('count')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    primary = load_primary(args, pd)
    primary = add_bins(primary, args.prediction_column, args.target_column, pd)
    specs = [("primary", args.predictions, args.prediction_column), *args.model]
    frame, model_columns = merge_model_predictions(primary, specs, args.target_column, pd)
    primary_model_column = "model__primary"
    overall = metric_frame(frame, primary_model_column, args.target_column, np)
    tail = error_tail(frame, primary_model_column, args.target_column, args.threshold_rmse, np)
    group_columns = args.group_column or list(DEFAULT_GROUP_COLUMNS)
    groups = {
        column: grouped_metrics(frame, column, primary_model_column, args.target_column, np, pd, args.limit)
        for column in group_columns
    }
    groups["spot_id+lead_time_minutes"] = grouped_pair_metrics(
        frame,
        ["spot_id", "lead_time_minutes"],
        primary_model_column,
        args.target_column,
        np,
        pd,
        args.limit,
    )
    oracle = model_oracles(frame, model_columns, args.target_column, np)
    composites = composite_counterfactuals(frame, primary_model_column, args.target_column, args.critical_spot, np)
    result = {
        "format": "corsewind.rmse09_gap_audit.v1",
        "generated_at_utc": utc_now(),
        "threshold_rmse": args.threshold_rmse,
        "prediction_path": str(args.predictions),
        "prediction_column": args.prediction_column,
        "target_column": args.target_column,
        "lead_minutes": args.lead_minute,
        "overall": overall,
        "rmse_gap_to_threshold": round(float(overall["rmse"]) - args.threshold_rmse, 6),
        "tail": tail,
        "groups": groups,
        "critical_spots": args.critical_spot,
        "composite_counterfactuals": composites,
        "model_oracle": oracle,
        "verdict": "not_achieved",
    }
    if float(overall["rmse"]) < args.threshold_rmse:
        result["verdict"] = "achieved"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-column", default="calibrated_wind_mean_ms")
    parser.add_argument("--target-column", default="actual_wind_mean_ms")
    parser.add_argument("--model", type=load_model_spec, action="append", default=[], help="name|parquet_path|prediction_column")
    parser.add_argument("--lead-minute", type=int, action="append", default=[15, 30, 45, 60])
    parser.add_argument("--start-utc")
    parser.add_argument("--end-utc")
    parser.add_argument("--threshold-rmse", type=float, default=0.9)
    parser.add_argument("--group-column", action="append", default=[])
    parser.add_argument("--critical-spot", action="append", default=[])
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, result)
    print(json.dumps({
        "verdict": result["verdict"],
        "rows": result["overall"].get("count"),
        "rmse": result["overall"].get("rmse"),
        "gap": result["rmse_gap_to_threshold"],
        "mse_reduction_needed_pct": result["tail"]["mse_reduction_needed_pct"],
        "rows_that_would_need_perfect_fix_pct": result["tail"]["rows_that_would_need_perfect_fix_pct"],
        "rowwise_existing_model_oracle_rmse": result["model_oracle"].get("rowwise_best_existing_model", {}).get("rmse"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
