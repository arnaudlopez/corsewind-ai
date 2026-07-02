#!/usr/bin/env python3
"""Route between two prediction models using a calibration-selected non-leaky gate."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KEY_COLUMNS = ("issue_time_utc", "spot_id", "lead_time_minutes")
SAFE_GATE_COLUMNS = (
    "base_corrected_wind_mean_ms",
    "alt_corrected_wind_mean_ms",
    "base_raw_wind_mean_ms",
    "base_predicted_residual_wind_mean_ms",
    "alt_predicted_residual_wind_mean_ms",
    "max_corrected_wind_mean_ms",
    "mean_corrected_wind_mean_ms",
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
    return {
        "count": int(len(errors)),
        "mae": round(float(np.mean(np.abs(errors))), 6),
        "rmse": round(float(math.sqrt(float(np.mean(errors * errors)))), 6),
        "bias": round(float(np.mean(errors)), 6),
        "p50_abs_error": round(float(np.quantile(np.abs(errors), 0.50)), 6),
        "p90_abs_error": round(float(np.quantile(np.abs(errors), 0.90)), 6),
        "p95_abs_error": round(float(np.quantile(np.abs(errors), 0.95)), 6),
    }


def metric_frame(frame: Any, prediction_column: str, np: Any) -> dict[str, Any]:
    valid = frame[[prediction_column, "actual_wind_mean_ms"]].dropna()
    return metric(
        valid[prediction_column].astype(float).to_numpy(),
        valid["actual_wind_mean_ms"].astype(float).to_numpy(),
        np,
    )


def grouped_metrics(frame: Any, group_columns: list[str], prediction_column: str, np: Any, pd: Any, limit: int | None = None) -> list[dict[str, Any]]:
    if any(column not in frame.columns for column in group_columns):
        return []
    rows = []
    valid = frame.dropna(subset=[prediction_column, "actual_wind_mean_ms"])
    for raw_key, group in valid.groupby(group_columns, dropna=False):
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        rows.append({
            "group": dict(zip(group_columns, [json_scalar(pd, value) for value in values], strict=True)),
            **metric_frame(group, prediction_column, np),
        })
    rows.sort(key=lambda item: (item.get("rmse") is None, item.get("rmse", -1)), reverse=True)
    return rows[:limit] if limit else rows


def json_scalar(pd: Any, value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def load_pair(base_path: Path, alt_path: Path, pd: Any, leads: list[int]) -> Any:
    base = pd.read_parquet(base_path)
    alt = pd.read_parquet(alt_path)
    for frame in (base, alt):
        frame["issue_time_utc"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
        if leads:
            frame.drop(frame[~frame["lead_time_minutes"].astype("Int64").isin([int(lead) for lead in leads])].index, inplace=True)
    base_columns = [
        *KEY_COLUMNS,
        "station_id",
        "actual_wind_mean_ms",
        "raw_wind_mean_ms",
        "predicted_residual_wind_mean_ms",
        "corrected_wind_mean_ms",
    ]
    alt_columns = [*KEY_COLUMNS, "predicted_residual_wind_mean_ms", "corrected_wind_mean_ms"]
    merged = base[base_columns].merge(
        alt[alt_columns],
        on=list(KEY_COLUMNS),
        how="inner",
        suffixes=("_base", "_alt"),
        validate="one_to_one",
    )
    merged = merged.rename(columns={
        "raw_wind_mean_ms": "base_raw_wind_mean_ms",
        "predicted_residual_wind_mean_ms_base": "base_predicted_residual_wind_mean_ms",
        "corrected_wind_mean_ms_base": "base_corrected_wind_mean_ms",
        "predicted_residual_wind_mean_ms_alt": "alt_predicted_residual_wind_mean_ms",
        "corrected_wind_mean_ms_alt": "alt_corrected_wind_mean_ms",
    })
    merged["max_corrected_wind_mean_ms"] = merged[["base_corrected_wind_mean_ms", "alt_corrected_wind_mean_ms"]].max(axis=1)
    merged["mean_corrected_wind_mean_ms"] = merged[["base_corrected_wind_mean_ms", "alt_corrected_wind_mean_ms"]].mean(axis=1)
    return merged


def add_bins(frame: Any, pd: Any) -> Any:
    out = frame.copy()
    out["actual_wind_bin_ms"] = pd.cut(
        out["actual_wind_mean_ms"].astype(float),
        bins=[-0.001, 2.0, 4.0, 6.0, 8.0, 999.0],
        labels=["0-2", "2-4", "4-6", "6-8", "8+"],
    ).astype(str)
    out["base_predicted_wind_bin_ms"] = pd.cut(
        out["base_corrected_wind_mean_ms"].astype(float),
        bins=[-0.001, 2.0, 4.0, 6.0, 8.0, 999.0],
        labels=["0-2", "2-4", "4-6", "6-8", "8+"],
    ).astype(str)
    return out


def route(frame: Any, gate_column: str, threshold: float) -> Any:
    use_alt = frame[gate_column].astype(float) >= float(threshold)
    return frame["base_corrected_wind_mean_ms"].where(~use_alt, frame["alt_corrected_wind_mean_ms"]), use_alt


def candidate_thresholds(frame: Any, columns: list[str], pd: Any) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        quantiles = values.quantile([0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]).tolist()
        rounded = sorted({round(float(value), 6) for value in quantiles})
        out.extend((column, value) for value in rounded)
    return out


def select_gate(calibration: Any, gate_columns: list[str], pd: Any, np: Any) -> dict[str, Any]:
    candidates = []
    base_metric = metric_frame(calibration, "base_corrected_wind_mean_ms", np)
    alt_metric = metric_frame(calibration, "alt_corrected_wind_mean_ms", np)
    best = {
        "gate_column": "__base_only__",
        "threshold": None,
        "use_alt_count": 0,
        "use_alt_rate": 0.0,
        **base_metric,
    }
    for gate_column, threshold in candidate_thresholds(calibration, gate_columns, pd):
        prediction, use_alt = route(calibration, gate_column, threshold)
        item = {
            "gate_column": gate_column,
            "threshold": threshold,
            "use_alt_count": int(use_alt.sum()),
            "use_alt_rate": round(float(use_alt.mean()), 6),
            **metric(prediction.astype(float).to_numpy(), calibration["actual_wind_mean_ms"].astype(float).to_numpy(), np),
        }
        candidates.append(item)
        if float(item["rmse"]) < float(best["rmse"]):
            best = item
    return {
        "base_metric": base_metric,
        "alt_metric": alt_metric,
        "best_gate": best,
        "candidate_count": len(candidates),
        "top_candidates": sorted(candidates, key=lambda item: item["rmse"])[:20],
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    gate = result["selection"]["best_gate"]
    lines = [
        "# Prediction Router",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Verdict: `{result['verdict']}`",
        f"Best gate: `{gate.get('gate_column')} >= {gate.get('threshold')}`",
        f"Calibration RMSE: `{gate.get('rmse')}`",
        f"Evaluation base RMSE: `{result['evaluation']['base_metric'].get('rmse')}`",
        f"Evaluation alt RMSE: `{result['evaluation']['alt_metric'].get('rmse')}`",
        f"Evaluation routed RMSE: `{result['evaluation']['routed_metric'].get('rmse')}`",
        f"Gain vs base: `{result['evaluation']['rmse_gain_pct_vs_base']}%`",
        f"Gap to threshold: `{result['evaluation']['rmse_gap_to_threshold']}`",
        "",
        "## Evaluation By Lead",
        "",
        "| Lead | Count | RMSE | MAE | Bias |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in result["evaluation"]["routed_by_lead"]:
        lines.append(f"| `{item['group'].get('lead_time_minutes')}` | {item.get('count')} | {item.get('rmse')} | {item.get('mae')} | {item.get('bias')} |")
    lines.extend(["", "## Evaluation Actual Wind Bins", "", "| Bin | Count | RMSE | MAE | Bias |", "| --- | ---: | ---: | ---: | ---: |"])
    for item in result["evaluation"]["routed_by_actual_wind_bin"]:
        lines.append(f"| `{item['group'].get('actual_wind_bin_ms')}` | {item.get('count')} | {item.get('rmse')} | {item.get('mae')} | {item.get('bias')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    gate_columns = args.gate_column or list(SAFE_GATE_COLUMNS)
    unsafe = sorted(set(gate_columns) - set(SAFE_GATE_COLUMNS))
    if unsafe:
        raise SystemExit(f"Unsafe/non-allowed gate columns requested: {unsafe}")
    calibration = add_bins(load_pair(args.calibration_base_predictions, args.calibration_alt_predictions, pd, args.lead_minute), pd)
    evaluation = add_bins(load_pair(args.evaluation_base_predictions, args.evaluation_alt_predictions, pd, args.lead_minute), pd)
    selection = select_gate(calibration, gate_columns, pd, np)
    gate = selection["best_gate"]
    if gate["gate_column"] == "__base_only__":
        evaluation["routed_wind_mean_ms"] = evaluation["base_corrected_wind_mean_ms"]
        use_alt = evaluation["base_corrected_wind_mean_ms"].astype(bool) & False
    else:
        evaluation["routed_wind_mean_ms"], use_alt = route(evaluation, gate["gate_column"], gate["threshold"])
    base_metric = metric_frame(evaluation, "base_corrected_wind_mean_ms", np)
    alt_metric = metric_frame(evaluation, "alt_corrected_wind_mean_ms", np)
    routed_metric = metric_frame(evaluation, "routed_wind_mean_ms", np)
    result = {
        "format": "corsewind.prediction_router.v1",
        "generated_at_utc": utc_now(),
        "threshold_rmse": args.threshold_rmse,
        "lead_minutes": args.lead_minute,
        "gate_columns": gate_columns,
        "calibration_rows": int(len(calibration)),
        "evaluation_rows": int(len(evaluation)),
        "selection": selection,
        "evaluation": {
            "base_metric": base_metric,
            "alt_metric": alt_metric,
            "routed_metric": routed_metric,
            "use_alt_count": int(use_alt.sum()),
            "use_alt_rate": round(float(use_alt.mean()), 6),
            "routed_by_lead": grouped_metrics(evaluation, ["lead_time_minutes"], "routed_wind_mean_ms", np, pd),
            "routed_by_actual_wind_bin": grouped_metrics(evaluation, ["actual_wind_bin_ms"], "routed_wind_mean_ms", np, pd),
            "routed_worst_spots": grouped_metrics(evaluation, ["spot_id"], "routed_wind_mean_ms", np, pd, limit=args.limit),
            "routed_worst_spot_leads": grouped_metrics(evaluation, ["spot_id", "lead_time_minutes"], "routed_wind_mean_ms", np, pd, limit=args.limit),
        },
        "verdict": "not_achieved",
    }
    result["evaluation"]["rmse_gap_to_threshold"] = round(float(routed_metric["rmse"]) - args.threshold_rmse, 6)
    result["evaluation"]["rmse_gain_pct_vs_base"] = round(
        (float(base_metric["rmse"]) - float(routed_metric["rmse"])) / float(base_metric["rmse"]) * 100.0,
        3,
    )
    if float(routed_metric["rmse"]) < args.threshold_rmse:
        result["verdict"] = "achieved"
    if args.output_predictions:
        args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
        evaluation.to_parquet(args.output_predictions, index=False)
        result["output_predictions"] = str(args.output_predictions)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-base-predictions", type=Path, required=True)
    parser.add_argument("--calibration-alt-predictions", type=Path, required=True)
    parser.add_argument("--evaluation-base-predictions", type=Path, required=True)
    parser.add_argument("--evaluation-alt-predictions", type=Path, required=True)
    parser.add_argument("--lead-minute", type=int, action="append", default=[15, 30, 45, 60])
    parser.add_argument("--gate-column", action="append", default=[])
    parser.add_argument("--threshold-rmse", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--output-predictions", type=Path)
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
        "calibration_rows": result["calibration_rows"],
        "evaluation_rows": result["evaluation_rows"],
        "best_gate": result["selection"]["best_gate"],
        "evaluation_base_rmse": result["evaluation"]["base_metric"].get("rmse"),
        "evaluation_alt_rmse": result["evaluation"]["alt_metric"].get("rmse"),
        "evaluation_routed_rmse": result["evaluation"]["routed_metric"].get("rmse"),
        "rmse_gain_pct_vs_base": result["evaluation"]["rmse_gain_pct_vs_base"],
        "rmse_gap_to_threshold": result["evaluation"]["rmse_gap_to_threshold"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
