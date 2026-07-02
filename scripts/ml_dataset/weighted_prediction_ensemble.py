#!/usr/bin/env python3
"""Evaluate a leakage-safe weighted ensemble between two prediction files."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KEY_COLUMNS = ("issue_time_utc", "spot_id", "lead_time_minutes")


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


def json_scalar(pd: Any, value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def grouped_metrics(
    frame: Any,
    group_columns: list[str],
    prediction_column: str,
    np: Any,
    pd: Any,
    limit: int | None = None,
) -> list[dict[str, Any]]:
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


def load_pair(
    base_path: Path,
    alt_path: Path,
    pd: Any,
    *,
    base_prediction_column: str,
    alt_prediction_column: str,
    leads: list[int],
) -> Any:
    base = pd.read_parquet(base_path)
    alt = pd.read_parquet(alt_path)
    for name, frame, prediction_column in (
        ("base", base, base_prediction_column),
        ("alt", alt, alt_prediction_column),
    ):
        missing = [column for column in (*KEY_COLUMNS, "actual_wind_mean_ms", prediction_column) if column not in frame.columns]
        if missing:
            raise SystemExit(f"{name} predictions missing required columns: {missing}")
        frame["issue_time_utc"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
        if leads:
            keep = frame["lead_time_minutes"].astype("Int64").isin([int(lead) for lead in leads])
            frame.drop(frame[~keep].index, inplace=True)

    base_columns = [*KEY_COLUMNS, "actual_wind_mean_ms", base_prediction_column]
    alt_columns = [*KEY_COLUMNS, "actual_wind_mean_ms", alt_prediction_column]
    base_ready = base[base_columns].rename(columns={
        "actual_wind_mean_ms": "actual_wind_mean_ms",
        base_prediction_column: "base_prediction_ms",
    })
    alt_ready = alt[alt_columns].rename(columns={
        "actual_wind_mean_ms": "alt_actual_wind_mean_ms",
        alt_prediction_column: "alt_prediction_ms",
    })
    merged = base_ready.merge(
        alt_ready,
        on=list(KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    actual_delta = (
        merged["actual_wind_mean_ms"].astype(float) - merged["alt_actual_wind_mean_ms"].astype(float)
    ).abs()
    if float(actual_delta.max(skipna=True) or 0.0) > 1e-6:
        raise SystemExit(f"Actual labels differ between files, max abs delta={actual_delta.max()}")
    merged = merged.drop(columns=["alt_actual_wind_mean_ms"])
    return merged


def prediction_for_weight(frame: Any, weight: float) -> Any:
    return (
        frame["base_prediction_ms"].astype(float) * (1.0 - float(weight))
        + frame["alt_prediction_ms"].astype(float) * float(weight)
    )


def candidate_weights(step: float) -> list[float]:
    if step <= 0 or step > 1:
        raise SystemExit("--weight-step must be in ]0,1].")
    count = int(round(1.0 / step))
    weights = sorted({round(index * step, 10) for index in range(count + 1)} | {0.0, 1.0})
    return [weight for weight in weights if 0.0 <= weight <= 1.0]


def select_weight(frame: Any, weights: list[float], np: Any) -> dict[str, Any]:
    metrics = []
    best: dict[str, Any] | None = None
    for weight in weights:
        prediction = prediction_for_weight(frame, weight)
        item = {"alt_weight": float(weight), **metric(prediction.to_numpy(), frame["actual_wind_mean_ms"].astype(float).to_numpy(), np)}
        metrics.append(item)
        if best is None or float(item["rmse"]) < float(best["rmse"]):
            best = item
    assert best is not None
    return {"best": best, "candidates": metrics}


def select_group_weights(
    calibration: Any,
    group_columns: list[str],
    weights: list[float],
    min_group_rows: int,
    fallback_weight: float,
    np: Any,
    pd: Any,
) -> dict[str, Any]:
    selected = []
    for raw_key, group in calibration.groupby(group_columns, dropna=False):
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        group_dict = dict(zip(group_columns, [json_scalar(pd, value) for value in values], strict=True))
        if len(group) < min_group_rows:
            selected.append({
                "group": group_dict,
                "rows": int(len(group)),
                "alt_weight": float(fallback_weight),
                "reason": "fallback_min_rows",
            })
            continue
        choice = select_weight(group, weights, np)["best"]
        selected.append({
            "group": group_dict,
            "rows": int(len(group)),
            "alt_weight": float(choice["alt_weight"]),
            "rmse": choice.get("rmse"),
            "reason": "selected",
        })
    return {"group_columns": group_columns, "min_group_rows": min_group_rows, "selected": selected}


def apply_group_weights(frame: Any, selection: dict[str, Any], fallback_weight: float, pd: Any) -> Any:
    output = frame.copy()
    output["selected_alt_weight"] = float(fallback_weight)
    group_columns = selection["group_columns"]
    for item in selection["selected"]:
        mask = pd.Series(True, index=output.index)
        for column, value in item["group"].items():
            if value is None:
                mask = mask & output[column].isna()
            else:
                mask = mask & (output[column] == value)
        output.loc[mask, "selected_alt_weight"] = float(item["alt_weight"])
    output["ensemble_prediction_ms"] = (
        output["base_prediction_ms"].astype(float) * (1.0 - output["selected_alt_weight"].astype(float))
        + output["alt_prediction_ms"].astype(float) * output["selected_alt_weight"].astype(float)
    )
    return output


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Weighted Prediction Ensemble",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Verdict: `{result['verdict']}`",
        f"Strategy: `{result['selected_strategy']}`",
        f"Calibration rows: `{result['calibration_rows']}`",
        f"Evaluation rows: `{result['evaluation_rows']}`",
        f"Evaluation base RMSE: `{result['evaluation']['base_metric'].get('rmse')}`",
        f"Evaluation alt RMSE: `{result['evaluation']['alt_metric'].get('rmse')}`",
        f"Evaluation ensemble RMSE: `{result['evaluation']['ensemble_metric'].get('rmse')}`",
        f"Gain vs base: `{result['evaluation']['rmse_gain_pct_vs_base']}%`",
        f"Gap to threshold: `{result['evaluation']['rmse_gap_to_threshold']}`",
        "",
        "## Strategy Selection",
        "",
        "| Strategy | Calibration RMSE | Evaluation RMSE |",
        "| --- | ---: | ---: |",
    ]
    for item in result["strategy_metrics"]:
        lines.append(f"| `{item['strategy']}` | {item.get('calibration_rmse')} | {item.get('evaluation_rmse')} |")
    lines.extend(["", "## Evaluation By Lead", "", "| Lead | Count | RMSE | MAE | Bias |", "| ---: | ---: | ---: | ---: | ---: |"])
    for item in result["evaluation"]["ensemble_by_lead"]:
        lines.append(f"| `{item['group'].get('lead_time_minutes')}` | {item.get('count')} | {item.get('rmse')} | {item.get('mae')} | {item.get('bias')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    weights = candidate_weights(args.weight_step)
    calibration = load_pair(
        args.calibration_base_predictions,
        args.calibration_alt_predictions,
        pd,
        base_prediction_column=args.base_prediction_column,
        alt_prediction_column=args.alt_prediction_column,
        leads=args.lead_minute,
    )
    evaluation = load_pair(
        args.evaluation_base_predictions,
        args.evaluation_alt_predictions,
        pd,
        base_prediction_column=args.base_prediction_column,
        alt_prediction_column=args.alt_prediction_column,
        leads=args.lead_minute,
    )
    global_selection = select_weight(calibration, weights, np)
    global_weight = float(global_selection["best"]["alt_weight"])

    candidates: dict[str, Any] = {
        "global": {
            "selection": global_selection,
            "calibration": calibration.assign(ensemble_prediction_ms=prediction_for_weight(calibration, global_weight), selected_alt_weight=global_weight),
            "evaluation": evaluation.assign(ensemble_prediction_ms=prediction_for_weight(evaluation, global_weight), selected_alt_weight=global_weight),
        }
    }
    if "lead" in args.strategy:
        lead_selection = select_group_weights(calibration, ["lead_time_minutes"], weights, args.min_group_rows, global_weight, np, pd)
        candidates["lead"] = {
            "selection": lead_selection,
            "calibration": apply_group_weights(calibration, lead_selection, global_weight, pd),
            "evaluation": apply_group_weights(evaluation, lead_selection, global_weight, pd),
        }
    if "spot_lead" in args.strategy:
        spot_lead_selection = select_group_weights(
            calibration,
            ["spot_id", "lead_time_minutes"],
            weights,
            args.min_group_rows,
            global_weight,
            np,
            pd,
        )
        candidates["spot_lead"] = {
            "selection": spot_lead_selection,
            "calibration": apply_group_weights(calibration, spot_lead_selection, global_weight, pd),
            "evaluation": apply_group_weights(evaluation, spot_lead_selection, global_weight, pd),
        }

    strategy_metrics = []
    selected_strategy = "global"
    selected_calibration_rmse = math.inf
    for strategy, payload in candidates.items():
        cal_metric = metric_frame(payload["calibration"], "ensemble_prediction_ms", np)
        eval_metric = metric_frame(payload["evaluation"], "ensemble_prediction_ms", np)
        item = {
            "strategy": strategy,
            "calibration_rmse": cal_metric.get("rmse"),
            "evaluation_rmse": eval_metric.get("rmse"),
        }
        strategy_metrics.append(item)
        if cal_metric.get("rmse") is not None and float(cal_metric["rmse"]) < selected_calibration_rmse:
            selected_calibration_rmse = float(cal_metric["rmse"])
            selected_strategy = strategy

    selected = candidates[selected_strategy]
    output = selected["evaluation"].copy()
    base_metric = metric_frame(output, "base_prediction_ms", np)
    alt_metric = metric_frame(output, "alt_prediction_ms", np)
    ensemble_metric = metric_frame(output, "ensemble_prediction_ms", np)
    result = {
        "format": "corsewind.weighted_prediction_ensemble.v1",
        "generated_at_utc": utc_now(),
        "threshold_rmse": args.threshold_rmse,
        "lead_minutes": args.lead_minute,
        "base_prediction_column": args.base_prediction_column,
        "alt_prediction_column": args.alt_prediction_column,
        "calibration_base_predictions": str(args.calibration_base_predictions),
        "calibration_alt_predictions": str(args.calibration_alt_predictions),
        "evaluation_base_predictions": str(args.evaluation_base_predictions),
        "evaluation_alt_predictions": str(args.evaluation_alt_predictions),
        "calibration_rows": int(len(calibration)),
        "evaluation_rows": int(len(output)),
        "weight_step": args.weight_step,
        "strategy_metrics": strategy_metrics,
        "selected_strategy": selected_strategy,
        "selection": selected["selection"],
        "evaluation": {
            "base_metric": base_metric,
            "alt_metric": alt_metric,
            "ensemble_metric": ensemble_metric,
            "ensemble_by_lead": grouped_metrics(output, ["lead_time_minutes"], "ensemble_prediction_ms", np, pd),
            "ensemble_worst_spots": grouped_metrics(output, ["spot_id"], "ensemble_prediction_ms", np, pd, limit=args.limit),
            "ensemble_worst_spot_leads": grouped_metrics(output, ["spot_id", "lead_time_minutes"], "ensemble_prediction_ms", np, pd, limit=args.limit),
        },
        "verdict": "not_achieved",
    }
    result["evaluation"]["rmse_gap_to_threshold"] = round(float(ensemble_metric["rmse"]) - args.threshold_rmse, 6)
    result["evaluation"]["rmse_gain_pct_vs_base"] = round(
        (float(base_metric["rmse"]) - float(ensemble_metric["rmse"])) / float(base_metric["rmse"]) * 100.0,
        3,
    )
    if float(ensemble_metric["rmse"]) < args.threshold_rmse:
        result["verdict"] = "achieved"
    if args.output_predictions:
        args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(args.output_predictions, index=False)
        result["output_predictions"] = str(args.output_predictions)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-base-predictions", type=Path, required=True)
    parser.add_argument("--calibration-alt-predictions", type=Path, required=True)
    parser.add_argument("--evaluation-base-predictions", type=Path, required=True)
    parser.add_argument("--evaluation-alt-predictions", type=Path, required=True)
    parser.add_argument("--base-prediction-column", default="calibrated_wind_mean_ms")
    parser.add_argument("--alt-prediction-column", default="calibrated_wind_mean_ms")
    parser.add_argument("--lead-minute", type=int, action="append", default=[15, 30, 45, 60])
    parser.add_argument("--strategy", choices=("global", "lead", "spot_lead"), action="append", default=["global", "lead"])
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--min-group-rows", type=int, default=400)
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
        "selected_strategy": result["selected_strategy"],
        "calibration_rows": result["calibration_rows"],
        "evaluation_rows": result["evaluation_rows"],
        "evaluation_base_rmse": result["evaluation"]["base_metric"].get("rmse"),
        "evaluation_alt_rmse": result["evaluation"]["alt_metric"].get("rmse"),
        "evaluation_ensemble_rmse": result["evaluation"]["ensemble_metric"].get("rmse"),
        "rmse_gain_pct_vs_base": result["evaluation"]["rmse_gain_pct_vs_base"],
        "rmse_gap_to_threshold": result["evaluation"]["rmse_gap_to_threshold"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
