#!/usr/bin/env python3
"""Select and apply a leakage-safe regime calibration to wind predictions."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PREDICTED_BIN_COLUMN = "__predicted_wind_bin_ms"
DEFAULT_GROUP_SPECS = (
    "global",
    "lead_time_minutes",
    "__predicted_wind_bin_ms",
    "lead_time_minutes,__predicted_wind_bin_ms",
    "spot_id",
    "spot_id,lead_time_minutes",
    "spot_id,__predicted_wind_bin_ms",
)


@dataclass(frozen=True)
class Candidate:
    correction: str
    group_columns: tuple[str, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def metric(prediction: Any, actual: Any, np: Any) -> dict[str, Any]:
    valid = ~(np.isnan(prediction) | np.isnan(actual))
    prediction = prediction[valid]
    actual = actual[valid]
    if len(prediction) == 0:
        return {"count": 0}
    errors = prediction - actual
    abs_errors = np.abs(errors)
    return {
        "count": int(len(errors)),
        "mae": round(float(np.mean(abs_errors)), 6),
        "rmse": round(float(math.sqrt(float(np.mean(errors * errors)))), 6),
        "bias": round(float(np.mean(errors)), 6),
        "p50_abs_error": round(float(np.quantile(abs_errors, 0.50)), 6),
        "p90_abs_error": round(float(np.quantile(abs_errors, 0.90)), 6),
        "p95_abs_error": round(float(np.quantile(abs_errors, 0.95)), 6),
    }


def metric_frame(frame: Any, prediction_column: str, actual_column: str, np: Any) -> dict[str, Any]:
    valid = frame[[prediction_column, actual_column]].dropna()
    return metric(
        valid[prediction_column].astype(float).to_numpy(),
        valid[actual_column].astype(float).to_numpy(),
        np,
    )


def grouped_metrics(frame: Any, group_columns: list[str], prediction_column: str, actual_column: str, np: Any, pd: Any, limit: int) -> list[dict[str, Any]]:
    available = [column for column in group_columns if column in frame.columns]
    if not available:
        return []
    rows = []
    valid = frame.dropna(subset=[prediction_column, actual_column])
    for raw_key, group in valid.groupby(available, dropna=False):
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        rows.append({
            "group": dict(zip(available, [json_scalar(pd, value) for value in values], strict=True)),
            **metric_frame(group, prediction_column, actual_column, np),
        })
    rows.sort(key=lambda item: (item.get("rmse") is None, item.get("rmse", -1)), reverse=True)
    return rows[:limit]


def json_scalar(pd: Any, value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def add_predicted_bins(frame: Any, prediction_column: str, pd: Any) -> Any:
    out = frame.copy()
    out[PREDICTED_BIN_COLUMN] = pd.cut(
        out[prediction_column].astype(float),
        bins=[-0.001, 2.0, 4.0, 6.0, 8.0, 999.0],
        labels=["0-2", "2-4", "4-6", "6-8", "8+"],
    ).astype(str)
    return out


def load_frame(path: Path, prediction_column: str, actual_column: str, pd: Any, leads: list[int]) -> Any:
    frame = pd.read_parquet(path)
    if leads:
        frame = frame[frame["lead_time_minutes"].astype("Int64").isin([int(lead) for lead in leads])].copy()
    frame = frame.dropna(subset=[prediction_column, actual_column]).copy()
    frame["lead_time_minutes"] = frame["lead_time_minutes"].astype(float)
    return add_predicted_bins(frame, prediction_column, pd)


def group_key(raw_key: Any, column_count: int) -> tuple[Any, ...]:
    if column_count == 0:
        return ()
    if isinstance(raw_key, tuple):
        return raw_key
    return (raw_key,)


def group_mask(frame: Any, pd: Any, columns: tuple[str, ...], values: tuple[Any, ...]) -> Any:
    if not columns:
        return pd.Series(True, index=frame.index)
    mask = pd.Series(True, index=frame.index)
    for column, value in zip(columns, values, strict=True):
        if pd.isna(value):
            mask = mask & frame[column].isna()
        else:
            mask = mask & (frame[column] == value)
    return mask


def fit_global_bias(frame: Any, prediction_column: str, actual_column: str) -> float:
    errors = frame[prediction_column].astype(float) - frame[actual_column].astype(float)
    return float(errors.mean())


def fit_global_affine(frame: Any, prediction_column: str, actual_column: str, np: Any, slope_clip: tuple[float, float]) -> tuple[float, float]:
    x = frame[prediction_column].astype(float).to_numpy()
    y = frame[actual_column].astype(float).to_numpy()
    if len(x) < 2 or float(np.nanstd(x)) < 1e-6:
        return 0.0, 1.0
    slope, intercept = np.polyfit(x, y, 1)
    slope = float(np.clip(slope, slope_clip[0], slope_clip[1]))
    return float(intercept), slope


def fit_candidate(
    frame: Any,
    candidate: Candidate,
    prediction_column: str,
    actual_column: str,
    args: argparse.Namespace,
    np: Any,
    pd: Any,
) -> dict[str, Any]:
    group_columns = candidate.group_columns
    global_bias = fit_global_bias(frame, prediction_column, actual_column)
    global_intercept, global_slope = fit_global_affine(frame, prediction_column, actual_column, np, (args.min_slope, args.max_slope))
    params: dict[str, Any] = {}
    if group_columns:
        grouped = frame.groupby(list(group_columns), dropna=False)
    else:
        grouped = [((), frame)]
    for raw_key, group in grouped:
        values = group_key(raw_key, len(group_columns))
        n = int(len(group))
        if candidate.correction == "bias":
            if n < args.min_group_rows:
                continue
            raw_bias = fit_global_bias(group, prediction_column, actual_column)
            weight = n / (n + float(args.shrinkage_rows))
            bias = weight * raw_bias + (1.0 - weight) * global_bias
            params[json.dumps([json_scalar(pd, value) for value in values], sort_keys=True)] = {
                "count": n,
                "bias": float(bias),
                "raw_bias": float(raw_bias),
                "shrink_weight": float(weight),
            }
        elif candidate.correction == "affine":
            if n < args.min_affine_group_rows:
                continue
            raw_intercept, raw_slope = fit_global_affine(group, prediction_column, actual_column, np, (args.min_slope, args.max_slope))
            weight = n / (n + float(args.shrinkage_rows))
            intercept = weight * raw_intercept + (1.0 - weight) * global_intercept
            slope = weight * raw_slope + (1.0 - weight) * global_slope
            params[json.dumps([json_scalar(pd, value) for value in values], sort_keys=True)] = {
                "count": n,
                "intercept": float(intercept),
                "slope": float(slope),
                "raw_intercept": float(raw_intercept),
                "raw_slope": float(raw_slope),
                "shrink_weight": float(weight),
            }
        else:
            raise ValueError(candidate.correction)
    return {
        "correction": candidate.correction,
        "group_columns": list(group_columns),
        "global_bias": float(global_bias),
        "global_intercept": float(global_intercept),
        "global_slope": float(global_slope),
        "params": params,
        "param_count": len(params),
    }


def apply_model(frame: Any, model: dict[str, Any], prediction_column: str, output_column: str, pd: Any) -> Any:
    out = frame.copy()
    out[output_column] = out[prediction_column].astype(float)
    columns = tuple(model["group_columns"])
    correction = model["correction"]
    fallback_bias = float(model["global_bias"])
    fallback_intercept = float(model["global_intercept"])
    fallback_slope = float(model["global_slope"])
    if columns:
        grouped = out.groupby(list(columns), dropna=False)
    else:
        grouped = [((), out)]
    for raw_key, group in grouped:
        values = group_key(raw_key, len(columns))
        key = json.dumps([json_scalar(pd, value) for value in values], sort_keys=True)
        params = model["params"].get(key)
        idx = group.index
        base = out.loc[idx, prediction_column].astype(float)
        if correction == "bias":
            bias = fallback_bias if params is None else float(params["bias"])
            out.loc[idx, output_column] = base - bias
        else:
            intercept = fallback_intercept if params is None else float(params["intercept"])
            slope = fallback_slope if params is None else float(params["slope"])
            out.loc[idx, output_column] = intercept + slope * base
    return out


def candidate_list(args: argparse.Namespace) -> list[Candidate]:
    specs = args.group_spec or list(DEFAULT_GROUP_SPECS)
    corrections = args.correction or ["bias", "affine"]
    out = []
    for correction in corrections:
        for spec in specs:
            columns = tuple(column for column in spec.split(",") if column and column != "global")
            out.append(Candidate(correction=correction, group_columns=columns))
    return out


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    best = result["selection"]["best"]
    lines = [
        "# Regime Prediction Calibration",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Verdict: `{result['verdict']}`",
        f"Best correction: `{best['correction']}`",
        f"Best groups: `{','.join(best['group_columns']) or 'global'}`",
        f"Validation base RMSE: `{result['selection']['validation_base_metric'].get('rmse')}`",
        f"Validation calibrated RMSE: `{best['validation_metric'].get('rmse')}`",
        f"Evaluation base RMSE: `{result['evaluation']['base_metric'].get('rmse')}`",
        f"Evaluation regime RMSE: `{result['evaluation']['regime_metric'].get('rmse')}`",
        f"Gain vs base: `{result['evaluation']['rmse_gain_pct_vs_base']}%`",
        f"Gap to threshold: `{result['evaluation']['rmse_gap_to_threshold']}`",
        "",
        "## Evaluation By Lead",
        "",
        "| Lead | Count | RMSE | MAE | Bias |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in result["evaluation"]["by_lead"]:
        lines.append(f"| `{item['group'].get('lead_time_minutes')}` | {item.get('count')} | {item.get('rmse')} | {item.get('mae')} | {item.get('bias')} |")
    lines.extend(["", "## Worst Spot/Lead", "", "| Spot | Lead | Count | RMSE | MAE | Bias |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for item in result["evaluation"]["worst_spot_leads"]:
        group = item["group"]
        lines.append(
            f"| `{group.get('spot_id')}` | `{group.get('lead_time_minutes')}` | {item.get('count')} | "
            f"{item.get('rmse')} | {item.get('mae')} | {item.get('bias')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    validation = load_frame(args.validation_predictions, args.prediction_column, args.actual_column, pd, args.lead_minute)
    evaluation = load_frame(args.evaluation_predictions, args.prediction_column, args.actual_column, pd, args.lead_minute)
    validation_base_metric = metric_frame(validation, args.prediction_column, args.actual_column, np)
    candidates = []
    for candidate in candidate_list(args):
        missing = [column for column in candidate.group_columns if column not in validation.columns or column not in evaluation.columns]
        if missing:
            continue
        model = fit_candidate(validation, candidate, args.prediction_column, args.actual_column, args, np, pd)
        scored = apply_model(validation, model, args.prediction_column, "__regime_calibrated_validation", pd)
        item = {
            "correction": candidate.correction,
            "group_columns": list(candidate.group_columns),
            "param_count": model["param_count"],
            "validation_metric": metric_frame(scored, "__regime_calibrated_validation", args.actual_column, np),
            "model": model,
        }
        candidates.append(item)
    if not candidates:
        raise SystemExit("No candidate regime calibrations could be evaluated.")
    candidates.sort(key=lambda item: item["validation_metric"].get("rmse", float("inf")))
    best = candidates[0]
    evaluation_scored = apply_model(evaluation, best["model"], args.prediction_column, args.output_prediction_column, pd)
    base_metric = metric_frame(evaluation_scored, args.prediction_column, args.actual_column, np)
    regime_metric = metric_frame(evaluation_scored, args.output_prediction_column, args.actual_column, np)
    result = {
        "format": "corsewind.regime_prediction_calibration.v1",
        "generated_at_utc": utc_now(),
        "threshold_rmse": args.threshold_rmse,
        "validation_predictions": str(args.validation_predictions),
        "evaluation_predictions": str(args.evaluation_predictions),
        "prediction_column": args.prediction_column,
        "actual_column": args.actual_column,
        "output_prediction_column": args.output_prediction_column,
        "lead_minutes": args.lead_minute,
        "validation_rows": int(len(validation)),
        "evaluation_rows": int(len(evaluation_scored)),
        "settings": {
            "min_group_rows": args.min_group_rows,
            "min_affine_group_rows": args.min_affine_group_rows,
            "shrinkage_rows": args.shrinkage_rows,
            "min_slope": args.min_slope,
            "max_slope": args.max_slope,
        },
        "selection": {
            "validation_base_metric": validation_base_metric,
            "best": {key: value for key, value in best.items() if key != "model"},
            "top_candidates": [{key: value for key, value in item.items() if key != "model"} for item in candidates[:20]],
        },
        "evaluation": {
            "base_metric": base_metric,
            "regime_metric": regime_metric,
            "by_lead": grouped_metrics(evaluation_scored, ["lead_time_minutes"], args.output_prediction_column, args.actual_column, np, pd, args.limit),
            "by_predicted_bin": grouped_metrics(evaluation_scored, [PREDICTED_BIN_COLUMN], args.output_prediction_column, args.actual_column, np, pd, args.limit),
            "worst_spots": grouped_metrics(evaluation_scored, ["spot_id"], args.output_prediction_column, args.actual_column, np, pd, args.limit),
            "worst_spot_leads": grouped_metrics(evaluation_scored, ["spot_id", "lead_time_minutes"], args.output_prediction_column, args.actual_column, np, pd, args.limit),
        },
        "verdict": "not_achieved",
    }
    result["evaluation"]["rmse_gap_to_threshold"] = round(float(regime_metric["rmse"]) - args.threshold_rmse, 6)
    result["evaluation"]["rmse_gain_pct_vs_base"] = round(
        (float(base_metric["rmse"]) - float(regime_metric["rmse"])) / float(base_metric["rmse"]) * 100.0,
        3,
    )
    if float(regime_metric["rmse"]) < args.threshold_rmse:
        result["verdict"] = "achieved"
    if args.output_predictions:
        args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
        evaluation_scored.to_parquet(args.output_predictions, index=False)
        result["output_predictions"] = str(args.output_predictions)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--evaluation-predictions", type=Path, required=True)
    parser.add_argument("--prediction-column", default="calibrated_wind_mean_ms")
    parser.add_argument("--actual-column", default="actual_wind_mean_ms")
    parser.add_argument("--output-prediction-column", default="regime_calibrated_wind_mean_ms")
    parser.add_argument("--lead-minute", type=int, action="append", default=[15, 30, 45, 60])
    parser.add_argument("--group-spec", action="append", default=[])
    parser.add_argument("--correction", choices=("bias", "affine"), action="append", default=[])
    parser.add_argument("--min-group-rows", type=int, default=200)
    parser.add_argument("--min-affine-group-rows", type=int, default=500)
    parser.add_argument("--shrinkage-rows", type=int, default=800)
    parser.add_argument("--min-slope", type=float, default=0.75)
    parser.add_argument("--max-slope", type=float, default=1.25)
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
        "validation_rows": result["validation_rows"],
        "evaluation_rows": result["evaluation_rows"],
        "best": result["selection"]["best"],
        "evaluation_base_rmse": result["evaluation"]["base_metric"].get("rmse"),
        "evaluation_regime_rmse": result["evaluation"]["regime_metric"].get("rmse"),
        "rmse_gain_pct_vs_base": result["evaluation"]["rmse_gain_pct_vs_base"],
        "rmse_gap_to_threshold": result["evaluation"]["rmse_gap_to_threshold"],
    }, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
