#!/usr/bin/env python3
"""Train a gust-factor calibrator: log(gust / wind_reference)."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import train_prediction_residual_calibrator as base


TARGET_COLUMN = "__gust_factor_target"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def prepare_frame(frame: Any, args: argparse.Namespace, pd: Any, np: Any) -> Any:
    required = [args.actual_gust_column, args.wind_reference_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(missing)}")
    frame = frame.dropna(subset=required).copy()
    actual_gust = pd.to_numeric(frame[args.actual_gust_column], errors="coerce")
    wind_reference = pd.to_numeric(frame[args.wind_reference_column], errors="coerce").clip(lower=args.min_wind_reference_ms)
    frame = frame[actual_gust.notna() & wind_reference.notna()].copy()
    actual_gust = pd.to_numeric(frame[args.actual_gust_column], errors="coerce").clip(lower=args.min_wind_reference_ms)
    wind_reference = pd.to_numeric(frame[args.wind_reference_column], errors="coerce").clip(lower=args.min_wind_reference_ms)
    frame[TARGET_COLUMN] = np.log((actual_gust / wind_reference).clip(lower=args.min_factor, upper=args.max_factor))
    return frame


def metric(prediction: Any, observation: Any, np: Any) -> dict[str, Any]:
    valid = ~(np.isnan(prediction) | np.isnan(observation))
    prediction = prediction[valid]
    observation = observation[valid]
    if len(prediction) == 0:
        return {"count": 0}
    err = prediction - observation
    pred_var = float(np.var(prediction))
    obs_var = float(np.var(observation))
    return {
        "count": int(len(prediction)),
        "mae": round(float(np.mean(np.abs(err))), 6),
        "rmse": round(float(math.sqrt(float(np.mean(err * err)))), 6),
        "bias": round(float(np.mean(err)), 6),
        "variance_ratio": None if obs_var <= 0.0 else round(pred_var / obs_var, 6),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = base.import_dependencies()
    pd = deps["pd"]
    np = deps["np"]
    calibration = base.load_predictions(
        args.calibration_predictions,
        pd,
        start=args.calibration_start_utc,
        end=args.calibration_end_utc,
        leads=args.lead_minute,
    )
    evaluation = base.load_predictions(
        args.evaluation_predictions,
        pd,
        start=args.evaluation_start_utc,
        end=args.evaluation_end_utc,
        leads=args.lead_minute,
    )
    calibration = prepare_frame(calibration, args, pd, np)
    evaluation = prepare_frame(evaluation, args, pd, np)
    numeric, categorical, dropped = base.infer_features(calibration, args.max_categorical_cardinality, pd)
    feature_columns = [*numeric, *categorical]
    if not feature_columns:
        raise SystemExit("No valid feature columns.")
    model = deps["Pipeline"]([
        ("preprocess", base.make_preprocessor(deps, numeric, categorical)),
        ("model", base.build_model(args, deps)),
    ])
    model.fit(calibration.loc[calibration[TARGET_COLUMN].notna(), feature_columns], calibration.loc[calibration[TARGET_COLUMN].notna(), TARGET_COLUMN])
    log_factor = model.predict(evaluation[feature_columns])
    wind_reference = pd.to_numeric(evaluation[args.wind_reference_column], errors="coerce").clip(lower=args.min_wind_reference_ms)
    predicted = wind_reference * np.exp(log_factor)
    predicted = predicted.clip(lower=wind_reference)
    output_column = args.output_column
    evaluation[output_column] = predicted
    result = {
        "format": "corsewind.gust_factor_calibrator.v1",
        "generated_at_utc": utc_now(),
        "objective": args.objective,
        "quantile_alpha": args.quantile_alpha if args.objective == "quantile" else None,
        "wind_reference_column": args.wind_reference_column,
        "actual_gust_column": args.actual_gust_column,
        "output_column": output_column,
        "calibration_rows": int(len(calibration)),
        "evaluation_rows": int(len(evaluation)),
        "feature_column_count": len(feature_columns),
        "numeric_column_count": len(numeric),
        "categorical_column_count": len(categorical),
        "dropped_columns": dropped,
        "metrics": metric(evaluation[output_column].astype(float).to_numpy(), evaluation[args.actual_gust_column].astype(float).to_numpy(), np),
    }
    if args.output_predictions:
        args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
        evaluation.to_parquet(args.output_predictions, index=False)
        result["output_predictions"] = str(args.output_predictions)
    if args.output_model:
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        deps["joblib"].dump(model, args.output_model)
        result["output_model"] = str(args.output_model)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-predictions", type=Path, required=True)
    parser.add_argument("--evaluation-predictions", type=Path, required=True)
    parser.add_argument("--actual-gust-column", default="actual_gust_ms")
    parser.add_argument("--wind-reference-column", required=True)
    parser.add_argument("--output-column", default="calibrated_gust_factor_ms")
    parser.add_argument("--calibration-start-utc")
    parser.add_argument("--calibration-end-utc")
    parser.add_argument("--evaluation-start-utc")
    parser.add_argument("--evaluation-end-utc")
    parser.add_argument("--lead-minute", type=int, action="append", default=[15, 30, 45, 60])
    parser.add_argument("--model-family", choices=("hist_gradient_boosting", "lightgbm"), default="lightgbm")
    parser.add_argument("--objective", choices=("squared_error", "quantile"), default="quantile")
    parser.add_argument("--quantile-alpha", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.1)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--lightgbm-max-bin", type=int, default=127)
    parser.add_argument("--lightgbm-feature-fraction", type=float, default=0.85)
    parser.add_argument("--lightgbm-bagging-fraction", type=float, default=0.85)
    parser.add_argument("--lightgbm-bagging-freq", type=int, default=1)
    parser.add_argument("--lightgbm-force-col-wise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-categorical-cardinality", type=int, default=100)
    parser.add_argument("--min-wind-reference-ms", type=float, default=0.5)
    parser.add_argument("--min-factor", type=float, default=1.0)
    parser.add_argument("--max-factor", type=float, default=4.0)
    parser.add_argument("--output-predictions", type=Path)
    parser.add_argument("--output-model", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.quantile_alpha < 1.0:
        raise SystemExit("--quantile-alpha must be between 0 and 1.")
    run(args)


if __name__ == "__main__":
    main()
