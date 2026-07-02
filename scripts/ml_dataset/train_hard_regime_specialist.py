#!/usr/bin/env python3
"""Train a conservative hard-regime residual specialist without 2026 leakage."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KEY_COLUMNS = ("issue_time_utc", "spot_id", "lead_time_minutes")
TARGET_COLUMN = "__target_residual_correction_ms"
FORCED_CATEGORICAL = {"spot_id", "station_id", "spot_kind"}
LEAKY_COLUMNS = {
    "actual_wind_mean_ms",
    "raw_error_ms",
    "corrected_error_ms",
    "abs_raw_error_ms",
    "abs_corrected_error_ms",
    "actual_wind_bin_ms",
    "raw_abs_error_bin_ms",
    "corrected_abs_error_bin_ms",
}


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


def import_dependencies() -> dict[str, Any]:
    try:
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OrdinalEncoder
    except ImportError as exc:
        raise SystemExit("Missing pandas/pyarrow/sklearn dependencies.") from exc
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        LGBMRegressor = None
    return {
        "joblib": joblib,
        "np": np,
        "pd": pd,
        "ColumnTransformer": ColumnTransformer,
        "ExtraTreesRegressor": ExtraTreesRegressor,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "SimpleImputer": SimpleImputer,
        "Pipeline": Pipeline,
        "OrdinalEncoder": OrdinalEncoder,
        "LGBMRegressor": LGBMRegressor,
    }


def load_frame(path: Path, pd: Any, base_prediction_column: str) -> Any:
    frame = pd.read_parquet(path)
    required = [*KEY_COLUMNS, "actual_wind_mean_ms", base_prediction_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"{path} missing required columns: {missing}")
    frame["issue_time_utc"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    frame = frame.copy()
    frame["issue_hour_utc"] = frame["issue_time_utc"].dt.hour.astype("float64")
    frame["issue_month_number"] = frame["issue_time_utc"].dt.month.astype("float64")
    dayofyear = frame["issue_time_utc"].dt.dayofyear.fillna(1).astype(float)
    frame["issue_dayofyear_sin"] = (2.0 * math.pi * dayofyear / 366.0).map(math.sin)
    frame["issue_dayofyear_cos"] = (2.0 * math.pi * dayofyear / 366.0).map(math.cos)
    frame[TARGET_COLUMN] = frame["actual_wind_mean_ms"].astype(float) - frame[base_prediction_column].astype(float)
    return frame


def hard_mask(frame: Any, args: argparse.Namespace) -> Any:
    masks = []
    if args.hard_spot:
        masks.append(frame["spot_id"].astype(str).isin(set(args.hard_spot)))
    if args.hard_min_lead is not None:
        masks.append(frame["lead_time_minutes"].astype(float) >= float(args.hard_min_lead))
    if args.hard_lead_minute:
        masks.append(frame["lead_time_minutes"].astype("Int64").isin([int(lead) for lead in args.hard_lead_minute]))
    if args.hard_min_prediction_ms is not None:
        masks.append(frame[args.base_prediction_column].astype(float) >= float(args.hard_min_prediction_ms))
    if not masks:
        return frame["lead_time_minutes"].notna()
    mask = masks[0]
    for item in masks[1:]:
        if args.hard_logic == "and":
            mask = mask & item
        else:
            mask = mask | item
    return mask


def allowed_feature(column: str, base_prediction_column: str) -> bool:
    if column in LEAKY_COLUMNS or column == TARGET_COLUMN:
        return False
    if column == base_prediction_column:
        return True
    if column in {"spot_id", "station_id", "spot_kind", "latitude", "longitude", "lead_time_minutes"}:
        return True
    if column in {"issue_hour_utc", "issue_month_number", "issue_dayofyear_sin", "issue_dayofyear_cos"}:
        return True
    if column in {"raw_wind_mean_ms", "predicted_residual_wind_mean_ms", "corrected_wind_mean_ms"}:
        return True
    if column.startswith("features__") or column.startswith("baselines__"):
        return True
    return False


def infer_features(train: Any, eval_frame: Any, args: argparse.Namespace, pd: Any) -> tuple[list[str], list[str], dict[str, Any]]:
    candidates = sorted(
        column
        for column in set(train.columns).intersection(eval_frame.columns)
        if allowed_feature(column, args.base_prediction_column)
    )
    numeric = []
    categorical = []
    dropped: dict[str, Any] = {"sparse": [], "constant": [], "high_cardinality": []}
    for column in candidates:
        non_null = int(train[column].notna().sum())
        if non_null < args.min_non_null_count:
            dropped["sparse"].append({"column": column, "non_null": non_null})
            continue
        unique_count = int(train[column].dropna().nunique())
        if unique_count <= 1:
            dropped["constant"].append({"column": column, "unique_count": unique_count})
            continue
        if column in FORCED_CATEGORICAL:
            if unique_count > args.max_categorical_cardinality:
                dropped["high_cardinality"].append({"column": column, "unique_count": unique_count})
                continue
            categorical.append(column)
            continue
        sample = train[column].dropna().head(1000)
        converted = pd.to_numeric(sample, errors="coerce")
        if len(sample) and converted.notna().all():
            numeric.append(column)
        else:
            if unique_count > args.max_categorical_cardinality:
                dropped["high_cardinality"].append({"column": column, "unique_count": unique_count})
                continue
            categorical.append(column)
    return numeric, categorical, dropped


def make_preprocessor(deps: dict[str, Any], numeric_columns: list[str], categorical_columns: list[str]) -> Any:
    transformers = []
    if numeric_columns:
        transformers.append((
            "numeric",
            deps["Pipeline"]([("imputer", deps["SimpleImputer"](strategy="median"))]),
            numeric_columns,
        ))
    if categorical_columns:
        transformers.append((
            "categorical",
            deps["Pipeline"]([
                ("imputer", deps["SimpleImputer"](strategy="constant", fill_value="__missing__")),
                ("ordinal", deps["OrdinalEncoder"](handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            categorical_columns,
        ))
    return deps["ColumnTransformer"](transformers=transformers, remainder="drop")


def build_model(args: argparse.Namespace, deps: dict[str, Any]) -> Any:
    if args.model_family == "extra_trees":
        return deps["ExtraTreesRegressor"](
            n_estimators=args.max_iter,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
        )
    if args.model_family == "lightgbm":
        if deps["LGBMRegressor"] is None:
            raise SystemExit("LightGBM is not installed.")
        return deps["LGBMRegressor"](
            n_estimators=args.max_iter,
            learning_rate=args.learning_rate,
            num_leaves=args.max_leaf_nodes,
            min_child_samples=args.min_samples_leaf,
            reg_lambda=args.l2_regularization,
            max_bin=args.lightgbm_max_bin,
            feature_fraction=args.lightgbm_feature_fraction,
            bagging_fraction=args.lightgbm_bagging_fraction,
            bagging_freq=args.lightgbm_bagging_freq,
            force_col_wise=True,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
            verbosity=-1,
        )
    return deps["HistGradientBoostingRegressor"](
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_seed,
    )


def fit_model(frame: Any, feature_columns: list[str], numeric: list[str], categorical: list[str], args: argparse.Namespace, deps: dict[str, Any]) -> Any:
    model = deps["Pipeline"]([
        ("preprocess", make_preprocessor(deps, numeric, categorical)),
        ("model", build_model(args, deps)),
    ])
    model.fit(frame[feature_columns], frame[TARGET_COLUMN].astype(float))
    return model


def scaled_prediction(frame: Any, base_prediction_column: str, raw_correction: Any, scale: float, clip: float | None) -> Any:
    correction = raw_correction.astype(float) * float(scale)
    if clip is not None:
        correction = correction.clip(lower=-float(clip), upper=float(clip))
    return frame[base_prediction_column].astype(float) + correction


def select_scale(validation: Any, raw_correction: Any, args: argparse.Namespace, np: Any) -> dict[str, Any]:
    candidates = args.scale_candidate or [round(index * 0.05, 2) for index in range(0, 21)]
    rows = []
    best = None
    for scale in candidates:
        prediction = scaled_prediction(validation, args.base_prediction_column, raw_correction, float(scale), args.clip_correction_ms)
        item = {
            "scale": float(scale),
            **metric(prediction.to_numpy(), validation["actual_wind_mean_ms"].astype(float).to_numpy(), np),
        }
        rows.append(item)
        if best is None or float(item["rmse"]) < float(best["rmse"]):
            best = item
    assert best is not None
    return {"selected_scale": float(best["scale"]), "selected_metric": best, "candidates": rows}


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Hard Regime Specialist",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Verdict: `{result['verdict']}`",
        f"Hard logic: `{result['hard_rule']}`",
        f"Fit rows: `{result['fit_rows']}`",
        f"Scale validation hard rows: `{result['scale_validation_hard_rows']}`",
        f"Evaluation rows: `{result['evaluation_rows']}`",
        f"Evaluation hard rows: `{result['evaluation_hard_rows']}`",
        f"Selected scale: `{result['scale_selection']['selected_scale']}`",
        f"Base RMSE: `{result['evaluation']['base_metric'].get('rmse')}`",
        f"Specialist RMSE: `{result['evaluation']['specialist_metric'].get('rmse')}`",
        f"Gain vs base: `{result['evaluation']['rmse_gain_pct_vs_base']}%`",
        f"Gap to threshold: `{result['evaluation']['rmse_gap_to_threshold']}`",
        "",
        "## Evaluation Hard Subset",
        "",
        f"- base RMSE: `{result['evaluation']['base_hard_metric'].get('rmse')}`",
        f"- specialist RMSE: `{result['evaluation']['specialist_hard_metric'].get('rmse')}`",
        "",
        "## By Lead",
        "",
        "| Lead | Count | RMSE | MAE | Bias |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in result["evaluation"]["specialist_by_lead"]:
        lines.append(f"| `{item['group'].get('lead_time_minutes')}` | {item.get('count')} | {item.get('rmse')} | {item.get('mae')} | {item.get('bias')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]
    np = deps["np"]
    calibration = load_frame(args.calibration_predictions, pd, args.base_prediction_column)
    evaluation = load_frame(args.evaluation_predictions, pd, args.base_prediction_column)
    calibration_hard = hard_mask(calibration, args)
    evaluation_hard = hard_mask(evaluation, args)
    fit = calibration[calibration_hard].copy()
    if args.fit_start_utc:
        fit = fit[fit["issue_time_utc"] >= pd.Timestamp(args.fit_start_utc, tz="UTC")]
    if args.fit_end_utc:
        fit = fit[fit["issue_time_utc"] < pd.Timestamp(args.fit_end_utc, tz="UTC")]
    scale_validation = calibration[calibration_hard].copy()
    if args.scale_validation_start_utc:
        scale_validation = scale_validation[scale_validation["issue_time_utc"] >= pd.Timestamp(args.scale_validation_start_utc, tz="UTC")]
    if args.scale_validation_end_utc:
        scale_validation = scale_validation[scale_validation["issue_time_utc"] < pd.Timestamp(args.scale_validation_end_utc, tz="UTC")]
    fit = fit.dropna(subset=[TARGET_COLUMN, args.base_prediction_column, "actual_wind_mean_ms"]).copy()
    scale_validation = scale_validation.dropna(subset=[TARGET_COLUMN, args.base_prediction_column, "actual_wind_mean_ms"]).copy()
    if len(fit) < args.min_fit_rows:
        raise SystemExit(f"Not enough hard-regime fit rows: {len(fit)} < {args.min_fit_rows}")
    if len(scale_validation) < args.min_scale_validation_rows:
        raise SystemExit(
            f"Not enough hard-regime scale-validation rows: {len(scale_validation)} < {args.min_scale_validation_rows}"
        )
    numeric, categorical, dropped = infer_features(fit, evaluation, args, pd)
    feature_columns = [*numeric, *categorical]
    if not feature_columns:
        raise SystemExit("No valid specialist features.")
    model = fit_model(fit, feature_columns, numeric, categorical, args, deps)
    validation_raw_correction = pd.Series(model.predict(scale_validation[feature_columns]), index=scale_validation.index)
    scale_selection = select_scale(scale_validation, validation_raw_correction, args, np)
    scale = float(scale_selection["selected_scale"])
    evaluation = evaluation.copy()
    evaluation["hard_regime_selected"] = evaluation_hard
    evaluation["specialist_raw_correction_ms"] = 0.0
    eval_hard_frame = evaluation[evaluation_hard].copy()
    if not eval_hard_frame.empty:
        raw_correction = pd.Series(model.predict(eval_hard_frame[feature_columns]), index=eval_hard_frame.index)
        evaluation.loc[eval_hard_frame.index, "specialist_raw_correction_ms"] = raw_correction
    evaluation["specialist_scaled_correction_ms"] = evaluation["specialist_raw_correction_ms"].astype(float) * scale
    if args.clip_correction_ms is not None:
        evaluation["specialist_scaled_correction_ms"] = evaluation["specialist_scaled_correction_ms"].clip(
            lower=-float(args.clip_correction_ms),
            upper=float(args.clip_correction_ms),
        )
    evaluation["specialist_wind_mean_ms"] = evaluation[args.base_prediction_column].astype(float)
    evaluation.loc[evaluation_hard, "specialist_wind_mean_ms"] = (
        evaluation.loc[evaluation_hard, args.base_prediction_column].astype(float)
        + evaluation.loc[evaluation_hard, "specialist_scaled_correction_ms"].astype(float)
    )
    hard_eval = evaluation[evaluation_hard].copy()
    base_metric = metric_frame(evaluation, args.base_prediction_column, np)
    specialist_metric = metric_frame(evaluation, "specialist_wind_mean_ms", np)
    result = {
        "format": "corsewind.hard_regime_specialist.v1",
        "generated_at_utc": utc_now(),
        "threshold_rmse": args.threshold_rmse,
        "calibration_predictions": str(args.calibration_predictions),
        "evaluation_predictions": str(args.evaluation_predictions),
        "base_prediction_column": args.base_prediction_column,
        "hard_rule": {
            "logic": args.hard_logic,
            "spots": args.hard_spot,
            "min_lead": args.hard_min_lead,
            "lead_minutes": args.hard_lead_minute,
            "min_prediction_ms": args.hard_min_prediction_ms,
        },
        "fit_window_utc": {"start": args.fit_start_utc, "end": args.fit_end_utc},
        "scale_validation_window_utc": {
            "start": args.scale_validation_start_utc,
            "end": args.scale_validation_end_utc,
        },
        "model_family": args.model_family,
        "feature_column_count": len(feature_columns),
        "numeric_column_count": len(numeric),
        "categorical_column_count": len(categorical),
        "dropped_columns": dropped,
        "fit_rows": int(len(fit)),
        "scale_validation_hard_rows": int(len(scale_validation)),
        "evaluation_rows": int(len(evaluation)),
        "evaluation_hard_rows": int(evaluation_hard.sum()),
        "scale_selection": scale_selection,
        "evaluation": {
            "base_metric": base_metric,
            "specialist_metric": specialist_metric,
            "base_hard_metric": metric_frame(hard_eval, args.base_prediction_column, np),
            "specialist_hard_metric": metric_frame(hard_eval, "specialist_wind_mean_ms", np),
            "specialist_by_lead": grouped_metrics(evaluation, ["lead_time_minutes"], "specialist_wind_mean_ms", np, pd),
            "specialist_by_spot": grouped_metrics(evaluation, ["spot_id"], "specialist_wind_mean_ms", np, pd, limit=args.limit),
            "specialist_worst_spot_leads": grouped_metrics(
                evaluation,
                ["spot_id", "lead_time_minutes"],
                "specialist_wind_mean_ms",
                np,
                pd,
                limit=args.limit,
            ),
        },
        "verdict": "not_achieved",
    }
    result["evaluation"]["rmse_gap_to_threshold"] = round(float(specialist_metric["rmse"]) - args.threshold_rmse, 6)
    result["evaluation"]["rmse_gain_pct_vs_base"] = round(
        (float(base_metric["rmse"]) - float(specialist_metric["rmse"])) / float(base_metric["rmse"]) * 100.0,
        3,
    )
    if float(specialist_metric["rmse"]) < args.threshold_rmse:
        result["verdict"] = "achieved"
    if args.output_predictions:
        args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
        evaluation.to_parquet(args.output_predictions, index=False)
        result["output_predictions"] = str(args.output_predictions)
    if args.output_model:
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        deps["joblib"].dump(model, args.output_model)
        result["output_model"] = str(args.output_model)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-predictions", type=Path, required=True)
    parser.add_argument("--evaluation-predictions", type=Path, required=True)
    parser.add_argument("--base-prediction-column", default="calibrated_wind_mean_ms")
    parser.add_argument("--hard-spot", action="append", default=[])
    parser.add_argument("--hard-min-lead", type=float)
    parser.add_argument("--hard-lead-minute", type=int, action="append", default=[])
    parser.add_argument("--hard-min-prediction-ms", type=float)
    parser.add_argument("--hard-logic", choices=("or", "and"), default="or")
    parser.add_argument("--fit-start-utc")
    parser.add_argument("--fit-end-utc")
    parser.add_argument("--scale-validation-start-utc")
    parser.add_argument("--scale-validation-end-utc")
    parser.add_argument("--model-family", choices=("hist_gradient_boosting", "extra_trees", "lightgbm"), default="hist_gradient_boosting")
    parser.add_argument("--max-iter", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--l2-regularization", type=float, default=0.1)
    parser.add_argument("--min-samples-leaf", type=int, default=80)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--lightgbm-max-bin", type=int, default=63)
    parser.add_argument("--lightgbm-feature-fraction", type=float, default=0.75)
    parser.add_argument("--lightgbm-bagging-fraction", type=float, default=0.85)
    parser.add_argument("--lightgbm-bagging-freq", type=int, default=1)
    parser.add_argument("--clip-correction-ms", type=float, default=1.5)
    parser.add_argument("--scale-candidate", type=float, action="append", default=[])
    parser.add_argument("--min-non-null-count", type=int, default=80)
    parser.add_argument("--max-categorical-cardinality", type=int, default=100)
    parser.add_argument("--min-fit-rows", type=int, default=800)
    parser.add_argument("--min-scale-validation-rows", type=int, default=300)
    parser.add_argument("--threshold-rmse", type=float, default=0.9)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--output-predictions", type=Path)
    parser.add_argument("--output-model", type=Path)
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
        "fit_rows": result["fit_rows"],
        "scale_validation_hard_rows": result["scale_validation_hard_rows"],
        "evaluation_rows": result["evaluation_rows"],
        "evaluation_hard_rows": result["evaluation_hard_rows"],
        "selected_scale": result["scale_selection"]["selected_scale"],
        "base_rmse": result["evaluation"]["base_metric"].get("rmse"),
        "specialist_rmse": result["evaluation"]["specialist_metric"].get("rmse"),
        "base_hard_rmse": result["evaluation"]["base_hard_metric"].get("rmse"),
        "specialist_hard_rmse": result["evaluation"]["specialist_hard_metric"].get("rmse"),
        "rmse_gain_pct_vs_base": result["evaluation"]["rmse_gain_pct_vs_base"],
        "rmse_gap_to_threshold": result["evaluation"]["rmse_gap_to_threshold"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
