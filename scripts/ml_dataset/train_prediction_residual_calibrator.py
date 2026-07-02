#!/usr/bin/env python3
"""Train a temporal second-stage calibrator from prediction parquet files."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import re


LEAKY_COLUMN_NAMES = {
    "actual_wind_mean_ms",
    "actual_gust_ms",
    "raw_error_ms",
    "corrected_error_ms",
    "abs_raw_error_ms",
    "abs_corrected_error_ms",
    "actual_wind_bin_ms",
    "actual_gust_bin_ms",
    "raw_abs_error_bin_ms",
    "corrected_abs_error_bin_ms",
}
BASE_FEATURES = {
    "spot_id",
    "station_id",
    "spot_kind",
    "latitude",
    "longitude",
    "lead_time_minutes",
    "issue_hour_utc",
    "raw_wind_mean_ms",
    "predicted_residual_wind_mean_ms",
    "corrected_wind_mean_ms",
    "raw_gust_ms",
    "predicted_residual_gust_ms",
    "corrected_gust_ms",
}
FORCED_CATEGORICAL = {"spot_id", "station_id", "spot_kind"}
TARGET_COLUMN = "__calibration_target_ms"
TARGET_CONFIGS = {
    "wind_mean": {
        "label": "wind mean",
        "actual": "actual_wind_mean_ms",
        "raw": "raw_wind_mean_ms",
        "corrected": "corrected_wind_mean_ms",
        "calibrated": "calibrated_wind_mean_ms",
        "predicted_second_stage": "predicted_second_stage_residual_wind_mean_ms",
    },
    "gust": {
        "label": "gust",
        "actual": "actual_gust_ms",
        "raw": "raw_gust_ms",
        "corrected": "corrected_gust_ms",
        "calibrated": "calibrated_gust_ms",
        "predicted_second_stage": "predicted_second_stage_residual_gust_ms",
    },
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
    return {
        "count": int(len(errors)),
        "mae": round(float(np.mean(np.abs(errors))), 6),
        "rmse": round(float(math.sqrt(float(np.mean(errors * errors)))), 6),
        "bias": round(float(np.mean(errors)), 6),
        "p50_abs_error": round(float(np.quantile(np.abs(errors), 0.50)), 6),
        "p90_abs_error": round(float(np.quantile(np.abs(errors), 0.90)), 6),
        "p95_abs_error": round(float(np.quantile(np.abs(errors), 0.95)), 6),
    }


def metric_frame(frame: Any, prediction_column: str, observation_column: str, np: Any) -> dict[str, Any]:
    valid = frame[[prediction_column, observation_column]].dropna()
    return metric(
        valid[prediction_column].astype(float).to_numpy(),
        valid[observation_column].astype(float).to_numpy(),
        np,
    )


def grouped_metrics(
    frame: Any,
    group_columns: list[str],
    prediction_column: str,
    observation_column: str,
    np: Any,
    pd: Any,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if any(column not in frame.columns for column in group_columns):
        return []
    rows = []
    valid = frame.dropna(subset=[prediction_column, observation_column])
    for raw_key, group in valid.groupby(group_columns, dropna=False):
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        rows.append({
            "group": dict(zip(group_columns, [json_scalar(pd, value) for value in values], strict=True)),
            **metric_frame(group, prediction_column, observation_column, np),
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


def load_predictions(path: Path, pd: Any, *, start: str | None, end: str | None, leads: list[int]) -> Any:
    frame = pd.read_parquet(path)
    frame["issue_time_utc"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    if start:
        frame = frame[frame["issue_time_utc"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        frame = frame[frame["issue_time_utc"] < pd.Timestamp(end, tz="UTC")]
    if leads:
        frame = frame[frame["lead_time_minutes"].astype("Int64").isin([int(lead) for lead in leads])]
    frame = frame.copy()
    frame["issue_hour_utc"] = frame["issue_time_utc"].dt.hour.astype("float64")
    frame["issue_month_number"] = frame["issue_time_utc"].dt.month.astype("float64")
    dayofyear = frame["issue_time_utc"].dt.dayofyear.fillna(1).astype(float)
    frame["issue_dayofyear_sin"] = (2.0 * math.pi * dayofyear / 366.0).map(math.sin)
    frame["issue_dayofyear_cos"] = (2.0 * math.pi * dayofyear / 366.0).map(math.cos)
    return frame


def allowed_feature(column: str) -> bool:
    if column in LEAKY_COLUMN_NAMES:
        return False
    if column in BASE_FEATURES:
        return True
    if column in {"issue_month_number", "issue_dayofyear_sin", "issue_dayofyear_cos"}:
        return True
    if column.startswith("features__") or column.startswith("baselines__"):
        return True
    return False


def infer_features(train: Any, max_categorical_cardinality: int, pd: Any) -> tuple[list[str], list[str], dict[str, Any]]:
    numeric: list[str] = []
    categorical: list[str] = []
    dropped: dict[str, Any] = {"missing_or_sparse": [], "constant": [], "high_cardinality_categorical": [], "leaky": []}
    for column in sorted(column for column in train.columns if allowed_feature(column)):
        if column in LEAKY_COLUMN_NAMES:
            dropped["leaky"].append(column)
            continue
        non_null = int(train[column].notna().sum())
        if non_null < 100:
            dropped["missing_or_sparse"].append({"column": column, "non_null_count": non_null})
            continue
        unique_count = int(train[column].dropna().nunique())
        if unique_count <= 1:
            dropped["constant"].append({"column": column, "unique_count": unique_count})
            continue
        if column in FORCED_CATEGORICAL:
            if unique_count > max_categorical_cardinality:
                dropped["high_cardinality_categorical"].append({"column": column, "unique_count": unique_count})
                continue
            categorical.append(column)
            continue
        sample = train[column].dropna().head(1000)
        converted = pd.to_numeric(sample, errors="coerce")
        if len(sample) and converted.notna().all():
            numeric.append(column)
        else:
            if unique_count > max_categorical_cardinality:
                dropped["high_cardinality_categorical"].append({"column": column, "unique_count": unique_count})
                continue
            categorical.append(column)
    return numeric, categorical, dropped


def make_preprocessor(deps: dict[str, Any], numeric_columns: list[str], categorical_columns: list[str]):
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


def build_model(args: argparse.Namespace, deps: dict[str, Any]):
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
            force_col_wise=args.lightgbm_force_col_wise,
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


def fit_pipeline(
    frame: Any,
    feature_columns: list[str],
    numeric: list[str],
    categorical: list[str],
    args: argparse.Namespace,
    deps: dict[str, Any],
) -> Any:
    model = deps["Pipeline"]([
        ("preprocess", make_preprocessor(deps, numeric, categorical)),
        ("model", build_model(args, deps)),
    ])
    train_mask = frame[TARGET_COLUMN].notna()
    model.fit(frame.loc[train_mask, feature_columns], frame.loc[train_mask, TARGET_COLUMN])
    return model


def group_key_tuple(key: Any, column_count: int) -> tuple[Any, ...]:
    if column_count == 1:
        if isinstance(key, tuple) and len(key) == 1:
            return key
        return (key,)
    return tuple(key)


def group_mask(frame: Any, pd: Any, group_columns: list[str], group_values: tuple[Any, ...]) -> Any:
    mask = None
    for column, value in zip(group_columns, group_values, strict=True):
        if pd.isna(value):
            part = frame[column].isna()
        else:
            part = frame[column] == value
        mask = part if mask is None else (mask & part)
    return mask


def safe_token(value: Any) -> str:
    text = "missing" if value is None else str(value)
    text = text.replace("/", "_").replace("\\", "_")
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_") or "value"


def group_label(group_columns: list[str], group_values: tuple[Any, ...]) -> str:
    return "|".join(
        f"{column}={safe_token(value)}"
        for column, value in zip(group_columns, group_values, strict=True)
    )


def fit_predict_residuals(
    calibration: Any,
    evaluation: Any,
    feature_columns: list[str],
    numeric: list[str],
    categorical: list[str],
    args: argparse.Namespace,
    deps: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    pd = deps["pd"]
    group_columns = [
        column
        for column in args.fit_group_column
        if column in calibration.columns and column in evaluation.columns
    ]
    skipped_group_columns = [
        column
        for column in args.fit_group_column
        if column not in calibration.columns or column not in evaluation.columns
    ]
    if not group_columns:
        model = fit_pipeline(calibration, feature_columns, numeric, categorical, args, deps)
        predictions = pd.Series(
            model.predict(evaluation[feature_columns]),
            index=evaluation.index,
            dtype="float64",
        )
        return predictions, model, {
            "fit_group_columns": [],
            "skipped_fit_group_columns": skipped_group_columns,
            "fit_groups": [],
            "skipped_fit_groups": [],
            "missing_prediction_count": 0,
        }

    predictions = pd.Series(index=evaluation.index, dtype="float64")
    models: dict[str, Any] = {}
    fit_groups: list[dict[str, Any]] = []
    skipped_groups: list[dict[str, Any]] = []
    calibration_groups = calibration[group_columns].copy()
    calibration_groups["__row_index"] = calibration_groups.index
    for raw_key, group in calibration_groups.groupby(group_columns, dropna=False):
        group_values = group_key_tuple(raw_key, len(group_columns))
        train_indices = group["__row_index"]
        eval_mask = group_mask(evaluation, pd, group_columns, group_values)
        eval_indices = evaluation.index[eval_mask]
        label = group_label(group_columns, group_values)
        if len(train_indices) < args.min_group_train_rows or len(eval_indices) < args.min_group_eval_rows:
            skipped_groups.append({
                "group": dict(zip(group_columns, [json_scalar(pd, value) for value in group_values], strict=True)),
                "train_rows": int(len(train_indices)),
                "evaluation_rows": int(len(eval_indices)),
                "reason": "not_enough_group_rows",
            })
            continue
        group_model = fit_pipeline(
            calibration.loc[train_indices],
            feature_columns,
            numeric,
            categorical,
            args,
            deps,
        )
        predictions.loc[eval_indices] = group_model.predict(evaluation.loc[eval_indices, feature_columns])
        models[label] = group_model
        fit_groups.append({
            "group": dict(zip(group_columns, [json_scalar(pd, value) for value in group_values], strict=True)),
            "train_rows": int(len(train_indices)),
            "evaluation_rows": int(len(eval_indices)),
        })
    return predictions, models, {
        "fit_group_columns": group_columns,
        "skipped_fit_group_columns": skipped_group_columns,
        "fit_groups": fit_groups,
        "skipped_fit_groups": skipped_groups,
        "missing_prediction_count": int(predictions.isna().sum()),
    }


def select_correction_scale(
    calibration: Any,
    feature_columns: list[str],
    numeric: list[str],
    categorical: list[str],
    args: argparse.Namespace,
    deps: dict[str, Any],
    target_config: dict[str, str],
) -> dict[str, Any]:
    pd = deps["pd"]
    np = deps["np"]
    actual_column = target_config["actual"]
    corrected_column = target_config["corrected"]
    if not args.scale_validation_start_utc:
        return {
            "method": "fixed",
            "selected_scale": float(args.correction_scale),
            "candidates": [],
            "validation_metrics": [],
        }
    start = pd.Timestamp(args.scale_validation_start_utc, tz="UTC")
    end = pd.Timestamp(args.scale_validation_end_utc, tz="UTC") if args.scale_validation_end_utc else None
    fit_frame = calibration[calibration["issue_time_utc"] < start].copy()
    validation = calibration[calibration["issue_time_utc"] >= start].copy()
    if end is not None:
        validation = validation[validation["issue_time_utc"] < end].copy()
    if len(fit_frame) < 1000 or len(validation) < 1000:
        raise SystemExit(
            "Not enough rows for calibration scale validation: "
            f"fit={len(fit_frame)}, validation={len(validation)}"
        )
    validation = validation.dropna(subset=[actual_column, corrected_column]).copy()
    correction, _, group_summary = fit_predict_residuals(fit_frame, validation, feature_columns, numeric, categorical, args, deps)
    validation = validation.assign(__scale_candidate_correction=correction)
    validation = validation.dropna(subset=["__scale_candidate_correction"]).copy()
    if len(validation) < 1000:
        raise SystemExit(
            "Not enough predicted rows for calibration scale validation: "
            f"predicted={len(validation)}, group_summary={group_summary}"
        )
    candidates = args.scale_candidate or [round(value * 0.05, 2) for value in range(0, 31)]
    metrics = []
    best: dict[str, Any] | None = None
    for scale in candidates:
        pred = validation[corrected_column].astype(float) + validation["__scale_candidate_correction"].astype(float) * float(scale)
        if args.clip_correction_ms is not None:
            delta = (pred - validation[corrected_column].astype(float)).clip(
                lower=-float(args.clip_correction_ms),
                upper=float(args.clip_correction_ms),
            )
            pred = validation[corrected_column].astype(float) + delta
        item = {
            "scale": float(scale),
            **metric(pred.to_numpy(), validation[actual_column].astype(float).to_numpy(), np),
        }
        metrics.append(item)
        if best is None or float(item["rmse"]) < float(best["rmse"]):
            best = item
    assert best is not None
    selected_group_scales: dict[str, Any] = {}
    group_columns = [
        column
        for column in args.fit_group_column
        if column in validation.columns and column in calibration.columns
    ]
    if args.scale_by_fit_group and group_columns:
        for raw_key, group in validation.groupby(group_columns, dropna=False):
            group_values = group_key_tuple(raw_key, len(group_columns))
            label = group_label(group_columns, group_values)
            if len(group) < args.min_scale_group_rows:
                selected_group_scales[label] = {
                    "group": dict(zip(group_columns, [json_scalar(pd, value) for value in group_values], strict=True)),
                    "selected_scale": float(best["scale"]),
                    "fallback": True,
                    "reason": "not_enough_validation_rows",
                    "validation_rows": int(len(group)),
                }
                continue
            group_best: dict[str, Any] | None = None
            group_metrics = []
            for scale in candidates:
                pred = group[corrected_column].astype(float) + group["__scale_candidate_correction"].astype(float) * float(scale)
                if args.clip_correction_ms is not None:
                    delta = (pred - group[corrected_column].astype(float)).clip(
                        lower=-float(args.clip_correction_ms),
                        upper=float(args.clip_correction_ms),
                    )
                    pred = group[corrected_column].astype(float) + delta
                item = {
                    "scale": float(scale),
                    **metric(pred.to_numpy(), group[actual_column].astype(float).to_numpy(), np),
                }
                group_metrics.append(item)
                if group_best is None or float(item["rmse"]) < float(group_best["rmse"]):
                    group_best = item
            assert group_best is not None
            selected_group_scales[label] = {
                "group": dict(zip(group_columns, [json_scalar(pd, value) for value in group_values], strict=True)),
                "selected_scale": float(group_best["scale"]),
                "selected_validation_metric": group_best,
                "validation_rows": int(len(group)),
                "fallback": False,
                "candidates": group_metrics,
            }
    return {
        "method": "calibration_validation",
        "fit_rows": int(len(fit_frame)),
        "validation_rows": int(len(validation)),
        "validation_start_utc": args.scale_validation_start_utc,
        "validation_end_utc": args.scale_validation_end_utc,
        "group_summary": group_summary,
        "selected_scale": float(best["scale"]),
        "scale_by_fit_group": bool(args.scale_by_fit_group and group_columns),
        "selected_group_scales": selected_group_scales,
        "selected_validation_metric": best,
        "validation_metrics": metrics,
    }


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


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Prediction Residual Calibrator",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Target: `{result['target']}`",
        f"Model family: `{result['model_family']}`",
        f"Verdict: `{result['verdict']}`",
        f"Calibration rows: `{result['calibration_row_count']}`",
        f"Evaluation rows: `{result['evaluation_row_count']}`",
        f"Base RMSE: `{result['base_metrics'].get('rmse')}`",
        f"Calibrated RMSE: `{result['calibrated_metrics'].get('rmse')}`",
        f"Gain vs base: `{result['rmse_gain_pct_vs_base']}%`",
        f"Gap to threshold: `{result['rmse_gap_to_threshold']}`",
        "",
        "## By Lead",
        "",
        "| Lead | Count | Base RMSE | Calibrated RMSE | Calibrated MAE | Bias |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_lead_base = {str(item["group"].get("lead_time_minutes")): item for item in result["base_by_lead"]}
    for item in result["calibrated_by_lead"]:
        lead = str(item["group"].get("lead_time_minutes"))
        base = by_lead_base.get(lead, {})
        lines.append(
            f"| `{lead}` | {item.get('count')} | {base.get('rmse')} | {item.get('rmse')} | "
            f"{item.get('mae')} | {item.get('bias')} |"
        )
    lines.extend(["", "## Worst Spots", "", "| Spot | Count | RMSE | MAE | Bias |", "| --- | ---: | ---: | ---: | ---: |"])
    for item in result["calibrated_worst_spots"]:
        lines.append(
            f"| `{item['group'].get('spot_id')}` | {item.get('count')} | {item.get('rmse')} | "
            f"{item.get('mae')} | {item.get('bias')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]
    np = deps["np"]
    target_config = TARGET_CONFIGS[args.target]
    actual_column = target_config["actual"]
    corrected_column = target_config["corrected"]
    calibrated_column = target_config["calibrated"]
    second_stage_column = target_config["predicted_second_stage"]
    calibration = load_predictions(
        args.calibration_predictions,
        pd,
        start=args.calibration_start_utc,
        end=args.calibration_end_utc,
        leads=args.lead_minute,
    )
    evaluation = load_predictions(
        args.evaluation_predictions,
        pd,
        start=args.evaluation_start_utc,
        end=args.evaluation_end_utc,
        leads=args.lead_minute,
    )
    calibration = calibration.dropna(subset=[actual_column, corrected_column]).copy()
    evaluation = evaluation.dropna(subset=[actual_column, corrected_column]).copy()
    calibration[TARGET_COLUMN] = calibration[actual_column].astype(float) - calibration[corrected_column].astype(float)
    numeric, categorical, dropped = infer_features(calibration, args.max_categorical_cardinality, pd)
    feature_columns = [*numeric, *categorical]
    if not feature_columns:
        raise SystemExit("No valid calibration features.")
    scale_selection = select_correction_scale(calibration, feature_columns, numeric, categorical, args, deps, target_config)
    correction_scale = float(scale_selection["selected_scale"])
    prediction_series, model, group_summary = fit_predict_residuals(
        calibration,
        evaluation,
        feature_columns,
        numeric,
        categorical,
        args,
        deps,
    )
    evaluation[second_stage_column] = prediction_series.fillna(0.0)
    selected_group_scales = scale_selection.get("selected_group_scales") or {}
    group_columns = group_summary.get("fit_group_columns") or []
    if selected_group_scales and group_columns:
        evaluation["selected_correction_scale"] = correction_scale
        for label, item in selected_group_scales.items():
            values = tuple((item.get("group") or {}).get(column) for column in group_columns)
            mask = group_mask(evaluation, pd, group_columns, values)
            evaluation.loc[mask, "selected_correction_scale"] = float(item.get("selected_scale", correction_scale))
        evaluation[second_stage_column] = (
            evaluation[second_stage_column].astype(float)
            * evaluation["selected_correction_scale"].astype(float)
        )
    else:
        evaluation["selected_correction_scale"] = correction_scale
        evaluation[second_stage_column] = (
            evaluation[second_stage_column].astype(float) * correction_scale
        )
    if args.clip_correction_ms is not None:
        evaluation[second_stage_column] = evaluation[second_stage_column].clip(
            lower=-float(args.clip_correction_ms),
            upper=float(args.clip_correction_ms),
        )
    evaluation[calibrated_column] = evaluation[corrected_column] + evaluation[second_stage_column]
    base_metrics = metric_frame(evaluation, corrected_column, actual_column, np)
    calibrated_metrics = metric_frame(evaluation, calibrated_column, actual_column, np)
    base_rmse = base_metrics.get("rmse")
    calibrated_rmse = calibrated_metrics.get("rmse")
    result = {
        "format": "corsewind.prediction_residual_calibrator.v1",
        "generated_at_utc": utc_now(),
        "target": args.target,
        "target_label": target_config["label"],
        "actual_column": actual_column,
        "base_prediction_column": corrected_column,
        "calibrated_prediction_column": calibrated_column,
        "model_family": args.model_family,
        "threshold_rmse": args.threshold_rmse,
        "calibration_predictions": str(args.calibration_predictions),
        "evaluation_predictions": str(args.evaluation_predictions),
        "calibration_window_utc": {"start": args.calibration_start_utc, "end": args.calibration_end_utc},
        "evaluation_window_utc": {"start": args.evaluation_start_utc, "end": args.evaluation_end_utc},
        "lead_minutes": args.lead_minute,
        "calibration_row_count": int(len(calibration)),
        "evaluation_row_count": int(len(evaluation)),
        "feature_column_count": len(feature_columns),
        "numeric_column_count": len(numeric),
        "categorical_column_count": len(categorical),
        "group_modeling": group_summary,
        "dropped_columns": dropped,
        "scale_selection": scale_selection,
        "base_metrics": base_metrics,
        "calibrated_metrics": calibrated_metrics,
        "base_by_lead": grouped_metrics(evaluation, ["lead_time_minutes"], corrected_column, actual_column, np, pd),
        "calibrated_by_lead": grouped_metrics(evaluation, ["lead_time_minutes"], calibrated_column, actual_column, np, pd),
        "calibrated_worst_spots": grouped_metrics(evaluation, ["spot_id"], calibrated_column, actual_column, np, pd, limit=args.limit),
        "calibrated_worst_spot_leads": grouped_metrics(
            evaluation,
            ["spot_id", "lead_time_minutes"],
            calibrated_column,
            actual_column,
            np,
            pd,
            limit=args.limit,
        ),
        "verdict": "not_achieved",
    }
    if calibrated_rmse is not None and calibrated_rmse < args.threshold_rmse:
        result["verdict"] = "achieved"
    result["rmse_gap_to_threshold"] = None if calibrated_rmse is None else round(float(calibrated_rmse) - args.threshold_rmse, 6)
    result["rmse_gain_pct_vs_base"] = (
        None
        if not base_rmse or calibrated_rmse is None
        else round((float(base_rmse) - float(calibrated_rmse)) / float(base_rmse) * 100.0, 3)
    )
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
    parser.add_argument("--calibration-start-utc")
    parser.add_argument("--calibration-end-utc")
    parser.add_argument("--evaluation-start-utc")
    parser.add_argument("--evaluation-end-utc")
    parser.add_argument("--lead-minute", type=int, action="append", default=[15, 30, 45, 60])
    parser.add_argument("--target", choices=sorted(TARGET_CONFIGS), default="wind_mean")
    parser.add_argument("--model-family", choices=("hist_gradient_boosting", "extra_trees", "lightgbm"), default="hist_gradient_boosting")
    parser.add_argument("--max-iter", type=int, default=240)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--l2-regularization", type=float, default=0.1)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--lightgbm-max-bin", type=int, default=127)
    parser.add_argument("--lightgbm-feature-fraction", type=float, default=0.9)
    parser.add_argument("--lightgbm-bagging-fraction", type=float, default=0.85)
    parser.add_argument("--lightgbm-bagging-freq", type=int, default=1)
    parser.add_argument("--lightgbm-force-col-wise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-correction-ms", type=float, default=2.0)
    parser.add_argument("--correction-scale", type=float, default=1.0)
    parser.add_argument("--scale-validation-start-utc")
    parser.add_argument("--scale-validation-end-utc")
    parser.add_argument("--scale-candidate", type=float, action="append", default=[])
    parser.add_argument("--max-categorical-cardinality", type=int, default=100)
    parser.add_argument(
        "--fit-group-column",
        action="append",
        default=[],
        help="Train one second-stage calibrator per value or value-combination of this column.",
    )
    parser.add_argument("--min-group-train-rows", type=int, default=1000)
    parser.add_argument("--min-group-eval-rows", type=int, default=100)
    parser.add_argument("--scale-by-fit-group", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-scale-group-rows", type=int, default=500)
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
        "calibration_rows": result["calibration_row_count"],
        "evaluation_rows": result["evaluation_row_count"],
        "feature_columns": result["feature_column_count"],
        "selected_scale": result["scale_selection"].get("selected_scale"),
        "base_rmse": result["base_metrics"].get("rmse"),
        "calibrated_rmse": result["calibrated_metrics"].get("rmse"),
        "rmse_gain_pct_vs_base": result["rmse_gain_pct_vs_base"],
        "rmse_gap_to_threshold": result["rmse_gap_to_threshold"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
