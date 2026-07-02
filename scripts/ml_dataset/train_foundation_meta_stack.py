#!/usr/bin/env python3
"""Train a same-key meta-stack over champion and foundation predictions."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGETS = {
    "wind_mean": {
        "actual": "actual_wind_mean_ms",
        "champion": "wind_champion_prediction_ms",
        "oracle": "wind_oracle_prediction_ms",
        "prefixes": ("raw_wind", "chronos", "chronos2", "timesfm", "moirai", "wind_champion"),
        "prediction_suffixes": ("wind_mean_ms", "wind_mean_prediction_ms"),
        "output": "meta_stack_wind_mean_ms",
    },
    "gust": {
        "actual": "actual_gust_ms",
        "champion": "gust_champion_prediction_ms",
        "oracle": "gust_oracle_prediction_ms",
        "prefixes": ("raw_gust", "chronos", "chronos2", "timesfm", "moirai", "gust_champion"),
        "prediction_suffixes": ("gust_ms", "gust_prediction_ms"),
        "output": "meta_stack_gust_ms",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def metric(frame: Any, prediction_column: str, actual_column: str, np: Any) -> dict[str, Any]:
    if prediction_column not in frame.columns:
        return {"count": 0}
    valid = frame[[prediction_column, actual_column]].dropna()
    if valid.empty:
        return {"count": 0}
    errors = valid[prediction_column].to_numpy(dtype=float) - valid[actual_column].to_numpy(dtype=float)
    return {
        "count": int(len(errors)),
        "rmse": round(float(math.sqrt(float(np.mean(errors * errors)))), 6),
        "mae": round(float(np.mean(np.abs(errors))), 6),
        "bias": round(float(np.mean(errors)), 6),
    }


def grouped_metrics(frame: Any, prediction_column: str, actual_column: str, group_column: str, np: Any, pd: Any) -> list[dict[str, Any]]:
    if group_column not in frame.columns:
        return []
    rows = []
    for value, group in frame.groupby(group_column, dropna=False):
        rows.append({"group": None if pd.isna(value) else value.item() if hasattr(value, "item") else value, **metric(group, prediction_column, actual_column, np)})
    return rows


def load_frame(path: Path, pd: Any) -> Any:
    frame = pd.read_parquet(path)
    frame["spot_id"] = frame["spot_id"].astype(str)
    frame["issue_time_utc"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    frame["lead_time_minutes"] = pd.to_numeric(frame["lead_time_minutes"], errors="coerce").astype("float64")
    frame["issue_hour_utc"] = frame["issue_time_utc"].dt.hour.astype("float64")
    frame["issue_month"] = frame["issue_time_utc"].dt.month.astype("float64")
    frame["issue_dayofyear"] = frame["issue_time_utc"].dt.dayofyear.astype("float64")
    return frame


def prediction_columns(frame: Any, target_config: dict[str, Any]) -> list[str]:
    blocked = ("oracle", "actual")
    candidates = []
    for column in frame.columns:
        lowered = column.lower()
        if any(token in lowered for token in blocked):
            continue
        if column.endswith("_ms") and any(suffix in column for suffix in target_config["prediction_suffixes"]):
            candidates.append(column)
    champion = target_config.get("champion")
    if champion in frame.columns:
        candidates.append(champion)
    return sorted(set(candidates))


def add_engineered_features(frame: Any, pred_cols: list[str], np: Any) -> Any:
    out = frame.copy()
    hour_angle = 2.0 * np.pi * out["issue_hour_utc"].astype(float) / 24.0
    month_angle = 2.0 * np.pi * out["issue_month"].astype(float) / 12.0
    day_angle = 2.0 * np.pi * out["issue_dayofyear"].fillna(1).astype(float) / 366.0
    out["issue_hour_sin"] = np.sin(hour_angle)
    out["issue_hour_cos"] = np.cos(hour_angle)
    out["issue_month_sin"] = np.sin(month_angle)
    out["issue_month_cos"] = np.cos(month_angle)
    out["issue_dayofyear_sin"] = np.sin(day_angle)
    out["issue_dayofyear_cos"] = np.cos(day_angle)
    if pred_cols:
        values = out[pred_cols].astype(float)
        out["expert_mean_ms"] = values.mean(axis=1, skipna=True)
        out["expert_std_ms"] = values.std(axis=1, skipna=True)
        out["expert_min_ms"] = values.min(axis=1, skipna=True)
        out["expert_max_ms"] = values.max(axis=1, skipna=True)
        out["expert_range_ms"] = out["expert_max_ms"] - out["expert_min_ms"]
        if "wind_champion_prediction_ms" in pred_cols:
            champion = out["wind_champion_prediction_ms"].astype(float)
            for column in pred_cols:
                if column != "wind_champion_prediction_ms":
                    out[f"{column}_minus_champion"] = out[column].astype(float) - champion
        if "gust_champion_prediction_ms" in pred_cols:
            champion = out["gust_champion_prediction_ms"].astype(float)
            for column in pred_cols:
                if column != "gust_champion_prediction_ms":
                    out[f"{column}_minus_champion"] = out[column].astype(float) - champion
    return out


def build_model(args: argparse.Namespace):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    def preprocess(numeric: list[str], categorical: list[str], scaled: bool = False) -> ColumnTransformer:
        if scaled:
            num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        else:
            num = SimpleImputer(strategy="median")
        return ColumnTransformer([
            ("num", num, numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ], remainder="drop")

    if args.model_family == "ridge":
        return lambda numeric, categorical: Pipeline([
            ("preprocess", preprocess(numeric, categorical, scaled=True)),
            ("model", Ridge(alpha=args.alpha)),
        ])
    if args.model_family == "extra_trees":
        return lambda numeric, categorical: Pipeline([
            ("preprocess", preprocess(numeric, categorical)),
            ("model", ExtraTreesRegressor(
                n_estimators=args.n_estimators,
                min_samples_leaf=args.min_samples_leaf,
                random_state=args.random_seed,
                n_jobs=args.n_jobs,
            )),
        ])
    return lambda numeric, categorical: Pipeline([
        ("preprocess", preprocess(numeric, categorical)),
        ("model", HistGradientBoostingRegressor(
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2_regularization,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_seed,
        )),
    ])


def feature_columns(frame: Any, actual: str, output: str) -> tuple[list[str], list[str]]:
    blocked_tokens = ("actual_", "oracle_", "target_", "meta_stack_")
    categorical = ["spot_id"]
    numeric = []
    for column in frame.columns:
        if column in categorical or column == actual or column == output:
            continue
        lowered = column.lower()
        if any(token in lowered for token in blocked_tokens):
            continue
        if column in {"issue_time_utc"}:
            continue
        if frame[column].dtype.kind in "biufc":
            numeric.append(column)
    return sorted(set(numeric)), categorical


def run(args: argparse.Namespace) -> dict[str, Any]:
    import joblib
    import numpy as np
    import pandas as pd

    config = TARGETS[args.target]
    actual = config["actual"]
    champion = config["champion"]
    output = config["output"]

    train = load_frame(args.train_superbench, pd)
    eval_frame = load_frame(args.eval_superbench, pd)
    pred_cols = [column for column in prediction_columns(train, config) if column in eval_frame.columns]
    if champion not in pred_cols:
        raise SystemExit(f"Champion column {champion} is required for target={args.target}")
    train = add_engineered_features(train, pred_cols, np)
    eval_frame = add_engineered_features(eval_frame, pred_cols, np)
    train = train.dropna(subset=[actual, champion]).copy()
    eval_frame = eval_frame.dropna(subset=[actual, champion]).copy()

    numeric, categorical = feature_columns(train, actual, output)
    numeric = [column for column in numeric if column in eval_frame.columns]
    categorical = [column for column in categorical if column in eval_frame.columns]
    model_factory = build_model(args)
    model = model_factory(numeric, categorical)
    target_values = train[actual].astype(float)
    if args.target_mode == "residual":
        target_values = target_values - train[champion].astype(float)
    model.fit(train[numeric + categorical], target_values)
    raw_prediction = model.predict(eval_frame[numeric + categorical])
    if args.target_mode == "residual":
        eval_frame[output] = eval_frame[champion].astype(float) + raw_prediction
    else:
        eval_frame[output] = raw_prediction
    if args.clip_delta_ms is not None:
        delta = (eval_frame[output].astype(float) - eval_frame[champion].astype(float)).clip(
            lower=-float(args.clip_delta_ms),
            upper=float(args.clip_delta_ms),
        )
        eval_frame[output] = eval_frame[champion].astype(float) + delta

    metrics = {
        column: metric(eval_frame, column, actual, np)
        for column in [*pred_cols, output, config["oracle"]]
        if column in eval_frame.columns
    }
    result = {
        "format": "corsewind.foundation_meta_stack.v1",
        "generated_at_utc": utc_now(),
        "target": args.target,
        "model_family": args.model_family,
        "target_mode": args.target_mode,
        "train_superbench": str(args.train_superbench),
        "eval_superbench": str(args.eval_superbench),
        "train_rows": int(len(train)),
        "eval_rows": int(len(eval_frame)),
        "prediction_columns": pred_cols,
        "numeric_feature_count": len(numeric),
        "categorical_features": categorical,
        "metrics": metrics,
        "by_lead": {
            column: grouped_metrics(eval_frame, column, actual, "lead_time_minutes", np, pd)
            for column in [champion, output, config["oracle"]]
            if column in eval_frame.columns
        },
    }
    champion_rmse = metrics.get(champion, {}).get("rmse")
    stack_rmse = metrics.get(output, {}).get("rmse")
    result["rmse_gain_pct_vs_champion"] = (
        None if champion_rmse is None or stack_rmse is None else round((champion_rmse - stack_rmse) / champion_rmse * 100.0, 3)
    )
    result["verdict"] = "improved" if stack_rmse is not None and champion_rmse is not None and stack_rmse < champion_rmse else "not_improved"

    args.output_root.mkdir(parents=True, exist_ok=True)
    eval_frame.to_parquet(args.output_root / "meta_stack_predictions.parquet", index=False)
    (args.output_root / "meta_stack_results.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_model:
        joblib.dump(model, args.output_root / "meta_stack_model.joblib")
    print(json.dumps({
        "output_root": str(args.output_root),
        "verdict": result["verdict"],
        "train_rows": result["train_rows"],
        "eval_rows": result["eval_rows"],
        "champion_rmse": champion_rmse,
        "stack_rmse": stack_rmse,
        "oracle_rmse": metrics.get(config["oracle"], {}).get("rmse"),
        "gain_pct": result["rmse_gain_pct_vs_champion"],
    }, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-superbench", type=Path, required=True)
    parser.add_argument("--eval-superbench", type=Path, required=True)
    parser.add_argument("--target", choices=sorted(TARGETS), default="wind_mean")
    parser.add_argument("--target-mode", choices=("direct", "residual"), default="residual")
    parser.add_argument("--model-family", choices=("ridge", "hist_gradient_boosting", "extra_trees"), default="ridge")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--l2-regularization", type=float, default=0.1)
    parser.add_argument("--min-samples-leaf", type=int, default=40)
    parser.add_argument("--clip-delta-ms", type=float, default=2.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-model", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
