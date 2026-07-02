#!/usr/bin/env python3
"""Train a historical router on foundation/champion superbench files.

This is a compact benchmark for the question: does training a router on
2024/2025 historical predictions generalize to 2026?
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KT_PER_MS = 1.9438444924406


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def metric(frame: Any, pred_col: str, actual_col: str) -> dict[str, Any]:
    values = frame[[pred_col, actual_col]].dropna()
    if values.empty:
        return {"n": 0}
    err = (values[pred_col].astype(float) - values[actual_col].astype(float)) * KT_PER_MS
    abs_err = err.abs()
    return {
        "n": int(len(values)),
        "rmse_kt": float((err.pow(2).mean()) ** 0.5),
        "mae_kt": float(abs_err.mean()),
        "bias_kt": float(err.mean()),
        "p90_abs_error_kt": float(abs_err.quantile(0.90)),
    }


def threshold_metric(frame: Any, pred_col: str, actual_col: str, threshold_kt: float) -> dict[str, Any]:
    values = frame[[pred_col, actual_col]].dropna()
    if values.empty:
        return {"n": 0, "threshold_kt": threshold_kt}
    pred = values[pred_col].astype(float) * KT_PER_MS >= threshold_kt
    actual = values[actual_col].astype(float) * KT_PER_MS >= threshold_kt
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    return {
        "n": int(len(values)),
        "threshold_kt": threshold_kt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
        "csi": None if tp + fp + fn == 0 else tp / (tp + fp + fn),
    }


def add_time_features(frame: Any, pd: Any, np: Any) -> Any:
    out = frame.copy()
    out["issue_time_utc"] = pd.to_datetime(out["issue_time_utc"], utc=True, errors="coerce")
    out["issue_hour_utc"] = out["issue_time_utc"].dt.hour.astype(float)
    out["issue_month"] = out["issue_time_utc"].dt.month.astype(float)
    dayofyear = out["issue_time_utc"].dt.dayofyear.fillna(1).astype(float)
    out["issue_dayofyear_sin"] = np.sin(2.0 * math.pi * dayofyear / 366.0)
    out["issue_dayofyear_cos"] = np.cos(2.0 * math.pi * dayofyear / 366.0)
    return out


def valid_candidates(train: Any, evaluation: Any, candidates: dict[str, str]) -> dict[str, str]:
    out = {}
    for name, column in candidates.items():
        if column not in train.columns or column not in evaluation.columns:
            continue
        if train[column].notna().sum() == 0 or evaluation[column].notna().sum() == 0:
            continue
        out[name] = column
    return out


def build_classifier(args: argparse.Namespace, numeric: list[str], categorical: list[str]) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    transformers = [("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric)]
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            categorical,
        ))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
        (
            "model",
            ExtraTreesClassifier(
                n_estimators=args.n_estimators,
                min_samples_leaf=args.min_samples_leaf,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                class_weight="balanced",
            ),
        ),
    ])


def build_regressor(args: argparse.Namespace, numeric: list[str], categorical: list[str]) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    transformers = [("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric)]
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            categorical,
        ))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
        (
            "model",
            HistGradientBoostingRegressor(
                max_iter=args.max_iter,
                learning_rate=args.learning_rate,
                max_leaf_nodes=args.max_leaf_nodes,
                min_samples_leaf=args.min_samples_leaf,
                l2_regularization=args.l2_regularization,
                random_state=args.random_state,
            ),
        ),
    ])


def add_oracle(frame: Any, target: str, actual_col: str, candidates: dict[str, str]) -> None:
    error_cols = []
    for name, column in candidates.items():
        err_col = f"{target}_err_{name}"
        frame[err_col] = (frame[column].astype(float) - frame[actual_col].astype(float)).abs()
        error_cols.append(err_col)
    frame[f"{target}_oracle_choice"] = (
        frame[error_cols].idxmin(axis=1).str.removeprefix(f"{target}_err_")
    )
    frame[f"{target}_oracle_prediction_ms"] = float("nan")
    for name, column in candidates.items():
        mask = frame[f"{target}_oracle_choice"] == name
        frame.loc[mask, f"{target}_oracle_prediction_ms"] = frame.loc[mask, column]


def run_target(
    args: argparse.Namespace,
    train: Any,
    evaluation: Any,
    *,
    target: str,
    actual_col: str,
    candidates: dict[str, str],
    pd: Any,
) -> tuple[dict[str, Any], Any]:
    candidates = valid_candidates(train, evaluation, candidates)
    train = train.dropna(subset=[actual_col, *candidates.values()]).copy()
    evaluation = evaluation.dropna(subset=[actual_col, *candidates.values()]).copy()
    add_oracle(train, target, actual_col, candidates)
    add_oracle(evaluation, target, actual_col, candidates)

    numeric = []
    for column in [
        "lead_time_minutes",
        "issue_hour_utc",
        "issue_month",
        "issue_dayofyear_sin",
        "issue_dayofyear_cos",
        *candidates.values(),
    ]:
        if column in train.columns and column in evaluation.columns and column not in numeric:
            numeric.append(column)
    categorical = [column for column in ("spot_id",) if column in train.columns and column in evaluation.columns]
    feature_columns = [*numeric, *categorical]

    y = train[f"{target}_oracle_choice"].astype(str)
    classifier = build_classifier(args, numeric, categorical)
    classifier.fit(train[feature_columns], y)
    evaluation[f"{target}_router_choice"] = classifier.predict(evaluation[feature_columns])
    evaluation[f"{target}_router_prediction_ms"] = float("nan")
    for name, column in candidates.items():
        mask = evaluation[f"{target}_router_choice"] == name
        evaluation.loc[mask, f"{target}_router_prediction_ms"] = evaluation.loc[mask, column]

    regressor = build_regressor(args, numeric, categorical)
    regressor.fit(train[feature_columns], train[actual_col].astype(float))
    evaluation[f"{target}_stacker_prediction_ms"] = regressor.predict(evaluation[feature_columns])

    prediction_columns = {
        **candidates,
        "router": f"{target}_router_prediction_ms",
        "stacker": f"{target}_stacker_prediction_ms",
        "oracle": f"{target}_oracle_prediction_ms",
    }
    thresholds = (12.0, 15.0, 20.0, 25.0) if target == "wind" else (15.0, 20.0, 25.0)
    summary = {
        "train_rows": int(len(train)),
        "eval_rows": int(len(evaluation)),
        "candidates": candidates,
        "metrics": {name: metric(evaluation, column, actual_col) for name, column in prediction_columns.items()},
        "train_oracle_choice_share": train[f"{target}_oracle_choice"].value_counts(normalize=True).round(6).to_dict(),
        "eval_oracle_choice_share": evaluation[f"{target}_oracle_choice"].value_counts(normalize=True).round(6).to_dict(),
        "router_choice_share": evaluation[f"{target}_router_choice"].value_counts(normalize=True).round(6).to_dict(),
        "thresholds": {
            f">={int(threshold)}kt": {
                name: threshold_metric(evaluation, column, actual_col, threshold)
                for name, column in prediction_columns.items()
            }
            for threshold in thresholds
        },
    }
    base = args.base_candidate if args.base_candidate in candidates else next(iter(candidates))
    base_rmse = summary["metrics"][base]["rmse_kt"]
    for name in ("router", "stacker", "oracle"):
        rmse = summary["metrics"][name]["rmse_kt"]
        summary[f"{name}_gain_pct_vs_{base}"] = (base_rmse - rmse) / base_rmse * 100.0
    return summary, evaluation


def parse_candidates(values: list[str]) -> dict[str, str]:
    out = {}
    for value in values:
        name, column = value.split(":", 1)
        out[name] = column
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    train = add_time_features(pd.read_parquet(args.train_parquet), pd, np)
    evaluation = add_time_features(pd.read_parquet(args.eval_parquet), pd, np)
    wind_summary, wind_predictions = run_target(
        args,
        train,
        evaluation,
        target="wind",
        actual_col=args.wind_actual_column,
        candidates=parse_candidates(args.wind_candidate),
        pd=pd,
    )
    gust_summary, gust_predictions = run_target(
        args,
        train,
        evaluation,
        target="gust",
        actual_col=args.gust_actual_column,
        candidates=parse_candidates(args.gust_candidate),
        pd=pd,
    )
    result = {
        "format": "corsewind.historical_foundation_router.v1",
        "generated_at_utc": utc_now(),
        "train_parquet": str(args.train_parquet),
        "eval_parquet": str(args.eval_parquet),
        "base_candidate": args.base_candidate,
        "wind": wind_summary,
        "gust": gust_summary,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "historical_router_results.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    wind_predictions.to_parquet(args.output_root / "historical_router_wind_predictions.parquet", index=False)
    gust_predictions.to_parquet(args.output_root / "historical_router_gust_predictions.parquet", index=False)
    print(json.dumps({
        "output_root": str(args.output_root),
        "wind": {
            "base_rmse": wind_summary["metrics"].get(args.base_candidate, next(iter(wind_summary["metrics"].values()))).get("rmse_kt"),
            "router_rmse": wind_summary["metrics"]["router"]["rmse_kt"],
            "stacker_rmse": wind_summary["metrics"]["stacker"]["rmse_kt"],
            "oracle_rmse": wind_summary["metrics"]["oracle"]["rmse_kt"],
        },
        "gust": {
            "base_rmse": gust_summary["metrics"].get(args.base_candidate, next(iter(gust_summary["metrics"].values()))).get("rmse_kt"),
            "router_rmse": gust_summary["metrics"]["router"]["rmse_kt"],
            "stacker_rmse": gust_summary["metrics"]["stacker"]["rmse_kt"],
            "oracle_rmse": gust_summary["metrics"]["oracle"]["rmse_kt"],
        },
    }, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--eval-parquet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-candidate", default="champion")
    parser.add_argument("--wind-actual-column", default="actual_wind_mean_ms")
    parser.add_argument("--gust-actual-column", default="actual_gust_ms")
    parser.add_argument("--wind-candidate", action="append", required=True, help="name:column")
    parser.add_argument("--gust-candidate", action="append", required=True, help="name:column")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--l2-regularization", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
