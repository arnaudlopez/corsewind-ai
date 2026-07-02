#!/usr/bin/env python3
"""Train spot-specific HGB wind residual models and score a sequence benchmark."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET = "labels__residual_wind_mean_ms"
OBSERVED = "labels__target_wind_mean_ms"
BASELINE = "baselines__baseline_wind_mean_ms"
OUTPUT_COLUMN = "spot_hgb_wind_mean_ms"


def import_dependencies() -> dict[str, Any]:
    try:
        import joblib
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OrdinalEncoder
    except ImportError as exc:
        raise SystemExit("Missing dependencies. Run in the ML dataset runner image.") from exc
    return locals()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def month_range(start_month: str, end_month: str) -> list[str]:
    year, month = [int(part) for part in start_month.split("-", 1)]
    end_year, end_month_num = [int(part) for part in end_month.split("-", 1)]
    out = []
    while (year, month) <= (end_year, end_month_num):
        out.append(f"{year:04d}_{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return out


def discover_paths(args: argparse.Namespace) -> list[Path]:
    paths = []
    for suffix in month_range(args.start_month, args.end_month):
        path = args.training_table_root / f"{args.run_id_prefix}_{suffix}" / "training_rows.parquet"
        if path.exists():
            paths.append(path)
    if not paths:
        raise SystemExit("No training shards found.")
    return paths


def add_time_features(frame: Any, pd: Any, np: Any) -> Any:
    issue_time = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    dayofyear = issue_time.dt.dayofyear.fillna(1).astype(float)
    frame["issue_hour_utc"] = issue_time.dt.hour.astype("float64")
    frame["issue_month"] = issue_time.dt.month.astype("float64")
    angle = 2.0 * math.pi * dayofyear / 366.0
    frame["issue_dayofyear_sin"] = np.sin(angle)
    frame["issue_dayofyear_cos"] = np.cos(angle)
    return frame


def load_rows(paths: list[Path], columns: list[str], spot_ids: set[str], args: argparse.Namespace, deps: dict[str, Any]) -> Any:
    pd = deps["pd"]
    np = deps["np"]
    pq = deps["pq"]
    frames = []
    read_columns = sorted(set(columns) | {"spot_id", "issue_time_utc", "lead_time_minutes", TARGET, OBSERVED, BASELINE})
    wanted_leads = {int(value) for value in args.lead_minutes.split(",") if value}
    train_end = pd.Timestamp(args.train_end, tz="UTC")
    for path in paths:
        pf = pq.ParquetFile(path)
        available = [column for column in read_columns if column in pf.schema.names]
        for batch in pf.iter_batches(batch_size=args.read_batch_size, columns=available):
            frame = batch.to_pandas().reindex(columns=read_columns)
            frame = frame[frame["spot_id"].isin(spot_ids)]
            frame = frame[frame["lead_time_minutes"].isin(wanted_leads)]
            if frame.empty:
                continue
            issue_time = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
            frame = frame[issue_time < train_end]
            frame = frame.dropna(subset=[TARGET, BASELINE, OBSERVED])
            if frame.empty:
                continue
            frames.append(add_time_features(frame, pd, np))
    if not frames:
        raise SystemExit("No matching training rows found.")
    return pd.concat(frames, ignore_index=True)


def make_model(deps: dict[str, Any], numeric_columns: list[str], categorical_columns: list[str], args: argparse.Namespace):
    transformers = []
    if numeric_columns:
        transformers.append(("num", deps["SimpleImputer"](strategy="median"), numeric_columns))
    if categorical_columns:
        transformers.append((
            "cat",
            deps["Pipeline"]([
                ("imputer", deps["SimpleImputer"](strategy="constant", fill_value="__missing__")),
                ("ordinal", deps["OrdinalEncoder"](handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            categorical_columns,
        ))
    return deps["Pipeline"]([
        ("preprocess", deps["ColumnTransformer"](transformers=transformers, remainder="drop")),
        ("model", deps["HistGradientBoostingRegressor"](
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2_regularization,
            random_state=args.random_seed,
        )),
    ])


def metric(np: Any, frame: Any, pred_col: str, actual_col: str) -> dict[str, Any]:
    valid = frame[[pred_col, actual_col]].dropna()
    if valid.empty:
        return {"count": 0}
    err = valid[pred_col].to_numpy(dtype=float) - valid[actual_col].to_numpy(dtype=float)
    return {
        "count": int(len(err)),
        "mae": round(float(np.mean(np.abs(err))), 6),
        "rmse": round(float(np.sqrt(np.mean(err * err))), 6),
        "bias": round(float(np.mean(err)), 6),
    }


def load_benchmark_features(paths: list[Path], predictions: Any, columns: list[str], deps: dict[str, Any], args: argparse.Namespace) -> Any:
    pd = deps["pd"]
    np = deps["np"]
    pq = deps["pq"]
    wanted = predictions[["spot_id", "issue_time_utc", "lead_time_minutes"]].copy()
    wanted["lead_time_minutes"] = wanted["lead_time_minutes"].astype(int)
    wanted_keys = set(zip(wanted["spot_id"].astype(str), wanted["issue_time_utc"].astype(str), wanted["lead_time_minutes"].astype(int)))
    read_columns = sorted(set(columns) | {"spot_id", "issue_time_utc", "lead_time_minutes", BASELINE})
    frames = []
    for path in paths:
        pf = pq.ParquetFile(path)
        available = [column for column in read_columns if column in pf.schema.names]
        for batch in pf.iter_batches(batch_size=args.read_batch_size, columns=available):
            frame = batch.to_pandas().reindex(columns=read_columns)
            frame["lead_time_minutes"] = frame["lead_time_minutes"].astype("Int64")
            keys = list(zip(frame["spot_id"].astype(str), frame["issue_time_utc"].astype(str), frame["lead_time_minutes"].astype(int)))
            mask = [key in wanted_keys for key in keys]
            if any(mask):
                frames.append(add_time_features(frame.loc[mask].copy(), pd, np))
    if not frames:
        raise SystemExit("No benchmark feature rows found.")
    return pd.concat(frames, ignore_index=True).drop_duplicates(["spot_id", "issue_time_utc", "lead_time_minutes"], keep="last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-table-root", type=Path, required=True)
    parser.add_argument("--run-id-prefix", default="residual_windsup_sst_prev")
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument("--end-month", default="2026-06")
    parser.add_argument("--train-end", default="2026-01-01T00:00:00Z")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--benchmark-predictions", default="predictions_final_all_models.parquet")
    parser.add_argument("--feature-columns-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lead-minutes", default="15,30,45,60")
    parser.add_argument("--read-batch-size", type=int, default=100_000)
    parser.add_argument("--max-rows-per-spot", type=int)
    parser.add_argument("--max-iter", type=int, default=240)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = import_dependencies()
    pd = deps["pd"]
    np = deps["np"]
    joblib = deps["joblib"]
    args.output_root.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(args.benchmark_root / args.benchmark_predictions)
    spot_ids = set(predictions["spot_id"].dropna().astype(str).unique())
    feature_info = json.loads(args.feature_columns_json.read_text(encoding="utf-8"))
    numeric_columns = feature_info["numeric"]
    categorical_columns = feature_info["categorical"]
    feature_columns = [*numeric_columns, *categorical_columns]
    paths = discover_paths(args)

    train = load_rows(paths, feature_columns, spot_ids, args, deps)
    benchmark_features = load_benchmark_features(paths, predictions, feature_columns, deps, args)

    scored = predictions.drop(columns=[OUTPUT_COLUMN], errors="ignore").copy()
    model_summaries = {}
    scored_parts = []
    for spot_id, spot_train in train.groupby("spot_id"):
        if args.max_rows_per_spot and len(spot_train) > args.max_rows_per_spot:
            spot_train = spot_train.sample(n=args.max_rows_per_spot, random_state=args.random_seed)
        if len(spot_train) < 200:
            model_summaries[str(spot_id)] = {"skipped": "not_enough_rows", "train_rows": int(len(spot_train))}
            continue
        model = make_model(deps, numeric_columns, categorical_columns, args)
        model.fit(spot_train[feature_columns], spot_train[TARGET])
        model_path = args.output_root / f"{spot_id}__{TARGET}.joblib"
        joblib.dump(model, model_path)
        spot_features = benchmark_features[benchmark_features["spot_id"] == spot_id].copy()
        if spot_features.empty:
            model_summaries[str(spot_id)] = {"skipped": "no_benchmark_rows", "train_rows": int(len(spot_train))}
            continue
        residual = model.predict(spot_features[feature_columns])
        spot_features[OUTPUT_COLUMN] = spot_features[BASELINE].astype(float).to_numpy() + residual
        scored_parts.append(spot_features[["spot_id", "issue_time_utc", "lead_time_minutes", OUTPUT_COLUMN]])
        model_summaries[str(spot_id)] = {"train_rows": int(len(spot_train)), "model_path": str(model_path)}

    if scored_parts:
        spot_predictions = pd.concat(scored_parts, ignore_index=True)
        scored = scored.merge(spot_predictions, on=["spot_id", "issue_time_utc", "lead_time_minutes"], how="left")
    else:
        scored[OUTPUT_COLUMN] = math.nan
    scored.to_parquet(args.output_root / "predictions_with_spot_hgb.parquet", index=False)

    result = {
        "generated_at_utc": utc_now(),
        "row_count": int(len(scored)),
        "spot_count": int(len(spot_ids)),
        "model_summaries": model_summaries,
        "metrics": {
            "spot_hgb": metric(np, scored, OUTPUT_COLUMN, "actual_wind_mean_ms"),
            "chronos2_univar": metric(np, scored, "chronos2_univar_wind_mean_ms_p50", "actual_wind_mean_ms"),
            "global_hgb": metric(np, scored, "hgb_wind_mean_ms", "actual_wind_mean_ms"),
            "raw_nwp": metric(np, scored, "raw_wind_mean_ms", "actual_wind_mean_ms"),
        },
    }
    (args.output_root / "spot_hgb_results.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
