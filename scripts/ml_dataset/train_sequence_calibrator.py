#!/usr/bin/env python3
"""Train/evaluate a leakage-safe wind calibrator from sequence benchmark outputs."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET = "actual_wind_mean_ms"
BASELINE = "raw_wind_mean_ms"
DEFAULT_FEATURES = [
    "raw_wind_mean_ms",
    "chronos2_univar_wind_mean_ms_p10",
    "chronos2_univar_wind_mean_ms_p50",
    "chronos2_univar_wind_mean_ms_p90",
    "timesfm_wind_mean_ms_p10",
    "timesfm_wind_mean_ms_p50",
    "timesfm_wind_mean_ms_p90",
    "moirai_wind_mean_ms_p10",
    "moirai_wind_mean_ms_p50",
    "moirai_wind_mean_ms_p90",
    "chronos_wind_mean_ms_p50",
    "persist_wind_mean_ms",
    "past_wind_mean_4",
    "past_wind_mean_8",
    "past_wind_trend_4",
    "past_wind_slope_4",
    "lead_time_minutes",
]
CATEGORICAL = ["spot_id"]
MODEL_P50_COLUMNS = [
    "raw_wind_mean_ms",
    "chronos2_univar_wind_mean_ms_p50",
    "timesfm_wind_mean_ms_p50",
    "moirai_wind_mean_ms_p50",
    "chronos_wind_mean_ms_p50",
    "persist_wind_mean_ms",
    "past_wind_mean_4",
    "past_wind_mean_8",
    "past_wind_trend_4",
]
SELECTOR_CANDIDATE_COLUMNS = [
    "raw_wind_mean_ms",
    "chronos_wind_mean_ms_p10",
    "chronos_wind_mean_ms_p50",
    "chronos_wind_mean_ms_p90",
    "chronos2_univar_wind_mean_ms_p10",
    "chronos2_univar_wind_mean_ms_p50",
    "chronos2_univar_wind_mean_ms_p90",
    "timesfm_wind_mean_ms_p10",
    "timesfm_wind_mean_ms_p50",
    "timesfm_wind_mean_ms_p90",
    "moirai_wind_mean_ms_p10",
    "moirai_wind_mean_ms_p50",
    "moirai_wind_mean_ms_p90",
    "persist_wind_mean_ms",
    "past_wind_mean_4",
    "past_wind_mean_8",
    "past_wind_trend_4",
]
DEFAULT_TRAINING_FEATURE_PREFIXES = [
    "features__obs_",
    "features__model_error_now_",
    "features__issue_hour_",
    "features__issue_dayofyear_",
    "features__previous_run_open_meteo_",
    "features__sst_",
    "features__eumetsat_",
    "features__context_nearest_",
    "features__context_coastal_",
    "features__context_inland_",
    "features__context_relief_",
    "features__context_global_",
    "features__context_agg_",
]
FORBIDDEN_FEATURE_PATTERNS = (
    "labels__",
    "target_feature_sources__",
    "actual_",
    "calibrated_",
    "target_time",
    "timestamp",
)


def import_dependencies() -> dict[str, Any]:
    try:
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.base import clone
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:
        raise SystemExit("Missing dependencies. Run in an ML Python environment with sklearn/pandas/pyarrow.") from exc
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        LGBMRegressor = None
    deps = locals()
    deps["LGBMRegressor"] = LGBMRegressor
    return deps


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def month_range(start_month: str, end_month: str) -> list[str]:
    start_year, start_m = [int(part) for part in start_month.split("-", 1)]
    end_year, end_m = [int(part) for part in end_month.split("-", 1)]
    months = []
    year, month = start_year, start_m
    while (year, month) <= (end_year, end_m):
        months.append(f"{year:04d}_{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return months


def discover_training_parquets(root: Path, prefix: str, start_month: str, end_month: str) -> list[Path]:
    paths = []
    for suffix in month_range(start_month, end_month):
        path = root / f"{prefix}_{suffix}" / "training_rows.parquet"
        if path.exists():
            paths.append(path)
    return paths


def metric(np: Any, frame: Any, pred_col: str, actual_col: str = TARGET) -> dict[str, Any]:
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


def metric_by_group(np: Any, frame: Any, pred_col: str, groups: list[str]) -> dict[str, dict[str, Any]]:
    out = {}
    available = [column for column in groups if column in frame.columns]
    if not available or pred_col not in frame.columns:
        return out
    for keys, group in frame.groupby(available, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key = "|".join(str(value) for value in keys)
        out[key] = metric(np, group, pred_col)
    return out


def coverage_summary(frame: Any, pd: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": int(len(frame))}
    if "spot_id" in frame.columns:
        summary["unique_spots"] = int(frame["spot_id"].nunique(dropna=True))
    if "lead_time_minutes" in frame.columns:
        summary["unique_leads"] = int(frame["lead_time_minutes"].nunique(dropna=True))
        summary["rows_by_lead"] = {
            str(int(lead)): int(count)
            for lead, count in frame["lead_time_minutes"].dropna().astype(float).round().astype(int).value_counts().sort_index().items()
        }
    if "issue_time" in frame.columns:
        issue_times = pd.to_datetime(frame["issue_time"], utc=True, errors="coerce").dropna()
        summary["unique_issue_days"] = int(issue_times.dt.date.nunique())
        summary["unique_issue_months"] = int(issue_times.dt.tz_convert(None).dt.to_period("M").nunique())
        if not issue_times.empty:
            summary["issue_time_min"] = issue_times.min().isoformat().replace("+00:00", "Z")
            summary["issue_time_max"] = issue_times.max().isoformat().replace("+00:00", "Z")
    return summary


def add_persistence_features(frame: Any, root: Path, pd: Any, np: Any) -> Any:
    context_path = root / "past_context.parquet"
    if not context_path.exists():
        return frame
    past = pd.read_parquet(context_path).sort_values(["item_id", "timestamp"])
    rows = []
    for item_id, group in past.groupby("item_id", sort=False):
        values = group["wind_mean_ms"].to_numpy(dtype=float)
        if len(values) == 0:
            continue
        last = float(values[-1])
        mean4 = float(np.nanmean(values[-4:]))
        mean8 = float(np.nanmean(values[-8:]))
        if len(values) >= 5 and not np.isnan(values[-1]) and not np.isnan(values[-5]):
            slope4 = float((values[-1] - values[-5]) / 4.0)
        else:
            slope4 = 0.0
        rows.append({
            "item_id": item_id,
            "persist_wind_mean_ms": last,
            "past_wind_mean_4": mean4,
            "past_wind_mean_8": mean8,
            "past_wind_slope_4": slope4,
        })
    if not rows:
        return frame
    features = pd.DataFrame(rows)
    frame = frame.merge(features, on="item_id", how="left")
    step = pd.to_numeric(frame["lead_time_minutes"], errors="coerce") / 15.0
    frame["past_wind_trend_4"] = frame["persist_wind_mean_ms"] + frame["past_wind_slope_4"] * step
    return frame


def load_frames(args: argparse.Namespace, pd: Any, np: Any) -> Any:
    frames = []
    for root in args.benchmark_root:
        path = root / args.predictions_file
        if not path.exists():
            for fallback in (
                "predictions_final_all_models.parquet",
                "predictions_with_chronos2_univariate.parquet",
                "predictions_with_timesfm.parquet",
                "predictions_with_hgb.parquet",
                "predictions.parquet",
            ):
                candidate = root / fallback
                if candidate.exists():
                    path = candidate
                    break
        if not path.exists():
            raise SystemExit(f"No predictions parquet found in {root}")
        frame = pd.read_parquet(path)
        frame = add_persistence_features(frame, root, pd, np)
        frame["benchmark_root"] = str(root)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["issue_time"] = pd.to_datetime(out["issue_time_utc"], utc=True, errors="coerce")
    out["target_time"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["issue_hour_utc"] = out["issue_time"].dt.hour.astype("float64")
    out["issue_month"] = out["issue_time"].dt.month.astype("float64")
    out["lead_time_minutes"] = out["lead_time_minutes"].astype("float64")
    return out


def selected_training_columns(all_columns: list[str], prefixes: list[str], limit: int) -> list[str]:
    blocked_prefixes = ("labels__", "target_feature_sources__")
    key_columns = ["spot_id", "issue_time_utc", "lead_time_minutes"]
    selected = []
    seen = set()
    for prefix in prefixes:
        for column in all_columns:
            if column in seen or column in key_columns:
                continue
            if column.startswith(blocked_prefixes):
                continue
            if column.startswith(prefix):
                selected.append(column)
                seen.add(column)
    return selected[:limit]


def canonical_issue_time(series: Any, pd: Any) -> Any:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_lead_minutes(series: Any, pd: Any) -> Any:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def merge_training_table_features(args: argparse.Namespace, frame: Any, pd: Any) -> tuple[Any, list[str], dict[str, Any]]:
    if args.training_table_root is None:
        return frame, [], {"enabled": False}
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Merging training-table features requires pyarrow.") from exc

    paths = discover_training_parquets(
        args.training_table_root,
        args.training_run_id_prefix,
        args.start_month,
        args.end_month,
    )
    if not paths:
        raise SystemExit(f"No training_rows.parquet shards found under {args.training_table_root}")

    key_columns = ["spot_id", "issue_time_utc", "lead_time_minutes"]
    wanted = frame[key_columns].copy()
    wanted["spot_id"] = wanted["spot_id"].astype(str)
    wanted["issue_time_utc"] = canonical_issue_time(wanted["issue_time_utc"], pd)
    wanted["lead_time_minutes"] = canonical_lead_minutes(wanted["lead_time_minutes"], pd)
    wanted = wanted.dropna(subset=key_columns)
    wanted_keys = set(zip(
        wanted["spot_id"],
        wanted["issue_time_utc"],
        wanted["lead_time_minutes"].astype("int64"),
    ))

    prefixes = args.training_feature_prefix or DEFAULT_TRAINING_FEATURE_PREFIXES
    all_schema_columns: list[str] = []
    seen_schema_columns = set()
    for path in paths:
        pf = pq.ParquetFile(path)
        for column in pf.schema.names:
            if column not in seen_schema_columns:
                all_schema_columns.append(column)
                seen_schema_columns.add(column)
    selected_columns = selected_training_columns(all_schema_columns, prefixes, args.max_training_features)
    missing_required = [
        pattern
        for pattern in args.require_selected_training_feature
        if not any(pattern in column for column in selected_columns)
    ]
    if missing_required:
        raise SystemExit(
            "Required training-table feature patterns were not selected; "
            f"increase --max-training-features or adjust prefix priority: {missing_required}"
        )
    selected_by_prefix = {
        prefix: sum(column.startswith(prefix) for column in selected_columns)
        for prefix in prefixes
    }

    frames = []
    for path in paths:
        pf = pq.ParquetFile(path)
        read_columns = [column for column in [*key_columns, *selected_columns] if column in pf.schema.names]
        if len(read_columns) <= len(key_columns):
            continue
        for batch in pf.iter_batches(batch_size=args.training_batch_size, columns=read_columns):
            shard = batch.to_pandas().reindex(columns=read_columns)
            shard["spot_id"] = shard["spot_id"].astype(str)
            shard["issue_time_utc"] = canonical_issue_time(shard["issue_time_utc"], pd)
            shard["lead_time_minutes"] = canonical_lead_minutes(shard["lead_time_minutes"], pd)
            shard = shard.dropna(subset=["spot_id", "issue_time_utc", "lead_time_minutes"])
            keys = list(zip(
                shard["spot_id"],
                shard["issue_time_utc"],
                shard["lead_time_minutes"].astype("int64"),
            ))
            keep = shard[[key in wanted_keys for key in keys]]
            if not keep.empty:
                frames.append(keep)
    if not frames:
        return frame, [], {
            "enabled": True,
            "paths": [str(path) for path in paths],
            "selected_column_count": len(selected_columns),
            "selected_by_prefix": selected_by_prefix,
            "required_selected_training_features": args.require_selected_training_feature,
            "merged_row_count": 0,
        }
    training = pd.concat(frames, ignore_index=True).drop_duplicates(key_columns, keep="last")
    training["lead_time_minutes"] = training["lead_time_minutes"].astype("float64")
    left = frame.copy()
    left["spot_id"] = left["spot_id"].astype(str)
    left["issue_time_utc"] = canonical_issue_time(left["issue_time_utc"], pd)
    left["lead_time_minutes"] = canonical_lead_minutes(left["lead_time_minutes"], pd).astype("float64")
    merged = left.merge(training, on=key_columns, how="left")
    feature_columns = [column for column in selected_columns if column in merged.columns]
    return merged, feature_columns, {
        "enabled": True,
        "paths": [str(path) for path in paths],
        "prefixes": prefixes,
        "selected_column_count": len(selected_columns),
        "selected_by_prefix": selected_by_prefix,
        "required_selected_training_features": args.require_selected_training_feature,
        "merged_feature_count": len(feature_columns),
        "merged_row_count": int(len(training)),
    }


def add_engineered_features(frame: Any, np: Any) -> Any:
    out = frame.copy()
    if "issue_hour_utc" in out.columns:
        angle = 2.0 * np.pi * out["issue_hour_utc"].astype(float) / 24.0
        out["issue_hour_sin"] = np.sin(angle)
        out["issue_hour_cos"] = np.cos(angle)
    if "issue_month" in out.columns:
        angle = 2.0 * np.pi * out["issue_month"].astype(float) / 12.0
        out["issue_month_sin"] = np.sin(angle)
        out["issue_month_cos"] = np.cos(angle)
    if "lead_time_minutes" in out.columns:
        out["lead_steps_15m"] = out["lead_time_minutes"].astype(float) / 15.0

    for prefix in ("chronos2_univar", "timesfm", "moirai"):
        p10 = f"{prefix}_wind_mean_ms_p10"
        p50 = f"{prefix}_wind_mean_ms_p50"
        p90 = f"{prefix}_wind_mean_ms_p90"
        if p10 in out.columns and p90 in out.columns:
            out[f"{prefix}_wind_mean_ms_spread_p90_p10"] = out[p90] - out[p10]
        if BASELINE in out.columns and p50 in out.columns:
            out[f"{prefix}_minus_raw_wind_mean_ms"] = out[p50] - out[BASELINE]
        if "persist_wind_mean_ms" in out.columns and p50 in out.columns:
            out[f"{prefix}_minus_persistence_wind_mean_ms"] = out[p50] - out["persist_wind_mean_ms"]

    if BASELINE in out.columns and "persist_wind_mean_ms" in out.columns:
        out["raw_minus_persistence_wind_mean_ms"] = out[BASELINE] - out["persist_wind_mean_ms"]
    if BASELINE in out.columns and "past_wind_mean_4" in out.columns:
        out["raw_minus_past_wind_mean_4"] = out[BASELINE] - out["past_wind_mean_4"]
    if "lead_steps_15m" in out.columns:
        for column in (
            BASELINE,
            "persist_wind_mean_ms",
            "past_wind_mean_4",
            "past_wind_trend_4",
        ):
            if column in out.columns:
                out[f"{column}_x_lead_steps_15m"] = out[column].astype(float) * out["lead_steps_15m"]

    present = [column for column in MODEL_P50_COLUMNS if column in out.columns]
    if len(present) >= 2:
        out["model_p50_mean_wind_mean_ms"] = out[present].mean(axis=1, skipna=True)
        out["model_p50_std_wind_mean_ms"] = out[present].std(axis=1, skipna=True)
        out["model_p50_min_wind_mean_ms"] = out[present].min(axis=1, skipna=True)
        out["model_p50_max_wind_mean_ms"] = out[present].max(axis=1, skipna=True)
    return out


def ordered_existing(columns: list[str], frame_columns: set[str]) -> list[str]:
    out = []
    seen = set()
    for column in columns:
        if column in frame_columns and column not in seen:
            out.append(column)
            seen.add(column)
    return out


def forbidden_feature_matches(columns: list[str]) -> list[str]:
    matches = []
    for column in columns:
        lowered = column.lower()
        if any(pattern in lowered for pattern in FORBIDDEN_FEATURE_PATTERNS):
            matches.append(column)
    return matches


def limit_training_rows(train: Any, args: argparse.Namespace, np: Any) -> tuple[Any, dict[str, Any]]:
    if args.max_train_rows is None or args.max_train_rows <= 0 or len(train) <= args.max_train_rows:
        return train, {
            "enabled": False,
            "original_rows": int(len(train)),
            "used_rows": int(len(train)),
        }
    sort_columns = [column for column in ("issue_time", "spot_id", "lead_time_minutes") if column in train.columns]
    ordered = train.sort_values(sort_columns, kind="mergesort") if sort_columns else train
    indices = np.linspace(0, len(ordered) - 1, num=args.max_train_rows, dtype=int)
    sampled = ordered.iloc[indices].copy()
    return sampled, {
        "enabled": True,
        "strategy": "deterministic_even_time_sample",
        "original_rows": int(len(train)),
        "used_rows": int(len(sampled)),
        "max_train_rows": int(args.max_train_rows),
    }


def build_preprocess(deps: dict[str, Any], numeric_pipe: Any, numeric: list[str], categorical: list[str]):
    return deps["ColumnTransformer"]([
        ("num", numeric_pipe, numeric),
        ("cat", deps["Pipeline"]([
            ("imputer", deps["SimpleImputer"](strategy="constant", fill_value="__missing__")),
            ("onehot", deps["OneHotEncoder"](handle_unknown="ignore")),
        ]), categorical),
    ], remainder="drop")


def build_model(args: argparse.Namespace, deps: dict[str, Any], numeric: list[str], categorical: list[str]):
    if args.model_family == "ridge":
        numeric_pipe = deps["Pipeline"]([
            ("imputer", deps["SimpleImputer"](strategy="median")),
            ("scaler", deps["StandardScaler"]()),
        ])
        estimator = deps["Ridge"](alpha=args.alpha)
    elif args.model_family == "random_forest":
        numeric_pipe = deps["SimpleImputer"](strategy="median")
        estimator = deps["RandomForestRegressor"](
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
        )
    elif args.model_family == "hist_gradient_boosting":
        numeric_pipe = deps["SimpleImputer"](strategy="median")
        estimator = deps["HistGradientBoostingRegressor"](
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2_regularization,
            random_state=args.random_seed,
        )
    elif args.model_family == "extra_trees":
        numeric_pipe = deps["SimpleImputer"](strategy="median")
        estimator = deps["ExtraTreesRegressor"](
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
        )
    else:
        if deps["LGBMRegressor"] is None:
            raise SystemExit("LightGBM is not installed in this Python environment.")
        numeric_pipe = deps["SimpleImputer"](strategy="median")
        estimator = deps["LGBMRegressor"](
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_child_samples=args.min_samples_leaf,
            reg_lambda=args.l2_regularization,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
            verbosity=-1,
        )
    preprocess = build_preprocess(deps, numeric_pipe, numeric, categorical)
    return deps["Pipeline"]([("preprocess", preprocess), ("model", estimator)])


def build_error_selector_model(args: argparse.Namespace, deps: dict[str, Any], numeric: list[str], categorical: list[str]):
    numeric_pipe = deps["SimpleImputer"](strategy="median")
    estimator = deps["ExtraTreesRegressor"](
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_seed,
        n_jobs=args.n_jobs,
    )
    preprocess = build_preprocess(deps, numeric_pipe, numeric, categorical)
    return deps["Pipeline"]([("preprocess", preprocess), ("model", estimator)])


def group_key(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def group_label(keys: tuple[Any, ...]) -> str:
    return "|".join(str(key) for key in keys)


def fit_predict_calibrator(args: argparse.Namespace, deps: dict[str, Any], train: Any, test: Any, numeric: list[str], categorical: list[str], target: Any) -> tuple[Any, Any, dict[str, Any]]:
    np = deps["np"]
    pd = deps["pd"]
    x_train = train[numeric + categorical]
    x_test = test[numeric + categorical]
    global_model = build_model(args, deps, numeric, categorical)
    global_model.fit(x_train, target)
    raw_prediction = pd.Series(np.asarray(global_model.predict(x_test), dtype=float), index=test.index)
    model_artifact: Any = global_model
    group_summary: dict[str, Any] = {"enabled": False}

    if not args.fit_group:
        return raw_prediction, model_artifact, group_summary

    missing_group_columns = [column for column in args.fit_group if column not in train.columns or column not in test.columns]
    if missing_group_columns:
        raise SystemExit(f"--fit-group columns are missing from train/test: {missing_group_columns}")

    group_models = {}
    group_rows = {}
    trained_group_count = 0
    skipped_group_count = 0
    for raw_keys, group in train.groupby(args.fit_group, dropna=False, sort=False):
        keys = group_key(raw_keys)
        if len(group) < args.min_group_train_rows:
            skipped_group_count += 1
            group_rows[group_label(keys)] = {"train_rows": int(len(group)), "status": "fallback_too_sparse"}
            continue
        estimator = deps["clone"](global_model)
        group_target = target.loc[group.index]
        estimator.fit(group[numeric + categorical], group_target)
        group_models[keys] = estimator
        trained_group_count += 1
        group_rows[group_label(keys)] = {"train_rows": int(len(group)), "status": "trained"}

    if group_models:
        test_grouped = test.groupby(args.fit_group, dropna=False, sort=False)
        for raw_keys, group in test_grouped:
            keys = group_key(raw_keys)
            estimator = group_models.get(keys)
            if estimator is None:
                continue
            raw_prediction.loc[group.index] = estimator.predict(group[numeric + categorical])

    group_summary = {
        "enabled": True,
        "fit_group": args.fit_group,
        "min_group_train_rows": args.min_group_train_rows,
        "trained_group_count": trained_group_count,
        "skipped_group_count": skipped_group_count,
        "groups": group_rows,
        "fallback": "global_model",
    }
    model_artifact = {"global_model": global_model, "group_models": group_models, "group_summary": group_summary}
    return raw_prediction.to_numpy(dtype=float), model_artifact, group_summary


def selector_candidates(train: Any, test: Any) -> list[str]:
    candidates = []
    for column in SELECTOR_CANDIDATE_COLUMNS:
        if column not in train.columns or column not in test.columns:
            continue
        if train[column].notna().any() and test[column].notna().any():
            candidates.append(column)
    return candidates


def fit_error_selector_models(args: argparse.Namespace, deps: dict[str, Any], train: Any, numeric: list[str], categorical: list[str], candidates: list[str]) -> dict[str, Any]:
    np = deps["np"]
    models = {}
    x_train = train[numeric + categorical]
    actual = train[TARGET].astype(float)
    for index, column in enumerate(candidates):
        usable = train[column].notna() & actual.notna()
        if int(usable.sum()) < args.min_group_train_rows:
            continue
        estimator = build_error_selector_model(args, deps, numeric, categorical)
        # Fit expected squared error so selector optimizes the RMSE objective rather than MAE.
        error_target = np.square(train.loc[usable, column].astype(float) - actual.loc[usable])
        estimator.fit(x_train.loc[usable], error_target)
        models[column] = estimator
    return models


def predict_error_selector(
    *,
    deps: dict[str, Any],
    models: dict[str, Any],
    test: Any,
    numeric: list[str],
    categorical: list[str],
    fallback_column: str,
) -> tuple[Any, Any, dict[str, Any]]:
    np = deps["np"]
    pd = deps["pd"]
    x_test = test[numeric + categorical]
    candidate_errors = {}
    for column, estimator in models.items():
        if column not in test.columns:
            continue
        predicted_error = np.asarray(estimator.predict(x_test), dtype=float)
        predicted_error = np.where(np.isfinite(predicted_error), predicted_error, np.inf)
        predicted_error = np.maximum(predicted_error, 0.0)
        missing_prediction = test[column].isna().to_numpy()
        predicted_error[missing_prediction] = np.inf
        candidate_errors[column] = predicted_error
    if not candidate_errors:
        fallback = test[fallback_column].astype(float).to_numpy()
        return fallback, pd.Series([fallback_column] * len(test), index=test.index), {"candidate_count": 0}

    columns = list(candidate_errors)
    matrix = np.column_stack([candidate_errors[column] for column in columns])
    all_missing = ~np.isfinite(matrix).any(axis=1)
    best_indices = np.argmin(matrix, axis=1)
    selected_columns = [columns[int(index)] for index in best_indices]
    if all_missing.any():
        selected_columns = [
            fallback_column if bool(missing) else column
            for column, missing in zip(selected_columns, all_missing, strict=True)
        ]
    selected = pd.Series(selected_columns, index=test.index)
    prediction = []
    for idx, column in selected.items():
        prediction.append(test.at[idx, column] if column in test.columns else test.at[idx, fallback_column])
    prediction_series = pd.Series(prediction, index=test.index, dtype="float64")
    diagnostics = {
        "candidate_count": len(columns),
        "candidates": columns,
        "selected_counts": selected.value_counts(dropna=False).sort_index().to_dict(),
        "fallback_column": fallback_column,
    }
    return prediction_series.to_numpy(dtype=float), selected, diagnostics


def fit_predict_error_selector(args: argparse.Namespace, deps: dict[str, Any], train: Any, test: Any, numeric: list[str], categorical: list[str]) -> tuple[Any, Any, Any, dict[str, Any]]:
    pd = deps["pd"]
    candidates = selector_candidates(train, test)
    if args.residual_baseline in train.columns and args.residual_baseline in test.columns and args.residual_baseline not in candidates:
        candidates.insert(0, args.residual_baseline)
    if not candidates:
        raise SystemExit("error_selector_extra_trees has no usable candidate prediction columns.")
    fallback_column = args.residual_baseline if args.residual_baseline in candidates else candidates[0]

    global_models = fit_error_selector_models(args, deps, train, numeric, categorical, candidates)
    prediction, selected, global_diag = predict_error_selector(
        deps=deps,
        models=global_models,
        test=test,
        numeric=numeric,
        categorical=categorical,
        fallback_column=fallback_column,
    )
    group_summary: dict[str, Any] = {"enabled": False, "selector": global_diag}
    artifact: Any = {"global_models": global_models, "candidates": candidates, "global_diagnostics": global_diag}

    if not args.fit_group:
        return prediction, selected, artifact, group_summary

    missing_group_columns = [column for column in args.fit_group if column not in train.columns or column not in test.columns]
    if missing_group_columns:
        raise SystemExit(f"--fit-group columns are missing from train/test: {missing_group_columns}")

    group_models: dict[tuple[Any, ...], dict[str, Any]] = {}
    group_rows = {}
    trained_group_count = 0
    skipped_group_count = 0
    for raw_keys, group in train.groupby(args.fit_group, dropna=False, sort=False):
        keys = group_key(raw_keys)
        if len(group) < args.min_group_train_rows:
            skipped_group_count += 1
            group_rows[group_label(keys)] = {"train_rows": int(len(group)), "status": "fallback_too_sparse"}
            continue
        models = fit_error_selector_models(args, deps, group, numeric, categorical, candidates)
        if not models:
            skipped_group_count += 1
            group_rows[group_label(keys)] = {"train_rows": int(len(group)), "status": "fallback_no_candidate_model"}
            continue
        group_models[keys] = models
        trained_group_count += 1
        group_rows[group_label(keys)] = {"train_rows": int(len(group)), "status": "trained", "candidate_model_count": len(models)}

    if group_models:
        prediction_series = pd.Series(prediction, index=test.index)
        selected_series = pd.Series(selected, index=test.index)
        for raw_keys, group in test.groupby(args.fit_group, dropna=False, sort=False):
            keys = group_key(raw_keys)
            models = group_models.get(keys)
            if models is None:
                continue
            group_prediction, group_selected, _ = predict_error_selector(
                deps=deps,
                models=models,
                test=group,
                numeric=numeric,
                categorical=categorical,
                fallback_column=fallback_column,
            )
            prediction_series.loc[group.index] = group_prediction
            selected_series.loc[group.index] = group_selected
        prediction = prediction_series.to_numpy(dtype=float)
        selected = selected_series

    group_summary = {
        "enabled": True,
        "fit_group": args.fit_group,
        "min_group_train_rows": args.min_group_train_rows,
        "trained_group_count": trained_group_count,
        "skipped_group_count": skipped_group_count,
        "groups": group_rows,
        "fallback": "global_error_selector",
        "selector": global_diag,
    }
    artifact = {
        "global_models": global_models,
        "group_models": group_models,
        "candidates": candidates,
        "group_summary": group_summary,
    }
    return prediction, selected, artifact, group_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, action="append", required=True)
    parser.add_argument("--predictions-file", default="predictions_final_all_models.parquet")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-end", default="2026-01-01T00:00:00Z")
    parser.add_argument("--eval-start", default="2026-01-01T00:00:00Z")
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--training-table-root", type=Path)
    parser.add_argument("--training-run-id-prefix", default="residual_windsup_sst_prev")
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument("--end-month", default="2026-06")
    parser.add_argument("--training-feature-prefix", action="append", default=[])
    parser.add_argument(
        "--require-selected-training-feature",
        action="append",
        default=[],
        help="Substring that must match at least one selected training-table feature.",
    )
    parser.add_argument("--max-training-features", type=int, default=1400)
    parser.add_argument("--training-batch-size", type=int, default=50000)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--target-mode", choices=("direct", "residual"), default="residual")
    parser.add_argument("--residual-baseline", default=BASELINE)
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float)
    parser.add_argument("--model-family", choices=("ridge", "hist_gradient_boosting", "random_forest", "extra_trees", "lightgbm", "error_selector_extra_trees"), default="ridge")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--l2-regularization", type=float, default=0.1)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("CORSEWIND_SKLEARN_N_JOBS", "2")))
    parser.add_argument("--fit-group", action="append", choices=("lead_time_minutes", "spot_id"), default=[])
    parser.add_argument("--min-group-train-rows", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = import_dependencies()
    pd = deps["pd"]
    np = deps["np"]
    joblib = deps["joblib"]
    args.output_root.mkdir(parents=True, exist_ok=True)

    frame = load_frames(args, pd, np)
    frame, training_feature_columns, training_merge_summary = merge_training_table_features(args, frame, pd)
    frame = add_engineered_features(frame, np)
    engineered_defaults = [
        "issue_hour_sin",
        "issue_hour_cos",
        "issue_month_sin",
        "issue_month_cos",
        "lead_steps_15m",
        "chronos2_univar_wind_mean_ms_spread_p90_p10",
        "timesfm_wind_mean_ms_spread_p90_p10",
        "moirai_wind_mean_ms_spread_p90_p10",
        "chronos2_univar_minus_raw_wind_mean_ms",
        "timesfm_minus_raw_wind_mean_ms",
        "moirai_minus_raw_wind_mean_ms",
        "chronos2_univar_minus_persistence_wind_mean_ms",
        "timesfm_minus_persistence_wind_mean_ms",
        "moirai_minus_persistence_wind_mean_ms",
        "raw_minus_persistence_wind_mean_ms",
        "raw_minus_past_wind_mean_4",
        "raw_wind_mean_ms_x_lead_steps_15m",
        "persist_wind_mean_ms_x_lead_steps_15m",
        "past_wind_mean_4_x_lead_steps_15m",
        "past_wind_trend_4_x_lead_steps_15m",
        "model_p50_mean_wind_mean_ms",
        "model_p50_std_wind_mean_ms",
        "model_p50_min_wind_mean_ms",
        "model_p50_max_wind_mean_ms",
    ]
    requested = args.feature or [*DEFAULT_FEATURES, *engineered_defaults, *training_feature_columns]
    forbidden = forbidden_feature_matches(requested)
    if forbidden:
        raise SystemExit(f"Forbidden leakage-prone feature columns requested: {forbidden}")
    numeric = [column for column in requested if column in frame.columns and column not in CATEGORICAL]
    categorical = [column for column in CATEGORICAL if column in frame.columns]
    for extra in ("issue_hour_utc", "issue_month"):
        if extra in frame.columns and extra not in numeric:
            numeric.append(extra)

    if args.target_mode == "residual" and args.residual_baseline not in frame.columns:
        raise SystemExit(f"--target-mode residual requires missing baseline column: {args.residual_baseline}")

    needed = ordered_existing([
        TARGET,
        args.residual_baseline,
        "issue_time",
        "issue_time_utc",
        "target_time",
        "timestamp",
        "lead_time_minutes",
        "benchmark_root",
        *numeric,
        *categorical,
    ], set(frame.columns))
    data = frame[needed].dropna(subset=[TARGET, "issue_time"]).copy()
    train_end = pd.Timestamp(args.train_end, tz="UTC")
    eval_start = pd.Timestamp(args.eval_start, tz="UTC")
    train = data[data["issue_time"] < train_end].copy()
    test = data[data["issue_time"] >= eval_start].copy()
    if args.target_mode == "residual":
        train = train.dropna(subset=[args.residual_baseline])
        test = test.dropna(subset=[args.residual_baseline])
    if train.empty or test.empty:
        raise SystemExit(f"Empty train/test split: train={len(train)} test={len(test)}")
    train, train_sampling_summary = limit_training_rows(train, args, np)

    usable_numeric = []
    for column in numeric:
        if column not in train.columns:
            continue
        train[column] = pd.to_numeric(train[column], errors="coerce")
        test[column] = pd.to_numeric(test[column], errors="coerce")
        if train[column].notna().any():
            usable_numeric.append(column)
    numeric = usable_numeric
    categorical = [column for column in categorical if column in train.columns and train[column].notna().any()]
    if not numeric and not categorical:
        raise SystemExit("No usable feature column after pruning empty train features.")

    if args.target_mode == "residual":
        y_train = train[TARGET].astype(float) - train[args.residual_baseline].astype(float)
    else:
        y_train = train[TARGET]
    predictions = test.copy()
    if args.model_family == "error_selector_extra_trees":
        raw_prediction, selected_source, model, group_fit_summary = fit_predict_error_selector(
            args,
            deps,
            train,
            test,
            numeric,
            categorical,
        )
        predictions["calibrated_wind_mean_ms"] = raw_prediction
        predictions["calibrated_source_prediction"] = selected_source
    else:
        raw_prediction, model, group_fit_summary = fit_predict_calibrator(args, deps, train, test, numeric, categorical, y_train)
    if args.model_family == "error_selector_extra_trees":
        pass
    elif args.target_mode == "residual":
        predictions["calibrated_residual_wind_mean_ms"] = raw_prediction
        predictions["calibrated_wind_mean_ms"] = predictions[args.residual_baseline].astype(float) + raw_prediction
    else:
        predictions["calibrated_wind_mean_ms"] = raw_prediction
    if args.clip_min is not None or args.clip_max is not None:
        predictions["calibrated_wind_mean_ms"] = predictions["calibrated_wind_mean_ms"].clip(
            lower=args.clip_min,
            upper=args.clip_max,
        )
    predictions.to_parquet(args.output_root / "calibrator_predictions.parquet", index=False)
    joblib.dump(model, args.output_root / "calibrator.joblib")

    metrics = {
        "calibrator": metric(np, predictions, "calibrated_wind_mean_ms"),
    }
    for column in requested:
        if column in predictions.columns:
            metrics[column] = metric(np, predictions, column)

    result = {
        "generated_at_utc": utc_now(),
        "model_family": args.model_family,
        "target_mode": args.target_mode,
        "residual_baseline": args.residual_baseline if args.target_mode == "residual" else None,
        "fit_group": args.fit_group,
        "group_fit_summary": group_fit_summary,
        "clip_min": args.clip_min,
        "clip_max": args.clip_max,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_sampling": train_sampling_summary,
        "split_coverage": {
            "train": coverage_summary(train, pd),
            "test": coverage_summary(test, pd),
        },
        "train_end": args.train_end,
        "eval_start": args.eval_start,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "training_table_feature_merge": training_merge_summary,
        "metrics": metrics,
        "calibrator_by_spot": metric_by_group(np, predictions, "calibrated_wind_mean_ms", ["spot_id"]),
        "calibrator_by_lead": metric_by_group(np, predictions, "calibrated_wind_mean_ms", ["lead_time_minutes"]),
        "calibrator_by_spot_lead": metric_by_group(np, predictions, "calibrated_wind_mean_ms", ["spot_id", "lead_time_minutes"]),
    }
    (args.output_root / "calibrator_results.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
