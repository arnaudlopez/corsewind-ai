#!/usr/bin/env python3
"""Benchmark a CorseWind SAPHIR-style sequence dataset with leakage-safe baselines."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGETS = {
    "wind_mean_ms": {
        "actual": "target_wind_mean_ms",
        "baseline": "baseline_wind_mean_ms",
        "residual": "residual_wind_mean_ms",
        "persistence": "history_target_wind_mean_ms_last_4",
    },
    "gust_ms": {
        "actual": "target_gust_ms",
        "baseline": "baseline_gust_ms",
        "residual": "residual_gust_ms",
        "persistence": "history_target_gust_ms_last_4",
    },
}

KEY_COLUMNS = {
    "sample_id",
    "spot_id",
    "issue_time_utc",
    "target_time_utc",
    "split",
    "benchmark_split",
}
FORBIDDEN_PREFIXES = (
    "target_",
    "residual_",
    "labels__target_",
)
FORBIDDEN_SUBSTRINGS = (
    "target_observation",
    "source_dataset",
    "source_project",
    "source_type",
)
DEFAULT_CATEGORICAL = [
    "spot_id",
    "spot_kind",
    "spot_source_type",
    "station_id",
    "lead_time_minutes",
]
HISTORY_BASE_COLUMNS = [
    "wind_mean_ms",
    "gust_ms",
    "wind_direction_deg",
    "nwp_wind_mean_ms",
    "nwp_gust_ms",
    "nwp_temperature_2m_c",
    "nwp_pressure_msl_hpa",
    "nwp_cloud_cover_pct",
    "nwp_shortwave_radiation",
    "wind_mean_error_ms",
    "gust_error_ms",
]
CONTEXT_COLUMNS = [
    "age_minutes",
    "altitude_delta_m",
    "altitude_m",
    "available",
    "bearing_from_spot_deg",
    "bearing_to_spot_deg",
    "delta_vs_target_gust_ms",
    "delta_vs_target_pressure_hpa",
    "delta_vs_target_temperature_c",
    "delta_vs_target_wind_mean_ms",
    "delta_vs_target_wind_u_ms",
    "delta_vs_target_wind_v_ms",
    "dewpoint_c",
    "distance_km",
    "gust_ms",
    "humidity_pct",
    "pressure_hpa",
    "sea_level_pressure_hpa",
    "temperature_c",
    "upwind_score_from_target_wind",
    "wind_direction_deg",
    "wind_mean_ms",
    "wind_u_ms",
    "wind_v_ms",
]
OFFSET_COLUMNS = [
    "available",
    "boundary_layer_height",
    "cape",
    "cloud_cover",
    "cloud_cover_low",
    "delta_vs_center_boundary_layer_height",
    "delta_vs_center_cape",
    "delta_vs_center_cloud_cover",
    "delta_vs_center_cloud_cover_low",
    "delta_vs_center_pressure_msl",
    "delta_vs_center_shortwave_radiation",
    "delta_vs_center_surface_pressure",
    "delta_vs_center_temperature_2m",
    "delta_vs_center_wind_direction_10m",
    "delta_vs_center_wind_gusts_10m",
    "delta_vs_center_wind_speed_10m",
    "dew_point_2m",
    "distance_km",
    "pressure_msl",
    "relative_humidity_2m",
    "shortwave_radiation",
    "surface_pressure",
    "temperature_2m",
    "valid_offset_minutes",
    "wind_direction_10m",
    "wind_gusts_10m",
    "wind_speed_10m",
]
VERTICAL_VALUE_COLUMNS = [
    "geopotential_height",
    "relative_humidity",
    "temperature",
    "wind_direction",
    "wind_speed",
    "wind_u_ms",
    "wind_v_ms",
]


def import_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:
        raise SystemExit("Missing pandas/numpy/sklearn/pyarrow dependencies.") from exc
    return locals()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_windows(value: str) -> list[int]:
    windows = [int(item.strip()) for item in value.split(",") if item.strip()]
    return sorted({window for window in windows if window > 0})


def to_numeric(series: Any, pd: Any) -> Any:
    return pd.to_numeric(series, errors="coerce")


def metric(np: Any, frame: Any, prediction_column: str, actual_column: str) -> dict[str, Any]:
    if prediction_column not in frame.columns or actual_column not in frame.columns:
        return {"count": 0}
    valid = frame[[prediction_column, actual_column]].dropna()
    if valid.empty:
        return {"count": 0}
    errors = valid[prediction_column].to_numpy(dtype=float) - valid[actual_column].to_numpy(dtype=float)
    return {
        "count": int(len(errors)),
        "mae": round(float(np.mean(np.abs(errors))), 6),
        "rmse": round(float(np.sqrt(np.mean(errors * errors))), 6),
        "bias": round(float(np.mean(errors)), 6),
    }


def metrics_by_group(np: Any, frame: Any, prediction_column: str, actual_column: str, group_column: str) -> dict[str, Any]:
    if group_column not in frame.columns:
        return {}
    out = {}
    for key, group in frame.groupby(group_column, dropna=False):
        out[str(key)] = metric(np, group, prediction_column, actual_column)
    return out


def read_table(root: Path, name: str, pd: Any) -> Any:
    path = root / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def drop_merge_metadata(frame: Any, keep_sample_id: bool = True) -> Any:
    drop = [column for column in ("spot_id", "issue_time_utc", "split") if column in frame.columns]
    if not keep_sample_id and "sample_id" in frame.columns:
        drop.append("sample_id")
    return frame.drop(columns=drop, errors="ignore")


def merge_sample_tables(base: Any, right: Any) -> Any:
    if right.empty or "sample_id" not in right.columns:
        return base
    right = drop_merge_metadata(right)
    return base.merge(right, on="sample_id", how="left")


def is_missing(value: Any, pd: Any) -> bool:
    return value is None or bool(pd.isna(value))


def history_features(history: Any, windows: list[int], deps: dict[str, Any]) -> Any:
    pd = deps["pd"]
    np = deps["np"]
    if history.empty:
        return pd.DataFrame()
    frame = history.copy()
    if {"wind_mean_ms", "nwp_wind_mean_ms"}.issubset(frame.columns):
        frame["wind_mean_error_ms"] = to_numeric(frame["wind_mean_ms"], pd) - to_numeric(frame["nwp_wind_mean_ms"], pd)
    if {"gust_ms", "nwp_gust_ms"}.issubset(frame.columns):
        frame["gust_error_ms"] = to_numeric(frame["gust_ms"], pd) - to_numeric(frame["nwp_gust_ms"], pd)
    value_columns = [column for column in HISTORY_BASE_COLUMNS if column in frame.columns]
    rows = []
    for sample_id, group in frame.sort_values(["sample_id", "time_index"]).groupby("sample_id", sort=False):
        item: dict[str, Any] = {"sample_id": sample_id}
        if "minutes_before_issue" in group.columns:
            item["history_target_minutes_before_issue_min"] = float(to_numeric(group["minutes_before_issue"], pd).min())
            item["history_target_minutes_before_issue_max"] = float(to_numeric(group["minutes_before_issue"], pd).max())
        for column in value_columns:
            values_all = to_numeric(group[column], pd)
            observed_column = f"{column}_observed"
            for window in windows:
                values = values_all.tail(window).dropna()
                prefix = f"history_target_{column}"
                suffix = str(window)
                item[f"{prefix}_count_{suffix}"] = int(values.shape[0])
                if values.empty:
                    item[f"{prefix}_last_{suffix}"] = np.nan
                    item[f"{prefix}_mean_{suffix}"] = np.nan
                    item[f"{prefix}_min_{suffix}"] = np.nan
                    item[f"{prefix}_max_{suffix}"] = np.nan
                    item[f"{prefix}_std_{suffix}"] = np.nan
                    item[f"{prefix}_trend_{suffix}"] = np.nan
                    continue
                item[f"{prefix}_last_{suffix}"] = float(values.iloc[-1])
                item[f"{prefix}_mean_{suffix}"] = float(values.mean())
                item[f"{prefix}_min_{suffix}"] = float(values.min())
                item[f"{prefix}_max_{suffix}"] = float(values.max())
                item[f"{prefix}_std_{suffix}"] = float(values.std(ddof=0)) if values.shape[0] > 1 else 0.0
                item[f"{prefix}_trend_{suffix}"] = float(values.iloc[-1] - values.iloc[0]) if values.shape[0] > 1 else 0.0
                if observed_column in group.columns:
                    observed = group[observed_column].tail(window).fillna(False).astype(bool)
                    item[f"{prefix}_observed_ratio_{suffix}"] = float(observed.mean()) if len(observed) else 0.0
        rows.append(item)
    return pd.DataFrame(rows)


def aggregate_long_table(
    frame: Any,
    *,
    value_columns: list[str],
    prefix: str,
    role_column: str | None,
    deps: dict[str, Any],
) -> Any:
    pd = deps["pd"]
    np = deps["np"]
    if frame.empty or "sample_id" not in frame.columns:
        return pd.DataFrame()
    columns = [column for column in value_columns if column in frame.columns]
    if not columns:
        return frame[["sample_id"]].drop_duplicates().copy()
    working = frame.copy()
    for column in columns:
        working[column] = to_numeric(working[column], pd)
    roles: list[tuple[str, Any]] = [("all", working)]
    if role_column and role_column in working.columns:
        for role, group in working.groupby(role_column, dropna=True):
            role_name = str(role).strip().lower().replace(" ", "_")
            if role_name:
                roles.append((role_name, group))
    merged = working[["sample_id"]].drop_duplicates().copy()
    for role_name, group in roles:
        agg = group.groupby("sample_id")[columns].agg(["count", "mean", "min", "max"])
        agg.columns = [f"{prefix}_{role_name}_{column}_{stat}" for column, stat in agg.columns]
        agg = agg.reset_index()
        merged = merged.merge(agg, on="sample_id", how="left")
    return merged.replace({np.inf: np.nan, -np.inf: np.nan})


def vertical_features(vertical: Any, deps: dict[str, Any]) -> Any:
    pd = deps["pd"]
    np = deps["np"]
    if vertical.empty or "pressure_hpa" not in vertical.columns:
        return pd.DataFrame()
    frame = vertical.copy()
    frame["pressure_hpa"] = to_numeric(frame["pressure_hpa"], pd).astype("Int64")
    value_columns = [column for column in VERTICAL_VALUE_COLUMNS if column in frame.columns]
    rows = []
    for sample_id, group in frame.groupby("sample_id", sort=False):
        item: dict[str, Any] = {"sample_id": sample_id}
        by_level: dict[int, dict[str, float]] = {}
        for row in group.itertuples(index=False):
            level = getattr(row, "pressure_hpa")
            if pd.isna(level):
                continue
            level_int = int(level)
            by_level[level_int] = {}
            for column in value_columns:
                value = getattr(row, column)
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    numeric = np.nan
                item[f"vertical_{level_int}hpa_{column}"] = numeric
                by_level[level_int][column] = numeric
        if 1000 in by_level and 850 in by_level:
            u0 = by_level[1000].get("wind_u_ms", np.nan)
            v0 = by_level[1000].get("wind_v_ms", np.nan)
            u1 = by_level[850].get("wind_u_ms", np.nan)
            v1 = by_level[850].get("wind_v_ms", np.nan)
            item["vertical_1000_850_wind_shear_ms"] = float(math.sqrt((u1 - u0) ** 2 + (v1 - v0) ** 2))
            item["vertical_1000_850_temperature_delta_c"] = by_level[1000].get("temperature", np.nan) - by_level[850].get("temperature", np.nan)
            item["vertical_1000_850_rh_delta_pct"] = by_level[1000].get("relative_humidity", np.nan) - by_level[850].get("relative_humidity", np.nan)
        rows.append(item)
    return pd.DataFrame(rows)


def safe_sample_features(samples: Any) -> Any:
    if samples.empty:
        return samples
    keep = []
    for column in samples.columns:
        if column in {"sample_id", "spot_name", "spot_kind", "spot_source_type", "station_id", "latitude", "longitude", "available_leads"}:
            keep.append(column)
            continue
        if column.startswith("baseline_"):
            keep.append(column)
    return samples[keep].copy()


def assign_benchmark_split(frame: Any, args: argparse.Namespace, deps: dict[str, Any]) -> Any:
    pd = deps["pd"]
    out = frame.copy()
    out["issue_time"] = pd.to_datetime(out["issue_time_utc"], utc=True, errors="coerce")
    split = out.get("split")
    if split is not None and (split.astype(str) != "train").any():
        out["benchmark_split"] = split.astype(str).where(split.astype(str) == "train", "eval")
        return out
    issue_times = out[["sample_id", "issue_time"]].drop_duplicates().dropna().sort_values("issue_time")
    if issue_times.empty:
        out["benchmark_split"] = "train"
        return out
    unique_samples = issue_times["sample_id"].drop_duplicates().tolist()
    eval_count = max(args.min_eval_samples, int(round(len(unique_samples) * args.fallback_eval_fraction)))
    eval_count = min(max(eval_count, 1), max(len(unique_samples) - 1, 1))
    eval_samples = set(unique_samples[-eval_count:])
    out["benchmark_split"] = out["sample_id"].map(lambda value: "eval" if value in eval_samples else "train")
    return out


def build_flat_frame(root: Path, args: argparse.Namespace, deps: dict[str, Any]) -> Any:
    pd = deps["pd"]
    future = read_table(root, "future_targets", pd)
    if future.empty:
        raise SystemExit(f"Missing or empty future_targets.parquet in {root}")
    future = future.copy()
    future["lead_time_minutes"] = to_numeric(future["lead_time_minutes"], pd)
    samples = safe_sample_features(read_table(root, "samples", pd))
    static = read_table(root, "static_context", pd)
    history = history_features(read_table(root, "station_history", pd), parse_windows(args.history_windows), deps)
    context = aggregate_long_table(
        read_table(root, "context_station_snapshot", pd),
        value_columns=CONTEXT_COLUMNS,
        prefix="context_snapshot",
        role_column="role",
        deps=deps,
    )
    offsets = aggregate_long_table(
        read_table(root, "nwp_surface_offsets", pd),
        value_columns=OFFSET_COLUMNS,
        prefix="nwp_offset",
        role_column=None,
        deps=deps,
    )
    vertical = vertical_features(read_table(root, "nwp_vertical_profile", pd), deps)

    frame = future
    for table in (samples, static, history, context, offsets, vertical):
        frame = merge_sample_tables(frame, table)
    frame = assign_benchmark_split(frame, args, deps)
    frame["issue_hour_utc"] = frame["issue_time"].dt.hour.astype("float64")
    frame["issue_month"] = frame["issue_time"].dt.month.astype("float64")
    dayofyear = frame["issue_time"].dt.dayofyear.fillna(1).astype(float)
    angle = 2.0 * math.pi * dayofyear / 366.0
    frame["issue_dayofyear_sin"] = deps["np"].sin(angle)
    frame["issue_dayofyear_cos"] = deps["np"].cos(angle)
    return frame


def forbidden_feature(column: str) -> bool:
    if column in KEY_COLUMNS or column == "issue_time":
        return True
    if any(column.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        return True
    if any(part in column for part in FORBIDDEN_SUBSTRINGS):
        return True
    return False


def candidate_columns(frame: Any, target_config: dict[str, str], args: argparse.Namespace, deps: dict[str, Any]) -> tuple[list[str], list[str]]:
    pd = deps["pd"]
    np = deps["np"]
    train = frame[frame["benchmark_split"] == "train"].copy()
    y = to_numeric(train[target_config["residual"]], pd)
    score_frame = train
    if args.feature_score_sample_size and len(score_frame) > args.feature_score_sample_size:
        score_frame = score_frame.sample(args.feature_score_sample_size, random_state=args.random_state)
        y = to_numeric(score_frame[target_config["residual"]], pd)

    categorical = [column for column in DEFAULT_CATEGORICAL if column in frame.columns and not forbidden_feature(column)]
    numeric_scores: list[tuple[float, float, str]] = []
    force_prefixes = (
        "baseline_",
        "history_target_",
        "vertical_",
        "thermal_",
        "nwp_offset_",
        "context_snapshot_",
        "context_agg_",
        "dem_",
        "fetch_",
        "landsea_",
    )
    for column in frame.columns:
        if forbidden_feature(column) or column in categorical:
            continue
        values = to_numeric(score_frame[column], pd)
        valid = values.notna() & y.notna()
        valid_count = int(valid.sum())
        if valid_count < args.min_feature_non_null:
            continue
        unique_count = int(values[valid].nunique(dropna=True))
        if unique_count <= 1:
            continue
        non_null_ratio = valid_count / max(1, len(score_frame))
        corr = 0.0
        if valid_count >= 3:
            corr_value = np.corrcoef(values[valid].to_numpy(dtype=float), y[valid].to_numpy(dtype=float))[0, 1]
            if not np.isnan(corr_value):
                corr = abs(float(corr_value))
        if column.startswith(force_prefixes):
            corr += 0.05
        numeric_scores.append((corr, non_null_ratio, column))
    numeric_scores.sort(reverse=True)
    numeric = [column for _corr, _ratio, column in numeric_scores[: args.max_numeric_features]]
    return numeric, categorical


def make_encoder(deps: dict[str, Any]) -> Any:
    OneHotEncoder = deps["OneHotEncoder"]
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_model_pipeline(model_family: str, numeric: list[str], categorical: list[str], args: argparse.Namespace, deps: dict[str, Any]) -> Any:
    Pipeline = deps["Pipeline"]
    ColumnTransformer = deps["ColumnTransformer"]
    SimpleImputer = deps["SimpleImputer"]
    StandardScaler = deps["StandardScaler"]
    HistGradientBoostingRegressor = deps["HistGradientBoostingRegressor"]
    ExtraTreesRegressor = deps["ExtraTreesRegressor"]
    Ridge = deps["Ridge"]
    transformers = []
    if numeric:
        if model_family == "ridge":
            numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        else:
            numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        transformers.append(("num", numeric_pipe, numeric))
    if categorical:
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_encoder(deps))])
        transformers.append(("cat", cat_pipe, categorical))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    if model_family == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=args.hgb_max_iter,
            learning_rate=args.hgb_learning_rate,
            max_leaf_nodes=args.hgb_max_leaf_nodes,
            l2_regularization=args.hgb_l2,
            random_state=args.random_state,
        )
    elif model_family == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=args.extra_trees_estimators,
            min_samples_leaf=args.extra_trees_min_samples_leaf,
            max_features=args.extra_trees_max_features,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
        )
    elif model_family == "ridge":
        model = Ridge(alpha=args.ridge_alpha)
    else:
        raise SystemExit(f"Unsupported model family: {model_family}")
    return Pipeline([("prep", preprocessor), ("model", model)])


def train_eval_target(frame: Any, target_name: str, args: argparse.Namespace, deps: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    pd = deps["pd"]
    np = deps["np"]
    config = TARGETS[target_name]
    needed = [config["actual"], config["baseline"], config["residual"]]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise SystemExit(f"Missing columns for {target_name}: {missing}")
    working = frame.dropna(subset=[config["actual"], config["baseline"], config["residual"]]).copy()
    train = working[working["benchmark_split"] == "train"].copy()
    eval_frame = working[working["benchmark_split"] != "train"].copy()
    if train.empty or eval_frame.empty:
        raise SystemExit(f"Not enough train/eval rows for {target_name}: train={len(train)} eval={len(eval_frame)}")

    predictions = working[[
        "sample_id",
        "spot_id",
        "issue_time_utc",
        "target_time_utc",
        "lead_time_minutes",
        "benchmark_split",
        config["actual"],
        config["baseline"],
    ]].copy()
    predictions = predictions.rename(columns={config["actual"]: f"actual_{target_name}", config["baseline"]: f"raw_{target_name}"})

    if config["persistence"] in working.columns:
        predictions[f"persist_{target_name}"] = to_numeric(working[config["persistence"]], pd)

    global_bias = float(to_numeric(train[config["residual"]], pd).mean())
    by_spot_lead = train.groupby(["spot_id", "lead_time_minutes"])[config["residual"]].mean().to_dict()
    by_lead = train.groupby("lead_time_minutes")[config["residual"]].mean().to_dict()

    def bias_prediction(row: Any) -> float:
        key = (row["spot_id"], row["lead_time_minutes"])
        bias = by_spot_lead.get(key, by_lead.get(row["lead_time_minutes"], global_bias))
        return float(row[config["baseline"]]) + float(bias)

    predictions[f"bias_spot_lead_{target_name}"] = working.apply(bias_prediction, axis=1)

    numeric, categorical = candidate_columns(working, config, args, deps)
    feature_columns = numeric + categorical
    if args.max_train_rows and len(train) > args.max_train_rows:
        train_fit = train.sample(args.max_train_rows, random_state=args.random_state)
    else:
        train_fit = train

    model_summaries: dict[str, Any] = {}
    trained_models = [item.strip() for item in args.model_family if item.strip()]
    for model_family in trained_models:
        pipeline = make_model_pipeline(model_family, numeric, categorical, args, deps)
        pipeline.fit(train_fit[feature_columns], to_numeric(train_fit[config["residual"]], pd))
        residual_pred = pipeline.predict(working[feature_columns])
        predictions[f"{model_family}_{target_name}"] = to_numeric(working[config["baseline"]], pd).to_numpy(dtype=float) + residual_pred
        model_summaries[model_family] = {
            "feature_count": len(feature_columns),
            "numeric_feature_count": len(numeric),
            "categorical_feature_count": len(categorical),
            "train_rows": int(len(train_fit)),
        }
        model = pipeline.named_steps.get("model")
        if hasattr(model, "feature_importances_"):
            model_summaries[model_family]["has_feature_importance"] = True

    eval_predictions = predictions[predictions["benchmark_split"] != "train"].copy()
    train_predictions = predictions[predictions["benchmark_split"] == "train"].copy()
    actual = f"actual_{target_name}"
    prediction_columns = [f"raw_{target_name}"]
    if f"persist_{target_name}" in predictions.columns:
        prediction_columns.append(f"persist_{target_name}")
    prediction_columns.append(f"bias_spot_lead_{target_name}")
    prediction_columns.extend([f"{model}_{target_name}" for model in trained_models])

    metrics = {
        "train": {column: metric(np, train_predictions, column, actual) for column in prediction_columns},
        "eval": {column: metric(np, eval_predictions, column, actual) for column in prediction_columns},
        "eval_by_lead": {
            column: metrics_by_group(np, eval_predictions, column, actual, "lead_time_minutes") for column in prediction_columns
        },
        "eval_by_spot": {
            column: metrics_by_group(np, eval_predictions, column, actual, "spot_id") for column in prediction_columns
        },
    }
    summary = {
        "target": target_name,
        "rows": int(len(working)),
        "train_rows": int(len(train)),
        "eval_rows": int(len(eval_frame)),
        "selected_numeric_features": numeric,
        "selected_categorical_features": categorical,
        "models": model_summaries,
        "metrics": metrics,
    }
    return predictions, summary


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# CorseWind SAPHIR-Style Sequence Benchmark",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Dataset root: `{result['dataset_root']}`",
        f"Output root: `{result['output_root']}`",
        "",
        "## Coverage",
        "",
        f"- Rows: `{result['row_count']}`",
        f"- Train rows: `{result['split_counts'].get('train', 0)}`",
        f"- Eval rows: `{sum(count for split, count in result['split_counts'].items() if split != 'train')}`",
        f"- Spots: `{result['spot_count']}`",
        "",
        "## Eval Metrics",
        "",
        "| Target | Model | RMSE | MAE | Bias | Count |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for target_name, target_result in result["targets"].items():
        for model_name, item in target_result["metrics"]["eval"].items():
            lines.append(
                f"| `{target_name}` | `{model_name}` | {item.get('rmse')} | {item.get('mae')} | {item.get('bias')} | {item.get('count')} |"
            )
    lines.extend(["", "## Best Eval Models", ""])
    for target_name, item in result["best_eval_models"].items():
        lines.append(
            f"- `{target_name}`: `{item.get('model')}` RMSE `{item.get('rmse')}`, MAE `{item.get('mae')}`, count `{item.get('count')}`"
        )
    lines.extend(["", "## Feature Counts", ""])
    for target_name, target_result in result["targets"].items():
        lines.append(
            f"- `{target_name}`: `{len(target_result['selected_numeric_features'])}` numeric, "
            f"`{len(target_result['selected_categorical_features'])}` categorical"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id", default="corsewind_saphir_sequence_benchmark_v1")
    parser.add_argument("--target", action="append", choices=sorted(TARGETS), default=[])
    parser.add_argument("--model-family", action="append", choices=("hgb", "extra_trees", "ridge"), default=[])
    parser.add_argument("--history-windows", default="4,8,16")
    parser.add_argument("--fallback-eval-fraction", type=float, default=0.25)
    parser.add_argument("--min-eval-samples", type=int, default=2)
    parser.add_argument("--min-feature-non-null", type=int, default=20)
    parser.add_argument("--max-numeric-features", type=int, default=800)
    parser.add_argument("--feature-score-sample-size", type=int, default=50_000)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--hgb-max-iter", type=int, default=220)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.045)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--hgb-l2", type=float, default=0.05)
    parser.add_argument("--extra-trees-estimators", type=int, default=240)
    parser.add_argument("--extra-trees-min-samples-leaf", type=int, default=4)
    parser.add_argument("--extra-trees-max-features", default="sqrt")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--write-flat-table", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.target:
        args.target = ["wind_mean_ms", "gust_ms"]
    if not args.model_family:
        args.model_family = ["hgb", "extra_trees", "ridge"]
    deps = import_dependencies()
    pd = deps["pd"]
    np = deps["np"]

    dataset_root = args.dataset_root.resolve()
    output_root = (args.output_root or dataset_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    frame = build_flat_frame(dataset_root, args, deps)
    split_counts = {str(key): int(value) for key, value in frame["benchmark_split"].value_counts(dropna=False).sort_index().items()}
    if args.write_flat_table:
        frame.to_parquet(output_root / "benchmark_flat_features.parquet", index=False)

    result: dict[str, Any] = {
        "format": "corsewind.saphir_sequence_benchmark.v1",
        "generated_at_utc": utc_now(),
        "run_id": args.run_id,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "row_count": int(len(frame)),
        "sample_count": int(frame["sample_id"].nunique(dropna=True)),
        "spot_count": int(frame["spot_id"].nunique(dropna=True)),
        "split_counts": split_counts,
        "targets": {},
        "best_eval_models": {},
    }

    prediction_frames = []
    for target_name in args.target:
        predictions, target_result = train_eval_target(frame, target_name, args, deps)
        predictions.to_parquet(output_root / f"predictions_{target_name}.parquet", index=False)
        (output_root / f"features_{target_name}.json").write_text(
            json.dumps(
                {
                    "numeric": target_result["selected_numeric_features"],
                    "categorical": target_result["selected_categorical_features"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        result["targets"][target_name] = target_result
        eval_metrics = target_result["metrics"]["eval"]
        candidates = [
            (name, item)
            for name, item in eval_metrics.items()
            if item.get("count", 0) and item.get("rmse") is not None
        ]
        if candidates:
            best_name, best_item = min(candidates, key=lambda pair: float(pair[1]["rmse"]))
            result["best_eval_models"][target_name] = {"model": best_name, **best_item}
        prediction_frames.append(predictions)

    if prediction_frames:
        merged = prediction_frames[0]
        key = ["sample_id", "spot_id", "issue_time_utc", "target_time_utc", "lead_time_minutes", "benchmark_split"]
        for other in prediction_frames[1:]:
            merged = merged.merge(other, on=key, how="outer")
        merged.to_parquet(output_root / "predictions_all_targets.parquet", index=False)

    (output_root / "benchmark_results.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_markdown(output_root / "benchmark_results.md", result)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "output_root": result["output_root"],
                "row_count": result["row_count"],
                "split_counts": result["split_counts"],
                "best_eval_models": result["best_eval_models"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
