#!/usr/bin/env python3
"""Train a SAPHIR-style structured neural residual benchmark on CorseWind data."""

from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGETS = {
    "wind_mean_ms": {
        "actual": "target_wind_mean_ms",
        "baseline": "baseline_wind_mean_ms",
        "residual": "residual_wind_mean_ms",
    },
    "gust_ms": {
        "actual": "target_gust_ms",
        "baseline": "baseline_gust_ms",
        "residual": "residual_gust_ms",
    },
}

HISTORY_FEATURES = [
    "wind_mean_ms",
    "gust_ms",
    "wind_direction_sin",
    "wind_direction_cos",
    "nwp_wind_mean_ms",
    "nwp_gust_ms",
    "nwp_temperature_2m_c",
    "nwp_pressure_msl_hpa",
    "nwp_cloud_cover_pct",
    "nwp_shortwave_radiation",
    "wind_mean_error_ms",
    "gust_error_ms",
    "wind_mean_ms_observed",
    "gust_ms_observed",
]

CONTEXT_NUMERIC_FEATURES = [
    "age_minutes",
    "altitude_delta_m",
    "altitude_m",
    "available",
    "bearing_from_spot_sin",
    "bearing_from_spot_cos",
    "bearing_to_spot_sin",
    "bearing_to_spot_cos",
    "delta_vs_target_gust_ms",
    "delta_vs_target_pressure_hpa",
    "delta_vs_target_temperature_c",
    "delta_vs_target_wind_mean_ms",
    "delta_vs_target_wind_u_ms",
    "delta_vs_target_wind_v_ms",
    "dewpoint_c",
    "distance_km",
    "east_offset_km",
    "gust_ms",
    "humidity_pct",
    "north_offset_km",
    "pressure_hpa",
    "sea_level_pressure_hpa",
    "temperature_c",
    "upwind_score_from_target_wind",
    "wind_direction_sin",
    "wind_direction_cos",
    "wind_mean_ms",
    "wind_u_ms",
    "wind_v_ms",
]

OFFSET_NUMERIC_FEATURES = [
    "available",
    "bearing_sin",
    "bearing_cos",
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
    "wind_direction_10m_sin",
    "wind_direction_10m_cos",
    "wind_gusts_10m",
    "wind_speed_10m",
]

VERTICAL_FEATURES = [
    "geopotential_height",
    "relative_humidity",
    "temperature",
    "wind_direction_sin",
    "wind_direction_cos",
    "wind_speed",
    "wind_u_ms",
    "wind_v_ms",
]

FUTURE_FEATURES = [
    "lead_time_minutes",
    "baseline_wind_mean_ms",
    "baseline_gust_ms",
    "baseline_wind_direction_sin",
    "baseline_wind_direction_cos",
    "baseline_temperature_2m_c",
    "baseline_pressure_msl_hpa",
    "baseline_surface_pressure_hpa",
    "baseline_shortwave_radiation",
    "baseline_cloud_cover_pct",
    "baseline_cape",
    "issue_hour_sin",
    "issue_hour_cos",
    "issue_dayofyear_sin",
    "issue_dayofyear_cos",
]

STATIC_BLOCKED_PREFIXES = (
    "target_",
    "residual_",
    "baseline_",
)
STATIC_BLOCKED_COLUMNS = {
    "sample_id",
    "spot_id",
    "issue_time_utc",
    "split",
}


def import_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit("Missing pandas/numpy/torch/pyarrow dependencies.") from exc
    return locals()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_table(root: Path, name: str, pd: Any) -> Any:
    path = root / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def numeric(series: Any, pd: Any) -> Any:
    return pd.to_numeric(series, errors="coerce")


def add_direction_sin_cos(frame: Any, column: str, pd: Any, np: Any, prefix: str | None = None) -> Any:
    if column not in frame.columns:
        return frame
    base = prefix or column.removesuffix("_deg")
    radians = np.deg2rad(numeric(frame[column], pd))
    frame[f"{base}_sin"] = np.sin(radians)
    frame[f"{base}_cos"] = np.cos(radians)
    return frame


def metric(np: Any, frame: Any, prediction_column: str, actual_column: str) -> dict[str, Any]:
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


def metrics_by_lead(np: Any, frame: Any, prediction_column: str, actual_column: str) -> dict[str, Any]:
    out = {}
    for lead, group in frame.groupby("lead_time_minutes", dropna=False):
        out[str(int(float(lead)))] = metric(np, group, prediction_column, actual_column)
    return out


def standardize_array(array: Any, train_mask: Any, np: Any, eps: float = 1e-6) -> tuple[Any, dict[str, Any]]:
    train_values = array[train_mask]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(train_values, axis=tuple(range(train_values.ndim - 1)), keepdims=True)
        std = np.nanstd(train_values, axis=tuple(range(train_values.ndim - 1)), keepdims=True)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > eps), std, 1.0)
    out = (array - mean) / std
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    return out, {
        "mean": np.squeeze(mean).astype(float).tolist(),
        "std": np.squeeze(std).astype(float).tolist(),
    }


def standardize_2d(array: Any, train_mask: Any, np: Any, eps: float = 1e-6) -> tuple[Any, dict[str, Any]]:
    train_values = array[train_mask]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(train_values, axis=0, keepdims=True)
        std = np.nanstd(train_values, axis=0, keepdims=True)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > eps), std, 1.0)
    out = (array - mean) / std
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    return out, {
        "mean": np.squeeze(mean).astype(float).tolist(),
        "std": np.squeeze(std).astype(float).tolist(),
    }


def build_sample_index(samples: Any, future: Any) -> tuple[list[str], dict[str, int]]:
    sample_ids = list(samples["sample_id"].astype(str).drop_duplicates()) if not samples.empty else []
    missing = [sample for sample in future["sample_id"].astype(str).drop_duplicates() if sample not in set(sample_ids)]
    sample_ids.extend(missing)
    return sample_ids, {sample_id: index for index, sample_id in enumerate(sample_ids)}


def build_history_tensor(history: Any, sample_to_idx: dict[str, int], context_length: int, deps: dict[str, Any]) -> tuple[Any, list[str]]:
    pd = deps["pd"]
    np = deps["np"]
    tensor = np.full((len(sample_to_idx), context_length, len(HISTORY_FEATURES)), np.nan, dtype="float32")
    if history.empty:
        return tensor, HISTORY_FEATURES
    frame = history.copy()
    if "station_slot_name" in frame.columns:
        frame = frame[frame["station_slot_name"].astype(str).eq("target")].copy()
    add_direction_sin_cos(frame, "wind_direction_deg", pd, np, "wind_direction")
    if {"wind_mean_ms", "nwp_wind_mean_ms"}.issubset(frame.columns):
        frame["wind_mean_error_ms"] = numeric(frame["wind_mean_ms"], pd) - numeric(frame["nwp_wind_mean_ms"], pd)
    if {"gust_ms", "nwp_gust_ms"}.issubset(frame.columns):
        frame["gust_error_ms"] = numeric(frame["gust_ms"], pd) - numeric(frame["nwp_gust_ms"], pd)
    for observed_column in ("wind_mean_ms_observed", "gust_ms_observed"):
        if observed_column in frame.columns:
            frame[observed_column] = frame[observed_column].fillna(False).astype(float)
    for sample_id, group in frame.sort_values(["sample_id", "time_index"]).groupby("sample_id", sort=False):
        sample_idx = sample_to_idx.get(str(sample_id))
        if sample_idx is None:
            continue
        group = group.tail(context_length)
        start = context_length - len(group)
        for feature_idx, column in enumerate(HISTORY_FEATURES):
            if column in group.columns:
                tensor[sample_idx, start:, feature_idx] = numeric(group[column], pd).to_numpy(dtype="float32")
    return tensor, HISTORY_FEATURES


def build_context_tensor(context: Any, sample_to_idx: dict[str, int], max_context_stations: int, deps: dict[str, Any]) -> tuple[Any, Any, list[str]]:
    pd = deps["pd"]
    np = deps["np"]
    if context.empty:
        return (
            np.zeros((len(sample_to_idx), max_context_stations, 1), dtype="float32"),
            np.zeros((len(sample_to_idx), max_context_stations), dtype="float32"),
            ["missing_context"],
        )
    frame = context.copy()
    add_direction_sin_cos(frame, "bearing_from_spot_deg", pd, np, "bearing_from_spot")
    add_direction_sin_cos(frame, "bearing_to_spot_deg", pd, np, "bearing_to_spot")
    add_direction_sin_cos(frame, "wind_direction_deg", pd, np, "wind_direction")
    roles = sorted(str(value) for value in frame.get("role", pd.Series(dtype=str)).dropna().unique())
    for role in roles:
        frame[f"role_{role}"] = frame["role"].astype(str).eq(role).astype(float)
    features = [column for column in CONTEXT_NUMERIC_FEATURES if column in frame.columns] + [f"role_{role}" for role in roles]
    if not features:
        features = ["available"]
        frame["available"] = 0.0
    tensor = np.full((len(sample_to_idx), max_context_stations, len(features)), np.nan, dtype="float32")
    mask = np.zeros((len(sample_to_idx), max_context_stations), dtype="float32")
    for sample_id, group in frame.groupby("sample_id", sort=False):
        sample_idx = sample_to_idx.get(str(sample_id))
        if sample_idx is None:
            continue
        sort_columns = [column for column in ("available", "distance_km") if column in group.columns]
        if sort_columns:
            ascending = [False if column == "available" else True for column in sort_columns]
            group = group.sort_values(sort_columns, ascending=ascending)
        group = group.head(max_context_stations)
        for row_idx, row in enumerate(group.itertuples(index=False)):
            mask[sample_idx, row_idx] = 1.0
            row_dict = row._asdict()
            for feature_idx, column in enumerate(features):
                value = row_dict.get(column)
                try:
                    tensor[sample_idx, row_idx, feature_idx] = float(value)
                except (TypeError, ValueError):
                    tensor[sample_idx, row_idx, feature_idx] = np.nan
    return tensor, mask, features


def build_offset_tensor(offsets: Any, sample_to_idx: dict[str, int], max_offsets: int, deps: dict[str, Any]) -> tuple[Any, Any, list[str]]:
    pd = deps["pd"]
    np = deps["np"]
    if offsets.empty:
        return (
            np.zeros((len(sample_to_idx), max_offsets, 1), dtype="float32"),
            np.zeros((len(sample_to_idx), max_offsets), dtype="float32"),
            ["missing_offset"],
        )
    frame = offsets.copy()
    add_direction_sin_cos(frame, "bearing_deg", pd, np, "bearing")
    add_direction_sin_cos(frame, "wind_direction_10m", pd, np, "wind_direction_10m")
    names = sorted(str(value) for value in frame.get("offset_name", pd.Series(dtype=str)).dropna().unique())
    for name in names:
        frame[f"offset_{name}"] = frame["offset_name"].astype(str).eq(name).astype(float)
    features = [column for column in OFFSET_NUMERIC_FEATURES if column in frame.columns] + [f"offset_{name}" for name in names]
    if not features:
        features = ["available"]
        frame["available"] = 0.0
    frame = frame.drop_duplicates(["sample_id", "offset_name"], keep="last")
    tensor = np.full((len(sample_to_idx), max_offsets, len(features)), np.nan, dtype="float32")
    mask = np.zeros((len(sample_to_idx), max_offsets), dtype="float32")
    for sample_id, group in frame.groupby("sample_id", sort=False):
        sample_idx = sample_to_idx.get(str(sample_id))
        if sample_idx is None:
            continue
        group = group.sort_values("offset_name").head(max_offsets)
        for row_idx, row in enumerate(group.itertuples(index=False)):
            mask[sample_idx, row_idx] = 1.0
            row_dict = row._asdict()
            for feature_idx, column in enumerate(features):
                value = row_dict.get(column)
                try:
                    tensor[sample_idx, row_idx, feature_idx] = float(value)
                except (TypeError, ValueError):
                    tensor[sample_idx, row_idx, feature_idx] = np.nan
    return tensor, mask, features


def build_vertical_tensor(vertical: Any, sample_to_idx: dict[str, int], deps: dict[str, Any]) -> tuple[Any, Any, list[str], list[int]]:
    pd = deps["pd"]
    np = deps["np"]
    if vertical.empty:
        return (
            np.zeros((len(sample_to_idx), 1, 1), dtype="float32"),
            np.zeros((len(sample_to_idx), 1), dtype="float32"),
            ["missing_vertical"],
            [0],
        )
    frame = vertical.copy()
    add_direction_sin_cos(frame, "wind_direction", pd, np, "wind_direction")
    frame["pressure_hpa"] = numeric(frame["pressure_hpa"], pd)
    levels = sorted(int(value) for value in frame["pressure_hpa"].dropna().unique())
    level_to_idx = {level: index for index, level in enumerate(levels)}
    features = [column for column in VERTICAL_FEATURES if column in frame.columns]
    tensor = np.full((len(sample_to_idx), len(levels), len(features)), np.nan, dtype="float32")
    mask = np.zeros((len(sample_to_idx), len(levels)), dtype="float32")
    for row in frame.itertuples(index=False):
        row_dict = row._asdict()
        sample_idx = sample_to_idx.get(str(row_dict.get("sample_id")))
        level = row_dict.get("pressure_hpa")
        try:
            level_float = float(level)
        except (TypeError, ValueError):
            continue
        if sample_idx is None or not math.isfinite(level_float):
            continue
        level_idx = level_to_idx.get(int(level_float))
        if level_idx is None:
            continue
        mask[sample_idx, level_idx] = 1.0
        for feature_idx, column in enumerate(features):
            value = row_dict.get(column)
            try:
                tensor[sample_idx, level_idx, feature_idx] = float(value)
            except (TypeError, ValueError):
                tensor[sample_idx, level_idx, feature_idx] = np.nan
    return tensor, mask, features, levels


def static_candidates(static: Any, pd: Any) -> list[str]:
    candidates = []
    for column in static.columns:
        if column in STATIC_BLOCKED_COLUMNS:
            continue
        if any(column.startswith(prefix) for prefix in STATIC_BLOCKED_PREFIXES):
            continue
        values = numeric(static[column], pd)
        if values.notna().sum() == 0 or values.nunique(dropna=True) <= 1:
            continue
        candidates.append(column)
    return candidates


def select_static_columns(static: Any, future: Any, max_features: int, deps: dict[str, Any]) -> list[str]:
    pd = deps["pd"]
    np = deps["np"]
    if static.empty or max_features <= 0:
        return []
    train = future[future["benchmark_split"].eq("train")].copy()
    merged = train[["sample_id", "residual_wind_mean_ms", "residual_gust_ms"]].merge(static, on="sample_id", how="left")
    y_wind = numeric(merged["residual_wind_mean_ms"], pd)
    y_gust = numeric(merged["residual_gust_ms"], pd)
    scores: list[tuple[float, float, str]] = []
    force_prefixes = ("thermal_", "dem_", "fetch_", "landsea_", "context_agg_")
    for column in static_candidates(static, pd):
        values = numeric(merged[column], pd)
        valid = values.notna() & (y_wind.notna() | y_gust.notna())
        count = int(valid.sum())
        if count < 25:
            continue
        score = 0.0
        for target in (y_wind, y_gust):
            valid_target = values.notna() & target.notna()
            if int(valid_target.sum()) >= 3:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    corr = np.corrcoef(values[valid_target].to_numpy(dtype=float), target[valid_target].to_numpy(dtype=float))[0, 1]
                if np.isfinite(corr):
                    score = max(score, abs(float(corr)))
        if column.startswith(force_prefixes):
            score += 0.03
        scores.append((score, count / max(1, len(merged)), column))
    scores.sort(reverse=True)
    return [column for _score, _coverage, column in scores[:max_features]]


def build_static_matrix(static: Any, sample_to_idx: dict[str, int], columns: list[str], deps: dict[str, Any]) -> Any:
    pd = deps["pd"]
    np = deps["np"]
    matrix = np.full((len(sample_to_idx), len(columns)), np.nan, dtype="float32")
    if static.empty or not columns:
        return matrix
    for row in static[["sample_id", *columns]].itertuples(index=False):
        row_dict = row._asdict()
        sample_idx = sample_to_idx.get(str(row_dict.get("sample_id")))
        if sample_idx is None:
            continue
        for column_idx, column in enumerate(columns):
            try:
                matrix[sample_idx, column_idx] = float(row_dict.get(column))
            except (TypeError, ValueError):
                matrix[sample_idx, column_idx] = np.nan
    return matrix


def prepare_future(future: Any, deps: dict[str, Any], fallback_eval_fraction: float) -> Any:
    pd = deps["pd"]
    np = deps["np"]
    frame = future.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["spot_id"] = frame["spot_id"].astype(str)
    frame["issue_time"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    frame["target_time"] = pd.to_datetime(frame["target_time_utc"], utc=True, errors="coerce")
    split = frame.get("split")
    if split is not None and (split.astype(str) != "train").any():
        frame["benchmark_split"] = split.astype(str).where(split.astype(str).eq("train"), "eval")
    else:
        issue_times = frame[["sample_id", "issue_time"]].drop_duplicates().sort_values("issue_time")
        eval_count = max(1, int(round(issue_times["sample_id"].nunique() * fallback_eval_fraction)))
        eval_samples = set(issue_times["sample_id"].drop_duplicates().tail(eval_count))
        frame["benchmark_split"] = frame["sample_id"].map(lambda value: "eval" if value in eval_samples else "train")
    add_direction_sin_cos(frame, "baseline_wind_direction_deg", pd, np, "baseline_wind_direction")
    hour = frame["issue_time"].dt.hour.astype(float) + frame["issue_time"].dt.minute.astype(float) / 60.0
    frame["issue_hour_sin"] = np.sin(2.0 * math.pi * hour / 24.0)
    frame["issue_hour_cos"] = np.cos(2.0 * math.pi * hour / 24.0)
    day = frame["issue_time"].dt.dayofyear.fillna(1).astype(float)
    frame["issue_dayofyear_sin"] = np.sin(2.0 * math.pi * day / 366.0)
    frame["issue_dayofyear_cos"] = np.cos(2.0 * math.pi * day / 366.0)
    for column in FUTURE_FEATURES:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


@dataclass
class PreparedData:
    future: Any
    sample_ids: list[str]
    sample_indices: Any
    spot_indices: Any
    future_features: Any
    history: Any
    context: Any
    context_mask: Any
    offsets: Any
    offset_mask: Any
    vertical: Any
    vertical_mask: Any
    static: Any
    y_scaled: Any
    y_mean: Any
    y_std: Any
    train_mask_rows: Any
    val_mask_rows: Any
    eval_mask_rows: Any
    feature_manifest: dict[str, Any]


def prepare_data(args: argparse.Namespace, deps: dict[str, Any]) -> PreparedData:
    pd = deps["pd"]
    np = deps["np"]
    root = args.dataset_root
    samples = read_table(root, "samples", pd)
    future = prepare_future(read_table(root, "future_targets", pd), deps, args.fallback_eval_fraction)
    if future.empty:
        raise SystemExit(f"Missing future_targets.parquet in {root}")
    future = future.dropna(
        subset=[
            "target_wind_mean_ms",
            "target_gust_ms",
            "baseline_wind_mean_ms",
            "baseline_gust_ms",
            "residual_wind_mean_ms",
            "residual_gust_ms",
        ]
    ).reset_index(drop=True)
    sample_ids, sample_to_idx = build_sample_index(samples, future)
    future["sample_index"] = future["sample_id"].map(sample_to_idx).astype(int)
    spots = sorted(future["spot_id"].dropna().astype(str).unique())
    spot_to_idx = {spot: index for index, spot in enumerate(spots)}
    future["spot_index"] = future["spot_id"].map(spot_to_idx).fillna(0).astype(int)

    train_rows_all = future["benchmark_split"].eq("train").to_numpy()
    train_sample_mask = np.zeros(len(sample_ids), dtype=bool)
    train_sample_mask[future.loc[future["benchmark_split"].eq("train"), "sample_index"].unique()] = True
    if not train_sample_mask.any():
        train_sample_mask[:] = True

    history, history_features = build_history_tensor(
        read_table(root, "station_history", pd), sample_to_idx, args.context_length, deps
    )
    context, context_mask, context_features = build_context_tensor(
        read_table(root, "context_station_snapshot", pd), sample_to_idx, args.max_context_stations, deps
    )
    offsets, offset_mask, offset_features = build_offset_tensor(
        read_table(root, "nwp_surface_offsets", pd), sample_to_idx, args.max_offsets, deps
    )
    vertical, vertical_mask, vertical_features, vertical_levels = build_vertical_tensor(
        read_table(root, "nwp_vertical_profile", pd), sample_to_idx, deps
    )
    static_table = read_table(root, "static_context", pd)
    static_columns = select_static_columns(static_table, future, args.max_static_features, deps)
    static = build_static_matrix(static_table, sample_to_idx, static_columns, deps)

    history, history_stats = standardize_array(history, train_sample_mask, np)
    context, context_stats = standardize_array(context, train_sample_mask, np)
    offsets, offset_stats = standardize_array(offsets, train_sample_mask, np)
    vertical, vertical_stats = standardize_array(vertical, train_sample_mask, np)
    if static.shape[1]:
        static, static_stats = standardize_2d(static, train_sample_mask, np)
    else:
        static = np.zeros((len(sample_ids), 1), dtype="float32")
        static_stats = {"mean": [0.0], "std": [1.0]}
        static_columns = ["missing_static"]

    future_matrix = future[FUTURE_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
    future_features, future_stats = standardize_2d(future_matrix, train_rows_all, np)

    y = future[["residual_wind_mean_ms", "residual_gust_ms"]].to_numpy(dtype="float32")
    y_mean = np.nanmean(y[train_rows_all], axis=0, keepdims=True)
    y_std = np.nanstd(y[train_rows_all], axis=0, keepdims=True)
    y_mean = np.where(np.isfinite(y_mean), y_mean, 0.0)
    y_std = np.where(np.isfinite(y_std) & (y_std > 1e-6), y_std, 1.0)
    y_scaled = np.nan_to_num((y - y_mean) / y_std, nan=0.0).astype("float32")

    train_indices = np.where(train_rows_all)[0]
    if len(train_indices) < 2:
        raise SystemExit("Need at least two training rows.")
    ordered_train = future.loc[train_rows_all, ["issue_time"]].reset_index()
    ordered_train = ordered_train.sort_values("issue_time")
    val_count = max(args.min_val_rows, int(round(len(ordered_train) * args.val_fraction)))
    val_count = min(max(val_count, 1), max(len(ordered_train) - 1, 1))
    val_indices = set(ordered_train["index"].tail(val_count).astype(int).tolist())
    val_mask = np.array([index in val_indices for index in range(len(future))], dtype=bool)
    train_mask = train_rows_all & ~val_mask
    eval_mask = ~future["benchmark_split"].eq("train").to_numpy()

    manifest = {
        "history_features": history_features,
        "context_features": context_features,
        "offset_features": offset_features,
        "vertical_features": vertical_features,
        "vertical_levels_hpa": vertical_levels,
        "static_features": static_columns,
        "future_features": FUTURE_FEATURES,
        "spots": spots,
        "normalization": {
            "history": history_stats,
            "context": context_stats,
            "offsets": offset_stats,
            "vertical": vertical_stats,
            "static": static_stats,
            "future": future_stats,
            "target_mean": y_mean.squeeze().astype(float).tolist(),
            "target_std": y_std.squeeze().astype(float).tolist(),
        },
    }
    return PreparedData(
        future=future,
        sample_ids=sample_ids,
        sample_indices=future["sample_index"].to_numpy(dtype="int64"),
        spot_indices=future["spot_index"].to_numpy(dtype="int64"),
        future_features=future_features,
        history=history,
        context=context,
        context_mask=context_mask,
        offsets=offsets,
        offset_mask=offset_mask,
        vertical=vertical,
        vertical_mask=vertical_mask,
        static=static,
        y_scaled=y_scaled,
        y_mean=y_mean.astype("float32"),
        y_std=y_std.astype("float32"),
        train_mask_rows=train_mask,
        val_mask_rows=val_mask,
        eval_mask_rows=eval_mask,
        feature_manifest=manifest,
    )


class StructuredDataset:
    def __init__(self, data: PreparedData, indices: Any, torch: Any) -> None:
        self.data = data
        self.indices = indices.astype("int64")
        self.torch = torch

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> dict[str, Any]:
        idx = int(self.indices[item])
        sample_idx = int(self.data.sample_indices[idx])
        return {
            "history": self.torch.from_numpy(self.data.history[sample_idx]),
            "context": self.torch.from_numpy(self.data.context[sample_idx]),
            "context_mask": self.torch.from_numpy(self.data.context_mask[sample_idx]),
            "offsets": self.torch.from_numpy(self.data.offsets[sample_idx]),
            "offset_mask": self.torch.from_numpy(self.data.offset_mask[sample_idx]),
            "vertical": self.torch.from_numpy(self.data.vertical[sample_idx]),
            "vertical_mask": self.torch.from_numpy(self.data.vertical_mask[sample_idx]),
            "static": self.torch.from_numpy(self.data.static[sample_idx]),
            "future": self.torch.from_numpy(self.data.future_features[idx]),
            "spot_index": self.torch.tensor(int(self.data.spot_indices[idx]), dtype=self.torch.long),
            "target": self.torch.from_numpy(self.data.y_scaled[idx]),
            "row_index": self.torch.tensor(idx, dtype=self.torch.long),
        }


class MaskedSetEncoder:
    def __init__(self, nn: Any, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        self.nn = nn
        self.module = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def __call__(self, values: Any, mask: Any) -> Any:
        emb = self.module(values)
        mask = mask.unsqueeze(-1).float()
        summed = (emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom


def build_model_class(deps: dict[str, Any]) -> Any:
    torch = deps["torch"]
    nn = deps["nn"]

    class StructuredResidualNet(nn.Module):
        def __init__(
            self,
            *,
            history_dim: int,
            context_dim: int,
            offset_dim: int,
            vertical_shape: tuple[int, int],
            static_dim: int,
            future_dim: int,
            spot_count: int,
            hidden_dim: int,
            dropout: float,
            spot_embedding_dim: int,
        ) -> None:
            super().__init__()
            self.history_gru = nn.GRU(history_dim, hidden_dim, batch_first=True)
            self.context_mlp = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.offset_mlp = nn.Sequential(
                nn.Linear(offset_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            vertical_dim = vertical_shape[0] * vertical_shape[1]
            self.vertical_mlp = nn.Sequential(
                nn.Linear(vertical_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.static_mlp = nn.Sequential(
                nn.Linear(static_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.future_mlp = nn.Sequential(
                nn.Linear(future_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.spot_embedding = nn.Embedding(max(spot_count, 1), spot_embedding_dim)
            merged_dim = hidden_dim * 6 + spot_embedding_dim
            self.head = nn.Sequential(
                nn.Linear(merged_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )

        @staticmethod
        def masked_mean(encoded: Any, mask: Any) -> Any:
            mask = mask.unsqueeze(-1).float()
            return (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        def forward(self, batch: dict[str, Any]) -> Any:
            _history_out, history_state = self.history_gru(batch["history"])
            history_emb = history_state[-1]
            context_emb = self.masked_mean(self.context_mlp(batch["context"]), batch["context_mask"])
            offset_emb = self.masked_mean(self.offset_mlp(batch["offsets"]), batch["offset_mask"])
            vertical_emb = self.vertical_mlp(batch["vertical"].flatten(start_dim=1))
            static_emb = self.static_mlp(batch["static"])
            future_emb = self.future_mlp(batch["future"])
            spot_emb = self.spot_embedding(batch["spot_index"])
            merged = torch.cat([history_emb, context_emb, offset_emb, vertical_emb, static_emb, future_emb, spot_emb], dim=1)
            return self.head(merged)

    return StructuredResidualNet


def move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def evaluate_model(model: Any, loader: Any, criterion: Any, device: Any, torch: Any) -> tuple[float, Any, Any]:
    model.eval()
    losses = []
    rows = []
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(batch)
            loss = criterion(output, batch["target"])
            losses.append(float(loss.item()))
            rows.append(batch["row_index"].detach().cpu())
            preds.append(output.detach().cpu())
    if not losses:
        return 0.0, torch.empty(0, dtype=torch.long), torch.empty((0, 2))
    return sum(losses) / len(losses), torch.cat(rows), torch.cat(preds)


def make_loader(data: PreparedData, indices: Any, batch_size: int, shuffle: bool, deps: dict[str, Any]) -> Any:
    Dataset = deps["Dataset"]
    DataLoader = deps["DataLoader"]
    torch = deps["torch"]

    class _Dataset(StructuredDataset, Dataset):
        pass

    return DataLoader(_Dataset(data, indices, torch), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_model(data: PreparedData, args: argparse.Namespace, deps: dict[str, Any]) -> tuple[Any, dict[str, Any], Any, Any]:
    torch = deps["torch"]
    nn = deps["nn"]
    seed_everything(args.random_state, torch)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_cls = build_model_class(deps)
    model = model_cls(
        history_dim=data.history.shape[2],
        context_dim=data.context.shape[2],
        offset_dim=data.offsets.shape[2],
        vertical_shape=(data.vertical.shape[1], data.vertical.shape[2]),
        static_dim=data.static.shape[1],
        future_dim=data.future_features.shape[1],
        spot_count=len(data.feature_manifest["spots"]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        spot_embedding_dim=args.spot_embedding_dim,
    ).to(device)
    criterion = nn.SmoothL1Loss(beta=args.huber_beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_indices = deps["np"].where(data.train_mask_rows)[0]
    val_indices = deps["np"].where(data.val_mask_rows)[0]
    train_loader = make_loader(data, train_indices, args.batch_size, True, deps)
    val_loader = make_loader(data, val_indices, args.batch_size, False, deps)

    best_state = None
    best_val = float("inf")
    patience_left = args.patience
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = criterion(output, batch["target"])
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss, _rows, _preds = evaluate_model(model, val_loader, criterion, device, torch)
        train_loss = sum(train_losses) / max(1, len(train_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if args.verbose:
            print(json.dumps(history[-1], sort_keys=True), flush=True)
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"epochs_ran": len(history), "best_val_loss": best_val, "history": history, "device": str(device)}, criterion, device


def predictions_frame(model: Any, data: PreparedData, criterion: Any, device: Any, args: argparse.Namespace, deps: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    pd = deps["pd"]
    np = deps["np"]
    torch = deps["torch"]
    all_indices = np.arange(len(data.future))
    loader = make_loader(data, all_indices, args.batch_size, False, deps)
    _loss, row_tensor, pred_scaled = evaluate_model(model, loader, criterion, device, torch)
    preds = np.full((len(data.future), 2), np.nan, dtype="float32")
    row_indices = row_tensor.numpy().astype(int)
    pred_values = pred_scaled.numpy().astype("float32") * data.y_std + data.y_mean
    preds[row_indices] = pred_values
    future = data.future.copy()
    future["saphir_nn_residual_wind_mean_ms"] = preds[:, 0]
    future["saphir_nn_residual_gust_ms"] = preds[:, 1]
    future["saphir_nn_wind_mean_ms"] = numeric(future["baseline_wind_mean_ms"], pd) + future["saphir_nn_residual_wind_mean_ms"]
    future["saphir_nn_gust_ms"] = numeric(future["baseline_gust_ms"], pd) + future["saphir_nn_residual_gust_ms"]
    future["raw_wind_mean_ms"] = numeric(future["baseline_wind_mean_ms"], pd)
    future["raw_gust_ms"] = numeric(future["baseline_gust_ms"], pd)
    future["actual_wind_mean_ms"] = numeric(future["target_wind_mean_ms"], pd)
    future["actual_gust_ms"] = numeric(future["target_gust_ms"], pd)

    metrics = {}
    eval_frame = future[future["benchmark_split"].ne("train")]
    train_frame = future[future["benchmark_split"].eq("train")]
    for target, config in TARGETS.items():
        actual = f"actual_{target}"
        raw = f"raw_{target}"
        pred = f"saphir_nn_{target}"
        metrics[target] = {
            "train": {
                "raw_nwp": metric(np, train_frame, raw, actual),
                "saphir_nn": metric(np, train_frame, pred, actual),
            },
            "eval": {
                "raw_nwp": metric(np, eval_frame, raw, actual),
                "saphir_nn": metric(np, eval_frame, pred, actual),
            },
            "eval_by_lead": {
                "raw_nwp": metrics_by_lead(np, eval_frame, raw, actual),
                "saphir_nn": metrics_by_lead(np, eval_frame, pred, actual),
            },
        }
    keep = [
        "sample_id",
        "spot_id",
        "issue_time_utc",
        "target_time_utc",
        "lead_time_minutes",
        "benchmark_split",
        "actual_wind_mean_ms",
        "raw_wind_mean_ms",
        "saphir_nn_wind_mean_ms",
        "saphir_nn_residual_wind_mean_ms",
        "actual_gust_ms",
        "raw_gust_ms",
        "saphir_nn_gust_ms",
        "saphir_nn_residual_gust_ms",
    ]
    return future[keep].copy(), metrics


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# CorseWind SAPHIR-Style Neural Benchmark",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Dataset root: `{result['dataset_root']}`",
        f"Output root: `{result['output_root']}`",
        f"Device: `{result['training']['device']}`",
        f"Epochs: `{result['training']['epochs_ran']}`",
        "",
        "## Eval Metrics",
        "",
        "| Target | Model | RMSE | MAE | Bias | Count |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for target, target_metrics in result["metrics"].items():
        for model_name, item in target_metrics["eval"].items():
            lines.append(
                f"| `{target}` | `{model_name}` | {item.get('rmse')} | {item.get('mae')} | {item.get('bias')} | {item.get('count')} |"
            )
    lines.extend(["", "## Eval RMSE By Lead", ""])
    for target, target_metrics in result["metrics"].items():
        lines.append(f"### {target}")
        lines.append("")
        lines.append("| Lead | Raw NWP | SAPHIR NN |")
        lines.append("| ---: | ---: | ---: |")
        leads = sorted({int(lead) for model in target_metrics["eval_by_lead"].values() for lead in model.keys()})
        for lead in leads:
            raw = target_metrics["eval_by_lead"]["raw_nwp"].get(str(lead), {}).get("rmse")
            nn = target_metrics["eval_by_lead"]["saphir_nn"].get(str(lead), {}).get("rmse")
            lines.append(f"| {lead} | {raw} | {nn} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id", default="corsewind_saphir_structured_neural_v1")
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--max-context-stations", type=int, default=12)
    parser.add_argument("--max-offsets", type=int, default=8)
    parser.add_argument("--max-static-features", type=int, default=192)
    parser.add_argument("--fallback-eval-fraction", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-val-rows", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--spot-embedding-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=0.8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = import_dependencies()
    output_root = (args.output_root or args.dataset_root / "saphir_neural_benchmark").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data = prepare_data(args, deps)
    model, training_summary, criterion, device = train_model(data, args, deps)
    predictions, metrics = predictions_frame(model, data, criterion, device, args, deps)
    predictions.to_parquet(output_root / "predictions_saphir_neural.parquet", index=False)
    if args.save_model:
        deps["torch"].save(model.state_dict(), output_root / "model_state.pt")
    result = {
        "format": "corsewind.saphir_structured_neural_benchmark.v1",
        "generated_at_utc": utc_now(),
        "run_id": args.run_id,
        "dataset_root": str(args.dataset_root.resolve()),
        "output_root": str(output_root),
        "row_count": int(len(data.future)),
        "train_rows": int(data.train_mask_rows.sum()),
        "val_rows": int(data.val_mask_rows.sum()),
        "eval_rows": int(data.eval_mask_rows.sum()),
        "training": training_summary,
        "metrics": metrics,
        "feature_manifest": {
            key: value
            for key, value in data.feature_manifest.items()
            if key != "normalization"
        },
        "args": vars(args),
    }
    (output_root / "benchmark_results.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (output_root / "feature_manifest.json").write_text(json.dumps(data.feature_manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_markdown(output_root / "benchmark_results.md", result)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "output_root": result["output_root"],
                "train_rows": result["train_rows"],
                "val_rows": result["val_rows"],
                "eval_rows": result["eval_rows"],
                "metrics": {target: values["eval"] for target, values in metrics.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
