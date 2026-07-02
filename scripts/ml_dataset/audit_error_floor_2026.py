#!/usr/bin/env python3
"""Audit the current CorseWind error floor across wind, gust, and labels."""

from __future__ import annotations

import argparse
import glob
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KT_PER_MS = 1.9438444924406

DEFAULT_WIND_PREDICTIONS = Path(
    "/srv/data/corsewind/ml_dataset/benchmarks/"
    "prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1/"
    "calibrated_predictions_2026.parquet"
)
DEFAULT_GUST_PREDICTIONS = Path(
    "/srv/data/corsewind/ml_dataset/benchmarks/"
    "prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_gust_from_wind_champion_recipe_v1/"
    "calibrated_predictions_2026.parquet"
)
DEFAULT_TRAINING_GLOB = (
    "/srv/data/corsewind/ml_dataset/training_tables/residual_windsup_sst_prev_202[456]_*"
    "/training_rows.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "/srv/data/corsewind/ml_dataset/benchmarks/scientific_error_floor_audit_2026_07_01"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def finite_frame(frame: Any, columns: list[str]) -> Any:
    out = frame[columns].copy()
    for column in columns:
        if column not in ("spot_id", "issue_time_utc", "target_time_utc"):
            out[column] = out[column].astype(float)
    return out.dropna()


def metric_from_errors(errors: Any, np: Any) -> dict[str, Any]:
    errors = errors.dropna() if hasattr(errors, "dropna") else errors
    if len(errors) == 0:
        return {"count": 0}
    abs_errors = np.abs(errors)
    rmse = math.sqrt(float(np.mean(errors * errors)))
    mae = float(np.mean(abs_errors))
    return {
        "count": int(len(errors)),
        "rmse_ms": round(rmse, 6),
        "rmse_kt": round(rmse * KT_PER_MS, 6),
        "mae_ms": round(mae, 6),
        "mae_kt": round(mae * KT_PER_MS, 6),
        "bias_ms": round(float(np.mean(errors)), 6),
        "bias_kt": round(float(np.mean(errors)) * KT_PER_MS, 6),
        "p50_abs_ms": round(float(np.quantile(abs_errors, 0.50)), 6),
        "p90_abs_ms": round(float(np.quantile(abs_errors, 0.90)), 6),
        "p95_abs_ms": round(float(np.quantile(abs_errors, 0.95)), 6),
        "p99_abs_ms": round(float(np.quantile(abs_errors, 0.99)), 6),
    }


def metric(frame: Any, prediction: str, target: str, np: Any) -> dict[str, Any]:
    valid = finite_frame(frame, [prediction, target])
    return metric_from_errors(valid[prediction] - valid[target], np)


def add_time_and_bins(frame: Any, pd: Any, target_column: str, prefix: str) -> Any:
    out = frame.copy()
    out["issue_time_utc"] = pd.to_datetime(out["issue_time_utc"], utc=True, errors="coerce")
    lead_delta = pd.to_timedelta(out["lead_time_minutes"].astype(float), unit="m")
    out["target_time_utc"] = out["issue_time_utc"] + lead_delta
    out["target_month"] = out["target_time_utc"].dt.strftime("%Y-%m")
    out["target_hour_utc"] = out["target_time_utc"].dt.hour.astype("Int64")
    out["target_hour_local"] = out["target_time_utc"].dt.tz_convert("Europe/Paris").dt.hour.astype("Int64")
    out[f"{prefix}_actual_bin_ms"] = pd.cut(
        out[target_column].astype(float),
        [-0.001, 2, 4, 6, 8, 10, 999],
        labels=["0-2", "2-4", "4-6", "6-8", "8-10", "10+"],
    ).astype(str)
    out[f"{prefix}_actual_bin_kt"] = pd.cut(
        out[target_column].astype(float) * KT_PER_MS,
        [-0.001, 12, 15, 20, 25, 999],
        labels=["0-12", "12-15", "15-20", "20-25", "25+"],
    ).astype(str)
    return out


def stage_metrics(frame: Any, target: str, predictions: list[str], np: Any) -> dict[str, Any]:
    out = {}
    for column in predictions:
        if column in frame.columns:
            out[column] = metric(frame, column, target, np)
    return out


def improvement(base: dict[str, Any], new: dict[str, Any]) -> float | None:
    if not base.get("rmse_ms") or not new.get("rmse_ms"):
        return None
    return round((base["rmse_ms"] - new["rmse_ms"]) / base["rmse_ms"] * 100.0, 3)


def group_metrics(
    frame: Any,
    group_columns: list[str],
    prediction: str,
    target: str,
    np: Any,
    pd: Any,
    limit: int = 12,
) -> list[dict[str, Any]]:
    missing = [column for column in group_columns if column not in frame.columns]
    if missing:
        return []
    valid = frame.dropna(subset=[*group_columns, prediction, target]).copy()
    if valid.empty:
        return []
    errors = valid[prediction].astype(float) - valid[target].astype(float)
    total_sse = float((errors * errors).sum())
    total_count = int(len(valid))
    current_rmse = math.sqrt(total_sse / total_count)
    rows = []
    for raw_key, group in valid.groupby(group_columns, dropna=False):
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        group_errors = group[prediction].astype(float) - group[target].astype(float)
        sse = float((group_errors * group_errors).sum())
        rmse_if_perfect = math.sqrt(max(0.0, total_sse - sse) / total_count)
        rows.append(
            {
                "group": {column: scalar(pd, value) for column, value in zip(group_columns, values, strict=True)},
                "row_share_pct": round(len(group) / total_count * 100.0, 3),
                "sse_share_pct": round(sse / total_sse * 100.0, 3),
                "global_rmse_if_perfect_ms": round(rmse_if_perfect, 6),
                "global_rmse_gain_if_perfect_ms": round(current_rmse - rmse_if_perfect, 6),
                **metric_from_errors(group_errors, np),
            }
        )
    rows.sort(key=lambda row: (row["sse_share_pct"], row.get("rmse_ms", 0)), reverse=True)
    return rows[:limit]


def scalar(pd: Any, value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value) if not isinstance(value, (str, int, float, bool)) else value


def tail(frame: Any, prediction: str, target: str, threshold_rmse_ms: float, np: Any) -> dict[str, Any]:
    valid = finite_frame(frame, [prediction, target])
    squared = ((valid[prediction] - valid[target]) ** 2).sort_values(ascending=False).to_numpy()
    n = len(squared)
    total = float(squared.sum())
    target_sse = threshold_rmse_ms * threshold_rmse_ms * n
    excess = max(0.0, total - target_sse)
    cumsum = np.cumsum(squared)
    perfect_rows = int(np.searchsorted(cumsum, excess, side="left") + 1) if excess > 0 else 0
    out = {
        "row_count": int(n),
        "current_sse": round(total, 6),
        "target_rmse_ms": threshold_rmse_ms,
        "target_sse": round(target_sse, 6),
        "excess_sse": round(excess, 6),
        "mse_reduction_needed_pct": round(excess / total * 100.0, 3) if total else 0.0,
        "perfect_rows_needed": min(perfect_rows, n),
        "perfect_rows_needed_pct": round(min(perfect_rows, n) / n * 100.0, 3) if n else 0.0,
    }
    for pct in (1, 2, 5, 10, 20):
        count = max(1, math.ceil(n * pct / 100.0)) if n else 0
        out[f"top_{pct}_pct_sse_share_pct"] = round(float(squared[:count].sum() / total * 100.0), 3) if total else 0.0
    return out


def rowwise_existing_oracle(frame: Any, predictions: list[str], target: str, np: Any) -> dict[str, Any]:
    valid = frame.dropna(subset=[*predictions, target]).copy()
    if valid.empty:
        return {"count": 0}
    y = valid[target].astype(float).to_numpy()
    matrix = valid[predictions].astype(float).to_numpy()
    errors = np.abs(matrix - y[:, None])
    best = np.argmin(errors, axis=1)
    chosen = matrix[np.arange(len(valid)), best]
    counts = {predictions[index]: int((best == index).sum()) for index in range(len(predictions))}
    return {**metric_from_errors(chosen - y, np), "selection_counts": counts}


def bias_oracles(frame: Any, prediction: str, target: str, groups: list[list[str]], np: Any) -> dict[str, Any]:
    valid = frame.dropna(subset=[prediction, target]).copy()
    valid["_error"] = valid[prediction].astype(float) - valid[target].astype(float)
    out = {
        "global_mean_residual_removed": metric_from_errors(valid["_error"] - float(valid["_error"].mean()), np),
    }
    for group_columns in groups:
        missing = [column for column in group_columns if column not in valid.columns]
        if missing:
            continue
        means = valid.groupby(group_columns)["_error"].transform("mean")
        key = "+".join(group_columns)
        out[key] = metric_from_errors(valid["_error"] - means, np)
    return out


def perfect_segment_oracles(frame: Any, prediction: str, target: str, prefix: str, np: Any) -> dict[str, Any]:
    valid = frame.dropna(subset=[prediction, target]).copy()
    errors = valid[prediction].astype(float) - valid[target].astype(float)
    total_count = len(errors)
    total_sse = float((errors * errors).sum())

    def one(name: str, mask: Any) -> dict[str, Any]:
        sse_fixed = float((errors[~mask] * errors[~mask]).sum())
        return {
            "rows": int(mask.sum()),
            "row_share_pct": round(float(mask.mean() * 100.0), 3),
            "sse_share_pct": round(float((total_sse - sse_fixed) / total_sse * 100.0), 3) if total_sse else 0.0,
            "rmse_if_perfect_ms": round(math.sqrt(sse_fixed / total_count), 6) if total_count else 0.0,
        }

    abs_error = errors.abs()
    return {
        "lead_45_60": one("lead_45_60", valid["lead_time_minutes"].astype(float).isin([45, 60])),
        "critical_south_spots": one(
            "critical_south_spots",
            valid["spot_id"].astype(str).isin(["la_tonnara", "santa_manza", "balistra"]),
        ),
        f"{prefix}_actual_ge_8ms": one(f"{prefix}_actual_ge_8ms", valid[target].astype(float) >= 8.0),
        f"{prefix}_actual_le_2ms": one(f"{prefix}_actual_le_2ms", valid[target].astype(float) <= 2.0),
        "top_5pct_abs_error": one("top_5pct_abs_error", abs_error >= abs_error.quantile(0.95)),
    }


def threshold_metrics(frame: Any, prediction: str, target: str, thresholds_kt: list[float], np: Any) -> dict[str, Any]:
    valid = frame.dropna(subset=[prediction, target]).copy()
    pred_kt = valid[prediction].astype(float).to_numpy() * KT_PER_MS
    actual_kt = valid[target].astype(float).to_numpy() * KT_PER_MS
    out = {}
    for threshold in thresholds_kt:
        pred_event = pred_kt >= threshold
        actual_event = actual_kt >= threshold
        tp = int((pred_event & actual_event).sum())
        fp = int((pred_event & ~actual_event).sum())
        fn = int((~pred_event & actual_event).sum())
        tn = int((~pred_event & ~actual_event).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[f"ge_{int(threshold)}kt"] = {
            "support_actual_events": int(actual_event.sum()),
            "actual_event_rate_pct": round(float(actual_event.mean() * 100.0), 3),
            "predicted_event_rate_pct": round(float(pred_event.mean() * 100.0), 3),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    return out


def feature_coverage(frame: Any) -> dict[str, Any]:
    concepts = {
        "sst": ["sst"],
        "land_surface_temperature": ["land_surface_temperature", "lst"],
        "land_sea_or_thermal_delta": ["thermal_", "land_minus_sst", "air_minus_sst"],
        "cloud_type_or_cover": ["cloud_type", "cloud_cover"],
        "instability_or_cape": ["instability", "cape", "lifted_index", "k_index", "total_totals"],
        "radiation": ["shortwave", "radiation", "insolation"],
        "pressure": ["pressure", "pressure_msl", "sea_level_pressure"],
        "recent_obs_and_model_error": ["obs_delta", "obs_lag", "model_error_now"],
        "context_stations": ["context_"],
        "previous_runs": ["previous_run", "best_match_day"],
        "dem_static": ["dem_", "altitude_m", "slope", "aspect"],
        "fetch_static": ["fetch", "landsea", "land_sea"],
        "exposure_fetch_proxy_static": ["open_exposure_score", "low_or_sea_sample_share", "nearest_land_distance"],
        "vertical_profile": ["vertical", "isobaric", "pressure_level", "geopotential", "lapse_rate"],
    }
    columns = list(frame.columns)
    feature_columns = [
        column
        for column in columns
        if column.startswith("features__")
        or column.startswith("baselines__")
        or column.startswith("issue_feature_sources__")
        or column.startswith("target_feature_sources__")
    ]
    out = {}
    for concept, patterns in concepts.items():
        matches = [
            column
            for column in feature_columns
            if any(pattern.lower() in column.lower() for pattern in patterns)
            and "actual_" not in column.lower()
            and "abs_" not in column.lower()
        ]
        if matches:
            any_non_null = frame[matches].notna().any(axis=1).mean() * 100.0
            rates = [(column, float(frame[column].notna().mean() * 100.0)) for column in matches]
            rates.sort(key=lambda item: item[1], reverse=True)
            out[concept] = {
                "present": True,
                "column_count": len(matches),
                "any_non_null_pct": round(float(any_non_null), 3),
                "top_columns": [{"column": column, "non_null_pct": round(rate, 3)} for column, rate in rates[:8]],
            }
        else:
            out[concept] = {"present": False, "column_count": 0, "any_non_null_pct": 0.0, "top_columns": []}
    return out


def load_label_frame(paths: list[Path], pd: Any) -> Any:
    columns = [
        "spot_id",
        "station_id",
        "issue_time_utc",
        "target_time_utc",
        "lead_time_minutes",
        "labels__target_wind_mean_ms",
        "labels__target_gust_ms",
        "labels__target_observation_distance_minutes",
        "labels__target_observation_timestamp_utc",
        "labels__target_observation_source_resolution_minutes",
        "labels__target_observation_source_dataset",
        "labels__target_observation_source_project",
        "labels__target_observation_source_type",
        "labels__target_observation_station_id",
    ]
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_parquet(path, columns=columns))
        except Exception as exc:  # pragma: no cover - report-side resilience.
            print(f"warning: could not read labels from {path}: {exc}")
    if not frames:
        return pd.DataFrame(columns=columns)
    frame = pd.concat(frames, ignore_index=True)
    frame["issue_time_utc"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    frame["target_time_utc"] = pd.to_datetime(frame["target_time_utc"], utc=True, errors="coerce")
    frame["labels__target_observation_timestamp_utc"] = pd.to_datetime(
        frame["labels__target_observation_timestamp_utc"], utc=True, errors="coerce"
    )
    return frame


def quantiles(series: Any, np: Any) -> dict[str, Any]:
    valid = series.dropna().astype(float)
    if valid.empty:
        return {"count": 0}
    return {
        "count": int(len(valid)),
        "mean": round(float(valid.mean()), 6),
        "p50": round(float(np.quantile(valid, 0.50)), 6),
        "p90": round(float(np.quantile(valid, 0.90)), 6),
        "p95": round(float(np.quantile(valid, 0.95)), 6),
        "p99": round(float(np.quantile(valid, 0.99)), 6),
        "max": round(float(valid.max()), 6),
    }


def label_audit(paths: list[Path], pd: Any, np: Any) -> dict[str, Any]:
    frame = load_label_frame(paths, pd)
    out: dict[str, Any] = {
        "path_count": len(paths),
        "row_count": int(len(frame)),
        "spot_count": int(frame["spot_id"].nunique()) if not frame.empty else 0,
    }
    if frame.empty:
        return out
    out["target_months"] = sorted(frame["target_time_utc"].dt.strftime("%Y-%m").dropna().unique().tolist())
    out["target_observation_distance_minutes"] = quantiles(frame["labels__target_observation_distance_minutes"], np)
    out["source_resolution_minutes_counts"] = {
        str(key): int(value)
        for key, value in frame["labels__target_observation_source_resolution_minutes"].value_counts(dropna=False).head(20).items()
    }
    out["source_dataset_counts"] = {
        str(key): int(value)
        for key, value in frame["labels__target_observation_source_dataset"].value_counts(dropna=False).head(20).items()
    }

    dedup = frame.dropna(subset=["spot_id", "target_time_utc"]).copy()
    grouped = dedup.groupby(["spot_id", "target_time_utc"], dropna=False).agg(
        rows=("spot_id", "size"),
        wind_min=("labels__target_wind_mean_ms", "min"),
        wind_max=("labels__target_wind_mean_ms", "max"),
        gust_min=("labels__target_gust_ms", "min"),
        gust_max=("labels__target_gust_ms", "max"),
    )
    grouped["wind_range"] = grouped["wind_max"] - grouped["wind_min"]
    grouped["gust_range"] = grouped["gust_max"] - grouped["gust_min"]
    out["duplicate_target_consistency"] = {
        "target_groups": int(len(grouped)),
        "groups_with_multiple_rows": int((grouped["rows"] > 1).sum()),
        "wind_range": quantiles(grouped.loc[grouped["rows"] > 1, "wind_range"], np),
        "gust_range": quantiles(grouped.loc[grouped["rows"] > 1, "gust_range"], np),
        "wind_groups_range_gt_0_05_ms_pct": round(float((grouped["wind_range"] > 0.05).mean() * 100.0), 6),
        "gust_groups_range_gt_0_05_ms_pct": round(float((grouped["gust_range"] > 0.05).mean() * 100.0), 6),
    }

    series = (
        dedup.groupby(["spot_id", "target_time_utc"], as_index=False)
        .agg(wind=("labels__target_wind_mean_ms", "mean"), gust=("labels__target_gust_ms", "mean"))
        .sort_values(["spot_id", "target_time_utc"])
    )
    series["prev_time"] = series.groupby("spot_id")["target_time_utc"].shift(1)
    series["dt_minutes"] = (series["target_time_utc"] - series["prev_time"]).dt.total_seconds() / 60.0
    series["wind_abs_delta"] = series.groupby("spot_id")["wind"].diff().abs()
    series["gust_abs_delta"] = series.groupby("spot_id")["gust"].diff().abs()
    near_15 = series[(series["dt_minutes"] > 0) & (series["dt_minutes"] <= 20)]
    out["target_short_term_volatility_le_20min"] = {
        "pair_count": int(len(near_15)),
        "wind_abs_delta_ms": quantiles(near_15["wind_abs_delta"], np),
        "gust_abs_delta_ms": quantiles(near_15["gust_abs_delta"], np),
        "wind_delta_gt_1ms_pct": round(float((near_15["wind_abs_delta"] > 1.0).mean() * 100.0), 3) if len(near_15) else 0.0,
        "wind_delta_gt_2ms_pct": round(float((near_15["wind_abs_delta"] > 2.0).mean() * 100.0), 3) if len(near_15) else 0.0,
        "gust_delta_gt_2ms_pct": round(float((near_15["gust_abs_delta"] > 2.0).mean() * 100.0), 3) if len(near_15) else 0.0,
    }
    return out


def training_source_coverage(paths: list[Path], pd: Any) -> dict[str, Any]:
    wanted = [
        "issue_feature_sources__previous_run_open_meteo_best_match_day1",
        "issue_feature_sources__previous_run_open_meteo_best_match_day2",
        "target_feature_sources__previous_run_open_meteo_best_match_day1",
        "target_feature_sources__previous_run_open_meteo_best_match_day2",
        "issue_feature_sources__vertical_arome",
        "target_feature_sources__vertical_arome",
    ]
    rows = 0
    counts = {column: 0 for column in wanted}
    present_columns: set[str] = set()
    for path in paths:
        try:
            import pyarrow.parquet as pq

            schema_columns = set(pq.read_schema(path).names)
            existing = [column for column in wanted if column in schema_columns]
            if existing:
                frame = pd.read_parquet(path, columns=existing)
            else:
                metadata = pq.read_metadata(path)
                frame = pd.DataFrame(index=range(metadata.num_rows))
        except Exception as exc:  # pragma: no cover - report-side resilience.
            print(f"warning: could not read source coverage from {path}: {exc}")
            continue
        rows += len(frame)
        for column in frame.columns:
            present_columns.add(column)
            counts[column] += int(frame[column].notna().sum())
    return {
        "row_count": int(rows),
        "columns": {
            column: {
                "present": column in present_columns,
                "non_null_count": int(counts[column]),
                "non_null_pct": round((counts[column] / rows * 100.0), 3) if rows else 0.0,
            }
            for column in wanted
        },
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    wind = pd.read_parquet(args.wind_predictions)
    gust = pd.read_parquet(args.gust_predictions)
    wind = add_time_and_bins(wind, pd, "actual_wind_mean_ms", "wind")
    gust = add_time_and_bins(gust, pd, "actual_gust_ms", "gust")

    wind_predictions = ["raw_wind_mean_ms", "corrected_wind_mean_ms", "calibrated_wind_mean_ms"]
    gust_predictions = ["raw_gust_ms", "corrected_gust_ms", "calibrated_gust_ms"]

    result: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "wind_predictions_path": str(args.wind_predictions),
        "gust_predictions_path": str(args.gust_predictions),
        "training_glob": args.training_glob,
        "wind": {
            "rows": int(len(wind)),
            "spots": sorted(wind["spot_id"].astype(str).unique().tolist()),
            "stage_metrics": stage_metrics(wind, "actual_wind_mean_ms", wind_predictions, np),
        },
        "gust": {
            "rows": int(len(gust)),
            "spots": sorted(gust["spot_id"].astype(str).unique().tolist()),
            "stage_metrics": stage_metrics(gust, "actual_gust_ms", gust_predictions, np),
        },
    }
    result["wind"]["improvements"] = {
        "raw_to_corrected_rmse_gain_pct": improvement(
            result["wind"]["stage_metrics"]["raw_wind_mean_ms"],
            result["wind"]["stage_metrics"]["corrected_wind_mean_ms"],
        ),
        "corrected_to_calibrated_rmse_gain_pct": improvement(
            result["wind"]["stage_metrics"]["corrected_wind_mean_ms"],
            result["wind"]["stage_metrics"]["calibrated_wind_mean_ms"],
        ),
        "raw_to_calibrated_rmse_gain_pct": improvement(
            result["wind"]["stage_metrics"]["raw_wind_mean_ms"],
            result["wind"]["stage_metrics"]["calibrated_wind_mean_ms"],
        ),
    }
    result["gust"]["improvements"] = {
        "raw_to_corrected_rmse_gain_pct": improvement(
            result["gust"]["stage_metrics"]["raw_gust_ms"],
            result["gust"]["stage_metrics"]["corrected_gust_ms"],
        ),
        "corrected_to_calibrated_rmse_gain_pct": improvement(
            result["gust"]["stage_metrics"]["corrected_gust_ms"],
            result["gust"]["stage_metrics"]["calibrated_gust_ms"],
        ),
        "raw_to_calibrated_rmse_gain_pct": improvement(
            result["gust"]["stage_metrics"]["raw_gust_ms"],
            result["gust"]["stage_metrics"]["calibrated_gust_ms"],
        ),
    }

    for label, frame, prediction, target, prefix, thresholds in [
        ("wind", wind, "calibrated_wind_mean_ms", "actual_wind_mean_ms", "wind", [12, 15, 20, 25]),
        ("gust", gust, "calibrated_gust_ms", "actual_gust_ms", "gust", [15, 20, 25, 30]),
    ]:
        result[label]["tail_to_rmse_0_9"] = tail(frame, prediction, target, args.target_rmse_ms, np)
        result[label]["rowwise_existing_oracle"] = rowwise_existing_oracle(
            frame,
            [column for column in (wind_predictions if label == "wind" else gust_predictions) if column in frame.columns],
            target,
            np,
        )
        result[label]["bias_oracles_in_sample_diagnostic"] = bias_oracles(
            frame,
            prediction,
            target,
            [
                ["spot_id"],
                ["lead_time_minutes"],
                ["spot_id", "lead_time_minutes"],
                ["spot_id", "lead_time_minutes", "target_hour_local"],
                ["spot_id", "lead_time_minutes", f"{prefix}_actual_bin_ms"],
            ],
            np,
        )
        result[label]["perfect_segment_oracles"] = perfect_segment_oracles(frame, prediction, target, prefix, np)
        result[label]["threshold_metrics"] = threshold_metrics(frame, prediction, target, thresholds, np)
        result[label]["groups"] = {
            "by_spot": group_metrics(frame, ["spot_id"], prediction, target, np, pd),
            "by_lead": group_metrics(frame, ["lead_time_minutes"], prediction, target, np, pd),
            "by_actual_bin_ms": group_metrics(frame, [f"{prefix}_actual_bin_ms"], prediction, target, np, pd),
            "by_actual_bin_kt": group_metrics(frame, [f"{prefix}_actual_bin_kt"], prediction, target, np, pd),
            "by_spot_lead": group_metrics(frame, ["spot_id", "lead_time_minutes"], prediction, target, np, pd, limit=16),
            "by_target_hour_local": group_metrics(frame, ["target_hour_local"], prediction, target, np, pd, limit=24),
        }

    paths = sorted(Path(path) for path in glob.glob(args.training_glob))
    result["label_audit"] = label_audit(paths, pd, np)
    result["training_source_coverage"] = training_source_coverage(paths, pd)
    result["feature_coverage_current_wind_frame"] = feature_coverage(wind)
    return result


def fmt_metric(item: dict[str, Any]) -> str:
    if not item or not item.get("count"):
        return "n/a"
    return f"{item['rmse_ms']:.3f} m/s ({item['rmse_kt']:.2f} kt), MAE {item['mae_ms']:.3f}, bias {item['bias_ms']:+.3f}"


def table_group(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| Group | Rows | RMSE | MAE | Bias | SSE share | RMSE if perfect |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        group = ", ".join(f"{key}={value}" for key, value in row["group"].items())
        lines.append(
            f"| `{group}` | {row['count']} | {row['rmse_ms']:.3f} | {row['mae_ms']:.3f} | "
            f"{row['bias_ms']:+.3f} | {row['sse_share_pct']:.2f}% | {row['global_rmse_if_perfect_ms']:.3f} |"
        )
    return lines


def table_bias_oracles(items: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["| Oracle | RMSE | MAE | Bias | Rows |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in items.items():
        if not row or not row.get("count"):
            continue
        lines.append(
            f"| `{name}` | {row['rmse_ms']:.3f} | {row['mae_ms']:.3f} | {row['bias_ms']:+.3f} | {row['count']} |"
        )
    return lines


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    wind = result["wind"]
    gust = result["gust"]
    label = result["label_audit"]
    wind_champ = wind["stage_metrics"]["calibrated_wind_mean_ms"]
    gust_champ = gust["stage_metrics"]["calibrated_gust_ms"]
    wind_oracle = wind["rowwise_existing_oracle"]
    gust_oracle = gust["rowwise_existing_oracle"]

    lines = [
        "# CorseWind Scientific Error Floor Audit",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        "## Executive Verdict",
        "",
        f"- Current wind champion: {fmt_metric(wind_champ)} on {wind_champ['count']} rows.",
        f"- Current gust champion: {fmt_metric(gust_champ)} on {gust_champ['count']} rows.",
        f"- Wind RMSE 0.9 needs {wind['tail_to_rmse_0_9']['mse_reduction_needed_pct']}% MSE reduction; top 5% rows carry {wind['tail_to_rmse_0_9']['top_5_pct_sse_share_pct']}% of SSE.",
        f"- Gust RMSE 0.9 needs {gust['tail_to_rmse_0_9']['mse_reduction_needed_pct']}% MSE reduction; top 5% rows carry {gust['tail_to_rmse_0_9']['top_5_pct_sse_share_pct']}% of SSE.",
        f"- Existing wind row-oracle across raw/base/calibrated reaches {wind_oracle['rmse_ms']:.3f} m/s, so current model variants alone do not prove a path to 0.9.",
        f"- Existing gust row-oracle across raw/base/calibrated reaches {gust_oracle['rmse_ms']:.3f} m/s.",
        "",
        "## Champion Stage Metrics",
        "",
        "| Target | Raw | Base corrected | Calibrated champion | Raw to calibrated gain |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| wind_mean | {fmt_metric(wind['stage_metrics']['raw_wind_mean_ms'])} | {fmt_metric(wind['stage_metrics']['corrected_wind_mean_ms'])} | {fmt_metric(wind_champ)} | {wind['improvements']['raw_to_calibrated_rmse_gain_pct']}% |",
        f"| gust | {fmt_metric(gust['stage_metrics']['raw_gust_ms'])} | {fmt_metric(gust['stage_metrics']['corrected_gust_ms'])} | {fmt_metric(gust_champ)} | {gust['improvements']['raw_to_calibrated_rmse_gain_pct']}% |",
        "",
        "## Error Concentration",
        "",
        "| Target | Rows needed perfect to hit 0.9 | Top 1% SSE | Top 5% SSE | Top 10% SSE | Top 20% SSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| wind_mean | {wind['tail_to_rmse_0_9']['perfect_rows_needed_pct']}% | {wind['tail_to_rmse_0_9']['top_1_pct_sse_share_pct']}% | {wind['tail_to_rmse_0_9']['top_5_pct_sse_share_pct']}% | {wind['tail_to_rmse_0_9']['top_10_pct_sse_share_pct']}% | {wind['tail_to_rmse_0_9']['top_20_pct_sse_share_pct']}% |",
        f"| gust | {gust['tail_to_rmse_0_9']['perfect_rows_needed_pct']}% | {gust['tail_to_rmse_0_9']['top_1_pct_sse_share_pct']}% | {gust['tail_to_rmse_0_9']['top_5_pct_sse_share_pct']}% | {gust['tail_to_rmse_0_9']['top_10_pct_sse_share_pct']}% | {gust['tail_to_rmse_0_9']['top_20_pct_sse_share_pct']}% |",
        "",
        "## Diagnostic Bias Oracles",
        "",
        "These use observed 2026 labels to remove grouped mean residuals. They are diagnostic upper bounds, not deployable models.",
        "",
        "### Wind",
        "",
        *table_bias_oracles(wind["bias_oracles_in_sample_diagnostic"]),
        "",
        "### Gust",
        "",
        *table_bias_oracles(gust["bias_oracles_in_sample_diagnostic"]),
        "",
        "## Wind Hard Groups",
        "",
        "### By Spot",
        "",
        *table_group(wind["groups"]["by_spot"][:8], ["spot_id"]),
        "",
        "### By Actual Wind Bin",
        "",
        *table_group(wind["groups"]["by_actual_bin_ms"], ["wind_actual_bin_ms"]),
        "",
        "### By Spot + Lead",
        "",
        *table_group(wind["groups"]["by_spot_lead"][:10], ["spot_id", "lead_time_minutes"]),
        "",
        "## Gust Hard Groups",
        "",
        "### By Spot",
        "",
        *table_group(gust["groups"]["by_spot"][:8], ["spot_id"]),
        "",
        "### By Actual Gust Bin",
        "",
        *table_group(gust["groups"]["by_actual_bin_ms"], ["gust_actual_bin_ms"]),
        "",
        "## Windsurf Thresholds",
        "",
        "| Target | Threshold | Actual event rate | Pred event rate | Precision | Recall | F1 | FN | FP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target_name, metrics in [("wind", wind["threshold_metrics"]), ("gust", gust["threshold_metrics"])]:
        for threshold, row in metrics.items():
            lines.append(
                f"| {target_name} | `{threshold}` | {row['actual_event_rate_pct']}% | {row['predicted_event_rate_pct']}% | "
                f"{row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['fn']} | {row['fp']} |"
            )

    lines.extend(
        [
            "",
            "## Label And Observation Diagnostics",
            "",
            f"- Training parquet files audited: `{label.get('path_count')}`.",
            f"- Label rows audited: `{label.get('row_count')}` across `{label.get('spot_count')}` spots.",
            f"- Target observation distance minutes: `{label.get('target_observation_distance_minutes')}`.",
            f"- Duplicate target consistency: `{label.get('duplicate_target_consistency')}`.",
            f"- Short-term target volatility <=20 min: `{label.get('target_short_term_volatility_le_20min')}`.",
            f"- Training source coverage: `{result.get('training_source_coverage')}`.",
            "",
            "## Feature Coverage Snapshot",
            "",
            "| Concept | Present | Columns | Any non-null | Top columns |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for concept, item in result["feature_coverage_current_wind_frame"].items():
        top = ", ".join(f"`{col['column']}` {col['non_null_pct']}%" for col in item["top_columns"][:3])
        lines.append(
            f"| `{concept}` | `{item['present']}` | {item['column_count']} | {item['any_non_null_pct']}% | {top} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The current champions already remove most of the raw AROME/Open-Meteo error, especially for gusts; the remaining gap is concentrated in a limited set of high-energy rows and south-coast spots.",
            "- Static bias correction is not the missing magic lever: even diagnostic in-sample bias oracles should be treated as upper bounds, not production evidence.",
            "- The most credible next gains are not more blind model families; they are denser fresh observations, better strong-wind/thermal event labeling, candidate models with genuinely different errors, and probabilistic threshold heads.",
            "- If the label audit shows large observation-distance or short-term volatility, the RMSE floor may be partly imposed by target noise and by trying to predict a 6-15 minute turbulent signal with point labels.",
            "",
            "## Recommended Next Steps",
            "",
            "1. Build an event-weighted evaluation set for thermal start, thermal collapse, and strong wind bins >=12/15/20/25 kt; optimize these explicitly alongside RMSE.",
            "2. Add a label-quality gate: exclude or down-weight targets with stale observations, inconsistent duplicate labels, or extreme 15-minute jumps during baseline training.",
            "3. Train specialist heads for high-wind and threshold probability, but promote only if they improve both RMSE and recall/precision on windsurf thresholds.",
            "4. Keep collecting live data; the current 2026 test window is too short to prove subtle feature gains, especially for rare regimes.",
            "5. Re-run foundation/model-router work only when candidate predictions have dense overlap; sparse oracle wins are not enough for production promotion.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind-predictions", type=Path, default=DEFAULT_WIND_PREDICTIONS)
    parser.add_argument("--gust-predictions", type=Path, default=DEFAULT_GUST_PREDICTIONS)
    parser.add_argument("--training-glob", default=DEFAULT_TRAINING_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-rmse-ms", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "error_floor_audit.json"
    md_path = args.output_dir / "error_floor_audit.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, result)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
