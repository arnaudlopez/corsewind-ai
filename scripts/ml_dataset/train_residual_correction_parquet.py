#!/usr/bin/env python3
"""Train residual-correction models from monthly Parquet training shards."""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGRESSION_TARGETS = (
    "labels__residual_wind_mean_ms",
    "labels__residual_gust_ms",
)
CLASSIFICATION_PREFIXES = (
    "labels__target_wind_gt_",
    "labels__target_gust_gt_",
)
METRIC_LABEL_COLUMNS = (
    "labels__target_wind_mean_ms",
    "labels__target_gust_ms",
)
REGRESSION_TARGET_METADATA = {
    "labels__residual_wind_mean_ms": {
        "baseline_feature": "baselines__baseline_wind_mean_ms",
        "observed_label": "labels__target_wind_mean_ms",
    },
    "labels__residual_gust_ms": {
        "baseline_feature": "baselines__baseline_gust_ms",
        "observed_label": "labels__target_gust_ms",
    },
}
KNOTS_PER_MS = 1.9438444924406
DEFAULT_OBSERVED_REGIME_THRESHOLDS_KT = (12.0, 15.0, 20.0, 25.0)
BASE_FEATURE_COLUMNS = {
    "spot_id",
    "spot_kind",
    "spot_source_type",
    "station_id",
    "latitude",
    "longitude",
    "lead_time_minutes",
}
FORCED_CATEGORICAL_COLUMNS = {
    "spot_id",
    "spot_kind",
    "spot_source_type",
    "station_id",
    "baselines__baseline_model",
}
DERIVED_TIME_COLUMNS = (
    "issue_hour_utc",
    "issue_month",
    "issue_dayofyear_sin",
    "issue_dayofyear_cos",
)


def import_dependencies():
    try:
        import joblib
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OrdinalEncoder
    except ImportError as exc:
        raise SystemExit(
            "Missing ML dependencies. Run inside the ml dataset runner image or "
            "install requirements-ml-dataset.txt."
        ) from exc
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError:
        LGBMClassifier = None
        LGBMRegressor = None
    return {
        "joblib": joblib,
        "np": np,
        "pd": pd,
        "pq": pq,
        "ColumnTransformer": ColumnTransformer,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "ExtraTreesRegressor": ExtraTreesRegressor,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "SimpleImputer": SimpleImputer,
        "accuracy_score": accuracy_score,
        "brier_score_loss": brier_score_loss,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "Pipeline": Pipeline,
        "OrdinalEncoder": OrdinalEncoder,
        "LGBMClassifier": LGBMClassifier,
        "LGBMRegressor": LGBMRegressor,
    }


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


def discover_parquet_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    paths.extend(args.training_parquet or [])
    if args.training_table_root:
        if args.start_month and args.end_month:
            suffixes = set(month_range(args.start_month, args.end_month))
            for suffix in suffixes:
                candidate = args.training_table_root / f"{args.run_id_prefix}_{suffix}" / "training_rows.parquet"
                if candidate.exists():
                    paths.append(candidate)
        else:
            pattern = f"{args.run_id_prefix}_20*_??/training_rows.parquet"
            paths.extend(sorted(args.training_table_root.glob(pattern)))
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise SystemExit("No Parquet training shards found.")
    missing = [str(path) for path in unique if not path.exists()]
    if missing:
        raise SystemExit(f"Missing Parquet shards: {missing[:5]}")
    return unique


def schema_columns(paths: list[Path], pq: Any) -> list[str]:
    columns: set[str] = set()
    for path in paths:
        columns.update(pq.ParquetFile(path).schema.names)
    return sorted(columns)


def is_allowed_feature_column(column: str, include_issue_source_flags: bool) -> bool:
    if column in BASE_FEATURE_COLUMNS:
        return True
    if column.startswith("features__"):
        return True
    if column.startswith("baselines__"):
        return True
    if include_issue_source_flags and column.startswith("issue_feature_sources__"):
        return True
    return False


def pattern_matches(column: str, pattern: str) -> bool:
    if pattern.startswith("re:"):
        return re.search(pattern[3:], column) is not None
    return pattern in column


def passes_feature_filters(column: str, args: argparse.Namespace) -> bool:
    if args.include_feature_pattern and not any(pattern_matches(column, pattern) for pattern in args.include_feature_pattern):
        return False
    if args.exclude_feature_pattern and any(pattern_matches(column, pattern) for pattern in args.exclude_feature_pattern):
        return False
    return True


def is_target_column(column: str) -> bool:
    return column in REGRESSION_TARGETS or any(column.startswith(prefix) for prefix in CLASSIFICATION_PREFIXES)


def lead_filter_mask(frame: Any, lead_minutes: list[int]) -> Any:
    if not lead_minutes or "lead_time_minutes" not in frame.columns:
        return None
    return frame["lead_time_minutes"].astype("Int64").isin(set(int(lead) for lead in lead_minutes))


def compute_split(
    paths: list[Path],
    pq: Any,
    test_fraction: float,
    split_time_utc: str | None,
    lead_minutes: list[int],
) -> tuple[str | None, dict[str, int]]:
    issue_times: list[str] = []
    total_rows = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        if "issue_time_utc" not in pf.schema.names:
            continue
        read_columns = ["issue_time_utc"]
        if lead_minutes and "lead_time_minutes" in pf.schema.names:
            read_columns.append("lead_time_minutes")
        frame = pf.read(columns=read_columns).to_pandas()
        mask = lead_filter_mask(frame, lead_minutes)
        if mask is not None:
            frame = frame[mask]
        values = [value for value in frame["issue_time_utc"].tolist() if value]
        total_rows += len(values)
        issue_times.extend(str(value) for value in values)
    unique_times = sorted(set(issue_times))
    if len(unique_times) < 2:
        return None, {"row_count": total_rows, "unique_issue_time_count": len(unique_times)}
    if split_time_utc:
        split_time = split_time_utc
    else:
        test_count = max(1, int(round(len(unique_times) * test_fraction)))
        split_time = unique_times[-test_count]
    train_rows = sum(1 for value in issue_times if value < split_time)
    test_rows = sum(1 for value in issue_times if value >= split_time)
    return split_time, {
        "row_count": total_rows,
        "unique_issue_time_count": len(unique_times),
        "pre_sample_train_rows": train_rows,
        "pre_sample_test_rows": test_rows,
    }


def sample_fraction(max_rows: int | None, row_count: int, oversample: float = 1.2) -> float:
    if not max_rows or max_rows <= 0 or row_count <= max_rows:
        return 1.0
    return min(1.0, max_rows / row_count * oversample)


def add_time_features(frame: Any, pd: Any, np: Any) -> Any:
    issue_time = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    dayofyear = issue_time.dt.dayofyear.fillna(1).astype(float)
    frame["issue_hour_utc"] = issue_time.dt.hour.astype("float64")
    frame["issue_month"] = issue_time.dt.month.astype("float64")
    angle = 2.0 * math.pi * dayofyear / 366.0
    frame["issue_dayofyear_sin"] = np.sin(angle)
    frame["issue_dayofyear_cos"] = np.cos(angle)
    return frame


def stable_sample(frame: Any, pd: Any, max_rows: int | None, fraction: float) -> Any:
    if frame.empty:
        return frame
    keys = frame[["spot_id", "issue_time_utc", "lead_time_minutes"]].astype(str).agg("|".join, axis=1)
    hashes = pd.util.hash_pandas_object(keys, index=False).astype("uint64")
    frame = frame.assign(__sample_hash=hashes)
    if fraction < 1.0:
        threshold = int(fraction * (2**64 - 1))
        frame = frame[frame["__sample_hash"] <= threshold]
    if max_rows and max_rows > 0 and len(frame) > max_rows:
        frame = frame.sort_values("__sample_hash").head(max_rows)
    return frame.drop(columns=["__sample_hash"])


def read_sampled_frames(
    paths: list[Path],
    columns: list[str],
    split_time: str | None,
    args: argparse.Namespace,
    deps: dict[str, Any],
    counts: dict[str, int],
) -> tuple[Any, Any]:
    pd = deps["pd"]
    np = deps["np"]
    pq = deps["pq"]
    train_frames = []
    test_frames = []
    train_fraction = sample_fraction(args.max_train_rows, counts.get("pre_sample_train_rows", 0))
    test_fraction = sample_fraction(args.max_test_rows, counts.get("pre_sample_test_rows", 0))
    read_columns = sorted(set(columns) | {"issue_time_utc"})
    for path in paths:
        pf = pq.ParquetFile(path)
        available = [column for column in read_columns if column in pf.schema.names]
        for batch in pf.iter_batches(batch_size=args.read_batch_size, columns=available):
            frame = batch.to_pandas().reindex(columns=read_columns)
            mask = lead_filter_mask(frame, args.include_lead_minute)
            if mask is not None:
                frame = frame[mask]
                if frame.empty:
                    continue
            frame = add_time_features(frame, pd, np)
            if split_time is None:
                train_part = frame
                test_part = frame.iloc[0:0].copy()
            else:
                train_mask = frame["issue_time_utc"].astype(str) < split_time
                train_part = frame[train_mask]
                test_part = frame[~train_mask]
            if not train_part.empty:
                train_frames.append(stable_sample(train_part, pd, args.max_train_rows, train_fraction))
            if not test_part.empty:
                test_frames.append(stable_sample(test_part, pd, args.max_test_rows, test_fraction))
    train = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame(columns=read_columns)
    test = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame(columns=read_columns)
    train = stable_sample(train, pd, args.max_train_rows, 1.0)
    test = stable_sample(test, pd, args.max_test_rows, 1.0)
    return train, test


def infer_feature_columns(
    train: Any,
    candidate_columns: list[str],
    min_non_null_ratio: float,
    min_non_null_count: int,
    max_categorical_cardinality: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    import pandas as pd

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    dropped: dict[str, Any] = {
        "too_sparse": [],
        "constant": [],
        "high_cardinality_categorical": [],
    }
    row_count = len(train)
    for column in candidate_columns:
        if column not in train.columns:
            dropped["too_sparse"].append({"column": column, "reason": "missing_from_sample"})
            continue
        series = train[column]
        non_null_count = int(series.notna().sum())
        non_null_ratio = non_null_count / row_count if row_count else 0.0
        if non_null_count < min_non_null_count or non_null_ratio < min_non_null_ratio:
            dropped["too_sparse"].append({
                "column": column,
                "non_null_count": non_null_count,
                "non_null_ratio": round(non_null_ratio, 8),
            })
            continue
        unique_count = int(series.dropna().nunique())
        if unique_count <= 1:
            dropped["constant"].append({"column": column, "unique_count": unique_count})
            continue
        if column in FORCED_CATEGORICAL_COLUMNS:
            if unique_count > max_categorical_cardinality:
                dropped["high_cardinality_categorical"].append({"column": column, "unique_count": unique_count})
                continue
            categorical_columns.append(column)
        else:
            sample = series.dropna().head(1000)
            converted = pd.to_numeric(sample, errors="coerce")
            is_numeric = bool(len(sample)) and bool(converted.notna().all())
            if not is_numeric:
                if unique_count > max_categorical_cardinality:
                    dropped["high_cardinality_categorical"].append({"column": column, "unique_count": unique_count})
                    continue
                categorical_columns.append(column)
            else:
                numeric_columns.append(column)
    return sorted(numeric_columns), sorted(categorical_columns), dropped


def make_preprocessor(deps: dict[str, Any], numeric_columns: list[str], categorical_columns: list[str]):
    transformers = []
    if numeric_columns:
        transformers.append((
            "numeric",
            deps["Pipeline"]([
                ("imputer", deps["SimpleImputer"](strategy="median")),
            ]),
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


def regression_metrics(deps: dict[str, Any], y_true: Any, y_pred: Any) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"count": 0}
    rmse = math.sqrt(deps["mean_squared_error"](y_true, y_pred))
    return {
        "count": int(len(y_true)),
        "mae": round(float(deps["mean_absolute_error"](y_true, y_pred)), 6),
        "rmse": round(float(rmse), 6),
        "bias": round(float((y_pred - y_true).mean()), 6),
    }


def regression_sample_weights(frame: Any, mask: Any, observed_label_column: str, args: argparse.Namespace, np: Any) -> tuple[Any | None, dict[str, Any]]:
    rules = parse_high_wind_weight_rules(args)
    if not rules and (not args.target_high_wind_weight_threshold_ms or args.target_high_wind_weight <= 1.0):
        return None, {"enabled": False}
    observed = frame.loc[mask, observed_label_column].astype(float)
    weights = np.ones(len(observed), dtype="float64")
    applied_rules = []
    if rules:
        for threshold_ms, weight, source in rules:
            high_mask = observed >= threshold_ms
            weights[high_mask.to_numpy()] = np.maximum(weights[high_mask.to_numpy()], float(weight))
            applied_rules.append({
                "threshold_ms": round(float(threshold_ms), 6),
                "threshold_kt": round(float(threshold_ms) * KNOTS_PER_MS, 6),
                "weight": float(weight),
                "source": source,
                "row_count": int(high_mask.sum()),
            })
    else:
        high_mask = observed >= float(args.target_high_wind_weight_threshold_ms)
        weights[high_mask.to_numpy()] = float(args.target_high_wind_weight)
        applied_rules.append({
            "threshold_ms": float(args.target_high_wind_weight_threshold_ms),
            "threshold_kt": round(float(args.target_high_wind_weight_threshold_ms) * KNOTS_PER_MS, 6),
            "weight": float(args.target_high_wind_weight),
            "source": "legacy_single_threshold_ms",
            "row_count": int(high_mask.sum()),
        })
    return weights, {
        "enabled": True,
        "observed_label_column": observed_label_column,
        "rules": applied_rules,
        "row_count": int(len(weights)),
        "mean_weight": round(float(weights.mean()), 6) if len(weights) else None,
        "max_weight": round(float(weights.max()), 6) if len(weights) else None,
    }


def parse_threshold_weight(value: str, *, unit: str) -> tuple[float, float]:
    if ":" not in value:
        raise SystemExit(f"Invalid weight rule {value!r}; expected THRESHOLD:WEIGHT.")
    threshold_text, weight_text = value.split(":", 1)
    threshold = float(threshold_text)
    weight = float(weight_text)
    if weight < 1.0:
        raise SystemExit(f"Invalid weight rule {value!r}; weight must be >= 1.")
    threshold_ms = threshold / KNOTS_PER_MS if unit == "kt" else threshold
    return threshold_ms, weight


def parse_high_wind_weight_rules(args: argparse.Namespace) -> list[tuple[float, float, str]]:
    rules: list[tuple[float, float, str]] = []
    for value in args.target_high_wind_weight_rule_ms:
        threshold_ms, weight = parse_threshold_weight(value, unit="ms")
        rules.append((threshold_ms, weight, value))
    for value in args.target_high_wind_weight_rule_kt:
        threshold_ms, weight = parse_threshold_weight(value, unit="kt")
        rules.append((threshold_ms, weight, value))
    rules.sort(key=lambda item: item[0])
    return rules


def prediction_metrics(np: Any, predictions: Any, observations: Any) -> dict[str, Any]:
    if len(predictions) == 0:
        return {"count": 0}
    errors = predictions - observations
    return {
        "count": int(len(errors)),
        "mae": round(float(np.mean(np.abs(errors))), 6),
        "rmse": round(float(np.sqrt(np.mean(errors * errors))), 6),
        "bias": round(float(np.mean(errors)), 6),
    }


def metrics_by_lead(np: Any, frame: Any, prediction_column: str, observation_column: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    valid = frame[[prediction_column, observation_column, "lead_time_minutes"]].dropna()
    for lead, group in valid.groupby("lead_time_minutes"):
        out[str(int(lead))] = prediction_metrics(np, group[prediction_column].to_numpy(), group[observation_column].to_numpy())
    return out


def metrics_by_columns(
    np: Any,
    pd: Any,
    frame: Any,
    prediction_column: str,
    observation_column: str,
    group_columns: list[str],
) -> dict[str, Any]:
    available_columns = [column for column in group_columns if column in frame.columns]
    if not available_columns:
        return {}
    valid = frame[[prediction_column, observation_column, *available_columns]].dropna(
        subset=[prediction_column, observation_column]
    )
    out: dict[str, Any] = {}
    for raw_key, group in valid.groupby(available_columns, dropna=False):
        values = group_key_tuple(raw_key, len(available_columns))
        key_parts = []
        for column, value in zip(available_columns, values):
            key_parts.append(f"{column}={safe_path_token(value)}")
        out["|".join(key_parts)] = {
            "group": dict(zip(available_columns, [json_safe_scalar(pd, value) for value in values])),
            **prediction_metrics(np, group[prediction_column].to_numpy(), group[observation_column].to_numpy()),
        }
    return dict(sorted(out.items()))


def observed_regime_label(value_ms: float, thresholds_kt: list[float]) -> str:
    value_kt = value_ms * KNOTS_PER_MS
    ordered = sorted(thresholds_kt)
    previous = None
    for threshold in ordered:
        if value_kt < threshold:
            return f"<{threshold:g}kt" if previous is None else f"{previous:g}-{threshold:g}kt"
        previous = threshold
    return f">={ordered[-1]:g}kt" if ordered else "all"


def metrics_by_observed_regime(
    np: Any,
    frame: Any,
    prediction_column: str,
    observation_column: str,
    thresholds_kt: list[float],
) -> dict[str, Any]:
    if not thresholds_kt:
        return {}
    valid = frame[[prediction_column, observation_column]].dropna().copy()
    if valid.empty:
        return {}
    valid["observed_regime_kt"] = valid[observation_column].astype(float).map(
        lambda value: observed_regime_label(value, thresholds_kt)
    )
    out: dict[str, Any] = {}
    for regime, group in valid.groupby("observed_regime_kt", dropna=False):
        out[str(regime)] = prediction_metrics(
            np,
            group[prediction_column].to_numpy(),
            group[observation_column].to_numpy(),
        )
    return dict(sorted(out.items()))


def threshold_detection_metrics(frame: Any, prediction_column: str, observation_column: str, threshold_kt: float) -> dict[str, Any]:
    valid = frame[[prediction_column, observation_column]].dropna()
    if valid.empty:
        return {"count": 0}
    threshold_ms = threshold_kt / KNOTS_PER_MS
    predicted = valid[prediction_column].astype(float) >= threshold_ms
    actual = valid[observation_column].astype(float) >= threshold_ms
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())
    return {
        "count": int(len(valid)),
        "threshold_kt": float(threshold_kt),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": None if tp + fp == 0 else round(tp / (tp + fp), 6),
        "recall": None if tp + fn == 0 else round(tp / (tp + fn), 6),
        "csi": None if tp + fp + fn == 0 else round(tp / (tp + fp + fn), 6),
    }


def threshold_detection_summary(
    frame: Any,
    prediction_column: str,
    observation_column: str,
    thresholds_kt: list[float],
) -> dict[str, Any]:
    return {
        f">={threshold:g}kt": threshold_detection_metrics(frame, prediction_column, observation_column, threshold)
        for threshold in thresholds_kt
    }


def prediction_metrics_for_leads(
    np: Any,
    frame: Any,
    prediction_column: str,
    observation_column: str,
    leads: list[int],
) -> dict[str, Any] | None:
    if not leads:
        return None
    valid = frame[[prediction_column, observation_column, "lead_time_minutes"]].dropna()
    subset = valid[valid["lead_time_minutes"].astype(int).isin(leads)]
    metrics = prediction_metrics(np, subset[prediction_column].to_numpy(), subset[observation_column].to_numpy())
    return {
        "lead_time_minutes": sorted(set(int(lead) for lead in leads)),
        **metrics,
    }


def classification_metrics(deps: dict[str, Any], y_true: Any, probabilities: Any) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"count": 0}
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "count": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "positive_rate": round(float(y_true.mean()), 6),
        "accuracy": round(float(deps["accuracy_score"](y_true, predictions)), 6),
        "brier": round(float(deps["brier_score_loss"](y_true, probabilities)), 6),
    }


def build_model(deps: dict[str, Any], model_family: str, target_type: str, args: argparse.Namespace):
    if model_family == "extra_trees":
        cls = deps["ExtraTreesRegressor"] if target_type == "regression" else deps["ExtraTreesClassifier"]
        return cls(
            n_estimators=args.max_iter,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
            min_samples_leaf=args.min_samples_leaf,
        )
    if model_family == "lightgbm":
        cls = deps["LGBMRegressor"] if target_type == "regression" else deps["LGBMClassifier"]
        if cls is None:
            raise SystemExit("LightGBM is not installed in this Python environment.")
        return cls(
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
    if target_type == "regression":
        return deps["HistGradientBoostingRegressor"](
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2_regularization,
            random_state=args.random_seed,
        )
    return deps["HistGradientBoostingClassifier"](
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        random_state=args.random_seed,
    )


def group_key_tuple(key: Any, column_count: int) -> tuple[Any, ...]:
    if column_count == 1:
        if isinstance(key, tuple) and len(key) == 1:
            return key
        return (key,)
    return tuple(key)


def safe_path_token(value: Any) -> str:
    text = "missing" if value is None else str(value)
    text = text.replace("/", "_").replace("\\", "_")
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_") or "value"


def json_safe_scalar(pd: Any, value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def group_model_path(output_root: Path, target: str, group_columns: list[str], group_values: tuple[Any, ...]) -> Path:
    parts = [f"{safe_path_token(column)}={safe_path_token(value)}" for column, value in zip(group_columns, group_values)]
    return output_root / f"{safe_path_token(target)}__{'__'.join(parts)}.joblib"


def group_mask(frame: Any, pd: Any, group_columns: list[str], group_values: tuple[Any, ...]) -> Any:
    mask = None
    for column, value in zip(group_columns, group_values):
        if pd.isna(value):
            part = frame[column].isna()
        else:
            part = frame[column] == value
        mask = part if mask is None else (mask & part)
    return mask


def train_models(train: Any, test: Any, feature_columns: list[str], categorical_columns: list[str], args: argparse.Namespace, deps: dict[str, Any]) -> dict[str, Any]:
    np = deps["np"]
    pd = deps["pd"]
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    preprocessor = make_preprocessor(deps, numeric_columns, categorical_columns)
    targets = [target for target in REGRESSION_TARGETS if target in train.columns]
    targets.extend(
        sorted(
            column
            for column in train.columns
            if any(column.startswith(prefix) for prefix in CLASSIFICATION_PREFIXES)
        )
    )
    if args.only_target:
        allowed = set(args.only_target)
        targets = [target for target in targets if target in allowed]
    if args.skip_classification:
        targets = [target for target in targets if target in REGRESSION_TARGETS]

    results: dict[str, Any] = {
        "models": {},
        "skipped_targets": {},
    }
    x_train_all = train[feature_columns]
    x_test_all = test[feature_columns]

    for target in targets:
        is_regression = target in REGRESSION_TARGETS
        target_type = "regression" if is_regression else "classification"
        train_mask = train[target].notna()
        test_mask = test[target].notna()
        if not is_regression:
            train_mask = train_mask & train[target].isin([0, 1])
            test_mask = test_mask & test[target].isin([0, 1])
            classes = Counter(int(value) for value in train.loc[train_mask, target].tolist())
            if len(classes) < 2:
                results["skipped_targets"][target] = f"single_training_class_{dict(classes)}"
                continue
        if int(train_mask.sum()) < args.min_target_train_rows or int(test_mask.sum()) < args.min_target_test_rows:
            results["skipped_targets"][target] = {
                "reason": "not_enough_train_or_test_rows",
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
            }
            continue
        model = deps["Pipeline"]([
            ("preprocess", preprocessor),
            ("model", build_model(deps, args.model_family, target_type, args)),
        ])
        x_train = x_train_all.loc[train_mask]
        x_test = x_test_all.loc[test_mask]
        y_train = train.loc[train_mask, target]
        y_test = test.loc[test_mask, target]
        if not is_regression:
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
        if is_regression:
            fit_group_columns = [column for column in args.fit_group_column if column in train.columns and column in test.columns]
            skipped_group_columns = [column for column in args.fit_group_column if column not in train.columns or column not in test.columns]
            group_results = []
            skipped_groups = []
            metadata = REGRESSION_TARGET_METADATA[target]
            train_sample_weights, sample_weight_summary = regression_sample_weights(
                train,
                train_mask,
                metadata["observed_label"],
                args,
                np,
            )
            if fit_group_columns:
                prediction_series = pd.Series(index=x_test.index, dtype="float64")
                train_groups = train.loc[train_mask, fit_group_columns].copy()
                train_groups["__row_index"] = train_groups.index
                train_weight_series = None
                if train_sample_weights is not None:
                    train_weight_series = pd.Series(train_sample_weights, index=train.index[train_mask])
                for raw_key, group in train_groups.groupby(fit_group_columns, dropna=False):
                    group_values = group_key_tuple(raw_key, len(fit_group_columns))
                    train_indices = group["__row_index"]
                    matching_test_mask = test_mask & group_mask(test, pd, fit_group_columns, group_values)
                    test_indices = test.index[matching_test_mask]
                    if len(train_indices) < args.min_group_train_rows or len(test_indices) < args.min_group_test_rows:
                        skipped_groups.append({
                            "group": dict(zip(fit_group_columns, [json_safe_scalar(pd, value) for value in group_values])),
                            "train_rows": int(len(train_indices)),
                            "test_rows": int(len(test_indices)),
                            "reason": "not_enough_group_rows",
                        })
                        continue
                    group_model = deps["Pipeline"]([
                        ("preprocess", make_preprocessor(deps, numeric_columns, categorical_columns)),
                        ("model", build_model(deps, args.model_family, target_type, args)),
                    ])
                    fit_kwargs = {}
                    if train_weight_series is not None:
                        fit_kwargs["model__sample_weight"] = train_weight_series.loc[train_indices].to_numpy()
                    group_model.fit(x_train_all.loc[train_indices], train.loc[train_indices, target], **fit_kwargs)
                    group_predictions = group_model.predict(x_test_all.loc[test_indices])
                    prediction_series.loc[test_indices] = group_predictions
                    model_path = group_model_path(output_root, target, fit_group_columns, group_values)
                    deps["joblib"].dump(group_model, model_path)
                    group_results.append({
                        "group": dict(zip(fit_group_columns, [json_safe_scalar(pd, value) for value in group_values])),
                        "model_path": str(model_path),
                        "train_rows": int(len(train_indices)),
                        "test_rows": int(len(test_indices)),
                    })
                predicted_mask = prediction_series.notna()
                if int(predicted_mask.sum()) < args.min_target_test_rows:
                    results["skipped_targets"][target] = {
                        "reason": "not_enough_group_predictions",
                        "predicted_test_rows": int(predicted_mask.sum()),
                        "fit_group_columns": fit_group_columns,
                        "skipped_group_columns": skipped_group_columns,
                        "skipped_groups": skipped_groups[:50],
                    }
                    continue
                predicted_indices = prediction_series.index[predicted_mask]
                residual_predictions = prediction_series.loc[predicted_indices].to_numpy()
                eval_test = test.loc[predicted_indices]
                y_test_eval = eval_test[target]
                model_path = None
            else:
                fit_kwargs = {}
                if train_sample_weights is not None:
                    fit_kwargs["model__sample_weight"] = train_sample_weights
                model.fit(x_train, y_train, **fit_kwargs)
                model_path = output_root / f"{target}.joblib"
                deps["joblib"].dump(model, model_path)
                residual_predictions = model.predict(x_test)
                eval_test = test.loc[test_mask]
                y_test_eval = y_test
                group_results = []
                skipped_groups = []
                skipped_group_columns = []
            residual_metrics = regression_metrics(deps, y_test_eval.to_numpy(), residual_predictions)
            baseline = eval_test[metadata["baseline_feature"]].astype(float)
            observed = eval_test[metadata["observed_label"]].astype(float)
            valid = baseline.notna() & observed.notna()
            eval_columns = [column for column in ("lead_time_minutes", "spot_id", "station_id", "spot_kind") if column in eval_test.columns]
            eval_frame = eval_test[eval_columns].copy()
            eval_frame["raw_prediction"] = baseline
            eval_frame["corrected_prediction"] = baseline + residual_predictions
            eval_frame["observation"] = observed
            raw_metrics = prediction_metrics(np, eval_frame.loc[valid, "raw_prediction"].to_numpy(), eval_frame.loc[valid, "observation"].to_numpy())
            corrected_metrics = prediction_metrics(
                np,
                eval_frame.loc[valid, "corrected_prediction"].to_numpy(),
                eval_frame.loc[valid, "observation"].to_numpy(),
            )
            raw_subset_metrics = prediction_metrics_for_leads(
                np,
                eval_frame,
                "raw_prediction",
                "observation",
                args.eval_lead_minute,
            )
            corrected_subset_metrics = prediction_metrics_for_leads(
                np,
                eval_frame,
                "corrected_prediction",
                "observation",
                args.eval_lead_minute,
            )
            results["models"][target] = {
                "type": "regression",
                "model_family": args.model_family,
                "model_path": str(model_path) if model_path else None,
                "fit_group_columns": fit_group_columns,
                "fit_groups": group_results,
                "skipped_fit_group_columns": skipped_group_columns,
                "skipped_fit_groups": skipped_groups,
                "sample_weighting": sample_weight_summary,
                "train_rows": int(train_mask.sum()),
                "test_rows": int(len(eval_test)),
                "residual_test": residual_metrics,
                "raw_nwp_test": raw_metrics,
                "corrected_nwp_test": corrected_metrics,
                "corrected_nwp_by_lead": metrics_by_lead(np, eval_frame, "corrected_prediction", "observation"),
                "raw_nwp_by_lead": metrics_by_lead(np, eval_frame, "raw_prediction", "observation"),
                "corrected_nwp_by_spot": metrics_by_columns(np, pd, eval_frame, "corrected_prediction", "observation", ["spot_id"]),
                "raw_nwp_by_spot": metrics_by_columns(np, pd, eval_frame, "raw_prediction", "observation", ["spot_id"]),
                "raw_nwp_by_observed_regime": metrics_by_observed_regime(
                    np,
                    eval_frame,
                    "raw_prediction",
                    "observation",
                    args.observed_regime_threshold_kt,
                ),
                "corrected_nwp_by_observed_regime": metrics_by_observed_regime(
                    np,
                    eval_frame,
                    "corrected_prediction",
                    "observation",
                    args.observed_regime_threshold_kt,
                ),
                "raw_nwp_threshold_detection": threshold_detection_summary(
                    eval_frame,
                    "raw_prediction",
                    "observation",
                    args.observed_regime_threshold_kt,
                ),
                "corrected_nwp_threshold_detection": threshold_detection_summary(
                    eval_frame,
                    "corrected_prediction",
                    "observation",
                    args.observed_regime_threshold_kt,
                ),
                "corrected_nwp_by_spot_lead": metrics_by_columns(
                    np,
                    pd,
                    eval_frame,
                    "corrected_prediction",
                    "observation",
                    ["spot_id", "lead_time_minutes"],
                ),
                "raw_nwp_by_spot_lead": metrics_by_columns(
                    np,
                    pd,
                    eval_frame,
                    "raw_prediction",
                    "observation",
                    ["spot_id", "lead_time_minutes"],
                ),
                "rmse_gain_pct_vs_raw": (
                    None
                    if not raw_metrics.get("rmse")
                    else round((raw_metrics["rmse"] - corrected_metrics.get("rmse", raw_metrics["rmse"])) / raw_metrics["rmse"] * 100.0, 3)
                ),
            }
            if raw_subset_metrics and corrected_subset_metrics:
                results["models"][target].update({
                    "raw_nwp_eval_leads": raw_subset_metrics,
                    "corrected_nwp_eval_leads": corrected_subset_metrics,
                    "rmse_gain_pct_vs_raw_eval_leads": (
                        None
                        if not raw_subset_metrics.get("rmse")
                        else round(
                            (raw_subset_metrics["rmse"] - corrected_subset_metrics.get("rmse", raw_subset_metrics["rmse"]))
                            / raw_subset_metrics["rmse"]
                            * 100.0,
                            3,
                        )
                    ),
                })
        else:
            model.fit(x_train, y_train)
            model_path = output_root / f"{target}.joblib"
            deps["joblib"].dump(model, model_path)
            probabilities = model.predict_proba(x_test)[:, 1]
            results["models"][target] = {
                "type": "classification",
                "model_family": args.model_family,
                "model_path": str(model_path),
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "test": classification_metrics(deps, y_test.to_numpy(), probabilities),
            }
    return results


def write_markdown_summary(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Parquet Residual Training Results",
        "",
        f"Run id: `{results['run_id']}`",
        "",
        f"- generated: `{results['generated_at_utc']}`",
        f"- model family: `{results['model_family']}`",
        f"- train rows: `{results['train_row_count']}`",
        f"- test rows: `{results['test_row_count']}`",
        f"- feature columns: `{results['feature_column_count']}`",
        f"- dropped sparse columns: `{len(results['dropped_columns']['too_sparse'])}`",
        "",
        "## Regression Models",
        "",
        "| Target | Raw RMSE | Corrected RMSE | Gain | Eval-lead RMSE | Eval-lead gain | Test rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target, item in sorted(results.get("models", {}).items()):
        if item.get("type") != "regression":
            continue
        raw = item.get("raw_nwp_test", {}).get("rmse")
        corrected = item.get("corrected_nwp_test", {}).get("rmse")
        gain = item.get("rmse_gain_pct_vs_raw")
        eval_corrected = item.get("corrected_nwp_eval_leads", {}).get("rmse")
        eval_gain = item.get("rmse_gain_pct_vs_raw_eval_leads")
        lines.append(
            f"| `{target}` | `{raw}` | `{corrected}` | `{gain}%` | "
            f"`{eval_corrected}` | `{eval_gain}%` | `{item.get('test_rows')}` |"
        )
    lines.extend(["", "## Regression By Observed Regime", ""])
    for target, item in sorted(results.get("models", {}).items()):
        if item.get("type") != "regression":
            continue
        lines.extend([
            f"### `{target}`",
            "",
            "| Observed regime | Raw RMSE | Corrected RMSE | Raw MAE | Corrected MAE | Count |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        raw_regimes = item.get("raw_nwp_by_observed_regime", {})
        corrected_regimes = item.get("corrected_nwp_by_observed_regime", {})
        for regime in sorted(set(raw_regimes) | set(corrected_regimes)):
            raw_item = raw_regimes.get(regime, {})
            corrected_item = corrected_regimes.get(regime, {})
            lines.append(
                f"| `{regime}` | `{raw_item.get('rmse')}` | `{corrected_item.get('rmse')}` | "
                f"`{raw_item.get('mae')}` | `{corrected_item.get('mae')}` | "
                f"`{corrected_item.get('count') or raw_item.get('count')}` |"
            )
        lines.extend(["", "| Detection threshold | Raw CSI | Corrected CSI | Raw recall | Corrected recall | Raw precision | Corrected precision |"])
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        raw_detection = item.get("raw_nwp_threshold_detection", {})
        corrected_detection = item.get("corrected_nwp_threshold_detection", {})
        for threshold in sorted(set(raw_detection) | set(corrected_detection)):
            raw_item = raw_detection.get(threshold, {})
            corrected_item = corrected_detection.get(threshold, {})
            lines.append(
                f"| `{threshold}` | `{raw_item.get('csi')}` | `{corrected_item.get('csi')}` | "
                f"`{raw_item.get('recall')}` | `{corrected_item.get('recall')}` | "
                f"`{raw_item.get('precision')}` | `{corrected_item.get('precision')}` |"
            )
    lines.extend(["", "## Classification Models", "", "| Target | Brier | Positive rate | Test rows |", "| --- | ---: | ---: | ---: |"])
    for target, item in sorted(results.get("models", {}).items()):
        if item.get("type") != "classification":
            continue
        test = item.get("test", {})
        lines.append(f"| `{target}` | `{test.get('brier')}` | `{test.get('positive_rate')}` | `{item.get('test_rows')}` |")
    if results.get("skipped_targets"):
        lines.extend(["", "## Skipped Targets", ""])
        for target, reason in sorted(results["skipped_targets"].items()):
            lines.append(f"- `{target}`: `{reason}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-parquet", type=Path, action="append", default=[])
    parser.add_argument("--training-table-root", type=Path)
    parser.add_argument("--run-id-prefix", default="residual_windsup_sst_prev")
    parser.add_argument("--start-month")
    parser.add_argument("--end-month")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", default="residual_windsup_sst_prev_full")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-time-utc", help="Explicit temporal split boundary; overrides --test-fraction when provided.")
    parser.add_argument("--read-batch-size", type=int, default=50000)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-test-rows", type=int)
    parser.add_argument(
        "--include-lead-minute",
        type=int,
        action="append",
        default=[],
        help="Restrict train/test rows to these lead times before sampling. Repeatable.",
    )
    parser.add_argument("--min-non-null-ratio", type=float, default=0.01)
    parser.add_argument("--min-non-null-count", type=int, default=100)
    parser.add_argument("--max-categorical-cardinality", type=int, default=100)
    parser.add_argument("--include-issue-source-flags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-allowlist-json", type=Path)
    parser.add_argument(
        "--include-feature-pattern",
        action="append",
        default=[],
        help="Keep only feature columns matching this substring or re:<regex>. Repeatable. Base columns still need an explicit matching pattern.",
    )
    parser.add_argument(
        "--exclude-feature-pattern",
        action="append",
        default=[],
        help="Drop feature columns matching this substring or re:<regex>. Repeatable.",
    )
    parser.add_argument("--model-family", choices=("hist_gradient_boosting", "extra_trees", "lightgbm"), default="hist_gradient_boosting")
    parser.add_argument("--max-iter", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--lightgbm-max-bin", type=int, default=255)
    parser.add_argument("--lightgbm-feature-fraction", type=float, default=1.0)
    parser.add_argument("--lightgbm-bagging-fraction", type=float, default=1.0)
    parser.add_argument("--lightgbm-bagging-freq", type=int, default=0)
    parser.add_argument("--lightgbm-force-col-wise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--target-high-wind-weight-threshold-ms",
        type=float,
        help="When set, regression training rows whose observed target wind is above this threshold receive extra sample weight.",
    )
    parser.add_argument("--target-high-wind-weight", type=float, default=1.0)
    parser.add_argument(
        "--target-high-wind-weight-rule-ms",
        action="append",
        default=[],
        help="Progressive regression sample-weight rule THRESHOLD_MS:WEIGHT. Repeatable.",
    )
    parser.add_argument(
        "--target-high-wind-weight-rule-kt",
        action="append",
        default=[],
        help="Progressive regression sample-weight rule THRESHOLD_KT:WEIGHT. Repeatable, e.g. 12:2.",
    )
    parser.add_argument(
        "--observed-regime-threshold-kt",
        type=float,
        action="append",
        default=list(DEFAULT_OBSERVED_REGIME_THRESHOLDS_KT),
        help="Observed wind/gust thresholds, in knots, used for regime and detection reporting.",
    )
    parser.add_argument("--min-target-train-rows", type=int, default=100)
    parser.add_argument("--min-target-test-rows", type=int, default=20)
    parser.add_argument("--only-target", action="append", default=[])
    parser.add_argument("--skip-classification", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--fit-group-column",
        action="append",
        default=[],
        help="Train one regression model per value or value-combination of this column. Repeatable, e.g. lead_time_minutes and/or spot_id.",
    )
    parser.add_argument("--min-group-train-rows", type=int, default=1000)
    parser.add_argument("--min-group-test-rows", type=int, default=100)
    parser.add_argument(
        "--eval-lead-minute",
        type=int,
        action="append",
        default=[],
        help="Optional lead minute to include in an additional short-horizon evaluation subset. Repeatable.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.observed_regime_threshold_kt = sorted({float(value) for value in args.observed_regime_threshold_kt})
    deps = import_dependencies()
    warnings.filterwarnings("ignore", category=deps["pd"].errors.PerformanceWarning)
    paths = discover_parquet_paths(args)
    all_columns = schema_columns(paths, deps["pq"])
    feature_candidates = [
        column for column in all_columns
        if is_allowed_feature_column(column, args.include_issue_source_flags)
    ]
    feature_candidates.extend(column for column in DERIVED_TIME_COLUMNS if column not in feature_candidates)
    feature_candidates = [
        column for column in feature_candidates
        if passes_feature_filters(column, args)
    ]
    allowlist: dict[str, Any] | None = None
    allowed_numeric: set[str] = set()
    allowed_categorical: set[str] = set()
    if args.feature_allowlist_json:
        allowlist = json.loads(args.feature_allowlist_json.read_text(encoding="utf-8"))
        allowed_numeric = set(allowlist.get("numeric") or [])
        allowed_categorical = set(allowlist.get("categorical") or [])
        allowed_features = allowed_numeric | allowed_categorical
        feature_candidates = [column for column in feature_candidates if column in allowed_features]
    target_columns = [
        column for column in all_columns
        if is_target_column(column) or column in METRIC_LABEL_COLUMNS
    ]
    required_columns = sorted(set(feature_candidates) | set(target_columns) | {"issue_time_utc"})
    split_time, source_counts = compute_split(paths, deps["pq"], args.test_fraction, args.split_time_utc, args.include_lead_minute)
    train, test = read_sampled_frames(paths, required_columns, split_time, args, deps, source_counts)
    numeric_columns, categorical_columns, dropped = infer_feature_columns(
        train,
        feature_candidates,
        args.min_non_null_ratio,
        args.min_non_null_count,
        args.max_categorical_cardinality,
    )
    if allowlist is not None:
        numeric_columns = [column for column in numeric_columns if column in allowed_numeric]
        categorical_columns = [column for column in categorical_columns if column in allowed_categorical]
        dropped["not_in_allowlist"] = [
            {"column": column}
            for column in sorted(set(feature_candidates) - allowed_numeric - allowed_categorical)
            if column in train.columns
        ]
    feature_columns = [*numeric_columns, *categorical_columns]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "feature_columns.json").write_text(
        json.dumps({
            "numeric": numeric_columns,
            "categorical": categorical_columns,
            "dropped": dropped,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model_results = train_models(train, test, feature_columns, categorical_columns, args, deps)
    results = {
        "format": "corsewind.residual_correction_parquet_training.v1",
        "generated_at_utc": utc_now(),
        "run_id": args.run_id,
        "model_family": args.model_family,
        "source_parquet_count": len(paths),
        "source_parquets": [str(path) for path in paths],
        "source_counts": source_counts,
        "temporal_split_issue_time_utc": split_time,
        "train_row_count": int(len(train)),
        "test_row_count": int(len(test)),
        "feature_column_count": len(feature_columns),
        "numeric_column_count": len(numeric_columns),
        "categorical_column_count": len(categorical_columns),
        "dropped_columns": dropped,
        "settings": {
            "test_fraction": args.test_fraction,
            "max_train_rows": args.max_train_rows,
            "max_test_rows": args.max_test_rows,
            "include_lead_minute": args.include_lead_minute,
            "min_non_null_ratio": args.min_non_null_ratio,
            "min_non_null_count": args.min_non_null_count,
            "skip_classification": args.skip_classification,
            "include_feature_pattern": args.include_feature_pattern,
            "exclude_feature_pattern": args.exclude_feature_pattern,
            "only_target": args.only_target,
            "fit_group_column": args.fit_group_column,
            "min_group_train_rows": args.min_group_train_rows,
            "min_group_test_rows": args.min_group_test_rows,
            "eval_lead_minute": args.eval_lead_minute,
            "target_high_wind_weight_threshold_ms": args.target_high_wind_weight_threshold_ms,
            "target_high_wind_weight": args.target_high_wind_weight,
            "target_high_wind_weight_rule_ms": args.target_high_wind_weight_rule_ms,
            "target_high_wind_weight_rule_kt": args.target_high_wind_weight_rule_kt,
            "observed_regime_threshold_kt": args.observed_regime_threshold_kt,
        },
        **model_results,
    }
    (args.output_root / "training_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown_summary(args.output_root / "training_results.md", results)
    print(json.dumps({
        "run_id": args.run_id,
        "train_row_count": results["train_row_count"],
        "test_row_count": results["test_row_count"],
        "feature_column_count": results["feature_column_count"],
        "models": {
            target: {
                "type": item.get("type"),
                "rmse_gain_pct_vs_raw": item.get("rmse_gain_pct_vs_raw"),
                "test": item.get("test"),
                "raw_nwp_test": item.get("raw_nwp_test"),
                "corrected_nwp_test": item.get("corrected_nwp_test"),
                "raw_nwp_eval_leads": item.get("raw_nwp_eval_leads"),
                "corrected_nwp_eval_leads": item.get("corrected_nwp_eval_leads"),
            }
            for target, item in results["models"].items()
        },
        "skipped_targets": results["skipped_targets"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
