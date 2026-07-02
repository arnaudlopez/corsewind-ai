#!/usr/bin/env python3
"""Train first sklearn residual-correction models from a training table."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGRESSION_TARGETS = ("residual_wind_mean_ms", "residual_gust_ms")
CLASSIFICATION_PREFIXES = ("target_wind_gt_", "target_gust_gt_")
REGRESSION_TARGET_METADATA = {
    "residual_wind_mean_ms": {
        "baseline_feature": "baselines.baseline_wind_mean_ms",
        "observed_label": "target_wind_mean_ms",
    },
    "residual_gust_ms": {
        "baseline_feature": "baselines.baseline_gust_ms",
        "observed_label": "target_gust_ms",
    },
}


def import_sklearn():
    try:
        import joblib
        import numpy as np
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except ImportError as exc:
        raise SystemExit(
            "Missing ML dependencies. Rebuild/install requirements-ml-dataset.txt "
            "so scikit-learn and joblib are available."
        ) from exc
    return {
        "joblib": joblib,
        "np": np,
        "ColumnTransformer": ColumnTransformer,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "ExtraTreesRegressor": ExtraTreesRegressor,
        "SimpleImputer": SimpleImputer,
        "accuracy_score": accuracy_score,
        "brier_score_loss": brier_score_loss,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def finite_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def flatten_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    features: dict[str, Any] = {
        "spot_id": row.get("spot_id"),
        "spot_kind": row.get("spot_kind"),
        "spot_source_type": row.get("spot_source_type"),
        "station_id": row.get("station_id"),
        "latitude": finite_float(row.get("latitude")),
        "longitude": finite_float(row.get("longitude")),
        "lead_time_minutes": finite_float(row.get("lead_time_minutes")),
    }
    for group in ("features", "baselines"):
        values = row.get(group, {}) if isinstance(row.get(group), dict) else {}
        for key, value in values.items():
            out_key = f"{group}.{key}"
            if isinstance(value, bool):
                features[out_key] = int(value)
            elif isinstance(value, (int, float)) or value is None:
                features[out_key] = finite_float(value)
            else:
                features[out_key] = str(value)
    labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}
    return features, labels


def split_by_time(rows: list[dict[str, Any]], test_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    issue_times = sorted({str(row.get("issue_time_utc")) for row in rows if row.get("issue_time_utc")})
    if len(issue_times) < 2:
        return rows, [], None
    test_count = max(1, int(round(len(issue_times) * test_fraction)))
    split_time = issue_times[-test_count]
    train = [row for row in rows if str(row.get("issue_time_utc")) < split_time]
    test = [row for row in rows if str(row.get("issue_time_utc")) >= split_time]
    if not train or not test:
        return rows, [], None
    return train, test, split_time


def infer_columns(records: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    categorical = set()
    numeric = set()
    for record in records:
        for key, value in record.items():
            if isinstance(value, str):
                categorical.add(key)
            else:
                numeric.add(key)
    return sorted(numeric - categorical), sorted(categorical)


def matrix(records: list[dict[str, Any]], numeric_columns: list[str], categorical_columns: list[str]) -> list[list[Any]]:
    rows = []
    for record in records:
        numeric_values = [
            math.nan if record.get(column) is None else finite_float(record.get(column))
            for column in numeric_columns
        ]
        categorical_values = [
            "__missing__" if record.get(column) in {None, ""} else str(record.get(column))
            for column in categorical_columns
        ]
        rows.append([*numeric_values, *categorical_values])
    return rows


def make_preprocessor(sklearn: dict[str, Any], numeric_columns: list[str], categorical_columns: list[str]):
    try:
        encoder = sklearn["OneHotEncoder"](handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = sklearn["OneHotEncoder"](handle_unknown="ignore", sparse=False)
    transformers = []
    if numeric_columns:
        transformers.append((
            "numeric",
            sklearn["Pipeline"]([
                ("imputer", sklearn["SimpleImputer"](strategy="median")),
            ]),
            list(range(len(numeric_columns))),
        ))
    if categorical_columns:
        cat_indices = list(range(len(numeric_columns), len(numeric_columns) + len(categorical_columns)))
        transformers.append((
            "categorical",
            sklearn["Pipeline"]([
                ("imputer", sklearn["SimpleImputer"](strategy="constant", fill_value="__missing__")),
                ("onehot", encoder),
            ]),
            cat_indices,
        ))
    return sklearn["ColumnTransformer"](transformers=transformers)


def regression_metrics(sklearn: dict[str, Any], y_true: list[float], y_pred: list[float]) -> dict[str, Any]:
    if not y_true:
        return {"count": 0}
    rmse = math.sqrt(sklearn["mean_squared_error"](y_true, y_pred))
    return {
        "count": len(y_true),
        "mae": round(float(sklearn["mean_absolute_error"](y_true, y_pred)), 6),
        "rmse": round(float(rmse), 6),
        "bias": round(float(sum(pred - true for pred, true in zip(y_pred, y_true)) / len(y_true)), 6),
    }


def prediction_metrics(predictions: list[float], observations: list[float]) -> dict[str, Any]:
    if not predictions:
        return {"count": 0}
    errors = [prediction - observation for prediction, observation in zip(predictions, observations)]
    return {
        "count": len(errors),
        "mae": round(float(sum(abs(error) for error in errors) / len(errors)), 6),
        "rmse": round(float(math.sqrt(sum(error * error for error in errors) / len(errors))), 6),
        "bias": round(float(sum(errors) / len(errors)), 6),
    }


def classification_metrics(sklearn: dict[str, Any], y_true: list[int], probabilities: list[float]) -> dict[str, Any]:
    if not y_true:
        return {"count": 0}
    predictions = [1 if value >= 0.5 else 0 for value in probabilities]
    return {
        "count": len(y_true),
        "positive_count": sum(y_true),
        "positive_rate": round(sum(y_true) / len(y_true), 6),
        "accuracy": round(float(sklearn["accuracy_score"](y_true, predictions)), 6),
        "brier": round(float(sklearn["brier_score_loss"](y_true, probabilities)), 6),
    }


def downsample_rows(rows: list[dict[str, Any]], max_rows: int | None) -> list[dict[str, Any]]:
    if not max_rows or max_rows <= 0 or len(rows) <= max_rows:
        return rows
    step = len(rows) / max_rows
    return [rows[min(len(rows) - 1, int(index * step))] for index in range(max_rows)]


def train_models(
    rows: list[dict[str, Any]],
    output_root: Path,
    test_fraction: float,
    max_iter: int,
    only_targets: set[str] | None = None,
    skip_classification: bool = False,
) -> dict[str, Any]:
    sklearn = import_sklearn()
    train_rows, test_rows, split_time = split_by_time(rows, test_fraction)
    train_records_labels = [flatten_row(row) for row in train_rows]
    test_records_labels = [flatten_row(row) for row in test_rows]
    train_records = [item[0] for item in train_records_labels]
    test_records = [item[0] for item in test_records_labels]
    numeric_columns, categorical_columns = infer_columns(train_records)
    x_train = matrix(train_records, numeric_columns, categorical_columns)
    x_test = matrix(test_records, numeric_columns, categorical_columns)
    output_root.mkdir(parents=True, exist_ok=True)

    labels_seen = set()
    for _, labels in train_records_labels:
        labels_seen.update(labels)
    regression_targets = [target for target in REGRESSION_TARGETS if target in labels_seen]
    classification_targets = sorted(
        target
        for target in labels_seen
        if any(target.startswith(prefix) for prefix in CLASSIFICATION_PREFIXES)
    )
    if only_targets:
        regression_targets = [target for target in regression_targets if target in only_targets]
        classification_targets = [target for target in classification_targets if target in only_targets]
    if skip_classification:
        classification_targets = []

    results: dict[str, Any] = {
        "format": "corsewind.residual_correction_sklearn_training.v1",
        "generated_at_utc": utc_now(),
        "row_count": len(rows),
        "train_row_count": len(train_rows),
        "test_row_count": len(test_rows),
        "temporal_split_issue_time_utc": split_time,
        "numeric_column_count": len(numeric_columns),
        "categorical_column_count": len(categorical_columns),
        "only_targets": sorted(only_targets) if only_targets else None,
        "skip_classification": skip_classification,
        "models": {},
        "skipped_targets": {},
    }
    (output_root / "feature_columns.json").write_text(
        json.dumps({"numeric": numeric_columns, "categorical": categorical_columns}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for target in regression_targets:
        y_train = [finite_float(labels.get(target)) for _, labels in train_records_labels]
        y_test = [finite_float(labels.get(target)) for _, labels in test_records_labels]
        train_pairs = [(x, y) for x, y in zip(x_train, y_train) if y is not None]
        test_pairs = []
        for x, y, feature_record, (_, labels) in zip(x_test, y_test, test_records, test_records_labels):
            if y is not None:
                test_pairs.append((x, y, feature_record, labels))
        if len(train_pairs) < 20 or len(test_pairs) < 1:
            results["skipped_targets"][target] = "not_enough_train_or_test_rows"
            continue
        model = sklearn["Pipeline"]([
            ("preprocess", make_preprocessor(sklearn, numeric_columns, categorical_columns)),
            ("model", sklearn["ExtraTreesRegressor"](n_estimators=max_iter, random_state=42, n_jobs=-1, min_samples_leaf=2)),
        ])
        model.fit([item[0] for item in train_pairs], [item[1] for item in train_pairs])
        predictions = model.predict([item[0] for item in test_pairs])
        corrected_metrics: dict[str, Any] = {"count": 0}
        raw_metrics: dict[str, Any] = {"count": 0}
        metadata = REGRESSION_TARGET_METADATA.get(target, {})
        baseline_feature = metadata.get("baseline_feature")
        observed_label = metadata.get("observed_label")
        if baseline_feature and observed_label:
            raw_predictions = []
            corrected_predictions = []
            observations = []
            for prediction, (_, _, feature_record, labels) in zip(predictions, test_pairs):
                baseline = finite_float(feature_record.get(baseline_feature))
                observed = finite_float(labels.get(observed_label))
                if baseline is None or observed is None:
                    continue
                raw_predictions.append(baseline)
                corrected_predictions.append(baseline + float(prediction))
                observations.append(observed)
            raw_metrics = prediction_metrics(raw_predictions, observations)
            corrected_metrics = prediction_metrics(corrected_predictions, observations)
        model_path = output_root / f"{target}.joblib"
        sklearn["joblib"].dump(model, model_path)
        results["models"][target] = {
            "type": "regression",
            "model_path": str(model_path),
            "residual_test": regression_metrics(sklearn, [item[1] for item in test_pairs], list(predictions)),
            "raw_nwp_test": raw_metrics,
            "corrected_nwp_test": corrected_metrics,
            "rmse_gain_pct_vs_raw": (
                None
                if not raw_metrics.get("rmse")
                else round((raw_metrics["rmse"] - corrected_metrics.get("rmse", raw_metrics["rmse"])) / raw_metrics["rmse"] * 100.0, 3)
            ),
        }

    for target in classification_targets:
        y_train = [labels.get(target) for _, labels in train_records_labels]
        y_test = [labels.get(target) for _, labels in test_records_labels]
        train_pairs = [(x, int(y)) for x, y in zip(x_train, y_train) if y in {0, 1}]
        test_pairs = [(x, int(y)) for x, y in zip(x_test, y_test) if y in {0, 1}]
        classes = Counter(y for _, y in train_pairs)
        if len(classes) < 2:
            results["skipped_targets"][target] = f"single_training_class_{dict(classes)}"
            continue
        if len(train_pairs) < 20 or len(test_pairs) < 1:
            results["skipped_targets"][target] = "not_enough_train_or_test_rows"
            continue
        model = sklearn["Pipeline"]([
            ("preprocess", make_preprocessor(sklearn, numeric_columns, categorical_columns)),
            ("model", sklearn["ExtraTreesClassifier"](n_estimators=max_iter, random_state=42, n_jobs=-1, min_samples_leaf=2)),
        ])
        model.fit([item[0] for item in train_pairs], [item[1] for item in train_pairs])
        probabilities = model.predict_proba([item[0] for item in test_pairs])[:, 1]
        model_path = output_root / f"{target}.joblib"
        sklearn["joblib"].dump(model, model_path)
        results["models"][target] = {
            "type": "classification",
            "model_path": str(model_path),
            "test": classification_metrics(sklearn, [item[1] for item in test_pairs], list(probabilities)),
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-rows", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--max-rows", type=int, help="Deterministically downsample rows for smoke tests.")
    parser.add_argument("--only-target", action="append", default=[], help="Train only this label target; can be repeated.")
    parser.add_argument("--skip-classification", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.training_rows)
    original_row_count = len(rows)
    rows = downsample_rows(rows, args.max_rows)
    results = train_models(
        rows,
        args.output_root,
        args.test_fraction,
        args.max_iter,
        only_targets=set(args.only_target) if args.only_target else None,
        skip_classification=args.skip_classification,
    )
    results["source_row_count"] = original_row_count
    results["max_rows"] = args.max_rows
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "training_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "row_count": results["row_count"],
        "train_row_count": results["train_row_count"],
        "test_row_count": results["test_row_count"],
        "models": results["models"],
        "skipped_targets": results["skipped_targets"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
