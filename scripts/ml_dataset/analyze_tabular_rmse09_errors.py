#!/usr/bin/env python3
"""Rebuild tabular holdout predictions and diagnose RMSE-0.9 error modes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_residual_correction_parquet import (  # noqa: E402
    REGRESSION_TARGET_METADATA,
    add_time_features,
    import_dependencies,
    json_safe_scalar,
    lead_filter_mask,
    sample_fraction,
    stable_sample,
)


DEFAULT_TARGET = "labels__residual_wind_mean_ms"
TARGET_OUTPUTS = {
    "labels__residual_wind_mean_ms": {
        "label": "Wind Mean",
        "value_label": "wind_mean",
        "predicted_residual": "predicted_residual_wind_mean_ms",
        "raw": "raw_wind_mean_ms",
        "actual": "actual_wind_mean_ms",
        "corrected": "corrected_wind_mean_ms",
        "actual_bin": "actual_wind_bin_ms",
        "default_feature_bin_columns": (
            "baselines__baseline_wind_mean_ms",
            "features__model_error_now_wind_mean_ms",
            "features__nwp_horizon_wind_ramp_ms",
            "features__nwp_error_persistence_plus_wind_ramp_ms",
            "features__obs_lag_15m_wind_mean_ms",
            "features__obs_delta_15m_wind_mean_ms",
            "baselines__baseline_shortwave_radiation",
            "baselines__baseline_temperature_2m_c",
            "baselines__baseline_pressure_msl_hpa",
            "baselines__baseline_cloud_cover_pct",
        ),
    },
    "labels__residual_gust_ms": {
        "label": "Gust",
        "value_label": "gust",
        "predicted_residual": "predicted_residual_gust_ms",
        "raw": "raw_gust_ms",
        "actual": "actual_gust_ms",
        "corrected": "corrected_gust_ms",
        "actual_bin": "actual_gust_bin_ms",
        "default_feature_bin_columns": (
            "baselines__baseline_gust_ms",
            "features__model_error_now_gust_ms",
            "features__nwp_horizon_gust_ramp_ms",
            "features__nwp_error_persistence_plus_gust_ramp_ms",
            "features__obs_lag_15m_gust_ms",
            "features__obs_delta_15m_gust_ms",
            "baselines__baseline_wind_mean_ms",
            "features__model_error_now_wind_mean_ms",
            "features__nwp_horizon_wind_ramp_ms",
            "baselines__baseline_shortwave_radiation",
            "baselines__baseline_temperature_2m_c",
            "baselines__baseline_pressure_msl_hpa",
            "baselines__baseline_cloud_cover_pct",
        ),
    },
}
DEFAULT_EXTRA_PATTERNS = (
    "sst",
    "thermal",
    "land",
    "surface",
    "shortwave",
    "temperature",
    "pressure",
    "cloud",
    "instability",
    "model_error_now",
    "nwp_horizon_wind_ramp",
    "nwp_error_persistence",
    "obs_lag_15m",
    "obs_delta_15m",
    "context_global_inland_1",
    "context_global_coastal_1",
    "context_global_relief_1",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def metric_frame(frame: Any, prediction_col: str, observation_col: str, np: Any) -> dict[str, Any]:
    valid = frame[[prediction_col, observation_col]].dropna()
    return metric(valid[prediction_col].astype(float).to_numpy(), valid[observation_col].astype(float).to_numpy(), np)


def grouped_metrics(
    frame: Any,
    group_columns: list[str],
    prediction_col: str,
    observation_col: str,
    np: Any,
    pd: Any,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if any(column not in frame.columns for column in group_columns):
        return []
    rows: list[dict[str, Any]] = []
    valid = frame.dropna(subset=[prediction_col, observation_col])
    for raw_key, group in valid.groupby(group_columns, dropna=False):
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        item = {
            "group": dict(zip(group_columns, [json_safe_scalar(pd, value) for value in values], strict=True)),
            **metric_frame(group, prediction_col, observation_col, np),
        }
        rows.append(item)
    rows.sort(key=lambda item: (item.get("rmse") is None, item.get("rmse", -1)), reverse=True)
    return rows[:limit] if limit else rows


def parquet_schema_columns(paths: list[Path], pq: Any) -> set[str]:
    columns: set[str] = set()
    for path in paths:
        columns.update(pq.ParquetFile(path).schema.names)
    return columns


def matching_extra_columns(all_columns: set[str], patterns: list[str]) -> list[str]:
    lowered_patterns = [pattern.lower() for pattern in patterns]
    return sorted(
        column
        for column in all_columns
        if column.startswith("features__") or column.startswith("baselines__")
        if any(pattern in column.lower() for pattern in lowered_patterns)
    )


def read_test_frame(
    paths: list[Path],
    read_columns: list[str],
    split_time: str,
    include_leads: list[int],
    max_test_rows: int | None,
    pre_sample_test_rows: int,
    batch_size: int,
    deps: dict[str, Any],
) -> Any:
    pd = deps["pd"]
    np = deps["np"]
    pq = deps["pq"]
    test_frames = []
    test_fraction = sample_fraction(max_test_rows, pre_sample_test_rows)
    read_columns = sorted(set(read_columns) | {"issue_time_utc"})
    for path in paths:
        pf = pq.ParquetFile(path)
        available = [column for column in read_columns if column in pf.schema.names]
        for batch in pf.iter_batches(batch_size=batch_size, columns=available):
            frame = batch.to_pandas().reindex(columns=read_columns)
            mask = lead_filter_mask(frame, include_leads)
            if mask is not None:
                frame = frame[mask]
            if frame.empty:
                continue
            frame = add_time_features(frame, pd, np)
            test_part = frame[frame["issue_time_utc"].astype(str) >= split_time]
            if not test_part.empty:
                test_frames.append(stable_sample(test_part, pd, max_test_rows, test_fraction))
    if not test_frames:
        return pd.DataFrame(columns=read_columns)
    test = pd.concat(test_frames, ignore_index=True)
    return stable_sample(test, pd, max_test_rows, 1.0)


def add_error_columns(frame: Any, target_output: dict[str, Any], pd: Any) -> Any:
    out = frame.copy()
    issue_time = pd.to_datetime(out["issue_time_utc"], utc=True, errors="coerce")
    out["issue_month"] = issue_time.dt.strftime("%Y-%m")
    out["issue_day"] = issue_time.dt.strftime("%Y-%m-%d")
    out["issue_hour_utc"] = issue_time.dt.hour.astype("Int64")
    out["corrected_error_ms"] = out[target_output["corrected"]] - out[target_output["actual"]]
    out["raw_error_ms"] = out[target_output["raw"]] - out[target_output["actual"]]
    out["abs_corrected_error_ms"] = out["corrected_error_ms"].abs()
    out["abs_raw_error_ms"] = out["raw_error_ms"].abs()
    out[target_output["actual_bin"]] = pd.cut(
        out[target_output["actual"]].astype(float),
        bins=[-0.001, 2.0, 4.0, 6.0, 8.0, 999.0],
        labels=["0-2", "2-4", "4-6", "6-8", "8+"],
    ).astype(str)
    out["raw_abs_error_bin_ms"] = pd.cut(
        out["abs_raw_error_ms"].astype(float),
        bins=[-0.001, 0.5, 1.0, 2.0, 999.0],
        labels=["0-0.5", "0.5-1", "1-2", "2+"],
    ).astype(str)
    out["corrected_abs_error_bin_ms"] = pd.cut(
        out["abs_corrected_error_ms"].astype(float),
        bins=[-0.001, 0.5, 1.0, 1.5, 2.0, 999.0],
        labels=["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2+"],
    ).astype(str)
    return out


def add_quantile_bin(frame: Any, column: str, pd: Any) -> str | None:
    values = pd.to_numeric(frame[column], errors="coerce")
    valid_count = int(values.notna().sum())
    if valid_count < 100:
        return None
    unique_count = int(values.dropna().nunique())
    if unique_count < 4:
        return None
    bin_column = f"__bin__{column}"
    try:
        frame[bin_column] = pd.qcut(values, q=4, duplicates="drop").astype(str)
    except ValueError:
        return None
    return bin_column


def analyze_feature_bins(
    frame: Any,
    columns: list[str],
    prediction_col: str,
    observation_col: str,
    np: Any,
    pd: Any,
    *,
    limit: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    working = frame.copy()
    for column in columns:
        if column not in working.columns:
            continue
        bin_column = add_quantile_bin(working, column, pd)
        if not bin_column:
            continue
        metrics = grouped_metrics(working, [bin_column], prediction_col, observation_col, np, pd)
        out[column] = metrics[:limit]
    return out


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Tabular RMSE09 Error Diagnosis",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Run: `{result['run_id']}`",
        f"Target: `{result['target']}`",
        f"Prediction column: `{result['prediction_column']}`",
        f"Prediction rows: `{result['prediction_row_count']}`",
        f"Metric rows: `{result['metric_row_count']}`",
        f"Metric leads: `{result['metric_lead_minute']}`",
        f"Corrected RMSE: `{result['overall']['corrected'].get('rmse')}`",
        f"Raw RMSE: `{result['overall']['raw'].get('rmse')}`",
        f"Gap to 0.9: `{result['gap_to_threshold']}`",
        "",
        "## Worst Groups",
    ]
    for title, key, fields in [
        ("Spot/Lead", "by_spot_lead", ("spot_id", "lead_time_minutes")),
        ("Spot", "by_spot", ("spot_id",)),
        ("Lead", "by_lead", ("lead_time_minutes",)),
        ("Hour UTC", "by_issue_hour", ("issue_hour_utc",)),
        ("Month", "by_issue_month", ("issue_month",)),
        (f"Actual {result['target_label']} Bin", "by_actual_value_bin", (result["actual_bin_column"],)),
        ("Raw Error Bin", "by_raw_abs_error_bin", ("raw_abs_error_bin_ms",)),
    ]:
        lines.extend(["", f"### {title}", "", "| Group | Count | RMSE | MAE | Bias | P90 abs |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for item in result.get(key, [])[:20]:
            group = item.get("group", {})
            label = " / ".join(str(group.get(field)) for field in fields)
            lines.append(
                f"| `{label}` | {item.get('count')} | {item.get('rmse')} | "
                f"{item.get('mae')} | {item.get('bias')} | {item.get('p90_abs_error')} |"
            )
    lines.extend(["", "## Feature Bins"])
    for column, metrics in result.get("by_feature_bin", {}).items():
        lines.extend(["", f"### `{column}`", "", "| Bin | Count | RMSE | MAE | Bias |", "| --- | ---: | ---: | ---: | ---: |"])
        for item in metrics:
            group = item.get("group", {})
            label = next(iter(group.values())) if group else ""
            lines.append(f"| `{label}` | {item.get('count')} | {item.get('rmse')} | {item.get('mae')} | {item.get('bias')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]
    np = deps["np"]
    joblib = deps["joblib"]

    training = load_json(args.training_results)
    feature_config = load_json(args.feature_columns)
    target = args.target
    if target not in TARGET_OUTPUTS:
        raise SystemExit(f"Unsupported target for diagnosis: {target}")
    target_output = TARGET_OUTPUTS[target]
    model_info = training.get("models", {}).get(target)
    if not model_info:
        raise SystemExit(f"Target not found in training_results models: {target}")
    model_path = Path(args.model_path or model_info.get("model_path", ""))
    if not model_path.exists():
        raise SystemExit(f"Missing model path: {model_path}")
    model = joblib.load(model_path)

    feature_columns = list(feature_config.get("numeric") or []) + list(feature_config.get("categorical") or [])
    metadata = REGRESSION_TARGET_METADATA[target]
    paths = [Path(path) for path in training["source_parquets"]]
    all_columns = parquet_schema_columns(paths, deps["pq"])
    include_leads = args.include_lead_minute
    extra_columns = matching_extra_columns(all_columns, args.extra_pattern)
    read_columns = sorted(
        set(feature_columns)
        | set(extra_columns)
        | {
            "issue_time_utc",
            "spot_id",
            "station_id",
            "spot_kind",
            "latitude",
            "longitude",
            "lead_time_minutes",
            target,
            metadata["baseline_feature"],
            metadata["observed_label"],
        }
    )
    test = read_test_frame(
        paths,
        read_columns,
        training["temporal_split_issue_time_utc"],
        [int(lead) for lead in include_leads],
        int(training.get("settings", {}).get("max_test_rows") or training.get("test_row_count") or 0) or None,
        int(training.get("source_counts", {}).get("pre_sample_test_rows") or 0),
        args.read_batch_size,
        deps,
    )
    if test.empty:
        raise SystemExit("Rebuilt test frame is empty.")
    missing_features = [column for column in feature_columns if column not in test.columns]
    if missing_features:
        raise SystemExit(f"Missing model features in rebuilt test frame: {missing_features[:20]}")
    residual_prediction = model.predict(test[feature_columns])
    predictions = test.copy()
    predictions[target_output["predicted_residual"]] = residual_prediction
    predictions[target_output["raw"]] = predictions[metadata["baseline_feature"]].astype(float)
    predictions[target_output["actual"]] = predictions[metadata["observed_label"]].astype(float)
    predictions[target_output["corrected"]] = predictions[target_output["raw"]] + predictions[target_output["predicted_residual"]]
    predictions = add_error_columns(predictions, target_output, pd)
    metric_leads = args.metric_lead_minute or training.get("settings", {}).get("eval_lead_minute", [])
    metric_predictions = predictions
    if metric_leads:
        metric_predictions = predictions[predictions["lead_time_minutes"].astype("Int64").isin([int(lead) for lead in metric_leads])]

    if args.output_predictions:
        args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
        output_columns = [
            "issue_time_utc",
            "issue_month",
            "issue_day",
            "issue_hour_utc",
            "spot_id",
            "station_id",
            "spot_kind",
            "latitude",
            "longitude",
            "lead_time_minutes",
            target_output["raw"],
            target_output["predicted_residual"],
            target_output["corrected"],
            target_output["actual"],
            "raw_error_ms",
            "corrected_error_ms",
            "abs_raw_error_ms",
            "abs_corrected_error_ms",
            *[column for column in extra_columns if column in predictions.columns],
        ]
        predictions[output_columns].to_parquet(args.output_predictions, index=False)

    feature_bin_columns = list(dict.fromkeys([*args.feature_bin_column, *target_output["default_feature_bin_columns"]]))
    result = {
        "format": "corsewind.tabular_rmse09_error_diagnosis.v1",
        "generated_at_utc": utc_now(),
        "training_results": str(args.training_results),
        "feature_columns": str(args.feature_columns),
        "model_path": str(model_path),
        "run_id": training.get("run_id"),
        "target": target,
        "target_label": target_output["label"],
        "prediction_column": target_output["corrected"],
        "actual_column": target_output["actual"],
        "actual_bin_column": target_output["actual_bin"],
        "threshold_rmse": args.threshold_rmse,
        "prediction_row_count": int(len(predictions)),
        "metric_row_count": int(len(metric_predictions)),
        "include_lead_minute": [int(lead) for lead in include_leads],
        "metric_lead_minute": [int(lead) for lead in metric_leads],
        "overall": {
            "raw": metric_frame(metric_predictions, target_output["raw"], target_output["actual"], np),
            "corrected": metric_frame(metric_predictions, target_output["corrected"], target_output["actual"], np),
        },
        "by_spot_lead": grouped_metrics(
            metric_predictions,
            ["spot_id", "lead_time_minutes"],
            target_output["corrected"],
            target_output["actual"],
            np,
            pd,
            limit=args.limit,
        ),
        "by_spot": grouped_metrics(metric_predictions, ["spot_id"], target_output["corrected"], target_output["actual"], np, pd, limit=args.limit),
        "by_lead": grouped_metrics(metric_predictions, ["lead_time_minutes"], target_output["corrected"], target_output["actual"], np, pd),
        "by_issue_hour": grouped_metrics(metric_predictions, ["issue_hour_utc"], target_output["corrected"], target_output["actual"], np, pd),
        "by_issue_month": grouped_metrics(metric_predictions, ["issue_month"], target_output["corrected"], target_output["actual"], np, pd),
        "by_actual_value_bin": grouped_metrics(metric_predictions, [target_output["actual_bin"]], target_output["corrected"], target_output["actual"], np, pd),
        "by_raw_abs_error_bin": grouped_metrics(metric_predictions, ["raw_abs_error_bin_ms"], target_output["corrected"], target_output["actual"], np, pd),
        "by_corrected_abs_error_bin": grouped_metrics(
            metric_predictions,
            ["corrected_abs_error_bin_ms"],
            target_output["corrected"],
            target_output["actual"],
            np,
            pd,
        ),
        "by_feature_bin": analyze_feature_bins(
            metric_predictions,
            feature_bin_columns,
            target_output["corrected"],
            target_output["actual"],
            np,
            pd,
            limit=args.limit,
        ),
        "output_predictions": str(args.output_predictions) if args.output_predictions else None,
    }
    corrected_rmse = result["overall"]["corrected"].get("rmse")
    result["gap_to_threshold"] = None if corrected_rmse is None else round(float(corrected_rmse) - args.threshold_rmse, 6)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-results", type=Path, required=True)
    parser.add_argument("--feature-columns", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--include-lead-minute", type=int, action="append", default=[])
    parser.add_argument("--metric-lead-minute", type=int, action="append", default=[])
    parser.add_argument("--extra-pattern", action="append", default=list(DEFAULT_EXTRA_PATTERNS))
    parser.add_argument("--feature-bin-column", action="append", default=[])
    parser.add_argument("--threshold-rmse", type=float, default=0.9)
    parser.add_argument("--read-batch-size", type=int, default=50000)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output-predictions", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(args)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, result)
    print(json.dumps({
        "run_id": result["run_id"],
        "prediction_row_count": result["prediction_row_count"],
        "raw_rmse": result["overall"]["raw"].get("rmse"),
        "corrected_rmse": result["overall"]["corrected"].get("rmse"),
        "gap_to_threshold": result["gap_to_threshold"],
        "worst_spot_lead": result["by_spot_lead"][:5],
    }, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
