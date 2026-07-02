#!/usr/bin/env python3
"""Tabular same-sample benchmark for CorseWind SAPHIR dictionary V2."""

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
        "persistence_feature": "station_target_wind_mean_ms_last_4",
    },
    "gust_ms": {
        "actual": "target_gust_ms",
        "baseline": "baseline_gust_ms",
        "residual": "residual_gust_ms",
        "persistence_feature": "station_target_gust_ms_last_4",
    },
}
FORBIDDEN_PREFIXES = ("target_", "residual_", "labels__target_")
KEY_COLUMNS = {"sample_id", "spot_id", "issue_time_utc", "target_time_utc", "issue_time", "timestamp_utc", "split", "benchmark_split"}
DEFAULT_CATEGORICAL = ["spot_id", "spot_kind", "spot_source_type", "station_id", "lead_time_minutes"]
STATION_VALUE_COLUMNS = [
    "wind_mean_ms",
    "gust_ms",
    "wind_direction_deg",
    "wind_u_ms",
    "wind_v_ms",
    "temperature_c",
    "pressure_hpa",
    "humidity_pct",
    "nwp_wind_mean_ms",
    "nwp_gust_ms",
    "nwp_temperature_2m_c",
    "nwp_pressure_msl_hpa",
    "wind_mean_error_ms",
    "gust_error_ms",
]
NWP_OFFSET_COLUMNS = [
    "boundary_layer_height",
    "cape",
    "cloud_cover",
    "cloud_cover_low",
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
VERTICAL_COLUMNS = ["geopotential_height", "relative_humidity", "temperature", "wind_speed", "wind_u_ms", "wind_v_ms"]


def import_deps() -> dict[str, Any]:
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
        raise SystemExit("Missing pandas/numpy/sklearn dependencies.") from exc
    return locals()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_table(root: Path, name: str, pd: Any) -> Any:
    path = root / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def numeric(series: Any, pd: Any) -> Any:
    return pd.to_numeric(series, errors="coerce")


def parse_windows(value: str) -> list[int]:
    return sorted({int(part.strip()) for part in value.split(",") if part.strip()})


def metric(np: Any, frame: Any, prediction: str, actual: str) -> dict[str, Any]:
    if prediction not in frame.columns or actual not in frame.columns:
        return {"count": 0}
    valid = frame[[prediction, actual]].dropna()
    if valid.empty:
        return {"count": 0}
    err = valid[prediction].to_numpy(dtype=float) - valid[actual].to_numpy(dtype=float)
    return {
        "count": int(len(err)),
        "mae": round(float(np.mean(np.abs(err))), 6),
        "rmse": round(float(np.sqrt(np.mean(err * err))), 6),
        "bias": round(float(np.mean(err)), 6),
    }


def metrics_by_group(np: Any, frame: Any, prediction: str, actual: str, group_column: str) -> dict[str, Any]:
    if group_column not in frame.columns:
        return {}
    return {str(key): metric(np, group, prediction, actual) for key, group in frame.groupby(group_column, dropna=False)}


def station_sequence_features(station: Any, windows: list[int], deps: dict[str, Any]) -> Any:
    pd = deps["pd"]
    np = deps["np"]
    if station.empty:
        return pd.DataFrame()
    frame = station.copy()
    for column in STATION_VALUE_COLUMNS:
        if column in frame.columns:
            frame[column] = numeric(frame[column], pd)
    rows = []
    for sample_id, sample_group in frame.sort_values(["sample_id", "station_slot", "time_index"]).groupby("sample_id", sort=False):
        item: dict[str, Any] = {"sample_id": sample_id}
        for source_kind, source_group in sample_group.groupby("source_kind", dropna=False):
            source_name = str(source_kind or "unknown").lower()
            for column in [c for c in STATION_VALUE_COLUMNS if c in source_group.columns]:
                values_by_time = source_group.groupby("time_index")[column].mean()
                target_values = source_group[source_group["station_slot"].eq(0)].sort_values("time_index")[column]
                for window in windows:
                    suffix = str(window)
                    target_tail = target_values.tail(window).dropna()
                    prefix = f"station_{source_name}_{column}"
                    if not target_tail.empty:
                        item[f"{prefix}_last_{suffix}"] = float(target_tail.iloc[-1])
                        item[f"{prefix}_mean_{suffix}"] = float(target_tail.mean())
                        item[f"{prefix}_trend_{suffix}"] = float(target_tail.iloc[-1] - target_tail.iloc[0]) if len(target_tail) > 1 else 0.0
                    all_tail = values_by_time.tail(window).dropna()
                    if not all_tail.empty:
                        item[f"{prefix}_allslot_mean_{suffix}"] = float(all_tail.mean())
                        item[f"{prefix}_allslot_min_{suffix}"] = float(all_tail.min())
                        item[f"{prefix}_allslot_max_{suffix}"] = float(all_tail.max())
                        item[f"{prefix}_allslot_std_{suffix}"] = float(all_tail.std(ddof=0)) if len(all_tail) > 1 else 0.0
            if source_name == "context":
                last_context = source_group.sort_values("time_index").groupby("station_slot").tail(1)
                if "distance_km" in last_context.columns:
                    item["station_context_distance_mean"] = float(numeric(last_context["distance_km"], pd).mean())
                item["station_context_slot_count"] = int(last_context["station_slot"].nunique())
                for column in ("wind_mean_ms", "gust_ms", "temperature_c", "pressure_hpa"):
                    if column in last_context.columns:
                        vals = numeric(last_context[column], pd).dropna()
                        if not vals.empty:
                            item[f"station_context_{column}_last_mean"] = float(vals.mean())
                            item[f"station_context_{column}_last_min"] = float(vals.min())
                            item[f"station_context_{column}_last_max"] = float(vals.max())
        rows.append(item)
    return pd.DataFrame(rows).replace({np.inf: np.nan, -np.inf: np.nan})


def aggregate_long(frame: Any, value_columns: list[str], prefix: str, deps: dict[str, Any], group_extra: str | None = None) -> Any:
    pd = deps["pd"]
    np = deps["np"]
    if frame.empty or "sample_id" not in frame.columns:
        return pd.DataFrame()
    columns = [column for column in value_columns if column in frame.columns]
    if not columns:
        return frame[["sample_id"]].drop_duplicates()
    working = frame.copy()
    for column in columns:
        working[column] = numeric(working[column], pd)
    if group_extra and group_extra in working.columns:
        groups = working.groupby(["sample_id", group_extra])[columns].agg(["count", "mean", "min", "max"]).reset_index()
        groups.columns = ["sample_id", group_extra] + [f"{prefix}_{group_extra}_{column}_{stat}" for column, stat in groups.columns[2:]]
        # Keep this simple: aggregate again over the extra groups.
        numeric_cols = [c for c in groups.columns if c not in {"sample_id", group_extra}]
        out = groups.groupby("sample_id")[numeric_cols].mean().reset_index()
    else:
        out = working.groupby("sample_id")[columns].agg(["count", "mean", "min", "max"]).reset_index()
        out.columns = ["sample_id"] + [f"{prefix}_{column}_{stat}" for column, stat in out.columns[1:]]
    return out.replace({np.inf: np.nan, -np.inf: np.nan})


def static_features(static: Any, max_static: int, pd: Any) -> Any:
    if static.empty:
        return pd.DataFrame()
    keep = ["sample_id"]
    for column in static.columns:
        if column == "sample_id":
            continue
        if column in KEY_COLUMNS:
            continue
        if any(column.startswith(prefix) for prefix in ("target_", "residual_", "baseline_")):
            continue
        values = numeric(static[column], pd)
        if values.notna().sum() > 0 and values.nunique(dropna=True) > 1:
            keep.append(column)
        if len(keep) > max_static:
            break
    return static[keep].copy()


def merge_sample(base: Any, right: Any) -> Any:
    if right.empty or "sample_id" not in right.columns:
        return base
    drop_cols = [c for c in ("spot_id", "issue_time_utc", "split") if c in right.columns]
    return base.merge(right.drop(columns=drop_cols, errors="ignore"), on="sample_id", how="left")


def build_frame(root: Path, args: argparse.Namespace, deps: dict[str, Any]) -> Any:
    pd = deps["pd"]
    np = deps["np"]
    future = read_table(root, "future_targets", pd)
    if future.empty:
        raise SystemExit(f"Missing future_targets.parquet in {root}")
    future = future.copy()
    future["issue_time"] = pd.to_datetime(future["issue_time_utc"], utc=True, errors="coerce")
    future["lead_time_minutes"] = numeric(future["lead_time_minutes"], pd)
    samples = read_table(root, "samples", pd)
    sample_keep = [c for c in ("sample_id", "spot_id", "spot_kind", "spot_source_type", "station_id", "latitude", "longitude") if c in samples.columns]
    features = [
        samples[sample_keep].copy() if sample_keep else pd.DataFrame(),
        station_sequence_features(read_table(root, "station_sequence", pd), parse_windows(args.history_windows), deps),
        aggregate_long(read_table(root, "nwp_surface_offsets", pd), NWP_OFFSET_COLUMNS, "nwp_offset", deps),
        aggregate_long(read_table(root, "nwp_vertical_profile", pd), VERTICAL_COLUMNS, "vertical", deps, group_extra="pressure_hpa"),
        static_features(read_table(root, "static_context", pd), args.max_static_features, pd),
    ]
    frame = future
    for table in features:
        frame = merge_sample(frame, table)
    split = frame.get("split")
    frame["benchmark_split"] = split.astype(str).where(split.astype(str).eq("train"), "eval") if split is not None else "train"
    hour = frame["issue_time"].dt.hour.astype(float) + frame["issue_time"].dt.minute.astype(float) / 60.0
    frame["issue_hour_sin"] = np.sin(2.0 * math.pi * hour / 24.0)
    frame["issue_hour_cos"] = np.cos(2.0 * math.pi * hour / 24.0)
    day = frame["issue_time"].dt.dayofyear.fillna(1).astype(float)
    frame["issue_dayofyear_sin"] = np.sin(2.0 * math.pi * day / 366.0)
    frame["issue_dayofyear_cos"] = np.cos(2.0 * math.pi * day / 366.0)
    return frame


def forbidden(column: str) -> bool:
    if column in KEY_COLUMNS:
        return True
    if any(column.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        return True
    if "target_observation" in column:
        return True
    return False


def select_columns(frame: Any, target: dict[str, str], args: argparse.Namespace, deps: dict[str, Any]) -> tuple[list[str], list[str]]:
    pd = deps["pd"]
    np = deps["np"]
    train = frame[frame["benchmark_split"].eq("train")]
    if args.feature_score_sample_size and len(train) > args.feature_score_sample_size:
        train = train.sample(args.feature_score_sample_size, random_state=args.random_state)
    y = numeric(train[target["residual"]], pd)
    categorical = [c for c in DEFAULT_CATEGORICAL if c in frame.columns and not forbidden(c)]
    scores = []
    force_prefixes = ("baseline_", "station_", "nwp_offset_", "vertical_", "thermal_", "dem_", "fetch_", "landsea_")
    for column in frame.columns:
        if forbidden(column) or column in categorical:
            continue
        values = numeric(train[column], pd)
        valid = values.notna() & y.notna()
        count = int(valid.sum())
        if count < args.min_feature_non_null or values[valid].nunique(dropna=True) <= 1:
            continue
        corr = 0.0
        if count >= 3:
            value = np.corrcoef(values[valid].to_numpy(dtype=float), y[valid].to_numpy(dtype=float))[0, 1]
            if np.isfinite(value):
                corr = abs(float(value))
        if column.startswith(force_prefixes):
            corr += 0.05
        scores.append((corr, count / max(1, len(train)), column))
    scores.sort(reverse=True)
    return [column for _score, _coverage, column in scores[: args.max_numeric_features]], categorical


def one_hot(deps: dict[str, Any]) -> Any:
    OneHotEncoder = deps["OneHotEncoder"]
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def pipeline(model_family: str, numeric_cols: list[str], categorical_cols: list[str], args: argparse.Namespace, deps: dict[str, Any]) -> Any:
    Pipeline = deps["Pipeline"]
    ColumnTransformer = deps["ColumnTransformer"]
    SimpleImputer = deps["SimpleImputer"]
    StandardScaler = deps["StandardScaler"]
    Ridge = deps["Ridge"]
    HistGradientBoostingRegressor = deps["HistGradientBoostingRegressor"]
    ExtraTreesRegressor = deps["ExtraTreesRegressor"]
    transformers = []
    if numeric_cols:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if model_family == "ridge":
            steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot(deps))]), categorical_cols))
    prep = ColumnTransformer(transformers=transformers, remainder="drop")
    if model_family == "ridge":
        model = Ridge(alpha=args.ridge_alpha)
    elif model_family == "hgb":
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
    else:
        raise SystemExit(f"Unsupported model family: {model_family}")
    return Pipeline([("prep", prep), ("model", model)])


def train_target(frame: Any, target_name: str, args: argparse.Namespace, deps: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    pd = deps["pd"]
    np = deps["np"]
    cfg = TARGETS[target_name]
    working = frame.dropna(subset=[cfg["actual"], cfg["baseline"], cfg["residual"]]).copy()
    train = working[working["benchmark_split"].eq("train")]
    eval_frame = working[~working["benchmark_split"].eq("train")]
    if train.empty or eval_frame.empty:
        raise SystemExit(f"Not enough train/eval rows for {target_name}: {len(train)}/{len(eval_frame)}")
    numeric_cols, categorical_cols = select_columns(working, cfg, args, deps)
    feature_cols = numeric_cols + categorical_cols
    predictions = working[["sample_id", "spot_id", "issue_time_utc", "target_time_utc", "lead_time_minutes", "benchmark_split", cfg["actual"], cfg["baseline"]]].copy()
    predictions = predictions.rename(columns={cfg["actual"]: f"actual_{target_name}", cfg["baseline"]: f"raw_{target_name}"})
    if cfg["persistence_feature"] in working.columns:
        predictions[f"persist_{target_name}"] = numeric(working[cfg["persistence_feature"]], pd)
    global_bias = float(numeric(train[cfg["residual"]], pd).mean())
    lead_bias = train.groupby("lead_time_minutes")[cfg["residual"]].mean().to_dict()
    spot_lead_bias = train.groupby(["spot_id", "lead_time_minutes"])[cfg["residual"]].mean().to_dict()
    predictions[f"bias_spot_lead_{target_name}"] = [
        float(base) + float(spot_lead_bias.get((spot, lead), lead_bias.get(lead, global_bias)))
        for spot, lead, base in zip(working["spot_id"], working["lead_time_minutes"], numeric(working[cfg["baseline"]], pd))
    ]
    model_summaries = {}
    train_fit = train.sample(args.max_train_rows, random_state=args.random_state) if args.max_train_rows and len(train) > args.max_train_rows else train
    for family in args.model_family:
        pipe = pipeline(family, numeric_cols, categorical_cols, args, deps)
        pipe.fit(train_fit[feature_cols], numeric(train_fit[cfg["residual"]], pd))
        residual_pred = pipe.predict(working[feature_cols])
        predictions[f"{family}_{target_name}"] = numeric(working[cfg["baseline"]], pd).to_numpy(dtype=float) + residual_pred
        model_summaries[family] = {"feature_count": len(feature_cols), "numeric_feature_count": len(numeric_cols), "categorical_feature_count": len(categorical_cols), "train_rows": int(len(train_fit))}
    eval_pred = predictions[~predictions["benchmark_split"].eq("train")]
    train_pred = predictions[predictions["benchmark_split"].eq("train")]
    actual = f"actual_{target_name}"
    pred_cols = [f"raw_{target_name}", f"bias_spot_lead_{target_name}"]
    if f"persist_{target_name}" in predictions.columns:
        pred_cols.insert(1, f"persist_{target_name}")
    pred_cols.extend([f"{family}_{target_name}" for family in args.model_family])
    summary = {
        "target": target_name,
        "rows": int(len(working)),
        "train_rows": int(len(train)),
        "eval_rows": int(len(eval_frame)),
        "selected_numeric_features": numeric_cols,
        "selected_categorical_features": categorical_cols,
        "models": model_summaries,
        "metrics": {
            "train": {column: metric(np, train_pred, column, actual) for column in pred_cols},
            "eval": {column: metric(np, eval_pred, column, actual) for column in pred_cols},
            "eval_by_lead": {column: metrics_by_group(np, eval_pred, column, actual, "lead_time_minutes") for column in pred_cols},
            "eval_by_spot": {column: metrics_by_group(np, eval_pred, column, actual, "spot_id") for column in pred_cols},
        },
    }
    return predictions, summary


def best_eval_models(targets: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for target, result in targets.items():
        candidates = [(metric.get("rmse"), name, metric) for name, metric in result["metrics"]["eval"].items() if metric.get("count", 0) and metric.get("rmse") is not None]
        candidates = [item for item in candidates if item[0] is not None]
        if candidates:
            rmse, name, metric_item = sorted(candidates, key=lambda x: x[0])[0]
            out[target] = {"model": name, **metric_item}
    return out


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# CorseWind SAPHIR Dictionary V2 Tabular Benchmark",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Dataset root: `{result['dataset_root']}`",
        "",
        "## Eval Metrics",
        "",
        "| Target | Model | RMSE | MAE | Bias | Count |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for target, target_result in result["targets"].items():
        for model_name, item in target_result["metrics"]["eval"].items():
            lines.append(f"| `{target}` | `{model_name}` | {item.get('rmse')} | {item.get('mae')} | {item.get('bias')} | {item.get('count')} |")
    lines.extend(["", "## Best Eval Models", ""])
    for target, item in result["best_eval_models"].items():
        lines.append(f"- `{target}`: `{item.get('model')}` RMSE `{item.get('rmse')}`, MAE `{item.get('mae')}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-family", action="append", default=["ridge", "hgb"])
    parser.add_argument("--history-windows", default="4,8,16,32")
    parser.add_argument("--max-numeric-features", type=int, default=900)
    parser.add_argument("--max-static-features", type=int, default=256)
    parser.add_argument("--min-feature-non-null", type=int, default=50)
    parser.add_argument("--feature-score-sample-size", type=int, default=80000)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--hgb-max-iter", type=int, default=180)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.06)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--hgb-l2", type=float, default=0.01)
    parser.add_argument("--extra-trees-estimators", type=int, default=180)
    parser.add_argument("--extra-trees-min-samples-leaf", type=int, default=6)
    parser.add_argument("--extra-trees-max-features", default="sqrt")
    parser.add_argument("--ridge-alpha", type=float, default=30.0)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = import_deps()
    output_root = (args.output_root or args.dataset_root / "benchmark_v2_tabular").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frame = build_frame(args.dataset_root, args, deps)
    frame.to_parquet(output_root / "flat_features.parquet", index=False)
    targets = {}
    prediction_frames = []
    for target in TARGETS:
        predictions, summary = train_target(frame, target, args, deps)
        targets[target] = summary
        prediction_frames.append(predictions)
    pred = prediction_frames[0]
    for extra in prediction_frames[1:]:
        merge_cols = ["sample_id", "spot_id", "issue_time_utc", "target_time_utc", "lead_time_minutes", "benchmark_split"]
        pred = pred.merge(extra, on=merge_cols, how="outer")
    pred.to_parquet(output_root / "predictions_tabular.parquet", index=False)
    result = {
        "format": "corsewind.saphir_dictionary_v2_tabular_benchmark.v1",
        "generated_at_utc": utc_now(),
        "dataset_root": str(args.dataset_root.resolve()),
        "output_root": str(output_root),
        "row_count": int(len(frame)),
        "split_counts": frame["benchmark_split"].value_counts().to_dict(),
        "targets": targets,
        "best_eval_models": best_eval_models(targets),
        "args": vars(args),
    }
    (output_root / "benchmark_results.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_markdown(output_root / "benchmark_results.md", result)
    print(json.dumps({"output_root": str(output_root), "best_eval_models": result["best_eval_models"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
