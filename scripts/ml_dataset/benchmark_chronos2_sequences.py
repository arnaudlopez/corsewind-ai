#!/usr/bin/env python3
"""Benchmark Chronos-2 sequence forecasts against raw NWP and HGB residual models."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_COLUMNS = {
    "wind_mean_ms": "labels__target_wind_mean_ms",
    "gust_ms": "labels__target_gust_ms",
}
BASELINE_COLUMNS = {
    "wind_mean_ms": "baselines__baseline_wind_mean_ms",
    "gust_ms": "baselines__baseline_gust_ms",
}
HGB_TARGET_MODELS = {
    "wind_mean_ms": "labels__residual_wind_mean_ms.joblib",
    "gust_ms": "labels__residual_gust_ms.joblib",
}
FUTURE_COVARIATES = {
    "nwp_wind_mean_ms": "baselines__baseline_wind_mean_ms",
    "nwp_gust_ms": "baselines__baseline_gust_ms",
    "nwp_temperature_2m_c": "baselines__baseline_temperature_2m_c",
    "nwp_pressure_msl_hpa": "baselines__baseline_pressure_msl_hpa",
    "nwp_cloud_cover_pct": "baselines__baseline_cloud_cover_pct",
    "nwp_shortwave_radiation": "baselines__baseline_shortwave_radiation",
}


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def import_dependencies():
    try:
        import joblib
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
        import torch
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing benchmark dependencies. Run with the z2 Chronos virtualenv "
            "(/home/z2/corsewind-ml-smoke/.venv)."
        ) from exc
    return {
        "joblib": joblib,
        "np": np,
        "pd": pd,
        "pq": pq,
        "torch": torch,
        "Chronos2Pipeline": Chronos2Pipeline,
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


def discover_parquets(root: Path, prefix: str, start_month: str, end_month: str) -> list[Path]:
    paths = []
    for suffix in month_range(start_month, end_month):
        path = root / f"{prefix}_{suffix}" / "training_rows.parquet"
        if path.exists():
            paths.append(path)
    if not paths:
        raise SystemExit("No training_rows.parquet shards found.")
    return paths


def read_parquet_columns(paths: list[Path], columns: list[str], deps: dict[str, Any]):
    pd = deps["pd"]
    pq = deps["pq"]
    columns = unique_preserve_order(columns)
    frames = []
    for path in paths:
        pf = pq.ParquetFile(path)
        available = [column for column in columns if column in pf.schema.names]
        if not available:
            continue
        frame = pf.read(columns=available).to_pandas().reindex(columns=columns)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def load_actual_series(paths: list[Path], deps: dict[str, Any]):
    pd = deps["pd"]
    columns = unique_preserve_order([
        "spot_id",
        "target_time_utc",
        "lead_time_minutes",
        TARGET_COLUMNS["wind_mean_ms"],
        TARGET_COLUMNS["gust_ms"],
        *FUTURE_COVARIATES.values(),
    ])
    frame = read_parquet_columns(paths, columns, deps)
    frame = frame[frame["lead_time_minutes"] == 15].copy()
    frame["timestamp"] = pd.to_datetime(frame["target_time_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["spot_id", "timestamp"])
    frame = frame.rename(columns={
        TARGET_COLUMNS["wind_mean_ms"]: "wind_mean_ms",
        TARGET_COLUMNS["gust_ms"]: "gust_ms",
        **{source: name for name, source in FUTURE_COVARIATES.items()},
    })
    keep = ["spot_id", "timestamp", "wind_mean_ms", "gust_ms", *FUTURE_COVARIATES]
    frame = frame[keep].sort_values(["spot_id", "timestamp"])
    return frame.drop_duplicates(["spot_id", "timestamp"], keep="last")


def load_candidate_rows(paths: list[Path], deps: dict[str, Any], args: argparse.Namespace):
    pd = deps["pd"]
    horizon_minutes = [step * args.freq_minutes for step in range(1, args.prediction_length + 1)]
    columns = unique_preserve_order([
        "spot_id",
        "issue_time_utc",
        "target_time_utc",
        "lead_time_minutes",
        TARGET_COLUMNS["wind_mean_ms"],
        TARGET_COLUMNS["gust_ms"],
        *BASELINE_COLUMNS.values(),
        *FUTURE_COVARIATES.values(),
    ])
    frame = read_parquet_columns(paths, columns, deps)
    frame = frame[frame["lead_time_minutes"].isin(horizon_minutes)].copy()
    frame["issue_time"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    frame["target_time"] = pd.to_datetime(frame["target_time_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["spot_id", "issue_time", "target_time", TARGET_COLUMNS["wind_mean_ms"], TARGET_COLUMNS["gust_ms"]])
    if args.eval_start:
        frame = frame[frame["issue_time"] >= pd.Timestamp(args.eval_start, tz="UTC")]
    if args.eval_end:
        frame = frame[frame["issue_time"] <= pd.Timestamp(args.eval_end, tz="UTC")]
    if args.issue_hour_start is not None:
        frame = frame[frame["issue_time"].dt.hour >= args.issue_hour_start]
    if args.issue_hour_end is not None:
        frame = frame[frame["issue_time"].dt.hour <= args.issue_hour_end]
    if args.spot_id:
        frame = frame[frame["spot_id"].isin(set(args.spot_id))]
    if args.candidate_keys_parquet:
        keys = pd.read_parquet(args.candidate_keys_parquet)
        keys["spot_id"] = keys["spot_id"].astype(str)
        keys["issue_time"] = pd.to_datetime(keys["issue_time_utc"], utc=True, errors="coerce")
        key_columns = ["spot_id", "issue_time"]
        if "lead_time_minutes" in keys.columns:
            keys["lead_time_minutes"] = keys["lead_time_minutes"].astype("Int64")
            key_columns.append("lead_time_minutes")
        keys = keys.dropna(subset=["spot_id", "issue_time"]).drop_duplicates(key_columns)
        frame["spot_id"] = frame["spot_id"].astype(str)
        frame = frame.merge(keys[key_columns], on=key_columns, how="inner")
    return frame.sort_values(["spot_id", "issue_time", "lead_time_minutes"])


def interpolate_context(group, start_time, end_time, freq: str, context_length: int, covariates: list[str], pd: Any):
    index = pd.date_range(end=end_time, periods=context_length, freq=freq, tz="UTC")
    context = group.set_index("timestamp").sort_index().reindex(index)
    value_columns = ["wind_mean_ms", "gust_ms", *covariates]
    context[value_columns] = context[value_columns].interpolate(method="time", limit=4).ffill().bfill()
    if context[["wind_mean_ms", "gust_ms"]].isna().any().any():
        return None
    context = context.reset_index(names="timestamp")
    return context


def interpolate_future_rows(group, actual_group, issue_time, args: argparse.Namespace, pd: Any):
    freq = f"{args.freq_minutes}min"
    future_index = pd.date_range(
        start=issue_time + pd.Timedelta(minutes=args.freq_minutes),
        periods=args.prediction_length,
        freq=freq,
        tz="UTC",
    )
    actual = actual_group.set_index("timestamp").sort_index().reindex(future_index)
    if actual[["wind_mean_ms", "gust_ms"]].isna().any().any():
        return None

    value_columns = unique_preserve_order([*BASELINE_COLUMNS.values(), *FUTURE_COVARIATES.values()])
    source = group.set_index("target_time").sort_index()
    available = [column for column in value_columns if column in source.columns]
    if not available:
        return None
    covariates = source[available].apply(pd.to_numeric, errors="coerce")
    if covariates.dropna(how="all").empty:
        return None
    interpolated_index = covariates.index.union(future_index).sort_values()
    covariates = covariates.reindex(interpolated_index).interpolate(method="time").ffill().bfill().reindex(future_index)
    if covariates[available].isna().any().any():
        return None

    future = covariates.reset_index(names="target_time")
    future["spot_id"] = group["spot_id"].iloc[0]
    future["issue_time_utc"] = issue_time.isoformat().replace("+00:00", "Z")
    future["target_time_utc"] = future["target_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    future["lead_time_minutes"] = [
        int((timestamp - issue_time).total_seconds() // 60)
        for timestamp in future["target_time"]
    ]
    future[TARGET_COLUMNS["wind_mean_ms"]] = actual["wind_mean_ms"].to_numpy(dtype=float)
    future[TARGET_COLUMNS["gust_ms"]] = actual["gust_ms"].to_numpy(dtype=float)
    return future


def select_cases(candidate_rows, actual_series, args: argparse.Namespace, deps: dict[str, Any]):
    pd = deps["pd"]
    freq = f"{args.freq_minutes}min"
    by_spot_actual = {spot: group.copy() for spot, group in actual_series.groupby("spot_id")}
    cases = []
    skipped = defaultdict(int)
    required_leads = set(step * args.freq_minutes for step in range(1, args.prediction_length + 1))
    for (spot_id, issue_time), group in candidate_rows.groupby(["spot_id", "issue_time"]):
        leads = set(int(value) for value in group["lead_time_minutes"].dropna().tolist())
        if args.allow_interpolated_future:
            max_horizon = args.prediction_length * args.freq_minutes
            if args.freq_minutes not in leads or max(leads or {0}) < max_horizon:
                skipped["missing_future_interpolation_anchors"] += 1
                continue
        elif not required_leads.issubset(leads):
            skipped["missing_horizon_rows"] += 1
            continue
        actual_group = by_spot_actual.get(spot_id)
        if actual_group is None:
            skipped["missing_spot_actual_series"] += 1
            continue
        context = interpolate_context(
            actual_group,
            issue_time - pd.Timedelta(minutes=args.freq_minutes * (args.context_length - 1)),
            issue_time,
            freq,
            args.context_length,
            list(FUTURE_COVARIATES),
            pd,
        )
        if context is None or len(context) != args.context_length:
            skipped["bad_context"] += 1
            continue
        if args.allow_interpolated_future:
            future = interpolate_future_rows(group, actual_group, issue_time, args, pd)
            if future is None or len(future) != args.prediction_length:
                skipped["bad_interpolated_future"] += 1
                continue
        else:
            future = group.sort_values("lead_time_minutes").head(args.prediction_length).copy()
        item_id = f"{spot_id}|{issue_time.isoformat()}"
        cases.append({
            "item_id": item_id,
            "spot_id": spot_id,
            "issue_time": issue_time,
            "context": context,
            "future": future,
        })
    selected = []
    for spot_id, spot_cases in defaultdict(list, {spot: [case for case in cases if case["spot_id"] == spot] for spot in sorted({case["spot_id"] for case in cases})}).items():
        spot_cases = sorted(spot_cases, key=lambda item: item["issue_time"])
        if args.max_cutoffs_per_spot and len(spot_cases) > args.max_cutoffs_per_spot:
            if args.max_cutoffs_per_spot == 1:
                indexes = [len(spot_cases) - 1]
            else:
                indexes = [round(i * (len(spot_cases) - 1) / (args.max_cutoffs_per_spot - 1)) for i in range(args.max_cutoffs_per_spot)]
            spot_cases = [spot_cases[int(index)] for index in indexes]
        selected.extend(spot_cases)
    if args.max_cases and len(selected) > args.max_cases:
        selected = selected[:args.max_cases]
    return selected, dict(skipped)


def build_chronos_frames(cases: list[dict[str, Any]], deps: dict[str, Any]):
    pd = deps["pd"]
    past_frames = []
    future_frames = []
    truth_frames = []
    for case in cases:
        context = case["context"].copy()
        context["item_id"] = case["item_id"]
        past_frames.append(context[["item_id", "timestamp", "wind_mean_ms", "gust_ms", *FUTURE_COVARIATES]])
        future = case["future"].copy()
        future["item_id"] = case["item_id"]
        future["actual_wind_mean_ms"] = future[TARGET_COLUMNS["wind_mean_ms"]]
        future["actual_gust_ms"] = future[TARGET_COLUMNS["gust_ms"]]
        future["raw_wind_mean_ms"] = future[BASELINE_COLUMNS["wind_mean_ms"]]
        future["raw_gust_ms"] = future[BASELINE_COLUMNS["gust_ms"]]
        future = future.rename(columns={
            **{source: name for name, source in FUTURE_COVARIATES.items()},
        })
        future["timestamp"] = future["target_time"]
        future["spot_id"] = case["spot_id"]
        future["issue_time_utc"] = case["issue_time"].isoformat().replace("+00:00", "Z")
        future_frames.append(future[["item_id", "timestamp", *FUTURE_COVARIATES]])
        truth_frames.append(future[[
            "item_id",
            "spot_id",
            "issue_time_utc",
            "timestamp",
            "lead_time_minutes",
            "actual_wind_mean_ms",
            "actual_gust_ms",
            "raw_wind_mean_ms",
            "raw_gust_ms",
        ]])
    past_df = pd.concat(past_frames, ignore_index=True)
    future_df = pd.concat(future_frames, ignore_index=True)
    truth_df = pd.concat(truth_frames, ignore_index=True)
    for frame in (past_df, future_df, truth_df):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    return past_df, future_df, truth_df


def load_hgb_features(paths: list[Path], cases_truth, model_root: Path, deps: dict[str, Any]):
    pd = deps["pd"]
    pq = deps["pq"]
    feature_info = json.loads((model_root / "feature_columns.json").read_text())
    feature_columns = unique_preserve_order([*feature_info["numeric"], *feature_info["categorical"]])
    key_columns = ["spot_id", "issue_time_utc", "lead_time_minutes"]
    needed = set(
        f"{row.spot_id}|{row.issue_time_utc}|{int(row.lead_time_minutes)}"
        for row in cases_truth.itertuples(index=False)
    )
    frames = []
    for path in paths:
        pf = pq.ParquetFile(path)
        selected_columns = unique_preserve_order([*key_columns, *feature_columns])
        available = [column for column in selected_columns if column in pf.schema.names]
        for batch in pf.iter_batches(batch_size=50000, columns=available):
            frame = batch.to_pandas().reindex(columns=selected_columns)
            keys = frame["spot_id"].astype(str) + "|" + frame["issue_time_utc"].astype(str) + "|" + frame["lead_time_minutes"].astype("Int64").astype(str)
            keep = frame[keys.isin(needed)]
            if not keep.empty:
                frames.append(keep)
    if not frames:
        return pd.DataFrame(columns=[*key_columns, *feature_columns])
    return pd.concat(frames, ignore_index=True).drop_duplicates(key_columns, keep="last")


def add_hgb_predictions(paths: list[Path], truth, model_root: Path, deps: dict[str, Any]):
    joblib = deps["joblib"]
    features = load_hgb_features(paths, truth, model_root, deps)
    if features.empty:
        truth["hgb_wind_mean_ms"] = math.nan
        truth["hgb_gust_ms"] = math.nan
        return truth
    feature_info = json.loads((model_root / "feature_columns.json").read_text())
    feature_columns = unique_preserve_order([*feature_info["numeric"], *feature_info["categorical"]])
    merged = truth.merge(features, on=["spot_id", "issue_time_utc", "lead_time_minutes"], how="left")
    for target, filename in HGB_TARGET_MODELS.items():
        model = joblib.load(model_root / filename)
        residual = model.predict(merged[feature_columns])
        raw_column = "raw_wind_mean_ms" if target == "wind_mean_ms" else "raw_gust_ms"
        out_column = "hgb_wind_mean_ms" if target == "wind_mean_ms" else "hgb_gust_ms"
        merged[out_column] = merged[raw_column].astype(float) + residual
    keep_columns = list(truth.columns) + ["hgb_wind_mean_ms", "hgb_gust_ms"]
    return merged[keep_columns]


def metrics(np: Any, frame, prediction_column: str, actual_column: str) -> dict[str, Any]:
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


def summarize_metrics(frame, deps: dict[str, Any], *, include_by_lead: bool = True) -> dict[str, Any]:
    np = deps["np"]
    summary: dict[str, Any] = {}
    pairs = {
        "wind_mean_ms": {
            "actual": "actual_wind_mean_ms",
            "raw_nwp": "raw_wind_mean_ms",
            "hgb": "hgb_wind_mean_ms",
            "chronos_p50": "chronos_wind_mean_ms_p50",
        },
        "gust_ms": {
            "actual": "actual_gust_ms",
            "raw_nwp": "raw_gust_ms",
            "hgb": "hgb_gust_ms",
            "chronos_p50": "chronos_gust_ms_p50",
        },
    }
    for target, cols in pairs.items():
        target_summary = {}
        for name, column in cols.items():
            if name == "actual":
                continue
            target_summary[name] = metrics(np, frame, column, cols["actual"])
        summary[target] = target_summary
    if not include_by_lead:
        return summary

    by_lead = {}
    for lead, group in frame.groupby("lead_time_minutes"):
        by_lead[str(int(lead))] = summarize_metrics(group, deps, include_by_lead=False)
    return {"overall": summary, "by_lead": by_lead}


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Chronos-2 Sequence Benchmark",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Run id: `{result['run_id']}`",
        "",
        "## Overall",
        "",
        "| Target | Model | RMSE | MAE | Bias | Count |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for target, models in result["metrics"]["overall"].items():
        for model_name, item in models.items():
            lines.append(f"| `{target}` | `{model_name}` | `{item.get('rmse')}` | `{item.get('mae')}` | `{item.get('bias')}` | `{item.get('count')}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-table-root", type=Path, required=True)
    parser.add_argument("--run-id-prefix", default="residual_windsup_sst_prev")
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument("--end-month", default="2026-06")
    parser.add_argument("--eval-start", default="2026-01-01T00:00:00Z")
    parser.add_argument("--eval-end")
    parser.add_argument("--issue-hour-start", type=int, help="Inclusive UTC issue-hour filter, 0-23.")
    parser.add_argument("--issue-hour-end", type=int, help="Inclusive UTC issue-hour filter, 0-23.")
    parser.add_argument("--spot-id", action="append", default=[])
    parser.add_argument("--candidate-keys-parquet", type=Path, help="Optional spot_id/issue_time_utc[/lead_time_minutes] key file used to force same-sample cases.")
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--freq-minutes", type=int, default=15)
    parser.add_argument("--allow-interpolated-future", action=argparse.BooleanOptionalAction, default=False, help="Build a full 15-minute future grid by interpolating sparse NWP future covariates.")
    parser.add_argument("--max-cutoffs-per-spot", type=int, default=12)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--chronos-model", default="amazon/chronos-2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cross-learning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hgb-model-root", type=Path)
    parser.add_argument("--skip-hgb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", default="chronos2_sequence_benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = import_dependencies()
    pd = deps["pd"]
    torch = deps["torch"]
    paths = discover_parquets(args.training_table_root, args.run_id_prefix, args.start_month, args.end_month)
    actual_series = load_actual_series(paths, deps)
    candidates = load_candidate_rows(paths, deps, args)
    cases, skipped = select_cases(candidates, actual_series, args, deps)
    if not cases:
        raise SystemExit(f"No valid benchmark cases selected. skipped={dict(skipped)}")
    past_df, future_df, truth = build_chronos_frames(cases, deps)
    if args.skip_hgb:
        truth["hgb_wind_mean_ms"] = math.nan
        truth["hgb_gust_ms"] = math.nan
    else:
        if args.hgb_model_root is None:
            raise SystemExit("--hgb-model-root is required unless --skip-hgb is set.")
        truth = add_hgb_predictions(paths, truth, args.hgb_model_root, deps)

    pipeline = deps["Chronos2Pipeline"].from_pretrained(
        args.chronos_model,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
    )
    forecast = pipeline.predict_df(
        past_df,
        future_df=future_df,
        id_column="item_id",
        timestamp_column="timestamp",
        target=["wind_mean_ms", "gust_ms"],
        prediction_length=args.prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
        batch_size=args.batch_size,
        context_length=args.context_length,
        cross_learning=args.cross_learning,
        freq=f"{args.freq_minutes}min",
    )
    pivot = forecast.pivot_table(
        index=["item_id", "timestamp"],
        columns="target_name",
        values=["predictions", "0.1", "0.5", "0.9"],
        aggfunc="last",
    )
    pivot.columns = [f"chronos_{target}_{metric if metric != '0.5' else 'p50'}" for metric, target in pivot.columns]
    pivot = pivot.reset_index()
    merged = truth.merge(pivot, on=["item_id", "timestamp"], how="left")
    merged = merged.rename(columns={
        "chronos_wind_mean_ms_predictions": "chronos_wind_mean_ms_mean",
        "chronos_gust_ms_predictions": "chronos_gust_ms_mean",
        "chronos_wind_mean_ms_0.1": "chronos_wind_mean_ms_p10",
        "chronos_wind_mean_ms_0.9": "chronos_wind_mean_ms_p90",
        "chronos_gust_ms_0.1": "chronos_gust_ms_p10",
        "chronos_gust_ms_0.9": "chronos_gust_ms_p90",
    })
    result = {
        "format": "corsewind.chronos2_sequence_benchmark.v1",
        "generated_at_utc": utc_now(),
        "run_id": args.run_id,
        "chronos_model": args.chronos_model,
        "case_count": len(cases),
        "prediction_row_count": int(len(merged)),
        "spot_count": int(merged["spot_id"].nunique()),
        "spots": sorted(merged["spot_id"].dropna().unique().tolist()),
        "settings": vars(args) | {
            "training_table_root": str(args.training_table_root),
            "hgb_model_root": str(args.hgb_model_root) if args.hgb_model_root else None,
            "output_root": str(args.output_root),
        },
        "skipped_cases": skipped,
        "metrics": summarize_metrics(merged, deps),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output_root / "predictions.parquet", index=False)
    forecast.to_parquet(args.output_root / "chronos_forecast_raw.parquet", index=False)
    past_df.to_parquet(args.output_root / "past_context.parquet", index=False)
    future_df.to_parquet(args.output_root / "future_covariates.parquet", index=False)
    (args.output_root / "benchmark_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_root / "benchmark_results.md", result)
    print(json.dumps({
        "run_id": result["run_id"],
        "case_count": result["case_count"],
        "prediction_row_count": result["prediction_row_count"],
        "spots": result["spots"],
        "overall": result["metrics"]["overall"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
