#!/usr/bin/env python3
"""Score saved sequence benchmark cases with residual HGB models."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_CONFIG = {
    "wind_mean_ms": {
        "actual": "actual_wind_mean_ms",
        "baseline_prediction": "raw_wind_mean_ms",
        "training_baseline": "baselines__baseline_wind_mean_ms",
        "model_file": "labels__residual_wind_mean_ms.joblib",
        "output": "hgb_wind_mean_ms",
    },
    "gust_ms": {
        "actual": "actual_gust_ms",
        "baseline_prediction": "raw_gust_ms",
        "training_baseline": "baselines__baseline_gust_ms",
        "model_file": "labels__residual_gust_ms.joblib",
        "output": "hgb_gust_ms",
    },
}


def import_dependencies() -> dict[str, Any]:
    try:
        import joblib
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Run inside corsewind-ml-dataset-runner or install requirements-ml-dataset.txt."
        ) from exc
    return {"joblib": joblib, "np": np, "pd": pd, "pq": pq}


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


def discover_training_paths(args: argparse.Namespace) -> list[Path]:
    paths = []
    for suffix in month_range(args.start_month, args.end_month):
        path = args.training_table_root / f"{args.run_id_prefix}_{suffix}" / "training_rows.parquet"
        if path.exists():
            paths.append(path)
    if not paths:
        raise SystemExit("No training parquet shards found.")
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


def normalize_issue_time(frame: Any, pd: Any, column: str = "issue_time_utc") -> Any:
    dt = pd.to_datetime(frame[column], utc=True, errors="coerce")
    frame[column] = dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return frame


def read_matching_training_rows(paths: list[Path], columns: list[str], wanted_keys: set[tuple[str, str, int]], deps: dict[str, Any], args: argparse.Namespace) -> Any:
    pd = deps["pd"]
    np = deps["np"]
    pq = deps["pq"]
    frames = []
    read_columns = sorted(set(columns) | {"spot_id", "issue_time_utc", "lead_time_minutes"})
    for path in paths:
        pf = pq.ParquetFile(path)
        available = [column for column in read_columns if column in pf.schema.names]
        for batch in pf.iter_batches(batch_size=args.read_batch_size, columns=available):
            frame = batch.to_pandas().reindex(columns=read_columns)
            frame = normalize_issue_time(frame, pd)
            frame["lead_time_minutes"] = pd.to_numeric(frame["lead_time_minutes"], errors="coerce").astype("Int64")
            keys = list(zip(frame["spot_id"].astype(str), frame["issue_time_utc"].astype(str), frame["lead_time_minutes"].astype("int64")))
            mask = [key in wanted_keys for key in keys]
            if any(mask):
                frames.append(frame.loc[mask].copy())
    if not frames:
        raise SystemExit("No matching training rows found for benchmark cases.")
    out = pd.concat(frames, ignore_index=True)
    out = add_time_features(out, pd, np)
    return out.drop_duplicates(["spot_id", "issue_time_utc", "lead_time_minutes"], keep="last")


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


def summarize_metrics(frame: Any, deps: dict[str, Any], *, include_by_lead: bool = True) -> dict[str, Any]:
    np = deps["np"]
    summary: dict[str, Any] = {}
    for target, config in TARGET_CONFIG.items():
        target_summary = {
            "raw_nwp": metric(np, frame, config["baseline_prediction"], config["actual"]),
            "hgb": metric(np, frame, config["output"], config["actual"]),
        }
        chronos_column = f"chronos_{target}_p50"
        if chronos_column in frame.columns:
            target_summary["chronos_p50"] = metric(np, frame, chronos_column, config["actual"])
        summary[target] = target_summary
    if not include_by_lead:
        return summary

    by_lead = {}
    for lead, group in frame.groupby("lead_time_minutes"):
        by_lead[str(int(lead))] = summarize_metrics(group, deps, include_by_lead=False)
    return {"overall": summary, "by_lead": by_lead}


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Sequence Benchmark With HGB",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Run id: `{result['run_id']}`",
        "",
        "| Target | Model | RMSE | MAE | Bias | Count |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for target, models in result["metrics"]["overall"].items():
        for model_name, item in models.items():
            lines.append(
                f"| `{target}` | `{model_name}` | `{item.get('rmse')}` | `{item.get('mae')}` | `{item.get('bias')}` | `{item.get('count')}` |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--training-table-root", type=Path, required=True)
    parser.add_argument("--run-id-prefix", default="residual_windsup_sst_prev")
    parser.add_argument("--start-month", required=True)
    parser.add_argument("--end-month", required=True)
    parser.add_argument("--hgb-model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id", default="sequence_hgb_scored")
    parser.add_argument("--read-batch-size", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = import_dependencies()
    pd = deps["pd"]
    joblib = deps["joblib"]

    output_root = args.output_root or args.benchmark_root
    output_root.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(args.benchmark_root / "predictions.parquet")
    predictions = normalize_issue_time(predictions, pd)
    predictions["lead_time_minutes"] = pd.to_numeric(predictions["lead_time_minutes"], errors="coerce").astype("Int64")

    wanted_keys = set(
        zip(
            predictions["spot_id"].astype(str),
            predictions["issue_time_utc"].astype(str),
            predictions["lead_time_minutes"].astype("int64"),
        )
    )
    feature_config = json.loads((args.hgb_model_root / "feature_columns.json").read_text(encoding="utf-8"))
    feature_columns = sorted(set(feature_config.get("numeric", [])) | set(feature_config.get("categorical", [])))
    needed_columns = feature_columns + [config["training_baseline"] for config in TARGET_CONFIG.values()]

    training_rows = read_matching_training_rows(discover_training_paths(args), needed_columns, wanted_keys, deps, args)
    hgb_features = training_rows.reindex(columns=feature_columns)
    for _target, config in TARGET_CONFIG.items():
        model_path = args.hgb_model_root / config["model_file"]
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        residual = model.predict(hgb_features)
        training_rows[config["output"]] = training_rows[config["training_baseline"]].astype(float).to_numpy() + residual

    scored = predictions.drop(columns=[config["output"] for config in TARGET_CONFIG.values() if config["output"] in predictions.columns])
    predicted_columns = [config["output"] for config in TARGET_CONFIG.values() if config["output"] in training_rows.columns]
    hgb_columns = ["spot_id", "issue_time_utc", "lead_time_minutes"] + predicted_columns
    scored = scored.merge(training_rows[hgb_columns], on=["spot_id", "issue_time_utc", "lead_time_minutes"], how="left")
    scored.to_parquet(output_root / "predictions_with_hgb.parquet", index=False)

    result = {
        "generated_at_utc": utc_now(),
        "run_id": args.run_id,
        "benchmark_root": str(args.benchmark_root),
        "hgb_model_root": str(args.hgb_model_root),
        "row_count": int(len(scored)),
        "hgb_scored_rows": int(scored[predicted_columns].dropna(how="all").shape[0]) if predicted_columns else 0,
        "metrics": summarize_metrics(scored, deps),
    }
    (output_root / "benchmark_results_with_hgb.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_markdown(output_root / "benchmark_results_with_hgb.md", result)
    print(json.dumps({"run_id": result["run_id"], "row_count": result["row_count"], "hgb_scored_rows": result["hgb_scored_rows"], "overall": result["metrics"]["overall"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
