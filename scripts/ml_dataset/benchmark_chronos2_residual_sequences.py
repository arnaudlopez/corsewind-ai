#!/usr/bin/env python3
"""Benchmark Chronos-2 on observed-minus-NWP residual sequences."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGETS = {
    "wind_mean_ms": {
        "observed_context": "wind_mean_ms",
        "nwp_context": "nwp_wind_mean_ms",
        "actual": "actual_wind_mean_ms",
        "raw": "raw_wind_mean_ms",
        "output_prefix": "chronos2_residual_wind_mean_ms",
    },
    "gust_ms": {
        "observed_context": "gust_ms",
        "nwp_context": "nwp_gust_ms",
        "actual": "actual_gust_ms",
        "raw": "raw_gust_ms",
        "output_prefix": "chronos2_residual_gust_ms",
    },
}


def import_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
        import torch
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise SystemExit("Missing Chronos dependencies. Run with the Chronos virtualenv.") from exc
    return {"np": np, "pd": pd, "torch": torch, "Chronos2Pipeline": Chronos2Pipeline}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
    for target, config in TARGETS.items():
        target_summary = {
            "chronos2_residual_p50": metric(np, frame, f"{config['output_prefix']}_corrected_p50", config["actual"]),
        }
        for model_name, column in (
            ("raw_nwp", config["raw"]),
            ("hgb", f"hgb_{target}"),
            ("chronos2_univar_p50", f"chronos2_univar_{target}_p50"),
            ("chronos2_covariate_p50", f"chronos_{target}_p50"),
            ("timesfm_p50", f"timesfm_{target}_p50"),
            ("moirai_p50", f"moirai_{target}_p50"),
        ):
            if column in frame.columns:
                target_summary[model_name] = metric(np, frame, column, config["actual"])
        summary[target] = target_summary
    if not include_by_lead:
        return summary

    by_lead = {}
    for lead, group in frame.groupby("lead_time_minutes"):
        by_lead[str(int(lead))] = summarize_metrics(group, deps, include_by_lead=False)
    return {"overall": summary, "by_lead": by_lead}


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Chronos-2 Residual Sequence Benchmark",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Run id: `{result['run_id']}`",
        f"Cross learning: `{result['cross_learning']}`",
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
    parser.add_argument("--predictions-file", default="predictions_with_chronos2_univariate.parquet")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id", default="chronos2_residual_sequence_benchmark")
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--prediction-length", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--freq", default="15min")
    parser.add_argument("--cross-learning", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = import_dependencies()
    pd = deps["pd"]
    torch = deps["torch"]
    Chronos2Pipeline = deps["Chronos2Pipeline"]

    output_root = args.output_root or args.benchmark_root
    output_root.mkdir(parents=True, exist_ok=True)

    prediction_path = args.benchmark_root / args.predictions_file
    if not prediction_path.exists():
        for fallback in ("predictions_with_moirai.parquet", "predictions_with_hgb.parquet", "predictions.parquet"):
            candidate = args.benchmark_root / fallback
            if candidate.exists():
                prediction_path = candidate
                break
    predictions = pd.read_parquet(prediction_path).sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    past = pd.read_parquet(args.benchmark_root / "past_context.parquet").sort_values(["item_id", "timestamp"])
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], errors="coerce").dt.tz_localize(None)
    past["timestamp"] = pd.to_datetime(past["timestamp"], errors="coerce").dt.tz_localize(None)

    pipeline = Chronos2Pipeline.from_pretrained(
        args.model_id,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
    )

    for target, config in TARGETS.items():
        residual_context = past[["item_id", "timestamp", config["observed_context"], config["nwp_context"]]].copy()
        residual_context["target"] = residual_context[config["observed_context"]] - residual_context[config["nwp_context"]]
        residual_context = residual_context[["item_id", "timestamp", "target"]]
        forecast = pipeline.predict_df(
            residual_context,
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            prediction_length=args.prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=args.batch_size,
            context_length=args.context_length,
            cross_learning=args.cross_learning,
            freq=args.freq,
        )
        forecast = forecast.rename(columns={
            "predictions": f"{config['output_prefix']}_residual_mean",
            "0.1": f"{config['output_prefix']}_residual_p10",
            "0.5": f"{config['output_prefix']}_residual_p50",
            "0.9": f"{config['output_prefix']}_residual_p90",
        })
        keep = [
            "item_id",
            "timestamp",
            f"{config['output_prefix']}_residual_mean",
            f"{config['output_prefix']}_residual_p10",
            f"{config['output_prefix']}_residual_p50",
            f"{config['output_prefix']}_residual_p90",
        ]
        predictions = predictions.merge(forecast[keep], on=["item_id", "timestamp"], how="left")
        for suffix in ("mean", "p10", "p50", "p90"):
            predictions[f"{config['output_prefix']}_corrected_{suffix}"] = (
                predictions[config["raw"]] + predictions[f"{config['output_prefix']}_residual_{suffix}"]
            )

    predictions.to_parquet(output_root / "predictions_with_chronos2_residual.parquet", index=False)
    result = {
        "generated_at_utc": utc_now(),
        "run_id": args.run_id,
        "benchmark_root": str(args.benchmark_root),
        "model_id": args.model_id,
        "cross_learning": args.cross_learning,
        "row_count": int(len(predictions)),
        "item_count": int(predictions["item_id"].nunique()),
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "metrics": summarize_metrics(predictions, deps),
    }
    (output_root / "benchmark_results_chronos2_residual.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_markdown(output_root / "benchmark_results_chronos2_residual.md", result)
    print(json.dumps({"run_id": result["run_id"], "row_count": result["row_count"], "overall": result["metrics"]["overall"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
