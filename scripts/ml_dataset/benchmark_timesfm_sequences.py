#!/usr/bin/env python3
"""Benchmark TimesFM on saved sequence benchmark cases."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGETS = {
    "wind_mean_ms": {
        "context": "wind_mean_ms",
        "actual": "actual_wind_mean_ms",
        "raw": "raw_wind_mean_ms",
        "hgb": "hgb_wind_mean_ms",
    },
    "gust_ms": {
        "context": "gust_ms",
        "actual": "actual_gust_ms",
        "raw": "raw_gust_ms",
        "hgb": "hgb_gust_ms",
    },
}


def import_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
        import torch
        import timesfm
    except ImportError as exc:
        raise SystemExit("Missing TimesFM dependencies. Run with the TimesFM virtualenv.") from exc
    return {"np": np, "pd": pd, "torch": torch, "timesfm": timesfm}


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
            "raw_nwp": metric(np, frame, config["raw"], config["actual"]),
            "timesfm_p50": metric(np, frame, f"timesfm_{target}_p50", config["actual"]),
        }
        if config["hgb"] in frame.columns:
            target_summary["hgb"] = metric(np, frame, config["hgb"], config["actual"])
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
        "# TimesFM Sequence Benchmark",
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
    parser.add_argument("--predictions-file", default="predictions_with_hgb.parquet")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id", default="timesfm_sequence_benchmark")
    parser.add_argument("--model-id", default="google/timesfm-2.5-200m-pytorch")
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--prediction-length", type=int, default=4)
    parser.add_argument("--max-context", type=int, default=1024)
    parser.add_argument("--max-horizon", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = import_dependencies()
    np = deps["np"]
    pd = deps["pd"]
    torch = deps["torch"]
    timesfm = deps["timesfm"]

    output_root = args.output_root or args.benchmark_root
    output_root.mkdir(parents=True, exist_ok=True)

    prediction_path = args.benchmark_root / args.predictions_file
    if not prediction_path.exists():
        prediction_path = args.benchmark_root / "predictions.parquet"
    predictions = pd.read_parquet(prediction_path).sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    past = pd.read_parquet(args.benchmark_root / "past_context.parquet").sort_values(["item_id", "timestamp"])
    item_ids = list(predictions["item_id"].drop_duplicates())

    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(args.model_id)
    model.compile(
        timesfm.ForecastConfig(
            max_context=args.max_context,
            max_horizon=args.max_horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    for target, config in TARGETS.items():
        inputs = []
        for item_id in item_ids:
            values = past.loc[past["item_id"] == item_id, config["context"]].tail(args.context_length).to_numpy(dtype=float)
            if len(values) == 0:
                values = np.array([np.nan], dtype=float)
            values = pd.Series(values).interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=float)
            inputs.append(values)
        point_forecast, quantile_forecast = model.forecast(horizon=args.prediction_length, inputs=inputs)
        point_forecast = np.asarray(point_forecast)
        quantile_forecast = np.asarray(quantile_forecast)
        for item_index, item_id in enumerate(item_ids):
            row_index = predictions.index[predictions["item_id"] == item_id].tolist()
            for step, idx in enumerate(row_index[: args.prediction_length]):
                predictions.loc[idx, f"timesfm_{target}_mean"] = float(quantile_forecast[item_index, step, 0])
                predictions.loc[idx, f"timesfm_{target}_p10"] = float(quantile_forecast[item_index, step, 1])
                predictions.loc[idx, f"timesfm_{target}_p50"] = float(point_forecast[item_index, step])
                predictions.loc[idx, f"timesfm_{target}_p90"] = float(quantile_forecast[item_index, step, 9])

    predictions.to_parquet(output_root / "predictions_with_timesfm.parquet", index=False)
    result = {
        "generated_at_utc": utc_now(),
        "run_id": args.run_id,
        "benchmark_root": str(args.benchmark_root),
        "model_id": args.model_id,
        "row_count": int(len(predictions)),
        "item_count": int(len(item_ids)),
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "metrics": summarize_metrics(predictions, deps),
    }
    (output_root / "benchmark_results_timesfm.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_markdown(output_root / "benchmark_results_timesfm.md", result)
    print(json.dumps({"run_id": result["run_id"], "item_count": result["item_count"], "row_count": result["row_count"], "overall": result["metrics"]["overall"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
