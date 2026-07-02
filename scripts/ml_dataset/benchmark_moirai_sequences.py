#!/usr/bin/env python3
"""Benchmark Moirai on saved sequence benchmark cases."""

from __future__ import annotations

import argparse
import json
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
        from gluonts.dataset.common import ListDataset
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    except ImportError as exc:
        raise SystemExit("Missing Moirai dependencies. Run with the Moirai virtualenv.") from exc
    return {"np": np, "pd": pd, "ListDataset": ListDataset, "MoiraiForecast": MoiraiForecast, "MoiraiModule": MoiraiModule}


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
            "moirai_p50": metric(np, frame, f"moirai_{target}_p50", config["actual"]),
        }
        if config["hgb"] in frame.columns:
            target_summary["hgb"] = metric(np, frame, config["hgb"], config["actual"])
        for prefix in ("chronos", "timesfm"):
            column = f"{prefix}_{target}_p50"
            if column in frame.columns:
                target_summary[f"{prefix}_p50"] = metric(np, frame, column, config["actual"])
        summary[target] = target_summary
    if not include_by_lead:
        return summary

    by_lead = {}
    for lead, group in frame.groupby("lead_time_minutes"):
        by_lead[str(int(lead))] = summarize_metrics(group, deps, include_by_lead=False)
    return {"overall": summary, "by_lead": by_lead}


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Moirai Sequence Benchmark",
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
    parser.add_argument("--predictions-file", default="predictions_with_timesfm.parquet")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id", default="moirai_sequence_benchmark")
    parser.add_argument("--model-id", default="Salesforce/moirai-1.1-R-small")
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--prediction-length", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--freq", default="15min")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = import_dependencies()
    np = deps["np"]
    pd = deps["pd"]
    ListDataset = deps["ListDataset"]
    MoiraiForecast = deps["MoiraiForecast"]
    MoiraiModule = deps["MoiraiModule"]

    output_root = args.output_root or args.benchmark_root
    output_root.mkdir(parents=True, exist_ok=True)

    prediction_path = args.benchmark_root / args.predictions_file
    if not prediction_path.exists():
        prediction_path = args.benchmark_root / "predictions_with_hgb.parquet"
    if not prediction_path.exists():
        prediction_path = args.benchmark_root / "predictions.parquet"
    predictions = pd.read_parquet(prediction_path).sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    past = pd.read_parquet(args.benchmark_root / "past_context.parquet").sort_values(["item_id", "timestamp"])
    item_ids = list(predictions["item_id"].drop_duplicates())

    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(args.model_id),
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        patch_size="auto",
        num_samples=args.num_samples,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=args.batch_size)

    for target, config in TARGETS.items():
        records = []
        for item_id in item_ids:
            context = past.loc[past["item_id"] == item_id].tail(args.context_length)
            values = context[config["context"]].to_numpy(dtype=float)
            values = pd.Series(values).interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype="float32")
            start = pd.Period(context["timestamp"].iloc[0], freq=args.freq)
            records.append({"start": start, "target": values})
        dataset = ListDataset(records, freq=args.freq)
        forecasts = list(predictor.predict(dataset))
        for item_id, forecast in zip(item_ids, forecasts):
            row_index = predictions.index[predictions["item_id"] == item_id].tolist()
            mean = np.asarray(forecast.mean)
            p10 = np.asarray(forecast.quantile(0.1))
            p50 = np.asarray(forecast.quantile(0.5))
            p90 = np.asarray(forecast.quantile(0.9))
            for step, idx in enumerate(row_index[: args.prediction_length]):
                predictions.loc[idx, f"moirai_{target}_mean"] = float(mean[step])
                predictions.loc[idx, f"moirai_{target}_p10"] = float(p10[step])
                predictions.loc[idx, f"moirai_{target}_p50"] = float(p50[step])
                predictions.loc[idx, f"moirai_{target}_p90"] = float(p90[step])

    predictions.to_parquet(output_root / "predictions_with_moirai.parquet", index=False)
    result = {
        "generated_at_utc": utc_now(),
        "run_id": args.run_id,
        "benchmark_root": str(args.benchmark_root),
        "model_id": args.model_id,
        "row_count": int(len(predictions)),
        "item_count": int(len(item_ids)),
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "num_samples": args.num_samples,
        "metrics": summarize_metrics(predictions, deps),
    }
    (output_root / "benchmark_results_moirai.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_markdown(output_root / "benchmark_results_moirai.md", result)
    print(json.dumps({"run_id": result["run_id"], "item_count": result["item_count"], "row_count": result["row_count"], "overall": result["metrics"]["overall"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
