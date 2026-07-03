#!/usr/bin/env python3
"""Audit gust quantile rails for anti-smoothing behavior and threshold skill."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MS_PER_KT = 0.514444
DEFAULT_THRESHOLDS_KT = (12.0, 15.0, 20.0, 25.0)
DEFAULT_RAILS = (
    "raw_gust_ms",
    "corrected_gust_ms",
    "calibrated_gust_ms",
    "calibrated_gust_ms_q50",
    "calibrated_gust_ms_q75",
    "calibrated_gust_ms_q90",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def import_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Missing pandas/numpy. Run inside the CorseWind ML environment.") from exc
    return {"np": np, "pd": pd}


def event_counts(pred: Any, actual: Any) -> dict[str, Any]:
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    return {
        "n": int(len(actual)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": None if tp + fp == 0 else round(tp / (tp + fp), 6),
        "recall": None if tp + fn == 0 else round(tp / (tp + fn), 6),
        "csi": None if tp + fp + fn == 0 else round(tp / (tp + fp + fn), 6),
    }


def continuous_metric(frame: Any, prediction_column: str, actual_column: str, np: Any) -> dict[str, Any]:
    values = frame[[prediction_column, actual_column]].dropna()
    if values.empty:
        return {"count": 0}
    pred = values[prediction_column].astype(float).to_numpy()
    obs = values[actual_column].astype(float).to_numpy()
    err = pred - obs
    pred_var = float(np.var(pred))
    obs_var = float(np.var(obs))
    return {
        "count": int(len(values)),
        "mae_ms": round(float(np.mean(np.abs(err))), 6),
        "rmse_ms": round(float(math.sqrt(float(np.mean(err * err)))), 6),
        "bias_ms": round(float(np.mean(err)), 6),
        "prediction_std_ms": round(float(np.std(pred)), 6),
        "observation_std_ms": round(float(np.std(obs)), 6),
        "variance_ratio": None if obs_var <= 0.0 else round(pred_var / obs_var, 6),
    }


def threshold_metric(frame: Any, prediction_column: str, actual_column: str, threshold_kt: float) -> dict[str, Any]:
    values = frame[[prediction_column, actual_column]].dropna()
    if values.empty:
        return {"n": 0}
    threshold_ms = threshold_kt * MS_PER_KT
    pred = values[prediction_column].astype(float) >= threshold_ms
    actual = values[actual_column].astype(float) >= threshold_ms
    return event_counts(pred, actual)


def regime_name(value_ms: float) -> str:
    value_kt = value_ms / MS_PER_KT
    if value_kt < 12.0:
        return "<12kt"
    if value_kt < 15.0:
        return "12-15kt"
    if value_kt < 20.0:
        return "15-20kt"
    if value_kt < 25.0:
        return "20-25kt"
    return ">=25kt"


def discover_rails(frame: Any, requested: list[str]) -> list[str]:
    rails = [column for column in requested if column in frame.columns]
    for column in frame.columns:
        if column.startswith("calibrated_gust_ms_q") and column not in rails:
            rails.append(column)
    return rails


def audit(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]
    np = deps["np"]
    frame = pd.read_parquet(args.predictions)
    if args.start_utc or args.end_utc:
        if "issue_time_utc" not in frame.columns:
            raise SystemExit("--start-utc/--end-utc require issue_time_utc in predictions.")
        frame["issue_time_utc"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
        if args.start_utc:
            frame = frame[frame["issue_time_utc"] >= pd.Timestamp(args.start_utc, tz="UTC")]
        if args.end_utc:
            frame = frame[frame["issue_time_utc"] < pd.Timestamp(args.end_utc, tz="UTC")]
    if args.lead_minute:
        frame = frame[frame["lead_time_minutes"].astype("Int64").isin([int(item) for item in args.lead_minute])]
    if args.actual_column not in frame.columns:
        raise SystemExit(f"Missing actual column: {args.actual_column}")
    frame = frame.dropna(subset=[args.actual_column]).copy()
    frame["__actual_gust_regime"] = frame[args.actual_column].astype(float).map(regime_name)
    rails = discover_rails(frame, args.rail)
    result: dict[str, Any] = {
        "format": "corsewind.gust_quantile_anti_smoothing_audit.v1",
        "generated_at_utc": utc_now(),
        "predictions": str(args.predictions),
        "actual_column": args.actual_column,
        "rows": int(len(frame)),
        "lead_minutes": args.lead_minute,
        "thresholds_kt": args.threshold_kt,
        "rails": {},
    }
    for rail in rails:
        rail_result: dict[str, Any] = {
            "overall": continuous_metric(frame, rail, args.actual_column, np),
            "by_actual_regime": {},
            "thresholds": {},
        }
        for regime, group in frame.groupby("__actual_gust_regime", dropna=False):
            rail_result["by_actual_regime"][str(regime)] = continuous_metric(group, rail, args.actual_column, np)
        for threshold in args.threshold_kt:
            rail_result["thresholds"][f"gust_ge_{threshold:g}kt"] = threshold_metric(
                frame,
                rail,
                args.actual_column,
                threshold,
            )
        result["rails"][rail] = rail_result
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return result


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Gust Quantile Anti-Smoothing Audit",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Rows: `{result['rows']}`",
        "",
        "## Overall",
        "",
        "| Rail | RMSE m/s | MAE m/s | Bias m/s | Pred std | Obs std | Variance ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rail, item in result["rails"].items():
        metric = item.get("overall") or {}
        lines.append(
            f"| `{rail}` | {fmt(metric.get('rmse_ms'))} | {fmt(metric.get('mae_ms'))} | "
            f"{fmt(metric.get('bias_ms'))} | {fmt(metric.get('prediction_std_ms'))} | "
            f"{fmt(metric.get('observation_std_ms'))} | {fmt(metric.get('variance_ratio'))} |"
        )
    lines.extend(["", "## Threshold CSI", ""])
    thresholds = [f"gust_ge_{threshold:g}kt" for threshold in result["thresholds_kt"]]
    lines.append("| Rail | " + " | ".join(thresholds) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in thresholds) + " |")
    for rail, item in result["rails"].items():
        cells = [fmt((item.get("thresholds") or {}).get(threshold, {}).get("csi")) for threshold in thresholds]
        lines.append(f"| `{rail}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--actual-column", default="actual_gust_ms")
    parser.add_argument("--rail", action="append", default=list(DEFAULT_RAILS))
    parser.add_argument("--threshold-kt", type=float, action="append", default=list(DEFAULT_THRESHOLDS_KT))
    parser.add_argument("--lead-minute", type=int, action="append", default=[])
    parser.add_argument("--start-utc")
    parser.add_argument("--end-utc")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    audit(parse_args())


if __name__ == "__main__":
    main()
