#!/usr/bin/env python3
"""Audit gust quantile coverage, conformal offsets and threshold rail choice."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KT_PER_MS = 1.9438444924406
THRESHOLD_EPSILON = 1e-9
DEFAULT_THRESHOLDS_KT = (12.0, 15.0, 20.0, 25.0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def import_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Missing pandas/numpy. Run inside the CorseWind ML venv.") from exc
    return {"np": np, "pd": pd}


def quantile_alpha_from_suffix(suffix: str) -> float | None:
    match = re.fullmatch(r"q(\d+(?:p\d+)?)", suffix.strip().lower())
    if not match:
        return None
    return float(match.group(1).replace("p", ".")) / 100.0


def discover_rails(columns: list[str], prefix: str) -> dict[str, str]:
    rails = {}
    for column in sorted(columns):
        if not column.startswith(prefix + "_q"):
            continue
        suffix = column.rsplit("_", 1)[-1]
        if quantile_alpha_from_suffix(suffix) is not None:
            rails[suffix] = column
    return dict(sorted(rails.items(), key=lambda item: (quantile_alpha_from_suffix(item[0]) or math.inf, item[0])))


def actual_regime_kt(value: float) -> str:
    if value >= 25:
        return ">=25kt"
    if value >= 20:
        return "20-25kt"
    if value >= 15:
        return "15-20kt"
    if value >= 12:
        return "12-15kt"
    return "<12kt"


def ensure_actual_kt(frame: Any, actual_column: str) -> Any:
    frame = frame.copy()
    frame["__actual_gust_kt"] = frame[actual_column].astype(float) * KT_PER_MS
    frame["__actual_gust_regime_kt"] = frame["__actual_gust_kt"].map(actual_regime_kt)
    if "issue_time_utc" in frame.columns:
        frame["__issue_day_utc"] = frame["issue_time_utc"].astype(str).str.slice(0, 10)
    return frame


def continuous_metrics(frame: Any, prediction_column: str) -> dict[str, Any]:
    values = frame[[prediction_column, "__actual_gust_kt"]].dropna()
    if values.empty:
        return {"n": 0}
    err = values[prediction_column] * KT_PER_MS - values["__actual_gust_kt"]
    return {
        "n": int(len(values)),
        "mae_kt": float(err.abs().mean()),
        "rmse_kt": float((err.pow(2).mean()) ** 0.5),
        "bias_kt": float(err.mean()),
    }


def coverage_metrics(frame: Any, prediction_column: str, alpha: float | None) -> dict[str, Any]:
    values = frame[[prediction_column, "__actual_gust_kt"]].dropna()
    if values.empty:
        return {"n": 0, "expected_coverage": alpha}
    prediction_kt = values[prediction_column].astype(float) * KT_PER_MS
    residual = values["__actual_gust_kt"] - prediction_kt
    coverage = float((residual <= THRESHOLD_EPSILON).mean())
    return {
        "n": int(len(values)),
        "expected_coverage": alpha,
        "empirical_coverage": coverage,
        "coverage_error": None if alpha is None else coverage - alpha,
        "mean_excess_when_missed_kt": None if (residual > 0).sum() == 0 else float(residual[residual > 0].mean()),
        "p90_excess_when_missed_kt": None if (residual > 0).sum() == 0 else float(residual[residual > 0].quantile(0.9)),
        "conformal_additive_offset_kt": None if alpha is None else float(residual.quantile(alpha)),
    }


def threshold_metrics(frame: Any, prediction_column: str, threshold_kt: float) -> dict[str, Any]:
    values = frame[[prediction_column, "__actual_gust_kt"]].dropna()
    if values.empty:
        return {"n": 0}
    pred = values[prediction_column].astype(float) * KT_PER_MS >= threshold_kt - THRESHOLD_EPSILON
    actual = values["__actual_gust_kt"] >= threshold_kt - THRESHOLD_EPSILON
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    return {
        "n": int(len(values)),
        "threshold_kt": threshold_kt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
        "csi": None if tp + fp + fn == 0 else tp / (tp + fp + fn),
        "false_alarm_ratio": None if tp + fp == 0 else fp / (tp + fp),
        "false_alarm_rate": None if fp + tn == 0 else fp / (fp + tn),
    }


def rail_summary(frame: Any, rails: dict[str, str], thresholds_kt: list[float]) -> dict[str, Any]:
    out = {}
    for suffix, column in rails.items():
        alpha = quantile_alpha_from_suffix(suffix)
        out[suffix] = {
            "column": column,
            "alpha": alpha,
            "continuous": continuous_metrics(frame, column),
            "coverage": coverage_metrics(frame, column, alpha),
            "thresholds": {
                f"gust_ge_{threshold:g}kt": threshold_metrics(frame, column, threshold)
                for threshold in thresholds_kt
            },
        }
    return out


def best_threshold_rails(rails_summary: dict[str, Any], max_false_alarm_ratio: float | None) -> dict[str, Any]:
    out = {}
    threshold_names = sorted({name for rail in rails_summary.values() for name in rail.get("thresholds", {})})
    for threshold_name in threshold_names:
        candidates = []
        for suffix, rail in rails_summary.items():
            metric = dict((rail.get("thresholds") or {}).get(threshold_name) or {})
            metric.update({"rail": suffix, "alpha": rail.get("alpha"), "column": rail.get("column")})
            if max_false_alarm_ratio is not None:
                far = metric.get("false_alarm_ratio")
                metric["passes_far_cap"] = far is None or float(far) <= max_false_alarm_ratio
            else:
                metric["passes_far_cap"] = True
            candidates.append(metric)
        valid = [
            item for item in candidates
            if item.get("csi") is not None and item.get("passes_far_cap")
        ]
        best = max(valid, key=lambda item: float(item["csi"])) if valid else None
        out[threshold_name] = {
            "best": best,
            "candidates": sorted(
                candidates,
                key=lambda item: -1.0 if item.get("csi") is None else float(item["csi"]),
                reverse=True,
            ),
        }
    return out


def grouped_summary(frame: Any, rails: dict[str, str], group_columns: list[str], thresholds_kt: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for column in group_columns:
        if column == "actual_gust_regime_kt":
            group_column = "__actual_gust_regime_kt"
        elif column == "issue_day_utc":
            group_column = "__issue_day_utc"
        else:
            group_column = column
        if group_column not in frame.columns:
            continue
        groups = {}
        for key, group in frame.groupby(group_column, dropna=False):
            groups[str(key)] = rail_summary(group, rails, thresholds_kt)
        out[column] = groups
    return out


def audit(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]
    frame = pd.read_parquet(args.predictions)
    if args.actual_column not in frame.columns:
        raise SystemExit(f"Missing actual column: {args.actual_column}")
    frame = ensure_actual_kt(frame.dropna(subset=[args.actual_column]), args.actual_column)
    rails = discover_rails(list(frame.columns), args.rail_prefix)
    if args.rail:
        rails = {suffix: rails[suffix] for suffix in args.rail if suffix in rails}
    if not rails:
        raise SystemExit(f"No quantile rails found with prefix {args.rail_prefix!r}.")
    thresholds_kt = args.threshold_kt or list(DEFAULT_THRESHOLDS_KT)
    overall = rail_summary(frame, rails, thresholds_kt)
    result = {
        "format": "corsewind.gust_quantile_coverage_audit.v1",
        "generated_at_utc": utc_now(),
        "predictions": str(args.predictions),
        "actual_column": args.actual_column,
        "row_count": int(len(frame)),
        "rails": overall,
        "best_threshold_rails": best_threshold_rails(overall, args.max_false_alarm_ratio),
        "groups": grouped_summary(frame, rails, args.group_column, thresholds_kt),
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return result


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Gust Quantile Coverage Audit",
        "",
        f"- rows: `{result['row_count']}`",
        f"- predictions: `{result['predictions']}`",
        "",
        "## Coverage",
        "",
        "| Rail | n | Expected | Empirical | Error | Conformal offset kt |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for suffix, rail in result["rails"].items():
        coverage = rail.get("coverage") or {}
        lines.append(
            "| "
            + " | ".join([
                f"`{suffix}`",
                fmt(coverage.get("n"), 0),
                fmt(coverage.get("expected_coverage")),
                fmt(coverage.get("empirical_coverage")),
                fmt(coverage.get("coverage_error")),
                fmt(coverage.get("conformal_additive_offset_kt")),
            ])
            + " |"
        )
    lines.extend(["", "## Best Rail By Threshold", "", "| Threshold | Rail | CSI | FAR | TP | FP | FN |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for threshold_name, item in result["best_threshold_rails"].items():
        best = item.get("best") or {}
        lines.append(
            "| "
            + " | ".join([
                f"`{threshold_name}`",
                f"`{best.get('rail')}`" if best else "",
                fmt(best.get("csi")),
                fmt(best.get("false_alarm_ratio")),
                fmt(best.get("tp"), 0),
                fmt(best.get("fp"), 0),
                fmt(best.get("fn"), 0),
            ])
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--actual-column", default="actual_gust_ms")
    parser.add_argument("--rail-prefix", default="calibrated_gust_ms")
    parser.add_argument("--rail", action="append", default=[])
    parser.add_argument("--threshold-kt", type=float, action="append", default=[])
    parser.add_argument("--group-column", action="append", default=["spot_id", "lead_time_minutes", "actual_gust_regime_kt"])
    parser.add_argument("--max-false-alarm-ratio", type=float)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    audit(parse_args())


if __name__ == "__main__":
    main()
