#!/usr/bin/env python3
"""Evaluate raw NWP and simple residual-correction baselines."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def metrics(errors: list[float]) -> dict[str, float | int | None]:
    if not errors:
        return {"count": 0, "mae": None, "rmse": None, "bias": None}
    count = len(errors)
    mae = sum(abs(error) for error in errors) / count
    rmse = math.sqrt(sum(error * error for error in errors) / count)
    bias = sum(errors) / count
    return {
        "count": count,
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "bias": round(bias, 6),
    }


def add_error(bucket: dict[str, list[float]], name: str, prediction: float | None, target: float | None) -> None:
    if prediction is None or target is None:
        return
    bucket[name].append(prediction - target)


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    threshold_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in rows:
        lead = str(row.get("lead_time_minutes"))
        labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}
        baselines = row.get("baselines", {}) if isinstance(row.get("baselines"), dict) else {}
        features = row.get("features", {}) if isinstance(row.get("features"), dict) else {}

        target_wind = finite_float(labels.get("target_wind_mean_ms"))
        target_gust = finite_float(labels.get("target_gust_ms"))
        baseline_wind = finite_float(baselines.get("baseline_wind_mean_ms"))
        baseline_gust = finite_float(baselines.get("baseline_gust_ms"))
        error_now_wind = finite_float(features.get("model_error_now_wind_mean_ms"))
        error_now_gust = finite_float(features.get("model_error_now_gust_ms"))

        raw_wind = baseline_wind
        raw_gust = baseline_gust
        persistence_wind = None if baseline_wind is None or error_now_wind is None else baseline_wind + error_now_wind
        persistence_gust = None if baseline_gust is None or error_now_gust is None else baseline_gust + error_now_gust

        for key in ("overall", f"lead_{lead}m", f"spot_{row.get('spot_id')}"):
            add_error(buckets[key], "wind_raw_nwp", raw_wind, target_wind)
            add_error(buckets[key], "wind_error_persistence", persistence_wind, target_wind)
            add_error(buckets[key], "gust_raw_nwp", raw_gust, target_gust)
            add_error(buckets[key], "gust_error_persistence", persistence_gust, target_gust)

        for label_key, label_value in labels.items():
            if label_key.startswith("target_wind_gt_") or label_key.startswith("target_gust_gt_"):
                threshold_counts[label_key]["count"] += 1
                if label_value == 1:
                    threshold_counts[label_key]["positive"] += 1

    groups = {
        group: {
            metric_name: metrics(errors)
            for metric_name, errors in sorted(group_errors.items())
        }
        for group, group_errors in sorted(buckets.items())
    }

    comparisons: dict[str, Any] = {}
    overall = groups.get("overall", {})
    for variable in ("wind", "gust"):
        raw = overall.get(f"{variable}_raw_nwp", {})
        corrected = overall.get(f"{variable}_error_persistence", {})
        raw_rmse = finite_float(raw.get("rmse"))
        corrected_rmse = finite_float(corrected.get("rmse"))
        comparisons[f"{variable}_rmse_delta_error_persistence_minus_raw"] = (
            None if raw_rmse is None or corrected_rmse is None else round(corrected_rmse - raw_rmse, 6)
        )
        comparisons[f"{variable}_rmse_gain_pct_error_persistence_vs_raw"] = (
            None
            if raw_rmse in {None, 0.0} or corrected_rmse is None
            else round((raw_rmse - corrected_rmse) / raw_rmse * 100.0, 3)
        )

    return {
        "format": "corsewind.residual_training_table_evaluation.v1",
        "row_count": len(rows),
        "groups": groups,
        "comparisons": comparisons,
        "threshold_positive_counts": {
            key: {
                "count": value.get("count", 0),
                "positive": value.get("positive", 0),
                "positive_rate": None
                if not value.get("count")
                else round(value.get("positive", 0) / value["count"], 6),
            }
            for key, value in sorted(threshold_counts.items())
        },
    }


def write_markdown(path: Path, evaluation: dict[str, Any]) -> None:
    overall = evaluation.get("groups", {}).get("overall", {})
    lines = [
        "# Residual Training Table Evaluation",
        "",
        f"- rows: `{evaluation.get('row_count', 0)}`",
        f"- wind RMSE gain, error persistence vs raw: `{evaluation.get('comparisons', {}).get('wind_rmse_gain_pct_error_persistence_vs_raw')}`%",
        f"- gust RMSE gain, error persistence vs raw: `{evaluation.get('comparisons', {}).get('gust_rmse_gain_pct_error_persistence_vs_raw')}`%",
        "",
        "## Overall",
        "",
        "| Baseline | Count | MAE | RMSE | Bias |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, item in sorted(overall.items()):
        lines.append(
            f"| `{name}` | {item.get('count', 0)} | {item.get('mae')} | {item.get('rmse')} | {item.get('bias')} |"
        )

    lines.extend(["", "## By Lead", "", "| Lead | Baseline | Count | MAE | RMSE | Bias |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for group, values in sorted(evaluation.get("groups", {}).items()):
        if not group.startswith("lead_"):
            continue
        lead = group.replace("lead_", "").replace("m", " min")
        for name, item in sorted(values.items()):
            lines.append(
                f"| `{lead}` | `{name}` | {item.get('count', 0)} | {item.get('mae')} | {item.get('rmse')} | {item.get('bias')} |"
            )

    lines.extend(["", "## Threshold Positives", "", "| Label | Count | Positive | Rate |", "| --- | ---: | ---: | ---: |"])
    for name, item in evaluation.get("threshold_positive_counts", {}).items():
        lines.append(
            f"| `{name}` | {item.get('count', 0)} | {item.get('positive', 0)} | {item.get('positive_rate')} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-rows", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.training_rows)
    evaluation = evaluate(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(args.output_md, evaluation)
    print(json.dumps({
        "row_count": evaluation["row_count"],
        "comparisons": evaluation["comparisons"],
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
