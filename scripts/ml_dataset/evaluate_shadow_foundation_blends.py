#!/usr/bin/env python3
"""Evaluate champion vs shadow foundation-blend predictions once observations exist."""

from __future__ import annotations

import argparse
import glob
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KEY_COLUMNS = ("spot_id", "issue_time_utc", "lead_time_minutes")
TARGETS = {
    "wind_mean": {
        "actual_candidates": ("actual_wind_mean_ms", "labels__target_wind_mean_ms", "target_wind_mean_ms"),
        "champion": "champion_wind_mean_ms",
        "fallback_champion_candidates": ("calibrated_wind_mean_ms", "wind_champion_prediction_ms"),
        "guarded": "guarded_foundation_wind_mean_ms",
        "delta": "guarded_foundation_wind_mean_delta_ms",
        "used": "guarded_foundation_wind_mean_used_foundation",
        "bin_edges": [-math.inf, 2, 4, 6, 8, 10, math.inf],
        "bin_labels": ["0-2", "2-4", "4-6", "6-8", "8-10", "10+"],
    },
    "gust": {
        "actual_candidates": ("actual_gust_ms", "labels__target_gust_ms", "target_gust_ms"),
        "champion": "champion_gust_ms",
        "fallback_champion_candidates": ("calibrated_gust_ms", "gust_champion_prediction_ms"),
        "guarded": "guarded_foundation_gust_ms",
        "delta": "guarded_foundation_gust_delta_ms",
        "used": "guarded_foundation_gust_used_foundation",
        "bin_edges": [-math.inf, 4, 8, 12, 16, math.inf],
        "bin_labels": ["0-4", "4-8", "8-12", "12-16", "16+"],
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def import_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Missing pandas/numpy dependencies.") from exc
    return {"np": np, "pd": pd}


def expand_inputs(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in paths:
        matches = [Path(path) for path in glob.glob(item)]
        out.extend(matches if matches else [Path(item)])
    seen = set()
    unique = []
    for path in out:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def canonicalize_keys(frame: Any, pd: Any) -> Any:
    out = frame.copy()
    missing = [column for column in KEY_COLUMNS if column not in out.columns]
    if missing:
        raise SystemExit(f"Missing key columns: {missing}")
    out["spot_id"] = out["spot_id"].astype(str)
    out["issue_time_utc"] = pd.to_datetime(out["issue_time_utc"], utc=True, errors="coerce")
    out["lead_time_minutes"] = pd.to_numeric(out["lead_time_minutes"], errors="coerce").astype("Int64")
    return out.dropna(subset=list(KEY_COLUMNS))


def read_predictions(paths: list[Path], pd: Any) -> Any:
    frames = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Prediction parquet not found: {path}")
        frame = pd.read_parquet(path)
        frame["source_prediction_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise SystemExit("No prediction parquet provided.")
    return canonicalize_keys(pd.concat(frames, ignore_index=True), pd).drop_duplicates(list(KEY_COLUMNS), keep="last")


def read_actuals(path: Path | None, pd: Any) -> Any | None:
    if path is None:
        return None
    if not path.exists():
        raise SystemExit(f"Actuals parquet not found: {path}")
    actuals = canonicalize_keys(pd.read_parquet(path), pd)
    keep = list(KEY_COLUMNS)
    for config in TARGETS.values():
        keep.extend(column for column in config["actual_candidates"] if column in actuals.columns)
    keep = list(dict.fromkeys(keep))
    return actuals[keep].drop_duplicates(list(KEY_COLUMNS), keep="last")


def first_present(frame: Any, columns: tuple[str, ...], pd: Any) -> Any | None:
    present = [column for column in columns if column in frame.columns]
    if not present:
        return None
    value = pd.to_numeric(frame[present[0]], errors="coerce")
    for column in present[1:]:
        value = value.combine_first(pd.to_numeric(frame[column], errors="coerce"))
    return value


def metric(frame: Any, prediction_column: str, actual_column: str, np: Any, pd: Any) -> dict[str, Any]:
    if prediction_column not in frame.columns or actual_column not in frame.columns:
        return {"count": 0}
    prediction = pd.to_numeric(frame[prediction_column], errors="coerce").to_numpy(dtype=float)
    actual = pd.to_numeric(frame[actual_column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(prediction) & np.isfinite(actual)
    if not mask.any():
        return {"count": 0}
    errors = prediction[mask] - actual[mask]
    abs_errors = np.abs(errors)
    return {
        "count": int(mask.sum()),
        "rmse": round(float(np.sqrt(np.mean(errors * errors))), 6),
        "mae": round(float(np.mean(abs_errors)), 6),
        "bias": round(float(np.mean(errors)), 6),
        "p90_abs_error": round(float(np.quantile(abs_errors, 0.90)), 6),
        "sse": round(float(np.sum(errors * errors)), 6),
    }


def gain_pct(champion: dict[str, Any], guarded: dict[str, Any]) -> float | None:
    if not champion.get("rmse") or guarded.get("rmse") is None:
        return None
    return round(float(100.0 * (champion["rmse"] - guarded["rmse"]) / champion["rmse"]), 6)


def grouped_metrics(frame: Any, prediction_column: str, actual_column: str, group_columns: list[str], np: Any, pd: Any) -> list[dict[str, Any]]:
    if any(column not in frame.columns for column in group_columns):
        return []
    rows = []
    valid = frame.dropna(subset=[actual_column])
    for raw_key, group in valid.groupby(group_columns, dropna=False):
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        item = {
            "group": {
                column: None if pd.isna(value) else value.item() if hasattr(value, "item") else str(value)
                for column, value in zip(group_columns, values, strict=True)
            },
            **metric(group, prediction_column, actual_column, np, pd),
        }
        rows.append(item)
    rows.sort(key=lambda item: (item.get("rmse") is None, item.get("rmse", -1)), reverse=True)
    return rows


def add_bins(frame: Any, pd: Any) -> Any:
    out = frame.copy()
    for target, config in TARGETS.items():
        actual = f"actual_{target}_eval_ms" if target == "wind_mean" else "actual_gust_eval_ms"
        if actual in out.columns:
            out[f"{target}_actual_bin"] = pd.cut(
                pd.to_numeric(out[actual], errors="coerce"),
                bins=config["bin_edges"],
                labels=config["bin_labels"],
            )
    if "issue_time_utc" in out.columns:
        hour = pd.to_datetime(out["issue_time_utc"], utc=True, errors="coerce").dt.hour
        out["thermal_hour"] = hour.between(10, 16)
    return out


def evaluate_target(frame: Any, target: str, args: argparse.Namespace, np: Any, pd: Any) -> dict[str, Any]:
    config = TARGETS[target]
    actual_column = "actual_wind_mean_eval_ms" if target == "wind_mean" else "actual_gust_eval_ms"
    champion_column = config["champion"]
    guarded_column = config["guarded"]
    if champion_column not in frame.columns:
        champion_values = first_present(frame, config["fallback_champion_candidates"], pd)
        if champion_values is not None:
            frame[champion_column] = champion_values
    if guarded_column not in frame.columns and champion_column in frame.columns:
        frame[guarded_column] = frame[champion_column]
    champion = metric(frame, champion_column, actual_column, np, pd)
    guarded = metric(frame, guarded_column, actual_column, np, pd)
    result = {
        "target": target,
        "actual_column": actual_column if actual_column in frame.columns else None,
        "champion_column": champion_column,
        "guarded_column": guarded_column,
        "champion": champion,
        "guarded": guarded,
        "gain_pct": gain_pct(champion, guarded),
        "used_foundation_count": int(frame[config["used"]].sum()) if config["used"] in frame.columns else None,
        "evaluated_row_count": guarded.get("count", 0),
        "by_lead": {
            "champion": grouped_metrics(frame, champion_column, actual_column, ["lead_time_minutes"], np, pd),
            "guarded": grouped_metrics(frame, guarded_column, actual_column, ["lead_time_minutes"], np, pd),
        },
        "by_spot": {
            "champion": grouped_metrics(frame, champion_column, actual_column, ["spot_id"], np, pd),
            "guarded": grouped_metrics(frame, guarded_column, actual_column, ["spot_id"], np, pd),
        },
    }
    bin_col = f"{target}_actual_bin"
    if bin_col in frame.columns:
        result["by_actual_bin"] = {
            "champion": grouped_metrics(frame, champion_column, actual_column, [bin_col], np, pd),
            "guarded": grouped_metrics(frame, guarded_column, actual_column, [bin_col], np, pd),
        }
    if target == "gust" and "actual_gust_eval_ms" in frame.columns:
        high = frame[pd.to_numeric(frame["actual_gust_eval_ms"], errors="coerce") >= args.high_gust_threshold_ms]
        result["high_gust"] = {
            "threshold_ms": args.high_gust_threshold_ms,
            "champion": metric(high, champion_column, actual_column, np, pd),
            "guarded": metric(high, guarded_column, actual_column, np, pd),
        }
        result["high_gust"]["gain_pct"] = gain_pct(result["high_gust"]["champion"], result["high_gust"]["guarded"])
    hard_spots = set(args.hard_spot)
    if hard_spots:
        hard = frame[frame["spot_id"].astype(str).isin(hard_spots)]
        result["hard_spots"] = {
            "spots": sorted(hard_spots),
            "champion": metric(hard, champion_column, actual_column, np, pd),
            "guarded": metric(hard, guarded_column, actual_column, np, pd),
        }
        result["hard_spots"]["gain_pct"] = gain_pct(result["hard_spots"]["champion"], result["hard_spots"]["guarded"])
    return result


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Shadow Foundation Blend Evaluation",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Prediction rows: `{result['prediction_row_count']}`",
        f"Evaluable rows: `{result['evaluable_row_count']}`",
        "",
        "| Target | Champion RMSE | Guarded RMSE | Guarded MAE | Guarded bias | Gain | Used foundation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in result["targets"]:
        champion = item.get("champion", {})
        guarded = item.get("guarded", {})
        gain = item.get("gain_pct")
        lines.append(
            f"| `{item['target']}` | {champion.get('rmse')} | {guarded.get('rmse')} | "
            f"{guarded.get('mae')} | {guarded.get('bias')} | {'' if gain is None else str(gain) + '%'} | "
            f"{item.get('used_foundation_count')} |"
        )
    for item in result["targets"]:
        lines.extend(["", f"## {item['target']}", "", "### By Lead", "", "| Lead | Champion RMSE | Guarded RMSE | Delta | Count |", "| ---: | ---: | ---: | ---: | ---: |"])
        champion_by = {str(row["group"].get("lead_time_minutes")): row for row in item.get("by_lead", {}).get("champion", [])}
        for guarded in item.get("by_lead", {}).get("guarded", []):
            lead = str(guarded["group"].get("lead_time_minutes"))
            champion = champion_by.get(lead, {})
            delta = None if champion.get("rmse") is None or guarded.get("rmse") is None else round(guarded["rmse"] - champion["rmse"], 6)
            lines.append(f"| {lead} | {champion.get('rmse')} | {guarded.get('rmse')} | {delta} | {guarded.get('count')} |")
        if item.get("high_gust"):
            high = item["high_gust"]
            lines.extend([
                "",
                "### High Gust",
                "",
                f"- threshold: `{high['threshold_ms']} m/s`",
                f"- champion RMSE: `{high['champion'].get('rmse')}`",
                f"- guarded RMSE: `{high['guarded'].get('rmse')}`",
                f"- gain: `{high.get('gain_pct')}%`",
            ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]
    np = deps["np"]
    prediction_paths = expand_inputs(args.predictions)
    predictions = read_predictions(prediction_paths, pd)
    actuals = read_actuals(args.actuals_parquet, pd)
    if actuals is not None:
        predictions = predictions.merge(actuals, on=list(KEY_COLUMNS), how="left", suffixes=("", "__actuals"))
    predictions["actual_wind_mean_eval_ms"] = first_present(predictions, TARGETS["wind_mean"]["actual_candidates"], pd)
    predictions["actual_gust_eval_ms"] = first_present(predictions, TARGETS["gust"]["actual_candidates"], pd)
    predictions = add_bins(predictions, pd)

    targets = [evaluate_target(predictions, target, args, np, pd) for target in args.target]
    evaluable_mask = False
    for target in args.target:
        actual = "actual_wind_mean_eval_ms" if target == "wind_mean" else "actual_gust_eval_ms"
        if actual in predictions.columns:
            current = pd.to_numeric(predictions[actual], errors="coerce").notna()
            evaluable_mask = current if isinstance(evaluable_mask, bool) else (evaluable_mask | current)
    evaluable_row_count = int(evaluable_mask.sum()) if not isinstance(evaluable_mask, bool) else 0
    result = {
        "format": "corsewind.shadow_foundation_blend_evaluation.v1",
        "generated_at_utc": utc_now(),
        "prediction_files": [str(path) for path in prediction_paths],
        "actuals_parquet": str(args.actuals_parquet) if args.actuals_parquet else None,
        "prediction_row_count": int(len(predictions)),
        "evaluable_row_count": evaluable_row_count,
        "hard_spots": args.hard_spot,
        "targets": targets,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_markdown, result)
    print(json.dumps({
        "prediction_row_count": result["prediction_row_count"],
        "evaluable_row_count": result["evaluable_row_count"],
        "targets": [
            {
                "target": item["target"],
                "champion_rmse": item["champion"].get("rmse"),
                "guarded_rmse": item["guarded"].get("rmse"),
                "gain_pct": item.get("gain_pct"),
            }
            for item in targets
        ],
    }, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", action="append", required=True, help="Prediction parquet path or glob. Repeatable.")
    parser.add_argument("--actuals-parquet", type=Path)
    parser.add_argument("--target", choices=sorted(TARGETS), action="append", default=["wind_mean", "gust"])
    parser.add_argument("--high-gust-threshold-ms", type=float, default=16.0)
    parser.add_argument("--hard-spot", action="append", default=["la_tonnara", "santa_manza", "piantarella", "figari_eole"])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
