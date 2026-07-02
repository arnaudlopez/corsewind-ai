#!/usr/bin/env python3
"""Analyze RMSE-0.9 residual errors after a sequence calibrator sweep."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET = "actual_wind_mean_ms"
CALIBRATED = "calibrated_wind_mean_ms"
RAW = "raw_wind_mean_ms"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rmse(errors: Any) -> float | None:
    if len(errors) == 0:
        return None
    return float(math.sqrt(float((errors * errors).mean())))


def metric_frame(frame: Any, prediction: str, actual: str = TARGET) -> dict[str, Any]:
    valid = frame[[prediction, actual]].dropna()
    if valid.empty:
        return {"count": 0}
    errors = valid[prediction].astype(float) - valid[actual].astype(float)
    return {
        "count": int(len(errors)),
        "rmse": round(float(rmse(errors)), 6),
        "mae": round(float(errors.abs().mean()), 6),
        "bias": round(float(errors.mean()), 6),
        "p50_abs_error": round(float(errors.abs().quantile(0.50)), 6),
        "p90_abs_error": round(float(errors.abs().quantile(0.90)), 6),
    }


def grouped_metrics(frame: Any, group_columns: list[str], prediction: str, limit: int | None = None) -> list[dict[str, Any]]:
    if prediction not in frame.columns or any(column not in frame.columns for column in group_columns):
        return []
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        item = {column: str(value) for column, value in zip(group_columns, keys, strict=True)}
        item.update(metric_frame(group, prediction))
        rows.append(item)
    rows.sort(key=lambda item: (item.get("rmse") is None, item.get("rmse", -1)), reverse=True)
    return rows[:limit] if limit else rows


def add_bins(frame: Any, pd: Any) -> Any:
    out = frame.copy()
    if TARGET in out.columns:
        out["actual_wind_bin_ms"] = pd.cut(
            out[TARGET].astype(float),
            bins=[-0.001, 2.0, 4.0, 6.0, 8.0, 999.0],
            labels=["0-2", "2-4", "4-6", "6-8", "8+"],
        ).astype(str)
    if RAW in out.columns and TARGET in out.columns:
        raw_error = out[RAW].astype(float) - out[TARGET].astype(float)
        out["raw_error_abs_bin_ms"] = pd.cut(
            raw_error.abs(),
            bins=[-0.001, 0.5, 1.0, 2.0, 999.0],
            labels=["0-0.5", "0.5-1", "1-2", "2+"],
        ).astype(str)
    if "issue_time_utc" in out.columns:
        issue = pd.to_datetime(out["issue_time_utc"], utc=True, errors="coerce")
        out["issue_day"] = issue.dt.strftime("%Y-%m-%d")
        out["issue_hour_utc"] = issue.dt.hour.astype("Int64").astype(str)
        out["issue_month"] = issue.dt.strftime("%Y-%m")
    return out


def best_run_from_sweep(summary: dict[str, Any], preferred_run: str | None, preferred_model: str | None) -> dict[str, Any] | None:
    runs = summary.get("runs", [])
    if preferred_run:
        for run in runs:
            if run.get("run_name") == preferred_run:
                return run
    if preferred_model:
        for run in runs:
            if run.get("model_family") == preferred_model:
                return run
    best = None
    for run in runs:
        value = run.get("metrics", {}).get("calibrator", {}).get("rmse")
        if value is None:
            continue
        if best is None or float(value) < float(best.get("metrics", {}).get("calibrator", {}).get("rmse", float("inf"))):
            best = run
    return best


def compare_predictions(frame: Any) -> dict[str, Any]:
    out = {}
    for column in candidate_prediction_columns(frame, include_calibrated=True):
        out[column] = metric_frame(frame, column)
    if CALIBRATED in out and RAW in out and out[RAW].get("rmse"):
        out["calibrated_vs_raw"] = {
            "rmse_delta": round(float(out[CALIBRATED]["rmse"]) - float(out[RAW]["rmse"]), 6),
            "rmse_improvement_pct": round(100.0 * (float(out[RAW]["rmse"]) - float(out[CALIBRATED]["rmse"])) / float(out[RAW]["rmse"]), 3),
        }
    return out


def candidate_prediction_columns(frame: Any, *, include_calibrated: bool) -> list[str]:
    candidates = [
        CALIBRATED,
        RAW,
        "chronos_wind_mean_ms_p10",
        "chronos_wind_mean_ms_p50",
        "chronos_wind_mean_ms_p90",
        "chronos2_univar_wind_mean_ms_p10",
        "chronos2_univar_wind_mean_ms_p50",
        "chronos2_univar_wind_mean_ms_p90",
        "timesfm_wind_mean_ms_p10",
        "timesfm_wind_mean_ms_p50",
        "timesfm_wind_mean_ms_p90",
        "moirai_wind_mean_ms_p10",
        "moirai_wind_mean_ms_p50",
        "moirai_wind_mean_ms_p90",
        "persist_wind_mean_ms",
        "past_wind_mean_4",
        "past_wind_mean_8",
        "past_wind_trend_4",
    ]
    if not include_calibrated:
        candidates = [column for column in candidates if column != CALIBRATED]
    return [column for column in candidates if column in frame.columns]


def best_metric(metrics: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    best = None
    for name, metric in metrics.items():
        value = metric.get("rmse")
        if value is None:
            continue
        if best is None or float(value) < float(best["rmse"]):
            best = {"prediction": name, **metric}
    return best


def oracle_by_group(frame: Any, group_columns: list[str], candidates: list[str], prediction_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not candidates or any(column not in frame.columns for column in group_columns):
        return {"count": 0}, []
    oracle = frame.copy()
    choices = []
    oracle[prediction_name] = float("nan")
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metrics = {column: metric_frame(group, column) for column in candidates}
        best = best_metric(metrics)
        if not best:
            continue
        oracle.loc[group.index, prediction_name] = group[best["prediction"]]
        item = {column: str(value) for column, value in zip(group_columns, keys, strict=True)}
        item.update({
            "chosen_prediction": best["prediction"],
            "chosen_rmse": best["rmse"],
            "count": best["count"],
        })
        choices.append(item)
    choices.sort(key=lambda item: (item["chosen_prediction"], item["chosen_rmse"]))
    return metric_frame(oracle, prediction_name), choices


def row_oracle(frame: Any, candidates: list[str], prediction_name: str) -> dict[str, Any]:
    if not candidates:
        return {"count": 0}
    valid = frame[[TARGET, *candidates]].copy()
    values = valid[candidates].apply(lambda column: column.astype(float))
    target = valid[TARGET].astype(float)
    abs_errors = values.sub(target, axis=0).abs()
    best_columns = abs_errors.idxmin(axis=1)
    prediction = []
    for idx, column in best_columns.items():
        prediction.append(values.at[idx, column])
    oracle = frame.loc[best_columns.index].copy()
    oracle[prediction_name] = prediction
    return metric_frame(oracle, prediction_name)


def oracle_bounds(frame: Any, threshold_rmse: float) -> dict[str, Any]:
    candidates = candidate_prediction_columns(frame, include_calibrated=False)
    candidate_metrics = {column: metric_frame(frame, column) for column in candidates}
    global_best = best_metric(candidate_metrics)
    spot_metric, spot_choices = oracle_by_group(frame, ["spot_id"], candidates, "__oracle_by_spot")
    spot_lead_metric, spot_lead_choices = oracle_by_group(frame, ["spot_id", "lead_time_minutes"], candidates, "__oracle_by_spot_lead")
    lead_metric, lead_choices = oracle_by_group(frame, ["lead_time_minutes"], candidates, "__oracle_by_lead")
    row_metric = row_oracle(frame, candidates, "__oracle_by_row")
    return {
        "diagnostic_leaky_upper_bound": True,
        "note": "Oracle metrics use target values to choose predictors and are diagnostics only, never trainable results.",
        "candidate_predictions": candidates,
        "global_best": global_best,
        "oracle_by_lead": lead_metric,
        "oracle_by_spot": spot_metric,
        "oracle_by_spot_lead": spot_lead_metric,
        "oracle_by_row": row_metric,
        "oracle_by_spot_lead_below_threshold": bool(spot_lead_metric.get("rmse") is not None and float(spot_lead_metric["rmse"]) < threshold_rmse),
        "oracle_by_row_below_threshold": bool(row_metric.get("rmse") is not None and float(row_metric["rmse"]) < threshold_rmse),
        "lead_choices": lead_choices,
        "spot_choices": spot_choices,
        "spot_lead_choices": spot_lead_choices,
    }


def summarize_oracle(bounds: dict[str, Any]) -> dict[str, Any]:
    return {
        "global_best_prediction": (bounds.get("global_best") or {}).get("prediction"),
        "global_best_rmse": (bounds.get("global_best") or {}).get("rmse"),
        "oracle_by_lead_rmse": (bounds.get("oracle_by_lead") or {}).get("rmse"),
        "oracle_by_spot_rmse": (bounds.get("oracle_by_spot") or {}).get("rmse"),
        "oracle_by_spot_lead_rmse": (bounds.get("oracle_by_spot_lead") or {}).get("rmse"),
        "oracle_by_row_rmse": (bounds.get("oracle_by_row") or {}).get("rmse"),
        "oracle_by_spot_lead_below_threshold": bounds.get("oracle_by_spot_lead_below_threshold"),
        "oracle_by_row_below_threshold": bounds.get("oracle_by_row_below_threshold"),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas/pyarrow are required for RMSE09 error analysis.") from exc

    sweep = load_json(args.sweep_results)
    audit = load_json(args.audit_json) if args.audit_json and args.audit_json.exists() else {}
    best = best_run_from_sweep(sweep, audit.get("best_run_name"), audit.get("best_model_family"))
    if not best:
        raise SystemExit("No calibrator run with RMSE was found in the sweep.")
    model_family = str(best.get("model_family"))
    run_name = str(best.get("run_name") or model_family)
    predictions_path = args.predictions_path or args.sweep_results.parent / run_name / "calibrator_predictions.parquet"
    if not predictions_path.exists():
        raise SystemExit(f"Missing predictions parquet: {predictions_path}")

    frame = pd.read_parquet(predictions_path)
    missing = sorted({TARGET, CALIBRATED} - set(frame.columns))
    if missing:
        raise SystemExit(f"Missing required prediction columns: {missing}")
    frame = add_bins(frame, pd)
    overall = metric_frame(frame, CALIBRATED)
    threshold_gap = None
    if overall.get("rmse") is not None:
        threshold_gap = round(float(overall["rmse"]) - args.threshold_rmse, 6)

    bounds = oracle_bounds(frame, args.threshold_rmse)
    result = {
        "format": "corsewind.rmse09_error_analysis.v1",
        "generated_at_utc": utc_now(),
        "sweep_results": str(args.sweep_results),
        "audit_json": str(args.audit_json) if args.audit_json else None,
        "predictions_path": str(predictions_path),
        "model_family": model_family,
        "run_name": run_name,
        "fit_group": best.get("fit_group"),
        "threshold_rmse": args.threshold_rmse,
        "overall": overall,
        "threshold_gap": threshold_gap,
        "audit_verdict": audit.get("verdict"),
        "audit_effective_rmse": audit.get("effective_rmse"),
        "prediction_comparison": compare_predictions(frame),
        "oracle_summary": summarize_oracle(bounds),
        "oracle_bounds": bounds,
        "worst_spot_lead": grouped_metrics(frame, ["spot_id", "lead_time_minutes"], CALIBRATED, limit=args.limit),
        "worst_spots": grouped_metrics(frame, ["spot_id"], CALIBRATED, limit=args.limit),
        "by_lead": grouped_metrics(frame, ["lead_time_minutes"], CALIBRATED),
        "by_actual_wind_bin": grouped_metrics(frame, ["actual_wind_bin_ms"], CALIBRATED),
        "by_raw_error_bin": grouped_metrics(frame, ["raw_error_abs_bin_ms"], CALIBRATED),
        "by_issue_hour": grouped_metrics(frame, ["issue_hour_utc"], CALIBRATED),
        "worst_issue_days": grouped_metrics(frame, ["issue_day"], CALIBRATED, limit=args.limit),
    }
    return result


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# RMSE09 Error Analysis",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Model: `{result['model_family']}`",
        f"Run: `{result.get('run_name')}`",
        f"Fit group: `{result.get('fit_group')}`",
        f"Overall RMSE: `{result['overall'].get('rmse')}`",
        f"Threshold gap: `{result.get('threshold_gap')}`",
        f"Audit verdict: `{result.get('audit_verdict')}`",
        f"Audit effective RMSE: `{result.get('audit_effective_rmse')}`",
        "",
        "## Prediction Comparison",
        "",
        "| Prediction | Count | RMSE | MAE | Bias |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metric in result["prediction_comparison"].items():
        if name == "calibrated_vs_raw":
            continue
        lines.append(f"| `{name}` | {metric.get('count')} | {metric.get('rmse')} | {metric.get('mae')} | {metric.get('bias')} |")
    if "calibrated_vs_raw" in result["prediction_comparison"]:
        item = result["prediction_comparison"]["calibrated_vs_raw"]
        lines.extend([
            "",
            f"Calibrated vs raw RMSE delta: `{item.get('rmse_delta')}`",
            f"Calibrated vs raw RMSE improvement: `{item.get('rmse_improvement_pct')}%`",
        ])
    oracle = result.get("oracle_bounds", {})
    lines.extend([
        "",
        "## Oracle Bounds",
        "",
        "These are diagnostic upper bounds that use target values to choose predictors; they are not valid trainable scores.",
        "",
        "| Oracle | Prediction | Count | RMSE | MAE | Bias | Below 0.9 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for label, key in [
        ("Global best", "global_best"),
        ("By lead", "oracle_by_lead"),
        ("By spot", "oracle_by_spot"),
        ("By spot/lead", "oracle_by_spot_lead"),
        ("By row", "oracle_by_row"),
    ]:
        item = oracle.get(key) or {}
        below = ""
        if item.get("rmse") is not None:
            below = str(float(item["rmse"]) < float(result["threshold_rmse"]))
        lines.append(
            f"| `{label}` | `{item.get('prediction', '')}` | {item.get('count')} | "
            f"{item.get('rmse')} | {item.get('mae')} | {item.get('bias')} | {below} |"
        )
    lines.extend([
        "",
        f"Oracle by spot/lead below threshold: `{oracle.get('oracle_by_spot_lead_below_threshold')}`",
        f"Row oracle below threshold: `{oracle.get('oracle_by_row_below_threshold')}`",
    ])
    for title, key, columns in [
        ("Worst Spot/Lead", "worst_spot_lead", ["spot_id", "lead_time_minutes"]),
        ("Worst Spots", "worst_spots", ["spot_id"]),
        ("By Lead", "by_lead", ["lead_time_minutes"]),
        ("By Actual Wind Bin", "by_actual_wind_bin", ["actual_wind_bin_ms"]),
        ("By Raw Error Bin", "by_raw_error_bin", ["raw_error_abs_bin_ms"]),
        ("Worst Issue Days", "worst_issue_days", ["issue_day"]),
    ]:
        lines.extend(["", f"## {title}", "", "| Group | Count | RMSE | MAE | Bias |", "| --- | ---: | ---: | ---: | ---: |"])
        for item in result.get(key, []):
            group = " / ".join(str(item.get(column)) for column in columns)
            lines.append(f"| `{group}` | {item.get('count')} | {item.get('rmse')} | {item.get('mae')} | {item.get('bias')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_results", type=Path)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--predictions-path", type=Path)
    parser.add_argument("--threshold-rmse", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(args)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
