#!/usr/bin/env python3
"""Build a scientific error diagnostic report for CorseWind predictions."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PREDICTION_COLUMNS = (
    "raw_wind_mean_ms",
    "corrected_wind_mean_ms",
    "calibrated_wind_mean_ms",
)
TARGET_COLUMN = "actual_wind_mean_ms"

FEATURE_FAMILIES = {
    "sst": ("sst",),
    "cloud": ("cloud",),
    "instability": ("instability", "lifted_index", "k_index", "total_totals"),
    "land_surface_temperature": ("land_surface_temperature",),
    "surface_pressure": ("pressure", "sea_level_pressure", "pressure_msl"),
    "temperature": ("temperature",),
    "radiation": ("radiation", "shortwave", "insolation"),
    "context_wind": ("context", "wind_mean"),
    "recent_obs": ("obs_lag", "obs_delta", "model_error_now"),
    "vertical_profile": ("vertical", "isobaric", "pressure_level", "lapse_rate", "geopotential"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def import_deps() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas, pyarrow, and numpy are required.") from exc
    return {"np": np, "pd": pd}


def metric(errors: Any, np: Any) -> dict[str, Any]:
    errors = errors.dropna() if hasattr(errors, "dropna") else errors
    if len(errors) == 0:
        return {"count": 0}
    abs_errors = np.abs(errors)
    return {
        "count": int(len(errors)),
        "rmse": round(float(math.sqrt(float(np.mean(errors * errors)))), 6),
        "mae": round(float(np.mean(abs_errors)), 6),
        "bias": round(float(np.mean(errors)), 6),
        "p50_abs_error": round(float(np.quantile(abs_errors, 0.50)), 6),
        "p90_abs_error": round(float(np.quantile(abs_errors, 0.90)), 6),
        "p95_abs_error": round(float(np.quantile(abs_errors, 0.95)), 6),
        "p99_abs_error": round(float(np.quantile(abs_errors, 0.99)), 6),
    }


def metric_for(frame: Any, prediction_column: str, target_column: str, np: Any) -> dict[str, Any]:
    valid = frame[[prediction_column, target_column]].dropna()
    return metric(valid[prediction_column].astype(float) - valid[target_column].astype(float), np)


def json_scalar(pd: Any, value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def add_derived_columns(frame: Any, pd: Any) -> Any:
    out = frame.copy()
    issue = pd.to_datetime(out["issue_time_utc"], utc=True, errors="coerce")
    lead = pd.to_timedelta(out["lead_time_minutes"].astype(float), unit="m")
    target_time = issue + lead
    out["issue_month"] = issue.dt.strftime("%Y-%m")
    out["issue_hour_utc"] = issue.dt.hour.astype("Int64")
    out["target_hour_utc"] = target_time.dt.hour.astype("Int64")
    out["target_hour_local"] = target_time.dt.tz_convert("Europe/Paris").dt.hour.astype("Int64")
    out["target_date_local"] = target_time.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
    out["actual_wind_bin_ms"] = pd.cut(
        out[TARGET_COLUMN].astype(float),
        bins=[-0.001, 2, 4, 6, 8, 10, 999],
        labels=["0-2", "2-4", "4-6", "6-8", "8-10", "10+"],
    ).astype(str)
    out["calibrated_wind_bin_ms"] = pd.cut(
        out["calibrated_wind_mean_ms"].astype(float),
        bins=[-0.001, 2, 4, 6, 8, 10, 999],
        labels=["0-2", "2-4", "4-6", "6-8", "8-10", "10+"],
    ).astype(str)
    out["calibrated_error_ms"] = out["calibrated_wind_mean_ms"].astype(float) - out[TARGET_COLUMN].astype(float)
    out["abs_calibrated_error_ms"] = out["calibrated_error_ms"].abs()
    out["squared_calibrated_error"] = out["calibrated_error_ms"] ** 2
    out["error_sign_bin"] = pd.cut(
        out["calibrated_error_ms"],
        bins=[-999, -3, -2, -1, 1, 2, 3, 999],
        labels=["under_3plus", "under_2_3", "under_1_2", "ok_-1_1", "over_1_2", "over_2_3", "over_3plus"],
    ).astype(str)
    if "raw_wind_mean_ms" in out.columns:
        raw_error = out["raw_wind_mean_ms"].astype(float) - out[TARGET_COLUMN].astype(float)
        out["raw_abs_error_bin_ms"] = pd.cut(
            raw_error.abs(),
            bins=[-0.001, 0.5, 1, 2, 3, 999],
            labels=["0-0.5", "0.5-1", "1-2", "2-3", "3+"],
        ).astype(str)
    for column in [
        "features__model_error_now_wind_mean_ms",
        "features__obs_delta_15m_wind_mean_ms",
        "features__obs_delta_60m_wind_mean_ms",
        "features__context_agg_inland_delta_vs_target_wind_mean_ms_mean",
        "features__context_agg_inland_wind_mean_ms_mean",
        "features__model_open_meteo_meteofrance_arome_france_shortwave_radiation",
        "baselines__baseline_shortwave_radiation",
        "features__eumetsat_global_instability_indices_lifted_index",
        "features__eumetsat_global_instability_indices_k_index",
    ]:
        if column in out.columns:
            values = pd.to_numeric(out[column], errors="coerce")
            try:
                out[f"{column}__quartile"] = pd.qcut(values, q=4, duplicates="drop").astype(str)
            except ValueError:
                pass
    return out


def group_sse(frame: Any, columns: list[str], prediction_column: str, target_column: str, np: Any, pd: Any, limit: int) -> list[dict[str, Any]]:
    if any(column not in frame.columns for column in columns):
        return []
    valid = frame.dropna(subset=[*columns, prediction_column, target_column]).copy()
    if valid.empty:
        return []
    total_sse = float(((valid[prediction_column].astype(float) - valid[target_column].astype(float)) ** 2).sum())
    total_count = int(len(valid))
    current_rmse = math.sqrt(total_sse / total_count)
    rows = []
    for raw_key, group in valid.groupby(columns, dropna=False):
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        errors = group[prediction_column].astype(float) - group[target_column].astype(float)
        sse = float((errors * errors).sum())
        rmse_if_perfect = math.sqrt(max(0.0, total_sse - sse) / total_count)
        rows.append({
            "group": dict(zip(columns, [json_scalar(pd, value) for value in values], strict=True)),
            "row_share_pct": round(float(len(group) / total_count * 100.0), 3),
            "sse_share_pct": round(float(sse / total_sse * 100.0), 3),
            "global_rmse_if_perfect": round(float(rmse_if_perfect), 6),
            "global_rmse_gain_if_perfect": round(float(current_rmse - rmse_if_perfect), 6),
            **metric(errors, np),
        })
    rows.sort(key=lambda item: (item["sse_share_pct"], item["rmse"]), reverse=True)
    return rows[:limit]


def tail_analysis(frame: Any, np: Any, threshold_rmse: float) -> dict[str, Any]:
    squared = frame["squared_calibrated_error"].dropna().sort_values(ascending=False).to_numpy()
    n = len(squared)
    total = float(squared.sum())
    target = float(threshold_rmse * threshold_rmse * n)
    excess = max(0.0, total - target)
    cumulative = np.cumsum(squared)
    needed = int(np.searchsorted(cumulative, excess, side="left") + 1) if excess > 0 else 0
    out = {
        "row_count": int(n),
        "current_sse": round(total, 6),
        "target_sse": round(target, 6),
        "excess_sse": round(excess, 6),
        "mse_reduction_needed_pct": round(float(excess / total * 100.0), 3) if total else 0.0,
        "perfect_rows_needed": min(needed, n),
        "perfect_rows_needed_pct": round(float(min(needed, n) / n * 100.0), 3) if n else 0.0,
    }
    for pct in [1, 2, 5, 10, 20]:
        count = max(1, int(math.ceil(n * pct / 100.0)))
        out[f"top_{pct}_pct_sse_share_pct"] = round(float(squared[:count].sum() / total * 100.0), 3) if total else 0.0
    return out


def stage_metrics(frame: Any, np: Any) -> dict[str, Any]:
    out = {}
    for column in PREDICTION_COLUMNS:
        if column in frame.columns:
            out[column] = metric_for(frame, column, TARGET_COLUMN, np)
    raw = out.get("raw_wind_mean_ms", {}).get("rmse")
    corrected = out.get("corrected_wind_mean_ms", {}).get("rmse")
    calibrated = out.get("calibrated_wind_mean_ms", {}).get("rmse")
    out["improvements"] = {}
    if raw and corrected:
        out["improvements"]["raw_to_corrected_rmse_gain_pct"] = round((raw - corrected) / raw * 100.0, 3)
    if corrected and calibrated:
        out["improvements"]["corrected_to_calibrated_rmse_gain_pct"] = round((corrected - calibrated) / corrected * 100.0, 3)
    if raw and calibrated:
        out["improvements"]["raw_to_calibrated_rmse_gain_pct"] = round((raw - calibrated) / raw * 100.0, 3)
    return out


def availability_deltas(frame: Any, np: Any, pd: Any) -> dict[str, Any]:
    high_threshold = float(frame["squared_calibrated_error"].quantile(0.90))
    high = frame["squared_calibrated_error"] >= high_threshold
    out = {"high_error_threshold_squared": round(high_threshold, 6), "high_error_row_count": int(high.sum()), "families": {}}
    columns = [column for column in frame.columns if column.startswith("features__") or column.startswith("baselines__")]
    for family, patterns in FEATURE_FAMILIES.items():
        matches = [
            column for column in columns
            if any(pattern.lower() in column.lower() for pattern in patterns)
            and "actual_wind" not in column.lower()
            and "error" not in column.lower()
        ]
        if not matches:
            out["families"][family] = {"column_count": 0}
            continue
        any_available = frame[matches].notna().any(axis=1)
        high_rate = float(any_available[high].mean() * 100.0) if int(high.sum()) else 0.0
        rest_rate = float(any_available[~high].mean() * 100.0) if int((~high).sum()) else 0.0
        out["families"][family] = {
            "column_count": len(matches),
            "any_available_high_error_pct": round(high_rate, 3),
            "any_available_other_pct": round(rest_rate, 3),
            "delta_high_minus_other_pct": round(high_rate - rest_rate, 3),
            "top_columns_by_coverage": [
                {
                    "column": column,
                    "coverage_pct": round(float(frame[column].notna().mean() * 100.0), 3),
                }
                for column in sorted(matches, key=lambda item: float(frame[item].notna().mean()), reverse=True)[:8]
            ],
        }
    return out


def feature_error_correlations(frame: Any, pd: Any, limit: int) -> list[dict[str, Any]]:
    candidates = []
    for column in frame.columns:
        low = column.lower()
        if not (column.startswith("features__") or column.startswith("baselines__")):
            continue
        if any(token in low for token in ["actual_wind", "raw_error", "corrected_error", "abs_"]):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() < 1000 or values.nunique(dropna=True) < 5:
            continue
        corr = values.corr(frame["abs_calibrated_error_ms"], method="spearman")
        if pd.isna(corr):
            continue
        candidates.append({
            "column": column,
            "spearman_abs_error": round(float(corr), 6),
            "non_null_rate_pct": round(float(values.notna().mean() * 100.0), 3),
        })
    candidates.sort(key=lambda item: abs(item["spearman_abs_error"]), reverse=True)
    return candidates[:limit]


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Scientific Error Diagnostic",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Prediction file: `{result['prediction_path']}`",
        f"Rows: `{result['row_count']}`",
        f"Champion RMSE: `{result['stage_metrics']['calibrated_wind_mean_ms']['rmse']}`",
        f"Champion MAE: `{result['stage_metrics']['calibrated_wind_mean_ms']['mae']}`",
        f"MSE reduction needed for threshold: `{result['tail']['mse_reduction_needed_pct']}%`",
        "",
        "## Stage Metrics",
        "",
        "| Stage | Count | RMSE | MAE | Bias | P90 abs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for column in PREDICTION_COLUMNS:
        item = result["stage_metrics"].get(column)
        if item:
            lines.append(f"| `{column}` | {item['count']} | {item['rmse']} | {item['mae']} | {item['bias']} | {item['p90_abs_error']} |")
    lines.extend(["", "## Tail", ""])
    for key, value in result["tail"].items():
        lines.append(f"- `{key}`: `{value}`")
    for title, key in [
        ("SSE By Spot Lead", "by_spot_lead"),
        ("SSE By Spot", "by_spot"),
        ("SSE By Lead", "by_lead"),
        ("SSE By Local Target Hour", "by_target_hour_local"),
        ("SSE By Month", "by_month"),
        ("SSE By Actual Wind Bin", "by_actual_wind_bin"),
        ("SSE By Error Sign", "by_error_sign"),
    ]:
        lines.extend(["", f"## {title}", "", "| Group | Count | RMSE | MAE | Bias | SSE share | RMSE if perfect |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for item in result["groups"].get(key, []):
            lines.append(
                f"| `{item['group']}` | {item['count']} | {item['rmse']} | {item['mae']} | "
                f"{item['bias']} | {item['sse_share_pct']}% | {item['global_rmse_if_perfect']} |"
            )
    lines.extend(["", "## Feature Availability In High Error Tail", "", "| Family | Columns | High-error coverage | Other coverage | Delta |", "| --- | ---: | ---: | ---: | ---: |"])
    for family, item in result["availability_deltas"]["families"].items():
        lines.append(
            f"| `{family}` | {item.get('column_count')} | {item.get('any_available_high_error_pct')}% | "
            f"{item.get('any_available_other_pct')}% | {item.get('delta_high_minus_other_pct')}% |"
        )
    lines.extend(["", "## Top Spearman Correlations With Absolute Error", "", "| Feature | Spearman | Coverage |", "| --- | ---: | ---: |"])
    for item in result["feature_error_correlations"]:
        lines.append(f"| `{item['column']}` | {item['spearman_abs_error']} | {item['non_null_rate_pct']}% |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_deps()
    pd = deps["pd"]
    np = deps["np"]
    frame = pd.read_parquet(args.predictions)
    if args.lead_minute:
        frame = frame[frame["lead_time_minutes"].astype("Int64").isin([int(item) for item in args.lead_minute])]
    frame = add_derived_columns(frame, pd)
    groups = {
        "by_spot_lead": group_sse(frame, ["spot_id", "lead_time_minutes"], "calibrated_wind_mean_ms", TARGET_COLUMN, np, pd, args.limit),
        "by_spot": group_sse(frame, ["spot_id"], "calibrated_wind_mean_ms", TARGET_COLUMN, np, pd, args.limit),
        "by_lead": group_sse(frame, ["lead_time_minutes"], "calibrated_wind_mean_ms", TARGET_COLUMN, np, pd, args.limit),
        "by_target_hour_local": group_sse(frame, ["target_hour_local"], "calibrated_wind_mean_ms", TARGET_COLUMN, np, pd, 24),
        "by_month": group_sse(frame, ["issue_month"], "calibrated_wind_mean_ms", TARGET_COLUMN, np, pd, args.limit),
        "by_actual_wind_bin": group_sse(frame, ["actual_wind_bin_ms"], "calibrated_wind_mean_ms", TARGET_COLUMN, np, pd, args.limit),
        "by_calibrated_wind_bin": group_sse(frame, ["calibrated_wind_bin_ms"], "calibrated_wind_mean_ms", TARGET_COLUMN, np, pd, args.limit),
        "by_error_sign": group_sse(frame, ["error_sign_bin"], "calibrated_wind_mean_ms", TARGET_COLUMN, np, pd, args.limit),
    }
    for column in list(frame.columns):
        if column.endswith("__quartile"):
            groups[f"by_{column}"] = group_sse(frame, [column], "calibrated_wind_mean_ms", TARGET_COLUMN, np, pd, args.limit)
    result = {
        "format": "corsewind.scientific_error_diagnostic.v1",
        "generated_at_utc": utc_now(),
        "prediction_path": str(args.predictions),
        "row_count": int(len(frame)),
        "stage_metrics": stage_metrics(frame, np),
        "tail": tail_analysis(frame, np, args.threshold_rmse),
        "groups": groups,
        "availability_deltas": availability_deltas(frame, np, pd),
        "feature_error_correlations": feature_error_correlations(frame, pd, args.limit),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--lead-minute", type=int, action="append", default=[15, 30, 45, 60])
    parser.add_argument("--threshold-rmse", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, result)
    print(json.dumps({
        "rows": result["row_count"],
        "rmse": result["stage_metrics"]["calibrated_wind_mean_ms"]["rmse"],
        "mae": result["stage_metrics"]["calibrated_wind_mean_ms"]["mae"],
        "mse_reduction_needed_pct": result["tail"]["mse_reduction_needed_pct"],
        "top_spot_leads": result["groups"]["by_spot_lead"][:5],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
