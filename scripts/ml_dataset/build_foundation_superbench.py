#!/usr/bin/env python3
"""Build a same-key foundation superbench from existing prediction artifacts."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KEY_COLUMNS = ("spot_id", "issue_time_utc", "lead_time_minutes")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def metric(frame: Any, prediction_column: str, actual_column: str, np: Any) -> dict[str, Any]:
    if prediction_column not in frame.columns or actual_column not in frame.columns:
        return {"count": 0}
    valid = frame[[prediction_column, actual_column]].dropna()
    if valid.empty:
        return {"count": 0}
    errors = valid[prediction_column].to_numpy(dtype=float) - valid[actual_column].to_numpy(dtype=float)
    return {
        "count": int(len(errors)),
        "rmse": round(float(math.sqrt(float(np.mean(errors * errors)))), 6),
        "mae": round(float(np.mean(np.abs(errors))), 6),
        "bias": round(float(np.mean(errors)), 6),
    }


def json_scalar(pd: Any, value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def grouped_metrics(frame: Any, prediction_column: str, actual_column: str, group_columns: list[str], np: Any, pd: Any) -> list[dict[str, Any]]:
    rows = []
    if any(column not in frame.columns for column in group_columns):
        return rows
    for raw_key, group in frame.groupby(group_columns, dropna=False):
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        rows.append({
            "group": dict(zip(group_columns, [json_scalar(pd, value) for value in values], strict=True)),
            **metric(group, prediction_column, actual_column, np),
        })
    rows.sort(key=lambda item: (item.get("rmse") is None, item.get("rmse", -1)), reverse=True)
    return rows


def normalize_keys(frame: Any, pd: Any) -> Any:
    out = frame.copy()
    out["spot_id"] = out["spot_id"].astype(str)
    out["issue_time_utc"] = pd.to_datetime(out["issue_time_utc"], utc=True, errors="coerce")
    out["lead_time_minutes"] = out["lead_time_minutes"].astype("Int64")
    return out.dropna(subset=list(KEY_COLUMNS)).drop_duplicates(subset=list(KEY_COLUMNS), keep="first")


def read_foundation(path: Path, pd: Any) -> Any:
    frame = normalize_keys(pd.read_parquet(path), pd)
    keep = [
        *KEY_COLUMNS,
        "actual_wind_mean_ms",
        "actual_gust_ms",
        "raw_wind_mean_ms",
        "raw_gust_ms",
        "hgb_wind_mean_ms",
        "hgb_gust_ms",
        "chronos_wind_mean_ms_mean",
        "chronos_gust_ms_mean",
        "chronos_wind_mean_ms_p50",
        "chronos_gust_ms_p50",
        "chronos2_univar_wind_mean_ms_mean",
        "chronos2_univar_gust_ms_mean",
        "timesfm_wind_mean_ms_mean",
        "timesfm_gust_ms_mean",
        "timesfm_wind_mean_ms_p50",
        "timesfm_gust_ms_p50",
        "moirai_wind_mean_ms_mean",
        "moirai_gust_ms_mean",
        "moirai_wind_mean_ms_p50",
        "moirai_gust_ms_p50",
        "chronos2_residual_wind_mean_ms_corrected_mean",
        "chronos2_residual_gust_ms_corrected_mean",
    ]
    keep = [column for column in keep if column in frame.columns]
    return frame[keep].copy()


def read_champion(path: Path, prediction_column: str, actual_column: str, prefix: str, pd: Any) -> Any:
    frame = normalize_keys(pd.read_parquet(path), pd)
    keep = [*KEY_COLUMNS]
    for column in (prediction_column, actual_column):
        if column not in frame.columns:
            raise SystemExit(f"{path} missing {column}")
        keep.append(column)
    out = frame[keep].copy()
    return out.rename(columns={
        prediction_column: f"{prefix}_prediction_ms",
        actual_column: f"{prefix}_actual_ms",
    })


def add_oracle(frame: Any, columns: list[str], actual_column: str, output_column: str, np: Any) -> tuple[Any, dict[str, Any]]:
    present = [column for column in columns if column in frame.columns]
    valid = frame[[actual_column, *present]].dropna(subset=[actual_column]).copy()
    valid = valid[valid[present].notna().any(axis=1)]
    if valid.empty:
        frame[output_column] = np.nan
        return frame, {"count": 0, "columns": present}
    predictions = valid[present].to_numpy(dtype=float)
    errors = np.abs(predictions - valid[[actual_column]].to_numpy(dtype=float))
    errors = np.where(np.isnan(errors), np.inf, errors)
    best_idx = np.argmin(errors, axis=1)
    oracle = np.take_along_axis(predictions, best_idx[:, None], axis=1)[:, 0]
    frame[output_column] = np.nan
    frame.loc[valid.index, output_column] = oracle
    selected = {}
    for idx in best_idx:
        selected[present[int(idx)]] = selected.get(present[int(idx)], 0) + 1
    return frame, {"count": int(len(valid)), "columns": present, "selected": selected}


def prediction_columns() -> tuple[list[str], list[str]]:
    return [
        "raw_wind_mean_ms",
        "hgb_wind_mean_ms",
        "chronos_wind_mean_ms_mean",
        "chronos_wind_mean_ms_p50",
        "chronos2_univar_wind_mean_ms_mean",
        "timesfm_wind_mean_ms_mean",
        "timesfm_wind_mean_ms_p50",
        "moirai_wind_mean_ms_mean",
        "moirai_wind_mean_ms_p50",
        "chronos2_residual_wind_mean_ms_corrected_mean",
        "wind_champion_prediction_ms",
        "wind_oracle_prediction_ms",
    ], [
        "raw_gust_ms",
        "hgb_gust_ms",
        "chronos_gust_ms_mean",
        "chronos_gust_ms_p50",
        "chronos2_univar_gust_ms_mean",
        "timesfm_gust_ms_mean",
        "timesfm_gust_ms_p50",
        "moirai_gust_ms_mean",
        "moirai_gust_ms_p50",
        "chronos2_residual_gust_ms_corrected_mean",
        "gust_champion_prediction_ms",
        "gust_oracle_prediction_ms",
    ]


def prediction_metrics(frame: Any, wind_columns: list[str], gust_columns: list[str], np: Any, pd: Any) -> dict[str, Any]:
    return {
        "wind_mean": {
            column: {
                "overall": metric(frame, column, "actual_wind_mean_ms", np),
                "by_lead": grouped_metrics(frame, column, "actual_wind_mean_ms", ["lead_time_minutes"], np, pd),
            }
            for column in wind_columns
            if column in frame.columns
        },
        "gust": {
            column: {
                "overall": metric(frame, column, "actual_gust_ms", np),
                "by_lead": grouped_metrics(frame, column, "actual_gust_ms", ["lead_time_minutes"], np, pd),
            }
            for column in gust_columns
            if column in frame.columns
        },
    }


def summarize(frame: Any, np: Any, pd: Any) -> dict[str, Any]:
    wind_columns, gust_columns = prediction_columns()
    subsets = {}
    for name, column in (
        ("wind_champion_overlap", "wind_champion_prediction_ms"),
        ("gust_champion_overlap", "gust_champion_prediction_ms"),
    ):
        if column in frame.columns:
            subset = frame[frame[column].notna()]
            subsets[name] = {
                "row_count": int(len(subset)),
                **prediction_metrics(subset, wind_columns, gust_columns, np, pd),
            }
    if "wind_champion_prediction_ms" in frame.columns and "gust_champion_prediction_ms" in frame.columns:
        subset = frame[frame["wind_champion_prediction_ms"].notna() & frame["gust_champion_prediction_ms"].notna()]
        subsets["both_champion_overlap"] = {
            "row_count": int(len(subset)),
            **prediction_metrics(subset, wind_columns, gust_columns, np, pd),
        }

    return {
        "row_count": int(len(frame)),
        "spot_count": int(frame["spot_id"].nunique()) if "spot_id" in frame.columns else 0,
        "issue_count": int(frame[["spot_id", "issue_time_utc"]].drop_duplicates().shape[0]),
        "coverage": {
            column: int(frame[column].notna().sum())
            for column in [*wind_columns, *gust_columns]
            if column in frame.columns
        },
        **prediction_metrics(frame, wind_columns, gust_columns, np, pd),
        "subsets": subsets,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        "# Foundation Superbench",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Rows: `{summary['row_count']}`",
        f"Spots: `{summary['spot_count']}`",
        f"Issues: `{summary['issue_count']}`",
        "",
        "## Coverage",
        "",
        "| Column | Non-null rows |",
        "| --- | ---: |",
    ]
    for column, count in sorted(summary["coverage"].items()):
        lines.append(f"| `{column}` | {count} |")
    for section, title in (("wind_mean", "Wind Mean"), ("gust", "Gust")):
        lines.extend(["", f"## {title}", "", "| Prediction | Rows | RMSE | MAE | Bias |", "| --- | ---: | ---: | ---: | ---: |"])
        for column, payload in sorted(
            summary[section].items(),
            key=lambda item: item[1]["overall"].get("rmse", float("inf")) if item[1]["overall"].get("rmse") is not None else float("inf"),
        ):
            metric_payload = payload["overall"]
            lines.append(
                f"| `{column}` | {metric_payload.get('count')} | {metric_payload.get('rmse')} | "
                f"{metric_payload.get('mae')} | {metric_payload.get('bias')} |"
            )
    for subset_name, subset in summary.get("subsets", {}).items():
        lines.extend(["", f"## Subset {subset_name}", "", f"Rows: `{subset['row_count']}`"])
        for section, title in (("wind_mean", "Wind Mean"), ("gust", "Gust")):
            lines.extend(["", f"### {title}", "", "| Prediction | Rows | RMSE | MAE | Bias |", "| --- | ---: | ---: | ---: | ---: |"])
            for column, payload in sorted(
                subset[section].items(),
                key=lambda item: item[1]["overall"].get("rmse", float("inf")) if item[1]["overall"].get("rmse") is not None else float("inf"),
            ):
                metric_payload = payload["overall"]
                lines.append(
                    f"| `{column}` | {metric_payload.get('count')} | {metric_payload.get('rmse')} | "
                    f"{metric_payload.get('mae')} | {metric_payload.get('bias')} |"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    superbench = read_foundation(args.foundation_predictions, pd)
    if args.wind_champion_predictions:
        wind = read_champion(args.wind_champion_predictions, args.wind_champion_column, args.wind_actual_column, "wind_champion", pd)
        superbench = superbench.merge(wind, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
    if args.gust_champion_predictions:
        gust = read_champion(args.gust_champion_predictions, args.gust_champion_column, args.gust_actual_column, "gust_champion", pd)
        superbench = superbench.merge(gust, on=list(KEY_COLUMNS), how="left", validate="one_to_one")

    superbench, wind_oracle = add_oracle(
        superbench,
        [
            "raw_wind_mean_ms",
            "hgb_wind_mean_ms",
            "chronos_wind_mean_ms_mean",
            "chronos_wind_mean_ms_p50",
            "chronos2_univar_wind_mean_ms_mean",
            "timesfm_wind_mean_ms_mean",
            "timesfm_wind_mean_ms_p50",
            "moirai_wind_mean_ms_mean",
            "moirai_wind_mean_ms_p50",
            "chronos2_residual_wind_mean_ms_corrected_mean",
            "wind_champion_prediction_ms",
        ],
        "actual_wind_mean_ms",
        "wind_oracle_prediction_ms",
        np,
    )
    superbench, gust_oracle = add_oracle(
        superbench,
        [
            "raw_gust_ms",
            "hgb_gust_ms",
            "chronos_gust_ms_mean",
            "chronos_gust_ms_p50",
            "chronos2_univar_gust_ms_mean",
            "timesfm_gust_ms_mean",
            "timesfm_gust_ms_p50",
            "moirai_gust_ms_mean",
            "moirai_gust_ms_p50",
            "chronos2_residual_gust_ms_corrected_mean",
            "gust_champion_prediction_ms",
        ],
        "actual_gust_ms",
        "gust_oracle_prediction_ms",
        np,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_predictions = args.output_root / "foundation_superbench.parquet"
    superbench.to_parquet(output_predictions, index=False)
    result = {
        "format": "corsewind.foundation_superbench.v1",
        "generated_at_utc": utc_now(),
        "foundation_predictions": str(args.foundation_predictions),
        "wind_champion_predictions": str(args.wind_champion_predictions) if args.wind_champion_predictions else None,
        "gust_champion_predictions": str(args.gust_champion_predictions) if args.gust_champion_predictions else None,
        "output_predictions": str(output_predictions),
        "wind_oracle": wind_oracle,
        "gust_oracle": gust_oracle,
        "summary": summarize(superbench, np, pd),
    }
    (args.output_root / "foundation_superbench_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_markdown(args.output_root / "foundation_superbench_summary.md", result)
    print(json.dumps({
        "output_root": str(args.output_root),
        "rows": result["summary"]["row_count"],
        "wind": {k: v["overall"].get("rmse") for k, v in result["summary"]["wind_mean"].items()},
        "gust": {k: v["overall"].get("rmse") for k, v in result["summary"]["gust"].items()},
    }, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-predictions", type=Path, required=True)
    parser.add_argument("--wind-champion-predictions", type=Path)
    parser.add_argument("--wind-champion-column", default="calibrated_wind_mean_ms")
    parser.add_argument("--wind-actual-column", default="actual_wind_mean_ms")
    parser.add_argument("--gust-champion-predictions", type=Path)
    parser.add_argument("--gust-champion-column", default="calibrated_gust_ms")
    parser.add_argument("--gust-actual-column", default="actual_gust_ms")
    parser.add_argument("--output-root", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
