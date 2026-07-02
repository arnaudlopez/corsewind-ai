#!/usr/bin/env python3
"""Audit ML prediction feature-family coverage, especially hard-regime inputs."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_FAMILIES: dict[str, list[str]] = {
    "sst": ["sst"],
    "land_surface_temperature": ["land_surface_temperature", "LST", "lst"],
    "thermal_derived": ["thermal_"],
    "cloud": ["cloud"],
    "instability": ["instability", "lifted_index", "k_index", "total_totals", "cape"],
    "radiation": ["radiation", "shortwave", "insolation"],
    "surface_pressure": ["pressure", "sea_level_pressure", "surface_pressure", "pressure_msl"],
    "vertical_profile": ["vertical_arome_", "open_meteo_vertical_", "isobaric", "pressure_level", "geopotential_height", "lapse_rate"],
    "upwind": ["upwind"],
    "coastal_inland_relief": ["coastal", "inland", "relief"],
    "recent_obs_trends": ["obs_delta", "obs_lag", "model_error_now", "recent_"],
    "previous_runs": ["previous_run", "best_match_day"],
}

REQUIRED_CONCEPTS: dict[str, list[str]] = {
    "sea_surface_temperature": ["sst_c", "sst_k"],
    "land_sea_delta": ["thermal_land_minus_sst", "land_minus_sst"],
    "air_sea_delta": ["thermal_air_minus_sst", "air_minus_sst"],
    "land_air_delta": ["thermal_land_minus_air", "land_minus_air"],
    "shortwave_ramp": ["shortwave_ramp", "shortwave_radiation", "insolation"],
    "cloud_type": ["cloud_type"],
    "cloud_mask": ["cloud_mask"],
    "instability_indices": ["global_instability", "lifted_index", "k_index", "total_totals"],
    "upwind_station_aggregates": ["upwind_weighted", "upwind_score"],
    "coastal_inland_temperature_delta": ["inland_minus_coastal_temperature"],
    "coastal_relief_temperature_delta": ["relief_minus_coastal_temperature"],
    "coastal_inland_pressure_delta": ["inland_minus_coastal_pressure"],
    "coastal_relief_pressure_delta": ["relief_minus_coastal_pressure"],
    "recent_temperature_tendency": ["recent_heating_rate", "obs_delta_60m_temperature"],
    "recent_pressure_tendency": ["recent_pressure_tendency", "obs_delta_60m_pressure"],
    "vertical_temperature_profile": ["vertical_arome_temperature", "open_meteo_vertical_temperature", "lapse_rate"],
    "vertical_humidity_profile": ["vertical_arome_relative_humidity", "open_meteo_vertical_relative_humidity"],
    "vertical_motion_profile": ["vertical_arome_vertical_velocity"],
    "geopotential_thickness": ["geopotential_thickness", "vertical_arome_geopotential", "open_meteo_vertical_geopotential"],
}

LEAKY_KEYWORDS = (
    "actual_wind",
    "corrected_error",
    "raw_error",
    "abs_corrected_error",
    "abs_raw_error",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def metric(prediction: Any, observation: Any, np: Any) -> dict[str, Any]:
    valid = ~(np.isnan(prediction) | np.isnan(observation))
    prediction = prediction[valid]
    observation = observation[valid]
    if len(prediction) == 0:
        return {"count": 0}
    errors = prediction - observation
    return {
        "count": int(len(errors)),
        "mae": round(float(np.mean(np.abs(errors))), 6),
        "rmse": round(float(math.sqrt(float(np.mean(errors * errors)))), 6),
        "bias": round(float(np.mean(errors)), 6),
    }


def normalize(text: str) -> str:
    return text.lower()


def family_columns(columns: list[str], patterns: list[str]) -> list[str]:
    lowered_patterns = [normalize(pattern) for pattern in patterns]
    return [
        column
        for column in columns
        if any(pattern in normalize(column) for pattern in lowered_patterns)
        and not any(keyword in normalize(column) for keyword in LEAKY_KEYWORDS)
    ]


def coverage_for_columns(frame: Any, columns: list[str], np: Any) -> dict[str, Any]:
    if not columns:
        return {
            "column_count": 0,
            "columns": [],
            "any_non_null_rate_pct": 0.0,
            "mean_column_non_null_rate_pct": 0.0,
            "columns_with_50pct_coverage": 0,
            "columns_with_90pct_coverage": 0,
        }
    rates = []
    for column in columns:
        rates.append(float(frame[column].notna().mean() * 100.0))
    any_non_null = frame[columns].notna().any(axis=1).mean() * 100.0
    return {
        "column_count": int(len(columns)),
        "columns": columns[:80],
        "any_non_null_rate_pct": round(float(any_non_null), 3),
        "mean_column_non_null_rate_pct": round(float(np.mean(rates)), 3),
        "columns_with_50pct_coverage": int(sum(rate >= 50.0 for rate in rates)),
        "columns_with_90pct_coverage": int(sum(rate >= 90.0 for rate in rates)),
        "top_coverage_columns": [
            {"column": column, "non_null_rate_pct": round(rate, 3)}
            for column, rate in sorted(zip(columns, rates, strict=True), key=lambda item: item[1], reverse=True)[:20]
        ],
    }


def concept_presence(columns: list[str]) -> dict[str, Any]:
    out = {}
    for concept, patterns in REQUIRED_CONCEPTS.items():
        matches = family_columns(columns, patterns)
        out[concept] = {
            "present": bool(matches),
            "column_count": len(matches),
            "example_columns": matches[:12],
        }
    return out


def hard_mask(frame: Any, args: argparse.Namespace) -> Any:
    masks = []
    if args.hard_spot:
        masks.append(frame["spot_id"].astype(str).isin(set(args.hard_spot)))
    if args.hard_min_lead is not None and "lead_time_minutes" in frame.columns:
        masks.append(frame["lead_time_minutes"].astype(float) >= float(args.hard_min_lead))
    if args.hard_min_actual_ms is not None and "actual_wind_mean_ms" in frame.columns:
        masks.append(frame["actual_wind_mean_ms"].astype(float) >= float(args.hard_min_actual_ms))
    if args.hard_min_prediction_ms is not None and args.prediction_column in frame.columns:
        masks.append(frame[args.prediction_column].astype(float) >= float(args.hard_min_prediction_ms))
    if not masks:
        return frame.index == frame.index
    mask = masks[0]
    for item in masks[1:]:
        mask = mask | item
    return mask


def family_audit(frame: Any, families: dict[str, list[str]], np: Any) -> dict[str, Any]:
    columns = list(frame.columns)
    out = {}
    for name, patterns in families.items():
        matches = family_columns(columns, patterns)
        out[name] = coverage_for_columns(frame, matches, np)
    return out


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Feature Family Coverage Audit",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Prediction file: `{result['prediction_path']}`",
        f"Rows: `{result['row_count']}`",
        f"Hard rows: `{result['hard_row_count']}`",
        f"Prediction RMSE: `{result['overall_metric'].get('rmse')}`",
        f"Hard RMSE: `{result['hard_metric'].get('rmse')}`",
        "",
        "## Required Concepts",
        "",
        "| Concept | Present | Columns | Examples |",
        "| --- | --- | ---: | --- |",
    ]
    for concept, item in result["required_concepts"].items():
        examples = ", ".join(f"`{column}`" for column in item["example_columns"][:4])
        lines.append(f"| `{concept}` | `{item['present']}` | {item['column_count']} | {examples} |")
    lines.extend([
        "",
        "## Family Coverage",
        "",
        "| Family | Columns | Any coverage | Mean coverage | >=90% columns | Hard any coverage | Hard mean coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for family, item in result["families"].items():
        hard = result["hard_families"].get(family, {})
        lines.append(
            f"| `{family}` | {item['column_count']} | {item['any_non_null_rate_pct']}% | "
            f"{item['mean_column_non_null_rate_pct']}% | {item['columns_with_90pct_coverage']} | "
            f"{hard.get('any_non_null_rate_pct')}% | {hard.get('mean_column_non_null_rate_pct')}% |"
        )
    missing = [name for name, item in result["required_concepts"].items() if not item["present"]]
    lines.extend(["", "## Missing Concepts", ""])
    if missing:
        for name in missing:
            lines.append(f"- `{name}`")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    frame = pd.read_parquet(args.predictions)
    if args.start_utc and "issue_time_utc" in frame.columns:
        frame["issue_time_utc"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
        frame = frame[frame["issue_time_utc"] >= pd.Timestamp(args.start_utc, tz="UTC")]
    if args.end_utc and "issue_time_utc" in frame.columns:
        frame["issue_time_utc"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
        frame = frame[frame["issue_time_utc"] < pd.Timestamp(args.end_utc, tz="UTC")]
    if args.lead_minute and "lead_time_minutes" in frame.columns:
        frame = frame[frame["lead_time_minutes"].astype("Int64").isin([int(lead) for lead in args.lead_minute])]
    frame = frame.copy()
    if args.prediction_column not in frame.columns:
        raise SystemExit(f"Missing prediction column: {args.prediction_column}")
    if "actual_wind_mean_ms" not in frame.columns:
        raise SystemExit("Missing actual_wind_mean_ms")
    hard = hard_mask(frame, args)
    hard_frame = frame[hard].copy()
    result = {
        "format": "corsewind.feature_family_coverage_audit.v1",
        "generated_at_utc": utc_now(),
        "prediction_path": str(args.predictions),
        "prediction_column": args.prediction_column,
        "row_count": int(len(frame)),
        "hard_row_count": int(hard.sum()),
        "hard_rule": {
            "spots": args.hard_spot,
            "min_lead": args.hard_min_lead,
            "min_actual_ms": args.hard_min_actual_ms,
            "min_prediction_ms": args.hard_min_prediction_ms,
        },
        "overall_metric": metric(
            frame[args.prediction_column].astype(float).to_numpy(),
            frame["actual_wind_mean_ms"].astype(float).to_numpy(),
            np,
        ),
        "hard_metric": metric(
            hard_frame[args.prediction_column].astype(float).to_numpy(),
            hard_frame["actual_wind_mean_ms"].astype(float).to_numpy(),
            np,
        ),
        "required_concepts": concept_presence(list(frame.columns)),
        "families": family_audit(frame, DEFAULT_FAMILIES, np),
        "hard_families": family_audit(hard_frame, DEFAULT_FAMILIES, np),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-column", default="calibrated_wind_mean_ms")
    parser.add_argument("--lead-minute", type=int, action="append", default=[15, 30, 45, 60])
    parser.add_argument("--start-utc")
    parser.add_argument("--end-utc")
    parser.add_argument("--hard-spot", action="append", default=[])
    parser.add_argument("--hard-min-lead", type=float, default=45.0)
    parser.add_argument("--hard-min-actual-ms", type=float)
    parser.add_argument("--hard-min-prediction-ms", type=float)
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
    missing = [name for name, item in result["required_concepts"].items() if not item["present"]]
    print(json.dumps({
        "rows": result["row_count"],
        "hard_rows": result["hard_row_count"],
        "rmse": result["overall_metric"].get("rmse"),
        "hard_rmse": result["hard_metric"].get("rmse"),
        "missing_required_concepts": missing,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
