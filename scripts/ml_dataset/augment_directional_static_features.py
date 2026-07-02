#!/usr/bin/env python3
"""Augment monthly training Parquet shards with wind-direction-aware static features."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SECTORS = ("n", "ne", "e", "se", "s", "sw", "w", "nw")
SECTOR_CENTERS = {
    "n": 0.0,
    "ne": 45.0,
    "e": 90.0,
    "se": 135.0,
    "s": 180.0,
    "sw": 225.0,
    "w": 270.0,
    "nw": 315.0,
}
SECTOR_IDS = {sector: index for index, sector in enumerate(SECTORS)}
FETCH_PREFIX = "features__spot_static_fetch_sector"
DEM_PREFIX = "features__spot_static_dem_sector"
DIRECTION_CANDIDATES = (
    "baselines__baseline_wind_direction_deg",
    "features__model_open_meteo_meteofrance_arome_france_wind_direction_10m",
    "features__model_open_meteo_arome_france_wind_direction_10m",
    "features__obs_recent_wind_direction_deg",
    "features__obs_last_wind_direction_deg",
)
WIND_SPEED_CANDIDATES = (
    "baselines__baseline_wind_mean_ms",
    "features__model_open_meteo_meteofrance_arome_france_wind_speed_10m",
    "features__model_open_meteo_arome_france_wind_speed_10m",
)


def import_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Run inside the ML environment with pandas and pyarrow installed."
        ) from exc
    return {"np": np, "pd": pd, "pa": pa, "pq": pq}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def month_range(start_month: str, end_month: str) -> list[str]:
    start_year, start_m = [int(part) for part in start_month.split("-", 1)]
    end_year, end_m = [int(part) for part in end_month.split("-", 1)]
    months = []
    year, month = start_year, start_m
    while (year, month) <= (end_year, end_m):
        months.append(f"{year:04d}_{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return months


def source_target_paths(args: argparse.Namespace, month: str) -> tuple[Path, Path]:
    source = args.training_table_root / f"{args.source_run_id_prefix}_{month}" / "training_rows.parquet"
    target = args.training_table_root / f"{args.output_run_id_prefix}_{month}" / "training_rows.parquet"
    return source, target


def angular_diff_signed_deg(a: Any, b: Any, np: Any) -> Any:
    return ((a - b + 180.0) % 360.0) - 180.0


def choose_first_numeric(frame: Any, candidates: tuple[str, ...], pd: Any) -> tuple[Any, str | None]:
    series = None
    source = None
    for column in candidates:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if series is None:
            series = values
        else:
            series = series.fillna(values)
        if source is None and values.notna().any():
            source = column
    if series is None:
        series = pd.Series(float("nan"), index=frame.index, dtype="float64")
    return series.astype("float64"), source


def sector_index_from_direction(direction_deg: Any, np: Any) -> Any:
    normalized = np.mod(direction_deg.astype("float64"), 360.0)
    return np.floor((normalized + 22.5) / 45.0).astype("float64") % 8


def values_by_sector(frame: Any, template: str, sector_index: Any, np: Any, pd: Any) -> Any:
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    sector_codes = sector_index.to_numpy()
    for index, sector in enumerate(SECTORS):
        column = template.format(sector=sector)
        if column not in frame.columns:
            continue
        mask = sector_codes == float(index)
        if mask.any():
            out.loc[mask] = pd.to_numeric(frame.loc[mask, column], errors="coerce")
    return out


def sector_id_with_offset(sector_index: Any, offset: int, np: Any, pd: Any) -> Any:
    values = np.mod(sector_index.to_numpy(dtype="float64") + float(offset), 8.0)
    values[sector_index.isna().to_numpy()] = np.nan
    return pd.Series(values, index=sector_index.index, dtype="float64")


def max_fetch_sector(frame: Any, np: Any, pd: Any) -> tuple[Any, Any]:
    columns = [
        f"{FETCH_PREFIX}_{sector}_coastal_snapped_water_fetch_km"
        for sector in SECTORS
        if f"{FETCH_PREFIX}_{sector}_coastal_snapped_water_fetch_km" in frame.columns
    ]
    if not columns:
        empty = pd.Series(np.nan, index=frame.index, dtype="float64")
        return empty, empty
    matrix = frame[columns].apply(pd.to_numeric, errors="coerce")
    sector_lookup = {column: SECTOR_IDS[column.split("_sector_", 1)[1].split("_", 1)[0]] for column in columns}
    max_column = matrix.idxmax(axis=1)
    max_value = matrix.max(axis=1)
    sector_id = max_column.map(sector_lookup).astype("float64")
    sector_id[max_value.isna()] = np.nan
    return max_value.astype("float64"), sector_id


def add_directional_features(frame: Any, deps: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    np = deps["np"]
    pd = deps["pd"]
    direction, direction_source = choose_first_numeric(frame, DIRECTION_CANDIDATES, pd)
    wind_speed, wind_speed_source = choose_first_numeric(frame, WIND_SPEED_CANDIDATES, pd)
    direction = direction.where(direction.notna(), np.nan)
    sector_index = pd.Series(sector_index_from_direction(direction, np), index=frame.index, dtype="float64")
    sector_index[direction.isna()] = np.nan
    center_deg = sector_index.map(lambda value: None if pd.isna(value) else float(value) * 45.0).astype("float64")

    upwind = sector_index
    downwind = sector_id_with_offset(sector_index, 4, np, pd)
    cross_left = sector_id_with_offset(sector_index, 2, np, pd)
    cross_right = sector_id_with_offset(sector_index, -2, np, pd)

    out = frame.copy()
    out["features__directional_wind_from_sector_id"] = sector_index
    out["features__directional_wind_from_sector_center_deg"] = center_deg
    out["features__directional_wind_from_sector_delta_deg"] = angular_diff_signed_deg(direction, center_deg, np)

    fetch_template = f"{FETCH_PREFIX}_{{sector}}_coastal_snapped_water_fetch_km"
    direct_fetch_template = f"{FETCH_PREFIX}_{{sector}}_direct_water_fetch_km"
    water_share_template = f"{FETCH_PREFIX}_{{sector}}_water_share"
    land_share_template = f"{FETCH_PREFIX}_{{sector}}_land_share"
    first_land_template = f"{FETCH_PREFIX}_{{sector}}_first_land_distance_km"
    longest_water_template = f"{FETCH_PREFIX}_{{sector}}_longest_water_run_km"
    barrier_p90_template = f"{DEM_PREFIX}_{{sector}}_20km_barrier_p90_m"
    barrier_max_template = f"{DEM_PREFIX}_{{sector}}_20km_barrier_max_m"
    exposure_template = f"{DEM_PREFIX}_{{sector}}_20km_open_exposure_score"
    relief_mean_template = f"{DEM_PREFIX}_{{sector}}_20km_relief_mean"
    relief_p90_template = f"{DEM_PREFIX}_{{sector}}_20km_relief_p90"
    low_or_sea_template = f"{DEM_PREFIX}_{{sector}}_20km_low_or_sea_sample_share"
    nearest_barrier_template = f"{DEM_PREFIX}_{{sector}}_20km_nearest_barrier_distance_km"
    nearest_mountain_template = f"{DEM_PREFIX}_{{sector}}_20km_nearest_mountain_500m_distance_km"

    out["features__directional_upwind_fetch_km"] = values_by_sector(frame, fetch_template, upwind, np, pd)
    out["features__directional_downwind_fetch_km"] = values_by_sector(frame, fetch_template, downwind, np, pd)
    out["features__directional_crosswind_left_fetch_km"] = values_by_sector(frame, fetch_template, cross_left, np, pd)
    out["features__directional_crosswind_right_fetch_km"] = values_by_sector(frame, fetch_template, cross_right, np, pd)
    out["features__directional_upwind_direct_fetch_km"] = values_by_sector(frame, direct_fetch_template, upwind, np, pd)
    out["features__directional_upwind_water_share"] = values_by_sector(frame, water_share_template, upwind, np, pd)
    out["features__directional_upwind_land_share"] = values_by_sector(frame, land_share_template, upwind, np, pd)
    out["features__directional_upwind_first_land_distance_km"] = values_by_sector(frame, first_land_template, upwind, np, pd)
    out["features__directional_upwind_longest_water_run_km"] = values_by_sector(frame, longest_water_template, upwind, np, pd)
    out["features__directional_upwind_barrier_p90_m"] = values_by_sector(frame, barrier_p90_template, upwind, np, pd)
    out["features__directional_upwind_barrier_max_m"] = values_by_sector(frame, barrier_max_template, upwind, np, pd)
    out["features__directional_upwind_open_exposure_score"] = values_by_sector(frame, exposure_template, upwind, np, pd)
    out["features__directional_upwind_relief_mean_m"] = values_by_sector(frame, relief_mean_template, upwind, np, pd)
    out["features__directional_upwind_relief_p90_m"] = values_by_sector(frame, relief_p90_template, upwind, np, pd)
    out["features__directional_upwind_low_or_sea_share"] = values_by_sector(frame, low_or_sea_template, upwind, np, pd)
    out["features__directional_upwind_nearest_barrier_distance_km"] = values_by_sector(frame, nearest_barrier_template, upwind, np, pd)
    out["features__directional_upwind_nearest_mountain_500m_distance_km"] = values_by_sector(frame, nearest_mountain_template, upwind, np, pd)
    out["features__directional_downwind_barrier_p90_m"] = values_by_sector(frame, barrier_p90_template, downwind, np, pd)
    out["features__directional_crosswind_left_barrier_p90_m"] = values_by_sector(frame, barrier_p90_template, cross_left, np, pd)
    out["features__directional_crosswind_right_barrier_p90_m"] = values_by_sector(frame, barrier_p90_template, cross_right, np, pd)

    out["features__directional_upwind_fetch_minus_downwind_fetch_km"] = (
        out["features__directional_upwind_fetch_km"] - out["features__directional_downwind_fetch_km"]
    )
    out["features__directional_crosswind_fetch_asymmetry_km"] = (
        out["features__directional_crosswind_left_fetch_km"] - out["features__directional_crosswind_right_fetch_km"]
    )
    out["features__directional_upwind_fetch_x_baseline_wind_ms"] = (
        out["features__directional_upwind_fetch_km"] * wind_speed
    )
    out["features__directional_upwind_open_exposure_x_baseline_wind_ms"] = (
        out["features__directional_upwind_open_exposure_score"] * wind_speed
    )
    out["features__directional_upwind_barrier_x_baseline_wind_ms"] = (
        out["features__directional_upwind_barrier_p90_m"] * wind_speed
    )
    out["features__directional_lee_blocking_index"] = (
        out["features__directional_upwind_barrier_p90_m"]
        * (1.0 - out["features__directional_upwind_open_exposure_score"])
    )
    out["features__directional_marine_exposure_index"] = (
        out["features__directional_upwind_fetch_km"]
        * out["features__directional_upwind_water_share"]
        * out["features__directional_upwind_open_exposure_score"]
    )

    max_fetch, max_fetch_sector_id = max_fetch_sector(frame, np, pd)
    max_fetch_center = max_fetch_sector_id.map(lambda value: None if pd.isna(value) else float(value) * 45.0).astype("float64")
    wind_from_max_fetch_delta = angular_diff_signed_deg(direction, max_fetch_center, np)
    out["features__directional_max_fetch_km"] = max_fetch
    out["features__directional_max_fetch_sector_id"] = max_fetch_sector_id
    out["features__directional_max_fetch_sector_center_deg"] = max_fetch_center
    out["features__directional_wind_from_max_fetch_delta_deg"] = wind_from_max_fetch_delta
    out["features__directional_sea_breeze_alignment"] = np.cos(np.deg2rad(wind_from_max_fetch_delta))
    out["features__directional_sea_breeze_alignment_x_baseline_wind_ms"] = (
        out["features__directional_sea_breeze_alignment"] * wind_speed
    )

    added = [column for column in out.columns if column.startswith("features__directional_")]
    profile = {
        "added_column_count": len(added),
        "added_columns": added,
        "direction_source_column": direction_source,
        "wind_speed_source_column": wind_speed_source,
        "direction_non_null_ratio": round(float(direction.notna().mean()), 6) if len(direction) else 0.0,
        "coverage": {
            column: {
                "non_null_count": int(out[column].notna().sum()),
                "non_null_ratio": round(float(out[column].notna().mean()), 6) if len(out) else 0.0,
            }
            for column in added
        },
    }
    return out, profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-table-root", type=Path, required=True)
    parser.add_argument("--source-run-id-prefix", required=True)
    parser.add_argument("--output-run-id-prefix", required=True)
    parser.add_argument("--start-month", required=True)
    parser.add_argument("--end-month", required=True)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = import_dependencies()
    pd = deps["pd"]
    pa = deps["pa"]
    pq = deps["pq"]

    summaries = []
    for month in month_range(args.start_month, args.end_month):
        source, target = source_target_paths(args, month)
        if not source.exists():
            summaries.append({"month": month, "status": "missing_source", "source": str(source)})
            continue
        if target.exists() and not args.overwrite:
            summaries.append({"month": month, "status": "exists", "target": str(target)})
            continue
        frame = pd.read_parquet(source)
        augmented, profile = add_directional_features(frame, deps)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(augmented, preserve_index=False), target, compression=args.compression)
        profile.update({
            "month": month,
            "status": "written",
            "generated_at_utc": utc_now(),
            "source": str(source),
            "target": str(target),
            "row_count": int(len(augmented)),
            "source_column_count": int(len(frame.columns)),
            "target_column_count": int(len(augmented.columns)),
        })
        (target.parent / "directional_static_profile.json").write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summaries.append(profile)
        print(json.dumps({
            "month": month,
            "row_count": len(augmented),
            "added_column_count": profile["added_column_count"],
            "target": str(target),
        }, sort_keys=True))

    summary_path = args.training_table_root / f"{args.output_run_id_prefix}_directional_augmentation_summary.json"
    payload = {
        "format": "corsewind.directional_static_augmentation.v1",
        "generated_at_utc": utc_now(),
        "source_run_id_prefix": args.source_run_id_prefix,
        "output_run_id_prefix": args.output_run_id_prefix,
        "start_month": args.start_month,
        "end_month": args.end_month,
        "month_count": len(summaries),
        "written_month_count": sum(1 for item in summaries if item.get("status") == "written"),
        "months": summaries,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "written_month_count": payload["written_month_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
