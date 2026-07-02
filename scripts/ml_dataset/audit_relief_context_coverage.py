#!/usr/bin/env python3
"""Audit relief/mountain context coverage in residual training Parquet shards."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_COLUMNS = (
    "spot_id",
    "issue_time_utc",
    "lead_time_minutes",
    "features__context_global_relief_1_available",
    "features__context_global_relief_1_station_id",
    "features__context_global_relief_1_wind_mean_ms",
    "features__context_global_relief_1_temperature_c",
    "features__context_agg_relief_wind_mean_ms_count",
    "features__context_agg_relief_temperature_c_count",
    "features__thermal_relief_minus_coastal_temperature_c",
    "features__thermal_coastal_minus_relief_wind_ms",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def month_from_path(path: Path, prefix: str) -> str:
    name = path.parent.name
    if name.startswith(prefix + "_"):
        return name.removeprefix(prefix + "_").replace("_", "-")
    return name


def scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def rate(series: Any) -> float:
    if len(series) == 0:
        return 0.0
    return round(float(series.mean()) * 100.0, 3)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    import pandas as pd
    import pyarrow.parquet as pq

    paths = sorted(args.training_table_root.glob(f"{args.run_id_prefix}_20*_??/training_rows.parquet"))
    if args.month:
        wanted = set(args.month)
        paths = [path for path in paths if month_from_path(path, args.run_id_prefix) in wanted]
    if not paths:
        raise SystemExit("No matching training_rows.parquet shards found.")

    rows: list[dict[str, Any]] = []
    total_rows = 0
    critical_spots = set(args.critical_spot)
    for path in paths:
        schema_columns = set(pq.read_schema(path).names)
        columns = [column for column in DEFAULT_COLUMNS if column in schema_columns]
        if "spot_id" not in columns:
            continue
        frame = pd.read_parquet(path, columns=columns)
        if args.lead_minute and "lead_time_minutes" in frame.columns:
            frame = frame[frame["lead_time_minutes"].astype("Int64").isin(args.lead_minute)]
        if critical_spots:
            frame = frame[frame["spot_id"].astype(str).isin(critical_spots)]
        total_rows += int(len(frame))
        month = month_from_path(path, args.run_id_prefix)
        for spot_id, group in frame.groupby("spot_id", dropna=False):
            item: dict[str, Any] = {
                "month": month,
                "spot_id": scalar(spot_id),
                "rows": int(len(group)),
            }
            available_col = "features__context_global_relief_1_available"
            if available_col in group.columns:
                available = group[available_col].fillna(0).astype(float) > 0
                item["global_relief_1_available_rate_pct"] = rate(available)
            station_col = "features__context_global_relief_1_station_id"
            if station_col in group.columns:
                item["global_relief_1_station_ids"] = sorted(
                    str(value) for value in group[station_col].dropna().unique().tolist()
                )
            for column in (
                "features__context_global_relief_1_wind_mean_ms",
                "features__context_global_relief_1_temperature_c",
                "features__context_agg_relief_wind_mean_ms_count",
                "features__context_agg_relief_temperature_c_count",
                "features__thermal_relief_minus_coastal_temperature_c",
                "features__thermal_coastal_minus_relief_wind_ms",
            ):
                if column in group.columns:
                    item[column.removeprefix("features__") + "_nonnull_rate_pct"] = rate(group[column].notna())
            rows.append(item)

    summary: dict[str, Any] = {
        "format": "corsewind.relief_context_coverage_audit.v1",
        "generated_at_utc": utc_now(),
        "training_table_root": str(args.training_table_root),
        "run_id_prefix": args.run_id_prefix,
        "critical_spots": args.critical_spot,
        "lead_minutes": args.lead_minute,
        "month_count": len(paths),
        "row_count": total_rows,
        "by_month_spot": rows,
    }
    if rows:
        critical = pd.DataFrame(rows)
        grouped = critical.groupby("spot_id", dropna=False)
        summary["by_spot"] = []
        for spot_id, group in grouped:
            weighted_rows = float(group["rows"].sum())
            item = {"spot_id": scalar(spot_id), "rows": int(weighted_rows), "month_count": int(group["month"].nunique())}
            for column in group.columns:
                if column.endswith("_rate_pct"):
                    item[column] = round(float((group[column] * group["rows"]).sum() / weighted_rows), 3) if weighted_rows else 0.0
            stations: set[str] = set()
            for values in group.get("global_relief_1_station_ids", []):
                if isinstance(values, list):
                    stations.update(values)
            if stations:
                item["global_relief_1_station_ids"] = sorted(stations)
            summary["by_spot"].append(item)
    return summary


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Relief Context Coverage Audit",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Run prefix: `{result['run_id_prefix']}`",
        f"Months: `{result['month_count']}`",
        f"Rows: `{result['row_count']}`",
        "",
        "## By Spot",
        "",
        "| Spot | Rows | Months | Available | Wind | Temp | Relief temp delta | Relief wind delta | Stations |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in result.get("by_spot", []):
        lines.append(
            f"| `{item.get('spot_id')}` | {item.get('rows')} | {item.get('month_count')} | "
            f"{item.get('global_relief_1_available_rate_pct')}% | "
            f"{item.get('context_global_relief_1_wind_mean_ms_nonnull_rate_pct')}% | "
            f"{item.get('context_global_relief_1_temperature_c_nonnull_rate_pct')}% | "
            f"{item.get('thermal_relief_minus_coastal_temperature_c_nonnull_rate_pct')}% | "
            f"{item.get('thermal_coastal_minus_relief_wind_ms_nonnull_rate_pct')}% | "
            f"`{','.join(item.get('global_relief_1_station_ids', []))}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-table-root", type=Path, required=True)
    parser.add_argument("--run-id-prefix", default="residual_windsup_sst_prev_regime_v1")
    parser.add_argument("--critical-spot", action="append", default=["la_tonnara", "santa_manza", "balistra"])
    parser.add_argument("--lead-minute", type=int, action="append", default=[15, 30, 45, 60])
    parser.add_argument("--month", action="append", default=[])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, result)
    print(json.dumps({
        "month_count": result["month_count"],
        "row_count": result["row_count"],
        "by_spot": result.get("by_spot", []),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
