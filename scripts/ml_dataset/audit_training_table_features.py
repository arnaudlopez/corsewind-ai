#!/usr/bin/env python3
"""Audit monthly training-table Parquet shards for required feature columns."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REQUIRED_PATTERNS = [
    "features__context_agg_all_upwind_score_from_target_wind",
    "features__context_nearest_1_bearing_from_spot_deg",
    "features__context_nearest_1_east_offset_km",
    "features__context_nearest_1_north_offset_km",
    "features__previous_run_open_meteo_best_match_day1_wind_speed_10m",
    "features__previous_run_open_meteo_best_match_day2_wind_speed_10m",
    "features__sst_c",
    "features__thermal_air_minus_sst_c",
    "features__thermal_inland_minus_coastal_temperature_c",
    "features__open_meteo_vertical_temperature_lapse_rate_1000_850_c_per_km",
    "features__spot_static_dem_sector",
]


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


def import_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Missing pyarrow; cannot inspect Parquet schemas.") from exc
    return pq


def matching_columns(columns: list[str], pattern: str) -> list[str]:
    return [column for column in columns if pattern in column]


def audit(args: argparse.Namespace) -> dict[str, Any]:
    pq = import_pyarrow()
    shards = []
    required_patterns = args.required_pattern or DEFAULT_REQUIRED_PATTERNS
    for suffix in month_range(args.start_month, args.end_month):
        run_id = f"{args.run_id_prefix}_{suffix}"
        path = args.training_table_root / run_id / "training_rows.parquet"
        item: dict[str, Any] = {
            "run_id": run_id,
            "path": str(path),
            "exists": path.exists(),
            "column_count": 0,
            "required_matches": {},
            "missing_patterns": list(required_patterns),
        }
        if path.exists():
            pf = pq.ParquetFile(path)
            columns = list(pf.schema.names)
            item["column_count"] = len(columns)
            matches = {pattern: matching_columns(columns, pattern) for pattern in required_patterns}
            item["required_matches"] = matches
            item["missing_patterns"] = [pattern for pattern, values in matches.items() if not values]
        shards.append(item)

    existing = [item for item in shards if item["exists"]]
    missing_files = [item["run_id"] for item in shards if not item["exists"]]
    stale = [item["run_id"] for item in existing if item["missing_patterns"]]
    verdict = "pass"
    reasons = []
    if missing_files:
        verdict = "fail"
        reasons.append(f"Missing Parquet shards: {', '.join(missing_files[:10])}")
    if stale:
        verdict = "fail"
        reasons.append(f"Shards missing required feature patterns: {', '.join(stale[:10])}")
    return {
        "format": "corsewind.training_table_feature_audit.v1",
        "generated_at_utc": utc_now(),
        "training_table_root": str(args.training_table_root),
        "run_id_prefix": args.run_id_prefix,
        "start_month": args.start_month,
        "end_month": args.end_month,
        "required_patterns": required_patterns,
        "verdict": verdict,
        "reasons": reasons,
        "shard_count": len(shards),
        "existing_shard_count": len(existing),
        "missing_shard_count": len(missing_files),
        "stale_shard_count": len(stale),
        "shards": shards,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Training Table Feature Audit",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Verdict: `{result['verdict']}`",
        f"Existing shards: `{result['existing_shard_count']}/{result['shard_count']}`",
        f"Stale shards: `{result['stale_shard_count']}`",
        "",
        "## Reasons",
        "",
    ]
    if result["reasons"]:
        lines.extend(f"- {reason}" for reason in result["reasons"])
    else:
        lines.append("- None.")
    lines.extend(["", "## Shards", "", "| Run | Exists | Columns | Missing patterns |", "| --- | ---: | ---: | --- |"])
    for item in result["shards"]:
        missing = ", ".join(item["missing_patterns"])
        lines.append(f"| `{item['run_id']}` | `{item['exists']}` | {item['column_count']} | {missing} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-table-root", type=Path, required=True)
    parser.add_argument("--run-id-prefix", default="residual_windsup_sst_prev")
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument("--end-month", default="2026-06")
    parser.add_argument("--required-pattern", action="append", default=[])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--fail-on-non-pass", action=argparse.BooleanOptionalAction, default=False)
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
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if args.fail_on_non_pass and result["verdict"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
