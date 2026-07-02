#!/usr/bin/env python3
"""Audit which training-table features a sequence calibrator will select."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from train_sequence_calibrator import DEFAULT_TRAINING_FEATURE_PREFIXES, selected_training_columns


DEFAULT_REQUIRED_PATTERNS = [
    "features__model_error_now_",
    "features__previous_run_open_meteo_best_match_day1_wind_speed_10m",
    "features__previous_run_open_meteo_best_match_day2_wind_speed_10m",
    "features__sst_c",
    "features__eumetsat_",
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


def load_schema(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required for feature-selection audit.") from exc
    return list(pq.ParquetFile(path).schema_arrow.names)


def audit_path(path: Path, prefixes: list[str], required_patterns: list[str], max_features: int) -> dict[str, Any]:
    columns = load_schema(path)
    selected = selected_training_columns(columns, prefixes, max_features)
    return {
        "path": str(path),
        "exists": True,
        "total_columns": len(columns),
        "selected_count": len(selected),
        "selected_by_prefix": {
            prefix: sum(column.startswith(prefix) for column in selected)
            for prefix in prefixes
        },
        "available_by_prefix": {
            prefix: sum(column.startswith(prefix) for column in columns)
            for prefix in prefixes
        },
        "required_selected": {
            pattern: any(pattern in column for column in selected)
            for pattern in required_patterns
        },
        "required_available": {
            pattern: any(pattern in column for column in columns)
            for pattern in required_patterns
        },
        "last_selected": selected[-10:],
        "first_unselected_after_limit": [
            column
            for column in selected_training_columns(columns, prefixes, len(columns))
            if column not in set(selected)
        ][:10],
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Calibrator Feature Selection Audit",
        "",
        f"Generated: `{summary['generated_at_utc']}`",
        f"Max features: `{summary['max_training_features']}`",
        f"Verdict: `{summary['verdict']}`",
        "",
        "| Shard | Selected | Missing required |",
        "| --- | ---: | --- |",
    ]
    for shard in summary["shards"]:
        missing = [
            pattern
            for pattern, selected in shard.get("required_selected", {}).items()
            if not selected
        ]
        lines.append(
            f"| `{Path(shard['path']).parent.name}` | {shard.get('selected_count')} | "
            f"{', '.join(f'`{item}`' for item in missing) if missing else '-'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-table-root", type=Path, required=True)
    parser.add_argument("--run-id-prefix", required=True)
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument("--end-month", default="2026-06")
    parser.add_argument("--max-training-features", type=int, default=900)
    parser.add_argument("--training-feature-prefix", action="append", default=[])
    parser.add_argument("--required-pattern", action="append", default=[])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--fail-on-missing-required", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefixes = args.training_feature_prefix or DEFAULT_TRAINING_FEATURE_PREFIXES
    required_patterns = args.required_pattern or DEFAULT_REQUIRED_PATTERNS
    shards = []
    missing_paths = []
    for suffix in month_range(args.start_month, args.end_month):
        path = args.training_table_root / f"{args.run_id_prefix}_{suffix}" / "training_rows.parquet"
        if not path.exists():
            missing_paths.append(str(path))
            continue
        shards.append(audit_path(path, prefixes, required_patterns, args.max_training_features))
    failures = []
    for shard in shards:
        missing_selected = [
            pattern
            for pattern, selected in shard["required_selected"].items()
            if not selected
        ]
        if missing_selected:
            failures.append({"path": shard["path"], "missing_selected": missing_selected})
    summary = {
        "format": "corsewind.calibrator_feature_selection_audit.v1",
        "generated_at_utc": utc_now(),
        "training_table_root": str(args.training_table_root),
        "run_id_prefix": args.run_id_prefix,
        "start_month": args.start_month,
        "end_month": args.end_month,
        "max_training_features": args.max_training_features,
        "prefixes": prefixes,
        "required_patterns": required_patterns,
        "existing_shard_count": len(shards),
        "missing_shard_count": len(missing_paths),
        "missing_paths": missing_paths,
        "failure_count": len(failures),
        "failures": failures,
        "verdict": "pass" if not failures else "fail",
        "shards": shards,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.fail_on_missing_required and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
