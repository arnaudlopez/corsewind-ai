#!/usr/bin/env python3
"""Audit training-table Parquet shards for temporal leakage and alignment risks."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = [
    "spot_id",
    "issue_time_utc",
    "target_time_utc",
    "lead_time_minutes",
    "labels__target_wind_mean_ms",
    "labels__target_observation_timestamp_utc",
    "labels__target_observation_distance_minutes",
]

SUSPICIOUS_FEATURE_TOKENS = (
    "actual_wind",
    "actual_gust",
    "ground_truth",
    "truth",
    "future",
    "target_observation",
    "raw_error",
    "corrected_error",
    "abs_error",
)


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


def import_deps() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Missing pandas/pyarrow dependencies for temporal integrity audit.") from exc
    return {"np": np, "pd": pd, "pq": pq}


def finite_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def stats_for_numeric(series: Any, pd: Any) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"nonnull": 0}
    return {
        "nonnull": int(len(values)),
        "min": round(float(values.min()), 6),
        "p50": round(float(values.quantile(0.50)), 6),
        "p95": round(float(values.quantile(0.95)), 6),
        "max": round(float(values.max()), 6),
    }


def suspicious_feature_columns(columns: list[str]) -> list[str]:
    out = []
    for column in columns:
        lowered = column.lower()
        if not (lowered.startswith("features__") or lowered.startswith("baselines__")):
            continue
        if any(token in lowered for token in SUSPICIOUS_FEATURE_TOKENS):
            out.append(column)
    return sorted(out)


def age_columns(columns: list[str]) -> list[str]:
    return sorted(
        column
        for column in columns
        if column.startswith("features__")
        and (column.endswith("_age_minutes") or column.endswith("_nominal_forecast_age_hours"))
    )


def audit_shard(path: Path, args: argparse.Namespace, deps: dict[str, Any]) -> dict[str, Any]:
    pd = deps["pd"]
    pq = deps["pq"]
    item: dict[str, Any] = {
        "run_id": path.parent.name,
        "path": str(path),
        "exists": path.exists(),
        "verdict": "missing",
        "row_count": 0,
        "warnings": [],
        "failures": [],
    }
    if not path.exists():
        item["failures"].append("missing_parquet")
        return item

    pf = pq.ParquetFile(path)
    columns = list(pf.schema_arrow.names)
    item["row_count"] = int(pf.metadata.num_rows)
    item["column_count"] = len(columns)
    missing_required = [column for column in REQUIRED_COLUMNS if column not in columns]
    item["missing_required_columns"] = missing_required
    if missing_required:
        item["failures"].append(f"missing_required_columns:{','.join(missing_required)}")

    suspicious = suspicious_feature_columns(columns)
    item["suspicious_feature_columns"] = suspicious
    if suspicious:
        item["warnings"].append(f"suspicious_feature_columns:{len(suspicious)}")

    read_columns = sorted(set(REQUIRED_COLUMNS).intersection(columns) | set(age_columns(columns)) | {"station_id"})
    if not read_columns:
        item["verdict"] = "fail"
        return item

    frame = pd.read_parquet(path, columns=read_columns)
    if "station_id" in frame.columns:
        key_columns = ["spot_id", "issue_time_utc", "target_time_utc", "lead_time_minutes"]
    else:
        key_columns = ["spot_id", "issue_time_utc", "target_time_utc", "lead_time_minutes"]
    if all(column in frame.columns for column in key_columns):
        duplicates = int(frame.duplicated(key_columns).sum())
        item["duplicate_key_rows"] = duplicates
        if duplicates:
            item["warnings"].append(f"duplicate_key_rows:{duplicates}")

    if {"issue_time_utc", "target_time_utc", "lead_time_minutes"}.issubset(frame.columns):
        issue = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
        target = pd.to_datetime(frame["target_time_utc"], utc=True, errors="coerce")
        lead = pd.to_numeric(frame["lead_time_minutes"], errors="coerce")
        expected_target = issue + pd.to_timedelta(lead, unit="m")
        mismatch_seconds = (target - expected_target).dt.total_seconds().abs()
        invalid_times = int(issue.isna().sum() + target.isna().sum() + lead.isna().sum())
        lead_mismatch = int((mismatch_seconds > args.max_lead_mismatch_seconds).fillna(False).sum())
        target_before_issue = int((target < issue).fillna(False).sum())
        item["time_alignment"] = {
            "invalid_time_or_lead_values": invalid_times,
            "lead_mismatch_rows": lead_mismatch,
            "target_before_issue_rows": target_before_issue,
            "max_lead_mismatch_seconds": None if mismatch_seconds.dropna().empty else round(float(mismatch_seconds.max()), 6),
        }
        if invalid_times:
            item["failures"].append(f"invalid_time_or_lead_values:{invalid_times}")
        if lead_mismatch:
            item["failures"].append(f"lead_mismatch_rows:{lead_mismatch}")
        if target_before_issue:
            item["failures"].append(f"target_before_issue_rows:{target_before_issue}")

    if {"target_time_utc", "labels__target_observation_timestamp_utc"}.issubset(frame.columns):
        target = pd.to_datetime(frame["target_time_utc"], utc=True, errors="coerce")
        obs_ts = pd.to_datetime(frame["labels__target_observation_timestamp_utc"], utc=True, errors="coerce")
        distance_minutes = (obs_ts - target).dt.total_seconds().abs() / 60.0
        max_observed_distance = None if distance_minutes.dropna().empty else round(float(distance_minutes.max()), 6)
        far = int((distance_minutes > args.max_target_observation_distance_minutes).fillna(False).sum())
        missing_obs_ts = int(obs_ts.isna().sum())
        item["target_observation_alignment"] = {
            "missing_observation_timestamp_rows": missing_obs_ts,
            "far_observation_rows": far,
            "max_observed_distance_minutes": max_observed_distance,
        }
        if missing_obs_ts:
            item["failures"].append(f"missing_target_observation_timestamp:{missing_obs_ts}")
        if far:
            item["warnings"].append(f"far_target_observation_rows:{far}")
    if "labels__target_observation_distance_minutes" in frame.columns:
        stats = stats_for_numeric(frame["labels__target_observation_distance_minutes"], pd)
        item["target_observation_distance_minutes"] = stats
        max_distance = finite_float(stats.get("max"))
        if max_distance is not None and max_distance > args.max_target_observation_distance_minutes:
            item["warnings"].append(f"stored_target_observation_distance_above_limit:{max_distance}")

    age_summary = []
    negative_age_columns = []
    stale_age_columns = []
    for column in age_columns(columns):
        stats = stats_for_numeric(frame[column], pd)
        stats["column"] = column
        age_summary.append(stats)
        min_value = finite_float(stats.get("min"))
        max_value = finite_float(stats.get("max"))
        if min_value is not None and min_value < -args.age_negative_tolerance:
            negative_age_columns.append({"column": column, "min": min_value})
        if column.endswith("_age_minutes") and max_value is not None and max_value > args.warn_age_minutes:
            stale_age_columns.append({"column": column, "max": max_value})
    item["age_column_count"] = len(age_summary)
    item["negative_age_columns"] = negative_age_columns
    item["stale_age_columns"] = stale_age_columns[: args.limit]
    item["age_columns_worst"] = sorted(
        age_summary,
        key=lambda row: (finite_float(row.get("max")) is None, finite_float(row.get("max")) or -1),
        reverse=True,
    )[: args.limit]
    if negative_age_columns:
        item["failures"].append(f"negative_age_columns:{len(negative_age_columns)}")
    if stale_age_columns:
        item["warnings"].append(f"stale_age_columns:{len(stale_age_columns)}")

    item["verdict"] = "fail" if item["failures"] else ("warn" if item["warnings"] else "pass")
    return item


def audit(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_deps()
    shards = []
    for suffix in month_range(args.start_month, args.end_month):
        path = args.training_table_root / f"{args.run_id_prefix}_{suffix}" / "training_rows.parquet"
        shards.append(audit_shard(path, args, deps))

    verdict_counts = Counter(item["verdict"] for item in shards)
    failures = [item for item in shards if item["verdict"] == "fail"]
    warnings = [item for item in shards if item["verdict"] == "warn"]
    missing = [item for item in shards if not item["exists"]]
    verdict = "fail" if failures else ("warn" if warnings or missing else "pass")
    reasons = []
    if missing:
        reasons.append(f"Missing shards: {len(missing)}")
    if failures:
        reasons.append(f"Failing shards: {len(failures)}")
    if warnings:
        reasons.append(f"Warning shards: {len(warnings)}")
    if not reasons:
        reasons.append("No temporal integrity issue detected.")
    return {
        "format": "corsewind.training_table_temporal_integrity.v1",
        "generated_at_utc": utc_now(),
        "training_table_root": str(args.training_table_root),
        "run_id_prefix": args.run_id_prefix,
        "start_month": args.start_month,
        "end_month": args.end_month,
        "verdict": verdict,
        "reasons": reasons,
        "shard_count": len(shards),
        "existing_shard_count": sum(1 for item in shards if item["exists"]),
        "total_rows": sum(int(item.get("row_count") or 0) for item in shards),
        "verdict_counts": dict(verdict_counts),
        "parameters": {
            "max_lead_mismatch_seconds": args.max_lead_mismatch_seconds,
            "max_target_observation_distance_minutes": args.max_target_observation_distance_minutes,
            "warn_age_minutes": args.warn_age_minutes,
            "age_negative_tolerance": args.age_negative_tolerance,
        },
        "shards": shards,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Training Table Temporal Integrity Audit",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Verdict: `{result['verdict']}`",
        f"Rows: `{result['total_rows']}`",
        f"Existing shards: `{result['existing_shard_count']}/{result['shard_count']}`",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in result["reasons"])
    lines.extend([
        "",
        "## Shards",
        "",
        "| Run | Verdict | Rows | Failures | Warnings | Negative ages | Stale ages | Duplicates |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
    ])
    for item in result["shards"]:
        lines.append(
            f"| `{item['run_id']}` | `{item['verdict']}` | {item.get('row_count', 0)} | "
            f"{'; '.join(item.get('failures') or [])} | { '; '.join(item.get('warnings') or [])} | "
            f"{len(item.get('negative_age_columns') or [])} | {len(item.get('stale_age_columns') or [])} | "
            f"{item.get('duplicate_key_rows')} |"
        )
    worst_negative = [
        (item["run_id"], row)
        for item in result["shards"]
        for row in (item.get("negative_age_columns") or [])
    ]
    if worst_negative:
        lines.extend(["", "## Negative Age Columns", "", "| Run | Column | Min |", "| --- | --- | ---: |"])
        for run_id, row in worst_negative[:30]:
            lines.append(f"| `{run_id}` | `{row['column']}` | {row.get('min')} |")
    suspicious = [
        (item["run_id"], column)
        for item in result["shards"]
        for column in (item.get("suspicious_feature_columns") or [])
    ]
    if suspicious:
        lines.extend(["", "## Suspicious Feature Names", "", "| Run | Column |", "| --- | --- |"])
        for run_id, column in suspicious[:50]:
            lines.append(f"| `{run_id}` | `{column}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-table-root", type=Path, required=True)
    parser.add_argument("--run-id-prefix", default="residual_windsup_sst_prev_phys_v1")
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument("--end-month", default="2026-06")
    parser.add_argument("--max-lead-mismatch-seconds", type=float, default=1.0)
    parser.add_argument("--max-target-observation-distance-minutes", type=float, default=20.0)
    parser.add_argument("--warn-age-minutes", type=float, default=360.0)
    parser.add_argument("--age-negative-tolerance", type=float, default=1e-6)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--fail-on-fail", action=argparse.BooleanOptionalAction, default=False)
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
        "verdict": result["verdict"],
        "existing_shards": result["existing_shard_count"],
        "shards": result["shard_count"],
        "rows": result["total_rows"],
        "reasons": result["reasons"],
    }, indent=2, sort_keys=True))
    if args.fail_on_fail and result["verdict"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
