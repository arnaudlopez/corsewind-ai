#!/usr/bin/env python3
"""Run the chunked feature-store -> residual-training pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_CONTEXT_REGISTRY = ROOT / "configs/ml_context_stations.json"
DEFAULT_SPOT_STATIC_FEATURES = ROOT / "configs/ml_spot_static_features.json"
DEFAULT_LEAD_MINUTES = "60,120,180,360"

COLLECT_OPEN_METEO = ROOT / "scripts/ml_dataset/collect_open_meteo_historical_forecast.py"
GENERATE_OPEN_METEO_OFFSETS = ROOT / "scripts/ml_dataset/generate_open_meteo_offset_registry.py"
BUILD_FEATURE_STORE = ROOT / "scripts/ml_dataset/build_spot_feature_store.py"
BUILD_RESIDUAL_TABLE = ROOT / "scripts/ml_dataset/build_residual_training_table.py"
EVALUATE_RESIDUAL_TABLE = ROOT / "scripts/ml_dataset/evaluate_residual_training_table.py"
TRAIN_RESIDUAL_MODEL = ROOT / "scripts/ml_dataset/train_residual_correction_model.py"
EXPORT_TRAINING_PARQUET = ROOT / "scripts/ml_dataset/export_training_table_parquet.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def parse_int_list(value: str) -> list[int]:
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def date_ranges(start: date, end: date, chunk_days: int) -> list[tuple[date, date]]:
    if end < start:
        raise SystemExit("--end-date must be after or equal to --start-date")
    if chunk_days <= 0:
        raise SystemExit("--chunk-days must be greater than zero")
    ranges = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=chunk_days - 1))
        ranges.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return ranges


def issue_window(chunk_start: date, chunk_end: date, start_hour: int, end_hour: int) -> tuple[datetime, datetime]:
    if not 0 <= start_hour <= 23:
        raise SystemExit("--start-hour-utc must be between 0 and 23")
    if not 0 <= end_hour <= 23:
        raise SystemExit("--end-hour-utc must be between 0 and 23")
    if end_hour < start_hour:
        raise SystemExit("--end-hour-utc must be greater than or equal to --start-hour-utc")
    start = datetime.combine(chunk_start, dt_time(start_hour, tzinfo=timezone.utc))
    end = datetime.combine(chunk_end, dt_time(end_hour, tzinfo=timezone.utc))
    return start, end


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def run_command(cmd: list[str], timeout_sec: int) -> tuple[int, str, str]:
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_sec if timeout_sec > 0 else None,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def run_logged(
    *,
    name: str,
    cmd: list[str],
    log_path: Path,
    timeout_sec: int,
    dry_run: bool,
    continue_on_error: bool,
    metadata: dict[str, Any],
) -> None:
    row = {
        "format": "corsewind.training_backfill_command.v1",
        "name": name,
        "status": "dry_run" if dry_run else "started",
        "started_at_utc": utc_now(),
        "command": cmd,
        **metadata,
    }
    append_jsonl(log_path, row)
    if dry_run:
        return
    try:
        returncode, stdout, stderr = run_command(cmd, timeout_sec)
    except subprocess.TimeoutExpired as exc:
        row.update({
            "finished_at_utc": utc_now(),
            "status": "timeout",
            "returncode": None,
            "stdout_tail": (exc.stdout or "")[-6000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-6000:] if isinstance(exc.stderr, str) else "",
        })
    else:
        row.update({
            "finished_at_utc": utc_now(),
            "status": "ok" if returncode == 0 else "error",
            "returncode": returncode,
            "stdout_tail": stdout[-6000:],
            "stderr_tail": stderr[-6000:],
        })
    append_jsonl(log_path, row)
    if row["status"] != "ok" and not continue_on_error:
        raise SystemExit(f"{name} failed; see {log_path}")


def chunk_slug(start: date, end: date) -> str:
    return f"{start.isoformat()}_{end.isoformat()}"


def command_collect_open_meteo(args: argparse.Namespace, start: date, end: date) -> list[str]:
    cmd = [
        sys.executable,
        str(COLLECT_OPEN_METEO),
        "--registry",
        str(resolve_path(args.registry)),
        "--output-root",
        str(resolve_path(args.ml_root) / "open_meteo/historical_forecast"),
        "--start-date",
        start.isoformat(),
        "--end-date",
        end.isoformat(),
        "--model",
        args.open_meteo_model,
        "--max-days-per-request",
        str(args.open_meteo_max_days_per_request),
        "--request-sleep-sec",
        str(args.open_meteo_request_sleep_sec),
        "--timeout-sec",
        str(args.open_meteo_timeout_sec),
    ]
    if args.open_meteo_hourly:
        cmd.extend(["--hourly", args.open_meteo_hourly])
    if not args.open_meteo_skip_existing_complete:
        cmd.append("--no-skip-existing-complete")
    if args.include_context_spots:
        cmd.append("--include-context-spots")
    return cmd


def command_generate_open_meteo_offsets(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(GENERATE_OPEN_METEO_OFFSETS),
        "--registry",
        str(resolve_path(args.registry)),
        "--output",
        str(resolve_path(args.open_meteo_offset_registry)),
        "--offsets",
        args.open_meteo_offset_points,
    ]


def command_collect_open_meteo_offsets(args: argparse.Namespace, start: date, end: date) -> list[str]:
    cmd = command_collect_open_meteo(args, start, end)
    registry_index = cmd.index("--registry") + 1
    cmd[registry_index] = str(resolve_path(args.open_meteo_offset_registry))
    if "--include-context-spots" not in cmd:
        cmd.append("--include-context-spots")
    return cmd


def command_build_feature_store(
    args: argparse.Namespace,
    start: datetime,
    end: datetime,
    output_root: Path,
    schema_doc: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(BUILD_FEATURE_STORE),
        "--ml-root",
        str(resolve_path(args.ml_root)),
        "--registry",
        str(resolve_path(args.registry)),
        "--context-registry",
        str(resolve_path(args.context_registry)),
        "--output-root",
        str(output_root),
        "--schema-doc",
        str(schema_doc),
        "--start-datetime",
        iso_z(start),
        "--end-datetime",
        iso_z(end),
        "--step-minutes",
        str(args.step_minutes),
        "--target-tolerance-minutes",
        str(args.target_tolerance_minutes),
        "--forecast-valid-tolerance-minutes",
        str(args.forecast_valid_tolerance_minutes),
        "--open-meteo-models",
        args.open_meteo_model,
        "--open-meteo-offset-points",
        args.open_meteo_offset_points,
    ]
    if args.spot_static_features:
        cmd.extend(["--spot-static-features", str(resolve_path(args.spot_static_features))])
    return cmd


def command_build_residual_table(
    args: argparse.Namespace,
    feature_store: Path,
    output_root: Path,
    issue_start: datetime,
    issue_end: datetime,
) -> list[str]:
    return [
        sys.executable,
        str(BUILD_RESIDUAL_TABLE),
        "--feature-store",
        str(feature_store),
        "--output-root",
        str(output_root),
        "--lead-minutes",
        args.lead_minutes,
        "--model-prefix",
        f"model_open_meteo_{args.open_meteo_model}",
        "--issue-start-datetime",
        iso_z(issue_start),
        "--issue-end-datetime",
        iso_z(issue_end),
        "--issue-start-hour-utc",
        str(args.start_hour_utc),
        "--issue-end-hour-utc",
        str(args.end_hour_utc),
    ]


def command_evaluate(training_rows: Path, output_json: Path, output_md: Path) -> list[str]:
    return [
        sys.executable,
        str(EVALUATE_RESIDUAL_TABLE),
        "--training-rows",
        str(training_rows),
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]


def command_export_parquet(training_rows: Path, output_root: Path, batch_size: int, compression: str) -> list[str]:
    return [
        sys.executable,
        str(EXPORT_TRAINING_PARQUET),
        "--training-rows",
        str(training_rows),
        "--output-root",
        str(output_root),
        "--batch-size",
        str(batch_size),
        "--compression",
        compression,
    ]


def cleanup_jsonl_after_parquet(combined_training_root: Path, chunk_training_roots: list[Path], log_path: Path, run_id: str) -> dict[str, Any]:
    parquet_path = combined_training_root / "training_rows.parquet"
    if not parquet_path.exists():
        return {
            "format": "corsewind.training_backfill_cleanup.v1",
            "run_id": run_id,
            "status": "skipped_missing_parquet",
            "parquet_path": str(parquet_path),
        }
    deleted: list[dict[str, Any]] = []
    for path in [combined_training_root / "training_rows.jsonl", *(root / "training_rows.jsonl" for root in chunk_training_roots)]:
        if not path.exists():
            continue
        size_bytes = path.stat().st_size
        path.unlink()
        deleted.append({"path": str(path), "size_bytes": size_bytes})
    result = {
        "format": "corsewind.training_backfill_cleanup.v1",
        "generated_at_utc": utc_now(),
        "run_id": run_id,
        "status": "ok",
        "parquet_path": str(parquet_path),
        "deleted_file_count": len(deleted),
        "deleted_bytes": sum(item["size_bytes"] for item in deleted),
        "deleted": deleted,
    }
    append_jsonl(log_path, result)
    return result


def command_train(training_rows: Path, output_root: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(TRAIN_RESIDUAL_MODEL),
        "--training-rows",
        str(training_rows),
        "--output-root",
        str(output_root),
        "--max-iter",
        str(args.train_max_iter),
        "--test-fraction",
        str(args.train_test_fraction),
    ]
    if args.train_max_rows is not None:
        cmd.extend(["--max-rows", str(args.train_max_rows)])
    if args.train_skip_classification:
        cmd.append("--skip-classification")
    for target in args.train_only_target:
        cmd.extend(["--only-target", target])
    return cmd


def collect_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns: dict[str, dict[str, Any]] = {}
    for row in rows:
        for group in ("features", "baselines", "labels"):
            values = row.get(group, {}) if isinstance(row.get(group), dict) else {}
            for key, value in values.items():
                name = f"{group}.{key}"
                item = columns.setdefault(name, {"column": name, "group": group, "non_null_count": 0, "types": set()})
                if value is not None:
                    item["non_null_count"] += 1
                    item["types"].add(type(value).__name__)
    return [
        {
            "column": name,
            "group": item["group"],
            "non_null_count": item["non_null_count"],
            "types": "|".join(sorted(item["types"])) if item["types"] else "",
        }
        for name, item in sorted(columns.items())
    ]


def update_column_stats(row: dict[str, Any], columns: dict[str, dict[str, Any]]) -> None:
    for group in ("features", "baselines", "labels"):
        values = row.get(group, {}) if isinstance(row.get(group), dict) else {}
        for key, value in values.items():
            name = f"{group}.{key}"
            item = columns.setdefault(name, {"column": name, "group": group, "non_null_count": 0, "types": set()})
            if value is not None:
                item["non_null_count"] += 1
                item["types"].add(type(value).__name__)


def column_stats_rows(columns: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "column": name,
            "group": item["group"],
            "non_null_count": item["non_null_count"],
            "types": "|".join(sorted(item["types"])) if item["types"] else "",
        }
        for name, item in sorted(columns.items())
    ]


def combine_training_tables(chunk_roots: list[Path], output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    seen_keys: set[tuple[str, str, str, int]] = set()
    chunk_profiles = []
    columns: dict[str, dict[str, Any]] = {}
    rows_by_lead: Counter[int | None] = Counter()
    rows_by_spot: Counter[str] = Counter()
    row_count = 0
    duplicate_row_count = 0
    first_issue_time_utc = None
    last_issue_time_utc = None

    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "training_rows.jsonl"
    tmp_rows_path = output_root / "training_rows.jsonl.tmp"
    if tmp_rows_path.exists():
        tmp_rows_path.unlink()

    with tmp_rows_path.open("w", encoding="utf-8") as output:
        for chunk_root in chunk_roots:
            profile_path = chunk_root / "training_profile.json"
            if profile_path.exists():
                chunk_profiles.append(read_json(profile_path))
            for row in iter_jsonl(chunk_root / "training_rows.jsonl") or []:
                key = (
                    str(row.get("spot_id")),
                    str(row.get("issue_time_utc")),
                    str(row.get("target_time_utc")),
                    int(row.get("lead_time_minutes") or 0),
                )
                if key in seen_keys:
                    duplicate_row_count += 1
                    continue
                seen_keys.add(key)
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                row_count += 1
                issue_time = row.get("issue_time_utc")
                if first_issue_time_utc is None:
                    first_issue_time_utc = issue_time
                last_issue_time_utc = issue_time
                rows_by_lead.update([row.get("lead_time_minutes")])
                rows_by_spot.update([str(row.get("spot_id"))])
                update_column_stats(row, columns)
    tmp_rows_path.replace(rows_path)

    profile = {
        "format": "corsewind.combined_residual_correction_training_table.v1",
        "generated_at_utc": utc_now(),
        "run_id": args.run_id,
        "source_chunk_count": len(chunk_roots),
        "chunk_training_row_refs": sum(item.get("training_row_count", 0) for item in chunk_profiles),
        "training_row_count": row_count,
        "duplicate_row_count": duplicate_row_count,
        "training_rows_by_lead": dict(sorted(rows_by_lead.items())),
        "training_rows_by_spot": dict(sorted(rows_by_spot.items())),
        "first_issue_time_utc": first_issue_time_utc,
        "last_issue_time_utc": last_issue_time_utc,
        "dedupe_strategy": "first_row_by_spot_issue_target_lead_streaming",
        "settings": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "start_hour_utc": args.start_hour_utc,
            "end_hour_utc": args.end_hour_utc,
            "lead_minutes": parse_int_list(args.lead_minutes),
            "open_meteo_model": args.open_meteo_model,
        },
    }
    (output_root / "training_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with (output_root / "training_columns.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["column", "group", "non_null_count", "types"])
        writer.writeheader()
        writer.writerows(column_stats_rows(columns))
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-root", type=Path, default=DEFAULT_ML_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--context-registry", type=Path, default=DEFAULT_CONTEXT_REGISTRY)
    parser.add_argument("--spot-static-features", type=Path, default=DEFAULT_SPOT_STATIC_FEATURES)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--start-hour-utc", type=int, default=10)
    parser.add_argument("--end-hour-utc", type=int, default=18)
    parser.add_argument("--chunk-days", type=int, default=31)
    parser.add_argument("--lead-minutes", default=DEFAULT_LEAD_MINUTES)
    parser.add_argument("--step-minutes", type=int, default=15)
    parser.add_argument("--target-tolerance-minutes", type=float, default=8)
    parser.add_argument("--forecast-valid-tolerance-minutes", type=float, default=31)
    parser.add_argument("--run-id")
    parser.add_argument("--open-meteo-model", default="meteofrance_arome_france")
    parser.add_argument("--collect-open-meteo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--collect-open-meteo-offsets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-context-spots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--open-meteo-offset-registry", type=Path)
    parser.add_argument("--open-meteo-offset-points", default="n10:0:10,e10:90:10,s10:180:10,w10:270:10")
    parser.add_argument("--open-meteo-max-days-per-request", type=int, default=31)
    parser.add_argument("--open-meteo-request-sleep-sec", type=float, default=0.2)
    parser.add_argument("--open-meteo-timeout-sec", type=int, default=60)
    parser.add_argument("--open-meteo-hourly", help="Optional comma-separated Open-Meteo hourly variables override.")
    parser.add_argument("--open-meteo-skip-existing-complete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--command-timeout-sec", type=int, default=3600)
    parser.add_argument("--export-parquet", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--parquet-batch-size", type=int, default=25000)
    parser.add_argument("--parquet-compression", default="zstd")
    parser.add_argument("--cleanup-jsonl-after-parquet", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-models", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-max-iter", type=int, default=150)
    parser.add_argument("--train-test-fraction", type=float, default=0.2)
    parser.add_argument("--train-max-rows", type=int)
    parser.add_argument("--train-only-target", action="append", default=[])
    parser.add_argument("--train-skip-classification", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ml_root = resolve_path(args.ml_root)
    if args.open_meteo_offset_registry is None:
        args.open_meteo_offset_registry = ml_root / "open_meteo/offset_spots.json"
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    leads = parse_int_list(args.lead_minutes)
    max_lead = max(leads) if leads else 0
    if not args.run_id:
        args.run_id = f"residual_backfill_{start.isoformat()}_{end.isoformat()}_{args.start_hour_utc:02d}-{args.end_hour_utc:02d}z"

    run_root = ml_root / "training_runs" / args.run_id
    feature_root = ml_root / "feature_store" / args.run_id
    chunk_training_root = run_root / "chunks"
    combined_training_root = ml_root / "training_tables" / args.run_id
    log_path = run_root / "commands.jsonl"
    chunks = date_ranges(start, end, args.chunk_days)
    chunk_training_roots = []

    plan = {
        "format": "corsewind.training_backfill_pipeline_plan.v1",
        "generated_at_utc": utc_now(),
        "run_id": args.run_id,
        "ml_root": str(ml_root),
        "feature_root": str(feature_root),
        "chunk_training_root": str(chunk_training_root),
        "combined_training_root": str(combined_training_root),
        "log_path": str(log_path),
        "chunk_count": len(chunks),
        "collect_open_meteo": args.collect_open_meteo,
        "collect_open_meteo_offsets": args.collect_open_meteo_offsets,
        "open_meteo_offset_registry": str(resolve_path(args.open_meteo_offset_registry)),
        "open_meteo_offset_points": args.open_meteo_offset_points,
        "spot_static_features": str(resolve_path(args.spot_static_features)) if args.spot_static_features else None,
        "export_parquet": args.export_parquet,
        "train_models": args.train_models,
        "lead_minutes": leads,
        "dry_run": args.dry_run,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    append_jsonl(log_path, plan)

    if args.collect_open_meteo_offsets:
        run_logged(
            name="generate_open_meteo_offset_registry",
            cmd=command_generate_open_meteo_offsets(args),
            log_path=log_path,
            timeout_sec=args.command_timeout_sec,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
            metadata={"run_id": args.run_id},
        )

    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        issue_start, issue_end = issue_window(chunk_start, chunk_end, args.start_hour_utc, args.end_hour_utc)
        feature_end = issue_end + timedelta(minutes=max_lead)
        feature_output = feature_root / f"chunk={chunk_slug(chunk_start, chunk_end)}"
        chunk_training = chunk_training_root / f"chunk={chunk_slug(chunk_start, chunk_end)}"
        chunk_training_roots.append(chunk_training)
        metadata = {
            "run_id": args.run_id,
            "chunk_index": index,
            "chunk_count": len(chunks),
            "chunk_start_date": chunk_start.isoformat(),
            "chunk_end_date": chunk_end.isoformat(),
            "issue_start_utc": iso_z(issue_start),
            "issue_end_utc": iso_z(issue_end),
            "feature_end_utc": iso_z(feature_end),
        }
        if args.collect_open_meteo:
            run_logged(
                name="collect_open_meteo_historical_forecast",
                cmd=command_collect_open_meteo(args, chunk_start, feature_end.date()),
                log_path=log_path,
                timeout_sec=args.command_timeout_sec,
                dry_run=args.dry_run,
                continue_on_error=args.continue_on_error,
                metadata=metadata,
            )
        if args.collect_open_meteo_offsets:
            run_logged(
                name="collect_open_meteo_historical_forecast_offsets",
                cmd=command_collect_open_meteo_offsets(args, chunk_start, feature_end.date()),
                log_path=log_path,
                timeout_sec=args.command_timeout_sec,
                dry_run=args.dry_run,
                continue_on_error=args.continue_on_error,
                metadata=metadata,
            )
        run_logged(
            name="build_spot_feature_store",
            cmd=command_build_feature_store(
                args,
                issue_start,
                feature_end,
                feature_output,
                feature_output / "feature_store_schema.md",
            ),
            log_path=log_path,
            timeout_sec=args.command_timeout_sec,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
            metadata=metadata,
        )
        run_logged(
            name="build_residual_training_table",
            cmd=command_build_residual_table(
                args,
                feature_output / "spot_forecast_15min.jsonl",
                chunk_training,
                issue_start,
                issue_end,
            ),
            log_path=log_path,
            timeout_sec=args.command_timeout_sec,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
            metadata=metadata,
        )
        if args.sleep_sec:
            time.sleep(args.sleep_sec)

    if not args.dry_run:
        profile = combine_training_tables(chunk_training_roots, combined_training_root, args)
        run_logged(
            name="evaluate_combined_residual_training_table",
            cmd=command_evaluate(
                combined_training_root / "training_rows.jsonl",
                combined_training_root / "evaluation.json",
                combined_training_root / "evaluation.md",
            ),
            log_path=log_path,
            timeout_sec=args.command_timeout_sec,
            dry_run=False,
            continue_on_error=args.continue_on_error,
            metadata={"run_id": args.run_id, "training_row_count": profile["training_row_count"]},
        )
        if args.export_parquet:
            run_logged(
                name="export_training_table_parquet",
                cmd=command_export_parquet(
                    combined_training_root / "training_rows.jsonl",
                    combined_training_root,
                    args.parquet_batch_size,
                    args.parquet_compression,
                ),
                log_path=log_path,
                timeout_sec=args.command_timeout_sec,
                dry_run=False,
                continue_on_error=args.continue_on_error,
                metadata={"run_id": args.run_id, "training_row_count": profile["training_row_count"]},
            )
            if args.cleanup_jsonl_after_parquet and (combined_training_root / "training_rows.parquet").exists():
                cleanup = cleanup_jsonl_after_parquet(combined_training_root, chunk_training_roots, log_path, args.run_id)
                print(json.dumps(cleanup, ensure_ascii=False, indent=2, sort_keys=True))
        if args.train_models:
            run_logged(
                name="train_residual_correction_model",
                cmd=command_train(
                    combined_training_root / "training_rows.jsonl",
                    ml_root / "models" / args.run_id,
                    args,
                ),
                log_path=log_path,
                timeout_sec=args.command_timeout_sec,
                dry_run=False,
                continue_on_error=args.continue_on_error,
                metadata={"run_id": args.run_id, "training_row_count": profile["training_row_count"]},
            )
        print(json.dumps({
            "run_id": args.run_id,
            "training_row_count": profile["training_row_count"],
            "training_rows_by_lead": profile["training_rows_by_lead"],
            "combined_training_root": str(combined_training_root),
            "log_path": str(log_path),
        }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
