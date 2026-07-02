#!/usr/bin/env python3
"""Run monthly residual-training shards with optional Parquet export and smoke training."""

from __future__ import annotations

import argparse
import calendar
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "scripts/ml_dataset/run_training_backfill_pipeline.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_month(value: str) -> tuple[int, int]:
    try:
        parsed = datetime.strptime(value, "%Y-%m")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("month must be YYYY-MM") from exc
    return parsed.year, parsed.month


def iter_months(start: tuple[int, int], end: tuple[int, int]):
    year, month = start
    while (year, month) <= end:
        yield year, month
        month += 1
        if month == 13:
            year += 1
            month = 1


def month_bounds(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last_day:02d}"


def run_id(prefix: str, year: int, month: int) -> str:
    return f"{prefix}_{year:04d}_{month:02d}"


def parquet_exists(ml_root: Path, run_id_value: str) -> bool:
    return (ml_root / "training_tables" / run_id_value / "training_rows.parquet").exists()


def build_command(args: argparse.Namespace, year: int, month: int) -> list[str]:
    start_date, end_date = month_bounds(year, month)
    run_id_value = run_id(args.run_id_prefix, year, month)
    cmd = [
        sys.executable,
        str(PIPELINE),
        "--ml-root",
        str(args.ml_root),
        "--registry",
        str(args.registry),
        "--context-registry",
        str(args.context_registry),
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--start-hour-utc",
        str(args.start_hour_utc),
        "--end-hour-utc",
        str(args.end_hour_utc),
        "--chunk-days",
        str(args.chunk_days),
        "--run-id",
        run_id_value,
        "--lead-minutes",
        args.lead_minutes,
        "--open-meteo-model",
        args.open_meteo_model,
        "--command-timeout-sec",
        str(args.command_timeout_sec),
        "--export-parquet",
        "--parquet-batch-size",
        str(args.parquet_batch_size),
        "--parquet-compression",
        args.parquet_compression,
    ]
    if args.cleanup_jsonl_after_parquet:
        cmd.append("--cleanup-jsonl-after-parquet")
    if args.collect_open_meteo:
        cmd.append("--collect-open-meteo")
    if args.collect_open_meteo_offsets:
        cmd.append("--collect-open-meteo-offsets")
    if args.open_meteo_offset_points:
        cmd.extend(["--open-meteo-offset-points", args.open_meteo_offset_points])
    if args.open_meteo_offset_registry:
        cmd.extend(["--open-meteo-offset-registry", str(args.open_meteo_offset_registry)])
    if args.open_meteo_hourly:
        cmd.extend(["--open-meteo-hourly", args.open_meteo_hourly])
    if args.spot_static_features:
        cmd.extend(["--spot-static-features", str(args.spot_static_features)])
    if not args.open_meteo_skip_existing_complete:
        cmd.append("--no-open-meteo-skip-existing-complete")
    if args.train_smoke:
        cmd.extend([
            "--train-models",
            "--train-max-iter",
            str(args.train_max_iter),
            "--train-test-fraction",
            str(args.train_test_fraction),
            "--train-max-rows",
            str(args.train_max_rows),
            "--train-skip-classification",
        ])
    if args.continue_on_error:
        cmd.append("--continue-on-error")
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--context-registry", type=Path, required=True)
    parser.add_argument("--spot-static-features", type=Path)
    parser.add_argument("--start-month", required=True, type=parse_month)
    parser.add_argument("--end-month", required=True, type=parse_month)
    parser.add_argument("--run-id-prefix", default="residual_windsup_spots")
    parser.add_argument("--start-hour-utc", type=int, default=10)
    parser.add_argument("--end-hour-utc", type=int, default=18)
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--lead-minutes", default="15,30,45,60,120,180,360")
    parser.add_argument("--open-meteo-model", default="meteofrance_arome_france")
    parser.add_argument("--open-meteo-hourly", help="Optional comma-separated Open-Meteo hourly variables override.")
    parser.add_argument("--open-meteo-skip-existing-complete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--open-meteo-offset-registry", type=Path)
    parser.add_argument("--open-meteo-offset-points", default="n10:0:10,e10:90:10,s10:180:10,w10:270:10")
    parser.add_argument("--collect-open-meteo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--collect-open-meteo-offsets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-existing-parquet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parquet-batch-size", type=int, default=25000)
    parser.add_argument("--parquet-compression", default="zstd")
    parser.add_argument("--cleanup-jsonl-after-parquet", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-smoke", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-max-iter", type=int, default=40)
    parser.add_argument("--train-test-fraction", type=float, default=0.2)
    parser.add_argument("--train-max-rows", type=int, default=20000)
    parser.add_argument("--command-timeout-sec", type=int, default=7200)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = {
        "format": "corsewind.monthly_training_shards_plan.v1",
        "generated_at_utc": utc_now(),
        "ml_root": str(args.ml_root),
        "start_month": f"{args.start_month[0]:04d}-{args.start_month[1]:02d}",
        "end_month": f"{args.end_month[0]:04d}-{args.end_month[1]:02d}",
        "run_id_prefix": args.run_id_prefix,
        "skip_existing_parquet": args.skip_existing_parquet,
        "train_smoke": args.train_smoke,
        "dry_run": args.dry_run,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    for year, month in iter_months(args.start_month, args.end_month):
        run_id_value = run_id(args.run_id_prefix, year, month)
        if args.skip_existing_parquet and parquet_exists(args.ml_root, run_id_value):
            print(json.dumps({"run_id": run_id_value, "status": "skipped_existing_parquet"}, sort_keys=True))
            continue
        cmd = build_command(args, year, month)
        print(json.dumps({"run_id": run_id_value, "status": "started", "command": cmd}, ensure_ascii=False, sort_keys=True))
        if args.dry_run:
            continue
        completed = subprocess.run(cmd, cwd=ROOT, check=False)
        status = "ok" if completed.returncode == 0 else "error"
        print(json.dumps({"run_id": run_id_value, "status": status, "returncode": completed.returncode}, sort_keys=True))
        if completed.returncode != 0 and not args.continue_on_error:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
