#!/usr/bin/env python3
"""Run chunked EUMETSAT spot-product backfills over selected UTC hours."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_LOG_ROOT = DEFAULT_ML_ROOT / "source_inventories/eumetsat_backfill_runs"
COLLECTOR = ROOT / "scripts/ml_dataset/collect_eumetsat_spot_product.py"
PRODUCTS = ("cloud_type", "land_surface_temperature", "global_instability_indices")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise SystemExit("--end-date must be after or equal to --start-date")
    days = []
    cursor = start
    while cursor <= end:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def windows_for_day(day: date, start_hour: int, end_hour: int, window_hours: float) -> list[tuple[datetime, datetime]]:
    if not 0 <= start_hour <= 23:
        raise SystemExit("--start-hour-utc must be between 0 and 23")
    if not 1 <= end_hour <= 24:
        raise SystemExit("--end-hour-utc must be between 1 and 24")
    if end_hour <= start_hour:
        raise SystemExit("--end-hour-utc must be greater than --start-hour-utc")
    if window_hours <= 0:
        raise SystemExit("--window-hours must be greater than zero")
    cursor = datetime.combine(day, dt_time(start_hour, tzinfo=timezone.utc))
    end = datetime.combine(day, dt_time(0, tzinfo=timezone.utc)) + timedelta(hours=end_hour)
    windows = []
    while cursor < end:
        next_cursor = min(cursor + timedelta(hours=window_hours), end)
        windows.append((cursor, next_cursor))
        cursor = next_cursor
    return windows


def product_keys(args: argparse.Namespace) -> list[str]:
    keys = args.product or list(PRODUCTS)
    unknown = sorted(set(keys) - set(PRODUCTS))
    if unknown:
        raise SystemExit(f"Unknown product key(s): {', '.join(unknown)}")
    return keys


def command_for(product: str, start: datetime, end: datetime, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(COLLECTOR),
        "--product",
        product,
        "--start-datetime",
        iso_z(start),
        "--end-datetime",
        iso_z(end),
        "--bbox",
        args.bbox,
        "--max-products",
        str(args.max_products),
        "--radius-cells",
        str(args.radius_cells),
    ]
    if args.delete_raw_after_sample:
        cmd.append("--delete-raw-after-sample")
    if args.include_context_spots:
        cmd.append("--include-context-spots")
    for spot_id in args.spot_id:
        cmd.extend(["--spot-id", spot_id])
    return cmd


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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def output_log_path(args: argparse.Namespace) -> Path:
    if args.output_log:
        return resolve_path(args.output_log)
    stamp = utc_now().replace(":", "").replace("-", "").replace(".", "")
    return DEFAULT_LOG_ROOT / f"eumetsat_backfill_{stamp}.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", action="append", choices=PRODUCTS, default=[])
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--start-hour-utc", type=int, default=10)
    parser.add_argument("--end-hour-utc", type=int, default=18)
    parser.add_argument("--window-hours", type=float, default=2)
    parser.add_argument("--bbox", default="7.5,41.0,10.2,43.3")
    parser.add_argument("--max-products", type=int, default=0)
    parser.add_argument("--radius-cells", type=int, default=3)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--request-sleep-sec", type=float, default=0.0)
    parser.add_argument("--spot-id", action="append", default=[])
    parser.add_argument("--include-context-spots", action="store_true")
    parser.add_argument("--delete-raw-after-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-log", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    products = product_keys(args)
    days = date_range(parse_date(args.start_date), parse_date(args.end_date))
    output_log = output_log_path(args)
    planned = [
        (product, start, end)
        for day in days
        for start, end in windows_for_day(day, args.start_hour_utc, args.end_hour_utc, args.window_hours)
        for product in products
    ]
    print(
        json.dumps(
            {
                "generated_at_utc": utc_now(),
                "mode": "dry_run" if args.dry_run else "execute",
                "output_log": str(output_log),
                "command_count": len(planned),
                "products": products,
                "day_count": len(days),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.dry_run:
        for product, start, end in planned:
            print(" ".join(command_for(product, start, end, args)))
        return

    failures = 0
    for index, (product, start, end) in enumerate(planned, start=1):
        cmd = command_for(product, start, end, args)
        row: dict[str, Any] = {
            "format": "corsewind.eumetsat_backfill_run_command.v1",
            "generated_at_utc": utc_now(),
            "index": index,
            "command_count": len(planned),
            "product": product,
            "start_datetime_utc": iso_z(start),
            "end_datetime_utc": iso_z(end),
            "command": cmd,
            "status": "started",
        }
        append_jsonl(output_log, row)
        try:
            returncode, stdout, stderr = run_command(cmd, args.timeout_sec)
        except subprocess.TimeoutExpired as exc:
            failures += 1
            row.update(
                {
                    "finished_at_utc": utc_now(),
                    "status": "timeout",
                    "returncode": None,
                    "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                    "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
                }
            )
        else:
            if returncode != 0:
                failures += 1
            row.update(
                {
                    "finished_at_utc": utc_now(),
                    "status": "ok" if returncode == 0 else "error",
                    "returncode": returncode,
                    "stdout_tail": stdout[-4000:],
                    "stderr_tail": stderr[-4000:],
                }
            )
        append_jsonl(output_log, row)
        if args.request_sleep_sec:
            time.sleep(args.request_sleep_sec)
    print(json.dumps({"output_log": str(output_log), "failures": failures, "commands": len(planned)}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
