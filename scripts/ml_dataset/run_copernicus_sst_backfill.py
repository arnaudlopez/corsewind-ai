#!/usr/bin/env python3
"""Run chunked Copernicus Marine SST spot backfills over selected UTC hours."""

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
DEFAULT_LOG_ROOT = DEFAULT_ML_ROOT / "source_inventories/copernicus_backfill_runs"
COLLECTOR = ROOT / "scripts/ml_dataset/collect_copernicus_marine_sst.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise SystemExit("--end-date must be after or equal to --start-date")
    values = []
    cursor = start
    while cursor <= end:
        values.append(cursor)
        cursor += timedelta(days=1)
    return values


def iso_compact(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def command_for(start: datetime, end: datetime, args: argparse.Namespace) -> list[str]:
    filename = f"sst_corse_{iso_compact(start)}_{iso_compact(end)}.nc"
    cmd = [
        sys.executable,
        str(COLLECTOR),
        "--start-datetime",
        iso_z(start),
        "--end-datetime",
        iso_z(end),
        "--minimum-longitude",
        str(args.minimum_longitude),
        "--minimum-latitude",
        str(args.minimum_latitude),
        "--maximum-longitude",
        str(args.maximum_longitude),
        "--maximum-latitude",
        str(args.maximum_latitude),
        "--output-filename",
        filename,
        "--search-radius-cells",
        str(args.search_radius_cells),
        "--log-level",
        args.log_level,
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
    return DEFAULT_LOG_ROOT / f"copernicus_sst_backfill_{stamp}.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--start-hour-utc", type=int, default=10)
    parser.add_argument("--end-hour-utc", type=int, default=18)
    parser.add_argument("--window-hours", type=float, default=8)
    parser.add_argument("--minimum-longitude", type=float, default=7.5)
    parser.add_argument("--minimum-latitude", type=float, default=41.0)
    parser.add_argument("--maximum-longitude", type=float, default=10.2)
    parser.add_argument("--maximum-latitude", type=float, default=43.3)
    parser.add_argument("--search-radius-cells", type=int, default=4)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--request-sleep-sec", type=float, default=0.0)
    parser.add_argument("--spot-id", action="append", default=[])
    parser.add_argument("--include-context-spots", action="store_true")
    parser.add_argument("--delete-raw-after-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL", "QUIET"])
    parser.add_argument("--output-log", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    days = date_range(parse_date(args.start_date), parse_date(args.end_date))
    output_log = output_log_path(args)
    planned = [
        (start, end)
        for day in days
        for start, end in windows_for_day(day, args.start_hour_utc, args.end_hour_utc, args.window_hours)
    ]
    print(
        json.dumps(
            {
                "generated_at_utc": utc_now(),
                "mode": "dry_run" if args.dry_run else "execute",
                "output_log": str(output_log),
                "command_count": len(planned),
                "day_count": len(days),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.dry_run:
        for start, end in planned:
            print(" ".join(command_for(start, end, args)))
        return

    failures = 0
    for index, (start, end) in enumerate(planned, start=1):
        cmd = command_for(start, end, args)
        row: dict[str, Any] = {
            "format": "corsewind.copernicus_sst_backfill_run_command.v1",
            "generated_at_utc": utc_now(),
            "index": index,
            "command_count": len(planned),
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
