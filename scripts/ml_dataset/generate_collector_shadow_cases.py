#!/usr/bin/env python3
"""Generate collector hindcast shadow cases for one validation day."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date, expected YYYY-MM-DD: {value}") from exc


def parse_issue_times(value: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for item in value.split(","):
        part = item.strip()
        if not part:
            continue
        try:
            parsed = time.fromisoformat(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid issue time HH:MM: {part}") from exc
        out.append((parsed.hour, parsed.minute))
    if not out:
        raise argparse.ArgumentTypeError("At least one issue time is required.")
    return out


def iso_z(day: date, hour: int, minute: int) -> str:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def output_default(day: date, version: str) -> Path:
    compact = day.strftime("%Y%m%d")
    return Path("configs") / f"ml_collector_shadow_cases_{compact}_full_day_{version}.json"


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    compact = args.date.strftime("%Y%m%d")
    target_end = iso_z(args.date, args.target_end_hour, args.target_end_minute)
    run_time = iso_z(args.date, args.run_hour, args.run_minute)
    cases = []
    for hour, minute in args.issue_time:
        issue_time = iso_z(args.date, hour, minute)
        cases.append(
            {
                "run_id": f"collector_{compact}T{hour:02d}{minute:02d}_{args.run_id_suffix}",
                "run_time_utc": run_time,
                "issue_time_utc": issue_time,
                "target_end_utc": target_end,
            }
        )
    return {
        "format": "corsewind.collector_hindcast_cases.v1",
        "generated_at_utc": utc_now(),
        "description": (
            f"Unseen shadow validation cases for the {args.date.isoformat()} windsurf day. "
            f"Run after observations are available through {target_end}."
        ),
        "cases": cases,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    output = args.output or output_default(args.date, args.version)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}. Use --overwrite to replace it.")
    payload = build_payload(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "case_count": len(payload["cases"]), "cases": payload["cases"]}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=parse_date, required=True, help="Validation date in UTC, YYYY-MM-DD.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--run-id-suffix", default="unseen_full_day_v1")
    parser.add_argument("--run-hour", type=int, default=6)
    parser.add_argument("--run-minute", type=int, default=0)
    parser.add_argument("--issue-time", type=parse_issue_times, default=parse_issue_times("06:45,08:45,10:45,12:45"))
    parser.add_argument("--target-end-hour", type=int, default=17)
    parser.add_argument("--target-end-minute", type=int, default=0)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
