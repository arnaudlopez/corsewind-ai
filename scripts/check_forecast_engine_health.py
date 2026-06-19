#!/usr/bin/env python3
"""Healthcheck for the CorseWind forecast update engine."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS_PATH = ROOT / "data/processed/diagnostics/forecast_update_engine_status.json"


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fail(message: str, payload: dict[str, Any] | None = None) -> int:
    output = {"healthy": False, "error": message}
    if payload:
        output.update(payload)
    print(json.dumps(output, ensure_ascii=False), file=sys.stderr)
    return 1


def ok(payload: dict[str, Any]) -> int:
    print(json.dumps({"healthy": True, **payload}, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--max-status-age-sec", type=int, default=1800)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.status_file if args.status_file.is_absolute() else ROOT / args.status_file
    if not path.exists():
        return fail(f"status file does not exist: {path}")
    try:
        status = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return fail(f"status file is not readable JSON: {exc}")

    generated_at = status.get("generated_at_utc")
    if not generated_at:
        return fail("status file has no generated_at_utc")
    try:
        age_sec = int((datetime.now(timezone.utc) - parse_utc(str(generated_at))).total_seconds())
    except ValueError:
        return fail(f"status generated_at_utc is invalid: {generated_at}")
    if age_sec > args.max_status_age_sec:
        return fail(
            "status file is stale",
            {
                "status_age_sec": age_sec,
                "max_status_age_sec": args.max_status_age_sec,
                "result": status.get("result"),
            },
        )

    failures = int(status.get("consecutive_failures") or 0)
    result = str(status.get("result") or "unknown")
    if result == "failed" and failures >= args.max_consecutive_failures:
        return fail(
            "engine has too many consecutive failures",
            {
                "result": result,
                "consecutive_failures": failures,
                "max_consecutive_failures": args.max_consecutive_failures,
                "error": status.get("error"),
            },
        )

    sources = status.get("sources") or {}
    degraded_sources = sorted(
        source for source, source_status in sources.items() if source_status.get("status") == "failed"
    )
    return ok(
        {
            "result": result,
            "status_age_sec": age_sec,
            "current_run_time_utc": status.get("current_run_time_utc"),
            "consecutive_failures": failures,
            "degraded_sources": degraded_sources,
            "git_commit": (status.get("runtime") or {}).get("git_commit"),
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
