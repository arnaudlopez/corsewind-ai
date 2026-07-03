#!/usr/bin/env python3
"""Audit operational model-run archives and spot samples."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCES = ("arome", "aromepi", "moloch", "icon2i")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        return {"_error": str(exc), "_path": str(path)}


def count_jsonl_rows(path: Path) -> int:
    try:
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except FileNotFoundError:
        return 0


def model_run_summary(root: Path, source: str) -> dict[str, Any]:
    source_root = root / "model_runs" / source
    summaries = sorted(source_root.glob("run_*/summary.json"))
    run_times = []
    total_bytes = 0
    for summary_path in summaries:
        total_bytes += summary_path.stat().st_size
        payload = read_json(summary_path) or {}
        if payload.get("run_time_utc"):
            run_times.append(str(payload["run_time_utc"]))
        archive_path = payload.get("archive_path")
        if archive_path and Path(archive_path).exists():
            total_bytes += Path(archive_path).stat().st_size
    latest = read_json(source_root / "latest.json")
    return {
        "source": source,
        "run_count": len(summaries),
        "first_run_time_utc": min(run_times) if run_times else None,
        "last_run_time_utc": max(run_times) if run_times else None,
        "latest_run_time_utc": None if not latest else latest.get("run_time_utc"),
        "latest_exists": latest is not None,
        "size_mb": round(total_bytes / 1_000_000, 3),
    }


def sample_summary(root: Path, source: str) -> dict[str, Any]:
    source_root = root / "model_samples" / f"source={source}"
    files = sorted(source_root.glob("date=*/samples.jsonl"))
    rows = 0
    dates = []
    run_times: set[str] = set()
    spot_ids: set[str] = set()
    lead_minutes: set[int] = set()
    by_date: dict[str, int] = {}
    for path in files:
        date = path.parent.name.removeprefix("date=")
        dates.append(date)
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                count += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("run_time_utc"):
                    run_times.add(str(row["run_time_utc"]))
                if row.get("spot_id"):
                    spot_ids.add(str(row["spot_id"]))
                if isinstance(row.get("lead_minutes"), int):
                    lead_minutes.add(int(row["lead_minutes"]))
        rows += count
        by_date[date] = count
    return {
        "source": source,
        "file_count": len(files),
        "row_count": rows,
        "first_valid_date": min(dates) if dates else None,
        "last_valid_date": max(dates) if dates else None,
        "sampled_run_count": len(run_times),
        "spot_count": len(spot_ids),
        "lead_minutes_count": len(lead_minutes),
        "latest_dates": dict(sorted(by_date.items())[-7:]),
    }


def build_report(root: Path, max_latest_age_hours: float) -> dict[str, Any]:
    generated_at = utc_now()
    generated_dt = parse_utc(generated_at)
    runs = {source: model_run_summary(root, source) for source in SOURCES}
    samples = {source: sample_summary(root, source) for source in SOURCES}
    problems = []
    for source in ("arome", "aromepi"):
        run_count = runs[source]["run_count"]
        sampled_run_count = samples[source]["sampled_run_count"]
        latest_run_dt = parse_utc(runs[source].get("last_run_time_utc"))
        latest_age_hours = None
        if generated_dt and latest_run_dt:
            latest_age_hours = round((generated_dt - latest_run_dt).total_seconds() / 3600.0, 3)
        runs[source]["latest_age_hours"] = latest_age_hours
        if run_count == 0:
            problems.append(f"{source}: no archived operational runs")
        if sampled_run_count == 0:
            problems.append(f"{source}: no spot samples")
        if run_count and sampled_run_count and sampled_run_count < run_count:
            problems.append(f"{source}: sampled run count {sampled_run_count} < archived run count {run_count}")
        if latest_age_hours is not None and latest_age_hours > max_latest_age_hours:
            problems.append(f"{source}: latest archived run is stale ({latest_age_hours}h old > {max_latest_age_hours}h)")
    return {
        "format": "corsewind.operational_nwp_archive_audit.v1",
        "generated_at_utc": generated_at,
        "ml_root": str(root),
        "max_latest_age_hours": max_latest_age_hours,
        "runs": runs,
        "samples": samples,
        "problem_count": len(problems),
        "problems": problems,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Operational NWP Archive Audit",
        "",
        f"- generated: `{report['generated_at_utc']}`",
        f"- ml root: `{report['ml_root']}`",
        f"- max latest age hours: `{report['max_latest_age_hours']}`",
        f"- problem count: `{report['problem_count']}`",
        "",
        "## Runs",
        "",
        "| Source | Runs | First run | Last run | Latest age h | Latest pointer | Size MB |",
        "| --- | ---: | --- | --- | ---: | --- | ---: |",
    ]
    for source, item in report["runs"].items():
        lines.append(
            f"| `{source}` | {item['run_count']} | `{item['first_run_time_utc']}` | "
            f"`{item['last_run_time_utc']}` | {item.get('latest_age_hours')} | "
            f"`{item['latest_run_time_utc']}` | {item['size_mb']} |"
        )
    lines.extend(
        [
            "",
            "## Spot Samples",
            "",
            "| Source | Files | Rows | Sampled runs | Spots | Lead count | First date | Last date |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for source, item in report["samples"].items():
        lines.append(
            f"| `{source}` | {item['file_count']} | {item['row_count']} | {item['sampled_run_count']} | "
            f"{item['spot_count']} | {item['lead_minutes_count']} | `{item['first_valid_date']}` | `{item['last_valid_date']}` |"
        )
    if report.get("problems"):
        lines.extend(["", "## Problems", ""])
        for problem in report["problems"]:
            lines.append(f"- {problem}")
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-root", type=Path, required=True)
    parser.add_argument("--max-latest-age-hours", type=float, default=36.0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.ml_root, args.max_latest_age_hours)
    if args.output_json:
        write_json(args.output_json, report)
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
