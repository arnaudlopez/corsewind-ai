#!/usr/bin/env python3
"""Summarize the hPa backfill/watchers and current wind-mean RMSE state."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


DEFAULT_ML_ROOT = Path("/srv/data/corsewind/ml_dataset")
WAIT_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z .*waiting (?P<seconds>\d+)s")
START_RE = re.compile(r"repair watcher started")
TASK_RE = re.compile(
    r"repair_task index=(?P<index>\d+)/(?P<total>\d+) attempt=(?P<attempt>\d+)/(?P<max_attempts>\d+) "
    r"spot=(?P<spot>\S+) start=(?P<start>\S+) end=(?P<end>\S+) reason=(?P<reason>\S+)"
)
TASK_429_RE = re.compile(r"repair_task_hit_429 index=(?P<index>\d+) spot=(?P<spot>\S+)")
TASK_RETRY_RE = re.compile(r"repair_task_failed index=(?P<index>\d+) spot=(?P<spot>\S+)")
TASK_FAILED_PERMANENTLY_RE = re.compile(r"repair_task_failed_permanently index=(?P<index>\d+) spot=(?P<spot>\S+)")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def pid_cmdline(pid: int) -> str | None:
    path = Path("/proc") / str(pid) / "cmdline"
    try:
        return path.read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore").strip()
    except OSError:
        return None


def process_status(pidfile: Path, needle: str) -> dict[str, Any]:
    raw = read_text(pidfile)
    pid = None
    if raw:
        try:
            pid = int(raw)
        except ValueError:
            pid = None
    cmdline = pid_cmdline(pid) if pid is not None else None
    return {
        "pidfile": str(pidfile),
        "pid": pid,
        "running": bool(cmdline and needle in cmdline),
        "cmdline": cmdline,
        "needle": needle,
    }


def status_file(path: Path) -> dict[str, Any]:
    value = read_text(path)
    return {"path": str(path), "value": value or "missing"}


def last_wait(log_path: Path, now: datetime) -> dict[str, Any]:
    latest: tuple[datetime, int, str] | None = None
    text = read_text(log_path)
    if not text:
        return {"available": False, "path": str(log_path)}
    lines = text.splitlines()
    last_start = 0
    for idx, line in enumerate(lines):
        if START_RE.search(line):
            last_start = idx

    for line in lines[last_start:]:
        match = WAIT_RE.search(line)
        if not match:
            continue
        started = datetime.strptime(match.group("ts"), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        seconds = int(match.group("seconds"))
        if latest is None or started > latest[0]:
            latest = (started, seconds, line)
    if latest is None:
        return {"available": False, "path": str(log_path), "reason": "no_wait_line_found"}
    started, seconds, line = latest
    wake = started + timedelta(seconds=seconds)
    remaining = max(0, math.ceil((wake - now).total_seconds()))
    return {
        "available": True,
        "path": str(log_path),
        "line": line,
        "started_at_utc": iso(started),
        "wait_seconds": seconds,
        "expected_wake_utc": iso(wake),
        "remaining_seconds": remaining,
    }


def coverage_summary(ml_root: Path) -> dict[str, Any]:
    candidates = [
        ml_root / "source_inventories/open_meteo_pressure_level_repair_audit.json",
        ml_root / "source_inventories/open_meteo_pressure_level_progress_final.json",
        ml_root / "source_inventories/open_meteo_pressure_level_progress_latest.json",
    ]
    for path in candidates:
        payload = read_json(path)
        if not payload:
            continue
        observed = int(payload.get("observed_rows") or 0)
        complete = int(payload.get("required_feature_complete_rows") or 0)
        expected = int(payload.get("expected_rows") or 0)
        missing = payload.get("required_feature_missing_rows")
        return {
            "path": str(path),
            "expected_rows": expected,
            "observed_rows": observed,
            "missing_rows": payload.get("missing_rows"),
            "required_feature_complete_rows": complete,
            "required_feature_missing_rows": missing,
            "required_feature_complete_spot_count": payload.get("required_feature_complete_spot_count"),
            "coverage_observed_ratio": round(complete / observed, 8) if observed else None,
            "coverage_expected_ratio": round(complete / expected, 8) if expected else None,
        }
    return {"path": None}


def leaderboard_summary(ml_root: Path) -> dict[str, Any]:
    candidates = [
        ml_root / "benchmarks/hpa_calibrator_selection_v1/wind_mean_rmse_leaderboard.json",
        ml_root / "benchmarks/hpa_tabular_rmse09_selection_v1/wind_mean_rmse_leaderboard.json",
        ml_root / "benchmarks/wind_mean_rmse_leaderboard_current.json",
    ]
    for path in candidates:
        payload = read_json(path)
        if not payload:
            continue
        best = payload.get("best") or {}
        rmse = best.get("rmse")
        return {
            "path": str(path),
            "generated_at_utc": payload.get("generated_at_utc"),
            "decision": payload.get("decision"),
            "best_run_id": best.get("run_id"),
            "best_rmse": rmse,
            "gap_to_0_9": round(float(rmse) - 0.9, 6) if rmse is not None else None,
            "best_path": best.get("path"),
        }
    return {"path": None}


def line_count(path: Path) -> int | None:
    try:
        with path.open(encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return None


def task_log_summary(log_root: Path) -> dict[str, Any]:
    tasks_path = log_root / "open_meteo_pressure_repair_after_429_tasks.tsv"
    task_logs = sorted(
        log_root.glob("open_meteo_pressure_repair_task_*_attempt_*.log"),
        key=lambda path: path.stat().st_mtime,
    )
    latest = task_logs[-1] if task_logs else None
    latest_text = read_text(latest) if latest else None
    latest_tail = latest_text.splitlines()[-20:] if latest_text else []
    latest_hit_429 = any("HTTP 429" in line or "Hourly API request limit exceeded" in line for line in latest_tail)
    return {
        "tasks_path": str(tasks_path),
        "tasks_file_exists": tasks_path.exists(),
        "task_count": line_count(tasks_path) if tasks_path.exists() else 0,
        "task_log_count": len(task_logs),
        "latest_task_log": str(latest) if latest else None,
        "latest_task_log_tail": latest_tail,
        "latest_task_log_hit_429": latest_hit_429,
    }


def repair_progress_summary(log_root: Path) -> dict[str, Any]:
    log_path = log_root / "open_meteo_pressure_repair_after_429.log"
    text = read_text(log_path)
    if not text:
        return {"available": False, "path": str(log_path)}

    total = None
    current: dict[str, Any] | None = None
    started_indices: set[int] = set()
    hit_429_indices: set[int] = set()
    retry_indices: set[int] = set()
    permanently_failed_indices: set[int] = set()
    pre_repair_coverage = None
    post_repair_coverage = None
    last_event = None

    for line in text.splitlines():
        if "pre_repair_coverage=" in line:
            last_event = line
            match = re.search(r"pre_repair_coverage=(?P<coverage>[0-9.]+) task_count=(?P<count>\d+)", line)
            if match:
                pre_repair_coverage = float(match.group("coverage"))
                total = int(match.group("count"))
            continue
        if "post_repair_coverage=" in line:
            last_event = line
            match = re.search(r"post_repair_coverage=(?P<coverage>[0-9.]+)", line)
            if match:
                post_repair_coverage = float(match.group("coverage"))
            continue
        match = TASK_RE.search(line)
        if match:
            last_event = line
            idx = int(match.group("index"))
            total = int(match.group("total"))
            started_indices.add(idx)
            current = {
                "index": idx,
                "total": total,
                "attempt": int(match.group("attempt")),
                "max_attempts": int(match.group("max_attempts")),
                "spot": match.group("spot"),
                "start": match.group("start"),
                "end": match.group("end"),
                "reason": match.group("reason"),
                "line": line,
            }
            continue
        match = TASK_429_RE.search(line)
        if match:
            last_event = line
            hit_429_indices.add(int(match.group("index")))
            continue
        match = TASK_FAILED_PERMANENTLY_RE.search(line)
        if match:
            last_event = line
            permanently_failed_indices.add(int(match.group("index")))
            continue
        match = TASK_RETRY_RE.search(line)
        if match:
            last_event = line
            retry_indices.add(int(match.group("index")))
            continue

    current_index = int(current["index"]) if current else (max(started_indices) if started_indices else 0)
    retry_indices = {idx for idx in retry_indices if idx <= current_index}
    hit_429_indices = {idx for idx in hit_429_indices if idx <= current_index}
    permanently_failed_indices = {idx for idx in permanently_failed_indices if idx <= current_index}
    completed_estimate = max(0, current_index - (1 if current else 0))
    return {
        "available": True,
        "path": str(log_path),
        "task_total": total,
        "task_started_count": current_index,
        "task_completed_estimate": completed_estimate,
        "task_retry_count": len(retry_indices),
        "task_failed_count": len(permanently_failed_indices),
        "task_hit_429_count": len(hit_429_indices),
        "current_task": current,
        "pre_repair_coverage": pre_repair_coverage,
        "post_repair_coverage": post_repair_coverage,
        "last_event": last_event,
    }


def artifact_summary(ml_root: Path) -> dict[str, Any]:
    artifacts = {
        "primary_training_results": ml_root / "benchmarks/tabular_lgbm_300k_short_hpa_v1_2024_2025_to_2026_v1/training_results.json",
        "primary_predictions": ml_root / "benchmarks/tabular_lgbm_300k_short_hpa_v1_2024_2025_to_2026_v1/tabular_holdout_predictions.parquet",
        "extra_selection": ml_root / "benchmarks/hpa_tabular_rmse09_selection_v1/hpa_tabular_rmse09_selection.json",
        "calibrator_results": ml_root / "benchmarks/prediction_residual_calibrator_hpa_2025h2_to_2026_extratrees_scalegrid_v1/calibration_results.json",
        "calibrator_predictions": ml_root / "benchmarks/prediction_residual_calibrator_hpa_2025h2_to_2026_extratrees_scalegrid_v1/calibrated_predictions_2026.parquet",
        "calibrator_gap_audit": ml_root / "benchmarks/prediction_residual_calibrator_hpa_2025h2_to_2026_extratrees_scalegrid_v1/rmse09_gap_audit_hpa_calibrator_v1.json",
    }
    out = {}
    for name, path in artifacts.items():
        try:
            stat = path.stat()
            out[name] = {
                "path": str(path),
                "exists": True,
                "size_bytes": stat.st_size,
                "modified_at_utc": iso(datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)),
            }
        except OSError:
            out[name] = {"path": str(path), "exists": False}
    return out


def consistency_warnings(summary: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for name, status_payload in summary["watcher_statuses"].items():
        status = str(status_payload.get("value") or "missing")
        running = bool(summary["processes"][name].get("running"))
        if status == "running" and not running:
            warnings.append(f"{name}_status_running_but_process_not_matching_pidfile")
        if status.startswith("failed:") and running:
            warnings.append(f"{name}_status_failed_but_process_still_running")
    if summary["repair_tasks"].get("task_log_count", 0) > 0 and not summary["repair_tasks"].get("tasks_file_exists"):
        warnings.append("repair_task_logs_exist_but_task_file_missing")
    if summary["repair_tasks"].get("latest_task_log_hit_429"):
        warnings.append("latest_repair_task_hit_open_meteo_429")
    progress = summary.get("repair_progress") or {}
    if progress.get("task_retry_count", 0) > 0:
        warnings.append("repair_tasks_retrying")
    if progress.get("task_failed_count", 0) > 0:
        warnings.append("repair_tasks_failed_permanently")
    coverage = summary["hpa_coverage"].get("coverage_observed_ratio")
    if coverage is not None and coverage < summary["coverage_gate"] and summary["artifacts"]["primary_training_results"]["exists"]:
        warnings.append("primary_training_exists_before_hpa_coverage_gate")
    return warnings


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    ml_root = args.ml_root
    log_root = ml_root / "backfill_logs"
    now = utc_now()
    watcher_statuses = {
        "repair": status_file(log_root / "open_meteo_pressure_repair_after_429.status"),
        "primary": status_file(log_root / "open_meteo_pressure_rebuild_watcher.status"),
        "extra": status_file(log_root / "hpa_extra_benchmarks_watcher.status"),
        "calibrator": status_file(log_root / "hpa_calibrator_watcher.status"),
    }
    processes = {
        "repair": process_status(
            log_root / "open_meteo_pressure_repair_after_429.pid",
            "z2_repair_open_meteo_pressure_after_rate_limit.sh",
        ),
        "primary": process_status(
            log_root / "open_meteo_pressure_rebuild_watcher.pid",
            "z2_watch_open_meteo_pressure_then_rebuild.sh",
        ),
        "extra": process_status(
            log_root / "hpa_extra_benchmarks_watcher.pid",
            "z2_watch_hpa_then_extra_benchmarks.sh",
        ),
        "calibrator": process_status(
            log_root / "hpa_calibrator_watcher.pid",
            "z2_watch_hpa_calibrator_then_leaderboard.sh",
        ),
    }
    repair_wait = last_wait(log_root / "open_meteo_pressure_repair_after_429.log", now)
    repair_tasks = task_log_summary(log_root)
    repair_progress = repair_progress_summary(log_root)
    coverage = coverage_summary(ml_root)
    leaderboard = leaderboard_summary(ml_root)
    artifacts = artifact_summary(ml_root)
    next_action = "wait_for_repair_wake"
    primary_status = watcher_statuses["primary"]["value"]
    repair_status = watcher_statuses["repair"]["value"]
    repair_running = bool(processes["repair"].get("running"))
    repair_remaining = repair_wait.get("remaining_seconds") if repair_wait.get("available") else None
    if repair_status.startswith("failed:"):
        next_action = "inspect_repair_failure"
    elif primary_status.startswith("failed:"):
        next_action = "inspect_primary_rebuild_or_benchmark_failure"
    elif repair_progress.get("task_hit_429_count", 0) > 0 or repair_tasks.get("latest_task_log_hit_429"):
        next_action = "wait_for_open_meteo_quota_retry"
    elif repair_progress.get("task_started_count", 0) > 0 or repair_tasks.get("task_log_count"):
        next_action = "monitor_hpa_repair_tasks"
    elif primary_status == "complete":
        next_action = "wait_for_extra_and_calibrator_benchmarks"
    elif repair_running and isinstance(repair_remaining, int) and repair_remaining > 0:
        next_action = "wait_for_repair_wake"
    elif repair_running:
        next_action = "wait_for_repair_audit_or_task_generation"
    elif coverage.get("coverage_observed_ratio") and coverage["coverage_observed_ratio"] >= args.coverage_gate:
        next_action = "wait_for_primary_rebuild"
    elif artifacts["primary_training_results"]["exists"] and primary_status != "complete":
        next_action = "wait_for_primary_diagnostics_to_complete"
    statuses = {name: item["value"] for name, item in watcher_statuses.items()}
    summary = {
        "format": "corsewind.hpa_rmse09_status.v1",
        "generated_at_utc": iso(now),
        "ml_root": str(ml_root),
        "watcher_statuses": watcher_statuses,
        "statuses": statuses,
        "processes": processes,
        "repair_wait": repair_wait,
        "repair_tasks": repair_tasks,
        "repair_progress": repair_progress,
        "hpa_coverage": coverage,
        "leaderboard": leaderboard,
        "artifacts": artifacts,
        "coverage_gate": args.coverage_gate,
        "next_action": next_action,
    }
    summary["warnings"] = consistency_warnings(summary)
    if summary["warnings"] and next_action == "wait_for_repair_wake":
        summary["next_action"] = "inspect_status_warnings"
    return summary


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lb = summary["leaderboard"]
    cov = summary["hpa_coverage"]
    wait = summary["repair_wait"]
    progress = summary.get("repair_progress") or {}
    lines = [
        "# hPa / RMSE09 Status",
        "",
        f"Generated: `{summary['generated_at_utc']}`",
        f"Next action: `{summary['next_action']}`",
        f"Warnings: `{len(summary.get('warnings') or [])}`",
        "",
        "## RMSE",
        "",
        f"- Best RMSE: `{lb.get('best_rmse')}`",
        f"- Best run: `{lb.get('best_run_id')}`",
        f"- Gap to 0.9: `{lb.get('gap_to_0_9')}`",
        f"- Leaderboard: `{lb.get('path')}`",
        "",
        "## hPa Coverage",
        "",
        f"- Coverage observed ratio: `{cov.get('coverage_observed_ratio')}`",
        f"- Complete rows: `{cov.get('required_feature_complete_rows')}`",
        f"- Missing hPa rows: `{cov.get('required_feature_missing_rows')}`",
        f"- Source: `{cov.get('path')}`",
        "",
        "## Repair Wait",
        "",
        f"- Expected wake UTC: `{wait.get('expected_wake_utc')}`",
        f"- Remaining seconds: `{wait.get('remaining_seconds')}`",
        "",
        "## Repair Tasks",
        "",
        f"- Task count: `{summary['repair_tasks'].get('task_count')}`",
        f"- Task logs: `{summary['repair_tasks'].get('task_log_count')}`",
        f"- Latest task log: `{summary['repair_tasks'].get('latest_task_log')}`",
        f"- Latest task hit 429: `{summary['repair_tasks'].get('latest_task_log_hit_429')}`",
        f"- Started tasks: `{progress.get('task_started_count')}` / `{progress.get('task_total')}`",
        f"- Completed estimate: `{progress.get('task_completed_estimate')}`",
        f"- Task 429 count: `{progress.get('task_hit_429_count')}`",
        f"- Task retry count: `{progress.get('task_retry_count')}`",
        f"- Task permanent failure count: `{progress.get('task_failed_count')}`",
        f"- Current task: `{progress.get('current_task')}`",
        f"- Last repair event: `{progress.get('last_event')}`",
        "",
        "## Watchers",
        "",
        "| Watcher | Status | Running | PID |",
        "| --- | --- | ---: | ---: |",
    ]
    for name in ("repair", "primary", "extra", "calibrator"):
        status = summary["watcher_statuses"][name]["value"]
        process = summary["processes"][name]
        lines.append(f"| `{name}` | `{status}` | `{process.get('running')}` | `{process.get('pid')}` |")
    if summary.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- `{warning}`" for warning in summary["warnings"])
    lines.extend([
        "",
        "## Artifacts",
        "",
        "| Artifact | Exists | Updated |",
        "| --- | ---: | --- |",
    ])
    for name, item in summary["artifacts"].items():
        lines.append(f"| `{name}` | `{item.get('exists')}` | `{item.get('modified_at_utc')}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-root", type=Path, default=DEFAULT_ML_ROOT)
    parser.add_argument("--coverage-gate", type=float, default=0.995)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(args)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
