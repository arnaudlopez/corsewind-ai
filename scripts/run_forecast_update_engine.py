#!/usr/bin/env python3
"""Autonomous AROME polling engine for the Corsica WindNinja 50 m product."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from build_arome_corsica_wind_layer import latest_complete_run
from meteo_france_client import coverage_ids, endpoint, load_dotenv, request_api


ROOT = Path(__file__).resolve().parents[1]
AROME_LAYER = ROOT / "visualizations/wind2d/arome-corsica-latest.json"
MOLOCH_LAYER = ROOT / "visualizations/wind2d/moloch-corsica-latest.json"

DEFAULT_STATE_PATH = ROOT / "data/processed/diagnostics/forecast_update_engine_state.json"
DEFAULT_STATUS_PATH = ROOT / "data/processed/diagnostics/forecast_update_engine_status.json"
DEFAULT_LOCK_PATH = ROOT / "tmp/forecast_update_engine.lock"
DEFAULT_EXPORT_MANIFEST = ROOT / "data/processed/exports/beacon_live/windninja_50m_latest.json"
DEFAULT_LEAD_HOURS = tuple(str(hour) for hour in range(0, 49))
DEFAULT_SESSION_TIMEZONE = "Europe/Paris"

WINDNINJA_50M_ARTIFACTS = {
    "arome_layer": "visualizations/wind2d/arome-corsica-latest.json",
    "moloch_layer": "visualizations/wind2d/moloch-corsica-latest.json",
    "color_tiles_manifest": "visualizations/wind2d/windninja-corsica-tiles-50m/manifest.json",
    "data_tiles_manifest": "visualizations/wind2d/windninja-corsica-data-50m/manifest.json",
    "tile_plan_pattern": "data/processed/physics/corsica_windninja_tile_plan_50m_hHH.json",
    "batch_status_pattern": "data/processed/diagnostics/corsica_windninja_50m_batch_status_hHH.json",
    "automatic_process_report_pattern": "reports/corsica_windninja_50m_automatic_process_hHH.md",
    "color_tiles_report_pattern": "reports/corsica_windninja_50m_raster_tiles_report_hHH.md",
    "data_tiles_report_pattern": "reports/corsica_windninja_50m_data_tiles_report_hHH.md",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any] | None:
    path = resolve_path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_run_time(path: Path = AROME_LAYER) -> str | None:
    payload = read_json(path)
    if not payload:
        return None
    value = payload.get("run_time_utc")
    return str(value) if value else None


def parse_utc_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def lead_key(lead_hour: int) -> str:
    return f"h{int(lead_hour):02d}"


def lead_artifacts(lead_hour: int) -> dict[str, str]:
    key = lead_key(lead_hour)
    return {
        "key": key,
        "tile_plan": f"data/processed/physics/corsica_windninja_tile_plan_50m_{key}.json",
        "batch_status": f"data/processed/diagnostics/corsica_windninja_50m_batch_status_{key}.json",
        "automatic_process_report": f"reports/corsica_windninja_50m_automatic_process_{key}.md",
        "color_tiles_report": f"reports/corsica_windninja_50m_raster_tiles_report_{key}.md",
        "data_tiles_report": f"reports/corsica_windninja_50m_data_tiles_report_{key}.md",
    }


def available_forecast_steps() -> list[dict[str, Any]]:
    payload = read_json(AROME_LAYER) or {}
    return list(payload.get("forecast_steps") or [])


def select_windninja_steps(args: argparse.Namespace) -> list[dict[str, Any]]:
    steps = available_forecast_steps()
    if not steps:
        return []

    by_lead = {int(step["lead_hour"]): step for step in steps}
    if args.windninja_lead_hours:
        selected = [by_lead[lead] for lead in args.windninja_lead_hours if lead in by_lead]
        return sorted(selected, key=lambda step: int(step["lead_hour"]))

    tz = ZoneInfo(args.session_timezone)
    now_local = datetime.now(timezone.utc).astimezone(tz)
    stale_before = now_local - timedelta(hours=args.session_past_tolerance_hours)
    today = now_local.date()
    tomorrow = today + timedelta(days=1)
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for step in steps:
        lead = int(step["lead_hour"])
        valid_local = parse_utc_datetime(step["valid_time_utc"]).astimezone(tz)
        hour = valid_local.hour
        if valid_local < stale_before:
            continue
        if hour < args.session_start_hour or hour > args.session_end_hour:
            continue
        if valid_local.date() == today:
            if (hour - args.session_start_hour) % max(1, args.today_session_step_hours) == 0:
                selected.append(step)
                seen.add(lead)
        elif args.session_days == "today-and-tomorrow" and valid_local.date() == tomorrow:
            if (hour - args.session_start_hour) % max(1, args.tomorrow_session_step_hours) == 0:
                selected.append(step)
                seen.add(lead)

    if selected:
        return sorted(selected, key=lambda step: parse_utc_datetime(step["valid_time_utc"]))

    future_or_recent = [
        step
        for step in steps
        if parse_utc_datetime(step["valid_time_utc"]).astimezone(tz) >= stale_before
    ]
    fallback_pool = future_or_recent or steps
    fallback = min(
        fallback_pool,
        key=lambda step: abs((parse_utc_datetime(step["valid_time_utc"]).astimezone(tz) - now_local).total_seconds()),
    )
    return [fallback] if int(fallback["lead_hour"]) not in seen else []


def selected_step_summary(steps: list[dict[str, Any]], timezone_name: str) -> list[dict[str, Any]]:
    tz = ZoneInfo(timezone_name)
    return [
        {
            "key": lead_key(int(step["lead_hour"])),
            "lead_hour": int(step["lead_hour"]),
            "valid_time_utc": step["valid_time_utc"],
            "valid_time_local": parse_utc_datetime(step["valid_time_utc"]).astimezone(tz).isoformat(),
            "stats_ms": step.get("stats_ms"),
            "artifacts": lead_artifacts(int(step["lead_hour"])),
        }
        for step in steps
    ]


def latest_arome_run_time(auth_header: str = "apikey") -> datetime:
    load_dotenv(ROOT / ".env")
    response = request_api(
        endpoint("arome", "001", "GetCapabilities"),
        [("service", "WCS"), ("version", "2.0.1"), ("language", "eng")],
        auth_header,
    )
    run_time, _ = latest_complete_run(coverage_ids(response.text))
    return run_time


def session_lead_hours_for_run(run_time_utc: datetime, args: argparse.Namespace) -> tuple[str, ...]:
    tz = ZoneInfo(args.session_timezone)
    now_local = datetime.now(timezone.utc).astimezone(tz)
    stale_before = now_local - timedelta(hours=args.session_past_tolerance_hours)
    today = now_local.date()
    tomorrow = today + timedelta(days=1)
    selected: list[int] = []
    for lead_hour in range(0, 49):
        valid_local = (run_time_utc + timedelta(hours=lead_hour)).astimezone(tz)
        hour = valid_local.hour
        if valid_local < stale_before:
            continue
        if hour < args.session_start_hour or hour > args.session_end_hour:
            continue
        if valid_local.date() == today:
            if (hour - args.session_start_hour) % max(1, args.today_session_step_hours) == 0:
                selected.append(lead_hour)
        elif args.session_days == "today-and-tomorrow" and valid_local.date() == tomorrow:
            if (hour - args.session_start_hour) % max(1, args.tomorrow_session_step_hours) == 0:
                selected.append(lead_hour)

    if selected:
        return tuple(str(item) for item in selected)

    fallback = min(
        range(0, 49),
        key=lambda lead_hour: abs(
            ((run_time_utc + timedelta(hours=lead_hour)).astimezone(tz) - now_local).total_seconds()
        ),
    )
    return (str(fallback),)


def resolve_arome_lead_hours(args: argparse.Namespace) -> tuple[str, ...]:
    if args.lead_hours:
        return tuple(str(item) for item in args.lead_hours)
    if args.windninja_lead_hours:
        return tuple(str(item) for item in args.windninja_lead_hours)
    if args.arome_lead_hour_policy == "all-48":
        return DEFAULT_LEAD_HOURS
    if args.dry_run:
        return ("0",)
    return session_lead_hours_for_run(latest_arome_run_time(), args)


def synthetic_dry_run_steps(lead_hours: tuple[str, ...]) -> list[dict[str, Any]]:
    run_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    steps = []
    for item in lead_hours:
        lead_hour = int(item)
        valid_time = run_time + timedelta(hours=lead_hour)
        steps.append(
            {
                "key": lead_key(lead_hour),
                "lead_hour": lead_hour,
                "valid_time_utc": valid_time.isoformat().replace("+00:00", "Z"),
                "stats_ms": None,
            }
        )
    return steps


def command_line(args: tuple[str, ...]) -> list[str]:
    return [sys.executable, *args]


def printable_command(cmd: list[str]) -> str:
    try:
        display = [str(Path(cmd[0]).relative_to(ROOT)), *cmd[1:]]
    except ValueError:
        display = cmd
    return " ".join(shlex.quote(str(part)) for part in display)


def run_command(args: tuple[str, ...], dry_run: bool) -> dict[str, Any]:
    cmd = command_line(args)
    printable = printable_command(cmd)
    started = time.time()
    if dry_run:
        print(f"dry-run: {printable}", flush=True)
        return {"cmd": printable, "status": "dry_run", "elapsed_s": 0.0}

    print(f"running: {printable}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    elapsed = round(time.time() - started, 3)
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr, flush=True)
    result = {
        "cmd": printable,
        "status": "pass" if proc.returncode == 0 else "fail",
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {printable}")
    return result


def arome_refresh_command(lead_hours: tuple[str, ...], request_sleep_sec: float) -> tuple[str, ...]:
    return (
        "scripts/build_arome_corsica_wind_layer.py",
        "--lead-hours",
        *lead_hours,
        "--request-sleep-sec",
        str(request_sleep_sec),
    )


def moloch_refresh_command(source: str | None, lead_hours: tuple[str, ...]) -> tuple[str, ...]:
    source_args = ("--input", source) if source else ()
    return (
        "scripts/build_moloch_corsica_wind_layer.py",
        *source_args,
        "--lead-hours",
        *lead_hours,
    )


def windninja_50m_commands(
    lead_hour: int,
    max_runtime_min: float,
    parallel: int,
    force_batch: bool,
    append_tiles: bool,
) -> list[tuple[str, ...]]:
    artifacts = lead_artifacts(lead_hour)
    batch_command = (
        "scripts/run_corsica_windninja_batch.py",
        "--plan",
        artifacts["tile_plan"],
        "--status-output",
        artifacts["batch_status"],
        "--max-runtime-min",
        str(max_runtime_min),
        "--parallel",
        str(parallel),
    )
    if force_batch:
        batch_command = (*batch_command, "--force")

    append_arg = ("--append",) if append_tiles else ()
    return [
        (
            "scripts/prepare_corsica_windninja_tiles.py",
            "--cellsize-m",
            "50",
            "--mesh-resolution-m",
            "50",
            "--tile-size-km",
            "20",
            "--overlap-km",
            "2",
            "--output-height-m",
            "10",
            "--lead-hour",
            str(lead_hour),
            "--min-land-fraction",
            "0",
            "--plan-output",
            artifacts["tile_plan"],
            "--report-output",
            artifacts["automatic_process_report"],
            "--batch-status-output",
            artifacts["batch_status"],
        ),
        batch_command,
        (
            "scripts/build_corsica_windninja_raster_tiles.py",
            "--plan",
            artifacts["tile_plan"],
            "--output-root",
            "visualizations/wind2d/windninja-corsica-tiles-50m",
            "--report-output",
            artifacts["color_tiles_report"],
            "--url-template",
            "./windninja-corsica-tiles-50m/{step}/{mode}/{z}/{x}/{y}.png",
            "--zooms",
            "8",
            "9",
            "10",
            "11",
            "12",
            "--modes",
            "speed",
            "devente",
            "acceleration",
            *append_arg,
        ),
        (
            "scripts/build_corsica_windninja_raster_tiles.py",
            "--encoding",
            "data",
            "--plan",
            artifacts["tile_plan"],
            "--output-root",
            "visualizations/wind2d/windninja-corsica-data-50m",
            "--report-output",
            artifacts["data_tiles_report"],
            "--url-template",
            "./windninja-corsica-data-50m/{step}/data/{z}/{x}/{y}.png",
            "--zooms",
            "8",
            "9",
            "10",
            "11",
            "12",
            "--modes",
            "speed",
            "devente",
            "acceleration",
            *append_arg,
        ),
    ]


def acquire_lock(lock_path: Path, stale_after_sec: int) -> None:
    lock_path = resolve_path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        payload = read_json(lock_path) or {}
        created_at = payload.get("created_at_epoch")
        age = time.time() - float(created_at or 0)
        if age < stale_after_sec:
            raise RuntimeError(f"engine lock already exists: {lock_path}")
    write_json(lock_path, {"pid": os.getpid(), "created_at_utc": utc_now(), "created_at_epoch": time.time()})


def release_lock(lock_path: Path) -> None:
    lock_path = resolve_path(lock_path)
    try:
        if lock_path.exists():
            lock_path.unlink()
    except OSError:
        pass


def load_state(path: Path) -> dict[str, Any]:
    return read_json(path) or {
        "format": "corsewind.forecast_update_engine.state.v1",
        "created_at_utc": utc_now(),
        "last_seen_run_time_utc": None,
        "last_completed_run_time_utc": None,
        "last_success_at_utc": None,
        "last_failure_at_utc": None,
        "consecutive_failures": 0,
        "history": [],
    }


def append_history(state: dict[str, Any], event: dict[str, Any], max_items: int = 30) -> None:
    history = list(state.get("history") or [])
    history.append(event)
    state["history"] = history[-max_items:]


def read_manifest_summary(path: str) -> dict[str, Any] | None:
    payload = read_json(ROOT / path)
    if not payload:
        return None
    return {
        "tileCount": payload.get("tileCount"),
        "encoding": payload.get("encoding"),
        "bounds_wgs84": payload.get("bounds_wgs84"),
        "zooms": payload.get("zooms"),
        "modes": payload.get("modes"),
        "steps": payload.get("steps"),
        "tileCountByStep": payload.get("tileCountByStep"),
        "urlTemplate": payload.get("urlTemplate"),
        "source": payload.get("source"),
        "candidateBoundsPadDeg": payload.get("candidateBoundsPadDeg"),
        "speedScaleMaxKt": payload.get("speedScaleMaxKt"),
        "dataEncoding": payload.get("dataEncoding"),
    }


def write_beacon_export_manifest(path: Path, run_time: str, status: dict[str, Any]) -> None:
    artifacts = {key: value for key, value in WINDNINJA_50M_ARTIFACTS.items()}
    windninja_steps = list(status.get("published_windninja_steps") or status.get("windninja_steps") or [])
    payload = {
        "format": "corsewind.beacon_live.windninja_50m_export.v1",
        "generated_at_utc": utc_now(),
        "status": "ready",
        "arome_run_time_utc": run_time,
        "model": {
            "source": "Meteo-France public AROME WCS",
            "forcing_height_agl_m": 10,
            "windninja_resolution_m": 50,
            "windninja_output_height_agl_m": 10,
            "lead_hours": [step["lead_hour"] for step in windninja_steps],
            "selection_policy": status.get("selection_policy"),
        },
        "artifacts": artifacts,
        "windninja_steps": windninja_steps,
        "data_tiles": read_manifest_summary(artifacts["data_tiles_manifest"]),
        "color_tiles": read_manifest_summary(artifacts["color_tiles_manifest"]),
        "last_pipeline_status": {
            "generated_at_utc": status.get("generated_at_utc"),
            "result": status.get("result"),
            "partial": status.get("result") == "partial_updated",
            "commands": [
                {
                    "cmd": item.get("cmd"),
                    "status": item.get("status"),
                    "elapsed_s": item.get("elapsed_s"),
                    "returncode": item.get("returncode"),
                }
                for item in status.get("commands", [])
            ],
        },
    }
    write_json(path, payload)


def poll_once(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    commands: list[dict[str, Any]] = []
    arome_lead_hours = resolve_arome_lead_hours(args)
    moloch_lead_hours = tuple(str(item) for item in (args.moloch_lead_hours or arome_lead_hours))
    status: dict[str, Any] = {
        "format": "corsewind.forecast_update_engine.status.v1",
        "generated_at_utc": utc_now(),
        "result": "running",
        "dry_run": bool(args.dry_run),
        "poll_interval_sec": args.poll_interval_sec,
        "previous_completed_run_time_utc": state.get("last_completed_run_time_utc"),
        "current_run_time_utc": None,
        "changed": False,
        "forced": bool(args.force),
        "arome_lead_hours": list(arome_lead_hours),
        "moloch_enabled": bool(args.enable_moloch),
        "moloch_lead_hours": list(moloch_lead_hours),
        "selection_policy": {
            "timezone": args.session_timezone,
            "session_start_hour": args.session_start_hour,
            "session_end_hour": args.session_end_hour,
            "today_step_hours": args.today_session_step_hours,
            "tomorrow_step_hours": args.tomorrow_session_step_hours,
            "session_days": args.session_days,
            "past_tolerance_hours": args.session_past_tolerance_hours,
            "explicit_windninja_lead_hours": args.windninja_lead_hours,
        },
        "windninja_steps": [],
        "published_windninja_steps": [],
        "commands": commands,
        "artifacts": WINDNINJA_50M_ARTIFACTS,
        "export_manifest": str(args.export_manifest),
    }
    write_json(args.status_file, status)

    commands.append(run_command(arome_refresh_command(arome_lead_hours, args.arome_request_sleep_sec), args.dry_run))
    if args.enable_moloch:
        source = args.moloch_input or os.getenv("MOLOCH_SOURCE") or os.getenv("MOLOCH_SOURCE_URL")
        if source or not args.moloch_skip_if_missing:
            commands.append(run_command(moloch_refresh_command(source, moloch_lead_hours), args.dry_run))
        else:
            status["moloch_status"] = "skipped_missing_source"
            status["moloch_hint"] = "Set MOLOCH_SOURCE_URL or pass --moloch-input to build moloch-corsica-latest.json."
    current_run_time = read_run_time()
    status["current_run_time_utc"] = current_run_time
    state["last_seen_run_time_utc"] = current_run_time
    state["last_poll_at_utc"] = utc_now()
    selected_steps = select_windninja_steps(args)
    if args.dry_run and not selected_steps:
        selected_steps = synthetic_dry_run_steps(arome_lead_hours)
    status["windninja_steps"] = selected_step_summary(selected_steps, args.session_timezone)

    if args.dry_run:
        for index, step in enumerate(selected_steps):
            lead_hour = int(step["lead_hour"])
            for command in windninja_50m_commands(
                lead_hour,
                args.windninja_runtime_min,
                args.windninja_parallel,
                force_batch=True,
                append_tiles=index > 0,
            ):
                commands.append(run_command(command, dry_run=True))
        status.update({"result": "dry_run", "changed": True, "elapsed_s": round(time.time() - started, 3)})
        write_json(args.status_file, status)
        return status

    if not current_run_time:
        raise RuntimeError(f"AROME refresh did not write run_time_utc to {AROME_LAYER.relative_to(ROOT)}")

    changed = bool(args.force or current_run_time != state.get("last_completed_run_time_utc"))
    status["changed"] = changed
    if not changed:
        status.update({"result": "unchanged", "elapsed_s": round(time.time() - started, 3)})
        write_json(args.status_file, status)
        append_history(state, {"at_utc": utc_now(), "result": "unchanged", "run_time_utc": current_run_time})
        return status

    state["in_progress_run_time_utc"] = current_run_time
    write_json(args.state_file, state)
    if not selected_steps:
        raise RuntimeError("No WindNinja forecast step selected from the refreshed AROME layer")
    published_steps: list[dict[str, Any]] = []
    for index, step in enumerate(selected_steps):
        lead_hour = int(step["lead_hour"])
        print(f"selected WindNinja lead H+{lead_hour} valid={step.get('valid_time_utc')}", flush=True)
        for command in windninja_50m_commands(
            lead_hour,
            args.windninja_runtime_min,
            args.windninja_parallel,
            force_batch=True,
            append_tiles=index > 0,
        ):
            commands.append(run_command(command, dry_run=False))
        published_steps.append(selected_step_summary([step], args.session_timezone)[0])
        status.update(
            {
                "result": "partial_updated",
                "elapsed_s": round(time.time() - started, 3),
                "published_windninja_steps": published_steps,
                "last_published_step": published_steps[-1],
            }
        )
        write_beacon_export_manifest(args.export_manifest, current_run_time, status)
        write_json(args.status_file, status)
        print(
            f"published WindNinja lead H+{lead_hour} "
            f"({len(published_steps)}/{len(selected_steps)}) elapsed={status['elapsed_s']}s",
            flush=True,
        )

    status.update(
        {
            "result": "updated",
            "elapsed_s": round(time.time() - started, 3),
            "published_windninja_steps": published_steps,
            "last_published_step": published_steps[-1] if published_steps else None,
        }
    )
    write_beacon_export_manifest(args.export_manifest, current_run_time, status)
    state["last_completed_run_time_utc"] = current_run_time
    state["last_success_at_utc"] = utc_now()
    state["in_progress_run_time_utc"] = None
    state["consecutive_failures"] = 0
    append_history(
        state,
        {
            "at_utc": utc_now(),
            "result": "updated",
            "run_time_utc": current_run_time,
            "elapsed_s": status["elapsed_s"],
            "export_manifest": str(args.export_manifest),
        },
    )
    write_json(args.status_file, status)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-interval-sec", type=int, default=900)
    parser.add_argument("--once", action="store_true", help="Run one poll cycle and exit.")
    parser.add_argument("--force", action="store_true", help="Run the 50 m pipeline even if the run was already completed.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--lead-hours", nargs="+", default=None)
    parser.add_argument("--arome-lead-hour-policy", choices=["session", "all-48"], default="session")
    parser.add_argument("--arome-request-sleep-sec", type=float, default=1.3)
    parser.add_argument("--enable-moloch", action="store_true", help="Build the optional MOLOCH 1.2 km viewer layer.")
    parser.add_argument("--moloch-input", default=None, help="Local GRIB/NetCDF/JSON file or direct URL. Defaults to MOLOCH_SOURCE_URL.")
    parser.add_argument("--moloch-lead-hours", nargs="+", default=None)
    parser.add_argument("--moloch-skip-if-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--windninja-lead-hours", nargs="+", type=int, default=None)
    parser.add_argument("--windninja-parallel", type=int, default=6)
    parser.add_argument("--windninja-runtime-min", type=float, default=60.0)
    parser.add_argument("--session-timezone", default=DEFAULT_SESSION_TIMEZONE)
    parser.add_argument("--session-start-hour", type=int, default=11)
    parser.add_argument("--session-end-hour", type=int, default=17)
    parser.add_argument("--session-days", choices=["today", "today-and-tomorrow"], default="today-and-tomorrow")
    parser.add_argument("--today-session-step-hours", type=int, default=1)
    parser.add_argument("--tomorrow-session-step-hours", type=int, default=2)
    parser.add_argument("--session-past-tolerance-hours", type=float, default=1.0)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--export-manifest", type=Path, default=DEFAULT_EXPORT_MANIFEST)
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--lock-stale-after-sec", type=int, default=6 * 60 * 60)
    parser.add_argument("--sleep-on-error-sec", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.state_file = resolve_path(args.state_file)
    args.status_file = resolve_path(args.status_file)
    args.export_manifest = resolve_path(args.export_manifest)
    args.lock_file = resolve_path(args.lock_file)

    acquire_lock(args.lock_file, args.lock_stale_after_sec)
    try:
        while True:
            state = load_state(args.state_file)
            try:
                status = poll_once(args, state)
                write_json(args.state_file, state)
                print(
                    f"engine cycle {status['result']} run={status.get('current_run_time_utc')} "
                    f"elapsed={status.get('elapsed_s')}s",
                    flush=True,
                )
            except Exception as exc:
                state["last_failure_at_utc"] = utc_now()
                state["consecutive_failures"] = int(state.get("consecutive_failures") or 0) + 1
                state["in_progress_run_time_utc"] = None
                append_history(state, {"at_utc": utc_now(), "result": "failed", "error": str(exc)})
                write_json(args.state_file, state)
                failure_status = {
                    "format": "corsewind.forecast_update_engine.status.v1",
                    "generated_at_utc": utc_now(),
                    "result": "failed",
                    "error": str(exc),
                    "current_run_time_utc": read_run_time(),
                    "consecutive_failures": state["consecutive_failures"],
                }
                write_json(args.status_file, failure_status)
                print(f"engine cycle failed: {exc}", file=sys.stderr, flush=True)
                if args.once:
                    raise
                time.sleep(max(1, args.sleep_on_error_sec))
            if args.once:
                break
            time.sleep(max(30, args.poll_interval_sec))
    finally:
        release_lock(args.lock_file)


if __name__ == "__main__":
    main()
