#!/usr/bin/env python3
"""Autonomous AROME polling engine for the Corsica WindNinja 50 m product."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from build_arome_corsica_wind_layer import latest_complete_run
from meteo_france_client import coverage_ids, endpoint, load_dotenv, request_api


ROOT = Path(__file__).resolve().parents[1]
AROME_LAYER = ROOT / "visualizations/wind2d/arome-corsica-latest.json"
AROMEPI_LAYER = ROOT / "visualizations/wind2d/aromepi-corsica-latest.json"
MOLOCH_LAYER = ROOT / "visualizations/wind2d/moloch-corsica-latest.json"
ICON2I_LAYER = ROOT / "visualizations/wind2d/icon2i-corsica-latest.json"

DEFAULT_STATE_PATH = ROOT / "data/processed/diagnostics/forecast_update_engine_state.json"
DEFAULT_STATUS_PATH = ROOT / "data/processed/diagnostics/forecast_update_engine_status.json"
DEFAULT_LOCK_PATH = ROOT / "tmp/forecast_update_engine.lock"
DEFAULT_EXPORT_MANIFEST = ROOT / "data/processed/exports/beacon_live/windninja_50m_latest.json"
DEFAULT_LEAD_HOURS = tuple(str(hour) for hour in range(0, 49))
DEFAULT_SESSION_TIMEZONE = "Europe/Paris"
PUBLICATION_HISTORY_LIMIT = 40
PUBLICATION_PROFILES = {
    "arome": {
        "run_hours_utc": tuple(range(0, 24, 3)),
        "default_delay_sec": 60 * 60,
        "fast_window_before_sec": 15 * 60,
        "fast_window_after_sec": 2 * 60 * 60,
        "delayed_poll_interval_sec": 5 * 60,
        "max_usable_delay_sec": 5 * 60 * 60,
    },
    "aromepi": {
        "run_hours_utc": tuple(range(0, 24)),
        "default_delay_sec": 15 * 60,
        "fast_window_before_sec": 5 * 60,
        "fast_window_after_sec": 45 * 60,
        "delayed_poll_interval_sec": 60,
        "max_usable_delay_sec": 3 * 60 * 60,
    },
    "moloch": {
        "run_hours_utc": (3,),
        "default_delay_sec": 4 * 60 * 60 + 30 * 60,
        "fast_window_before_sec": 45 * 60,
        "fast_window_after_sec": 3 * 60 * 60,
        "delayed_poll_interval_sec": 10 * 60,
        "max_usable_delay_sec": 12 * 60 * 60,
    },
    "icon2i": {
        "run_hours_utc": (0, 12),
        "default_delay_sec": 2 * 60 * 60 + 30 * 60,
        "fast_window_before_sec": 45 * 60,
        "fast_window_after_sec": 3 * 60 * 60,
        "delayed_poll_interval_sec": 10 * 60,
        "max_usable_delay_sec": 8 * 60 * 60,
    },
}
SOURCE_LAYER_PATHS = {
    "arome": AROME_LAYER,
    "aromepi": AROMEPI_LAYER,
    "moloch": MOLOCH_LAYER,
    "icon2i": ICON2I_LAYER,
}
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_ML_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_ML_DATASET_ROOT = DEFAULT_ML_ROOT / "model_runs"
DEFAULT_ML_MODEL_SAMPLES_ROOT = DEFAULT_ML_ROOT / "model_samples"
DEFAULT_ML_NWP_EXTRA_FIELDS_RAW_ROOT = DEFAULT_ML_ROOT / "meteo_france_nwp/raw/extra_fields"
DEFAULT_ML_NWP_EXTRA_FIELDS_SAMPLES_ROOT = DEFAULT_ML_ROOT / "meteo_france_nwp/extra_field_samples"
DEFAULT_ML_NWP_VERTICAL_PROFILES_RAW_ROOT = DEFAULT_ML_ROOT / "meteo_france_nwp/raw/vertical_profiles"
DEFAULT_ML_NWP_VERTICAL_PROFILES_SAMPLES_ROOT = DEFAULT_ML_ROOT / "meteo_france_nwp/vertical_profiles"
DEFAULT_ML_FEATURE_STORE_ROOT = DEFAULT_ML_ROOT / "feature_store"
DEFAULT_ML_COPERNICUS_SST_RAW_ROOT = DEFAULT_ML_ROOT / "copernicus_marine/raw/sst"
DEFAULT_ML_COPERNICUS_SST_SAMPLES_ROOT = DEFAULT_ML_ROOT / "copernicus_marine/sst_samples"
DEFAULT_ML_EUMETSAT_CLOUD_MASK_RAW_ROOT = DEFAULT_ML_ROOT / "eumetsat/raw/cloud_mask"
DEFAULT_ML_EUMETSAT_CLOUD_MASK_SAMPLES_ROOT = DEFAULT_ML_ROOT / "eumetsat/cloud_mask_samples"
DEFAULT_ML_EUMETSAT_CLOUD_MASK_COLLECTION_ID = "EO:EUM:DAT:0678"
DEFAULT_ML_EUMETSAT_BBOX = "7.5,41.0,10.2,43.3"
EUMETSAT_THERMAL_PRODUCTS = {
    "cloud_type": {
        "status_key": "eumetsat_cloud_type",
        "collection_id": "EO:EUM:DAT:0680",
        "raw_root": DEFAULT_ML_ROOT / "eumetsat/raw/cloud_type",
        "samples_root": DEFAULT_ML_ROOT / "eumetsat/cloud_type_samples",
        "enable_attr": "enable_ml_eumetsat_cloud_type",
    },
    "land_surface_temperature": {
        "status_key": "eumetsat_land_surface_temperature",
        "collection_id": "EO:EUM:DAT:1088",
        "raw_root": DEFAULT_ML_ROOT / "eumetsat/raw/land_surface_temperature",
        "samples_root": DEFAULT_ML_ROOT / "eumetsat/land_surface_temperature_samples",
        "enable_attr": "enable_ml_eumetsat_land_surface_temperature",
    },
    "global_instability_indices": {
        "status_key": "eumetsat_global_instability_indices",
        "collection_id": "EO:EUM:DAT:0683",
        "raw_root": DEFAULT_ML_ROOT / "eumetsat/raw/global_instability_indices",
        "samples_root": DEFAULT_ML_ROOT / "eumetsat/global_instability_indices_samples",
        "enable_attr": "enable_ml_eumetsat_global_instability_indices",
    },
}
SHUTDOWN_REQUESTED = False
ACTIVE_PROCESS: subprocess.Popen[str] | None = None


class CommandFailed(RuntimeError):
    def __init__(self, message: str, result: dict[str, Any]):
        super().__init__(message)
        self.result = result

WINDNINJA_50M_ARTIFACTS = {
    "arome_layer": "visualizations/wind2d/arome-corsica-latest.json",
    "arome_layer_gzip": "visualizations/wind2d/arome-corsica-latest.json.gz",
    "aromepi_layer": "visualizations/wind2d/aromepi-corsica-latest.json",
    "aromepi_layer_gzip": "visualizations/wind2d/aromepi-corsica-latest.json.gz",
    "moloch_layer": "visualizations/wind2d/moloch-corsica-latest.json",
    "moloch_layer_gzip": "visualizations/wind2d/moloch-corsica-latest.json.gz",
    "icon2i_layer": "visualizations/wind2d/icon2i-corsica-latest.json",
    "icon2i_layer_gzip": "visualizations/wind2d/icon2i-corsica-latest.json.gz",
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


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_int_list(name: str, default: list[int]) -> list[int]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return list(default)
    parsed = []
    for token in re.split(r"[, ]+", value.strip()):
        if not token:
            continue
        try:
            parsed.append(int(token))
        except ValueError:
            return list(default)
    return parsed or list(default)


def request_shutdown(signum: int, _frame: Any) -> None:
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    print(f"shutdown requested by signal {signum}", flush=True)
    if ACTIVE_PROCESS and ACTIVE_PROCESS.poll() is None:
        ACTIVE_PROCESS.terminate()


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
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
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


def utc_datetime_or_none(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return parse_utc_datetime(str(value))
    except (TypeError, ValueError):
        return None


def seconds_until(target: datetime | None, now: datetime) -> int:
    if target is None:
        return 0
    return max(0, int((target - now).total_seconds()))


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def scheduled_runs_between(source: str, start: datetime, end: datetime) -> list[datetime]:
    profile = PUBLICATION_PROFILES[source]
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    day = (start - timedelta(days=1)).date()
    last_day = (end + timedelta(days=1)).date()
    runs: list[datetime] = []
    while day <= last_day:
        for hour in profile["run_hours_utc"]:
            run = datetime(day.year, day.month, day.day, int(hour), tzinfo=timezone.utc)
            if start <= run <= end:
                runs.append(run)
        day += timedelta(days=1)
    return sorted(runs)


def usable_publication_delays(source: str, source_state: dict[str, Any]) -> list[int]:
    profile = PUBLICATION_PROFILES[source]
    delays: list[int] = []
    for item in source_state.get("publication_history") or []:
        if not item.get("usable_for_schedule", True):
            continue
        delay = item.get("delay_after_run_sec")
        try:
            delay_int = int(delay)
        except (TypeError, ValueError):
            continue
        if 0 <= delay_int <= int(profile["max_usable_delay_sec"]):
            delays.append(delay_int)
    return delays[-12:]


def learned_publication_delay_sec(source: str, source_state: dict[str, Any]) -> int:
    delays = usable_publication_delays(source, source_state)
    if not delays:
        return int(PUBLICATION_PROFILES[source]["default_delay_sec"])
    delays = sorted(delays)
    return delays[len(delays) // 2]


def publication_window(source: str, source_state: dict[str, Any], run_time: datetime) -> dict[str, Any]:
    profile = PUBLICATION_PROFILES[source]
    learned_delay = learned_publication_delay_sec(source, source_state)
    expected_publication = run_time + timedelta(seconds=learned_delay)
    start = expected_publication - timedelta(seconds=int(profile["fast_window_before_sec"]))
    end = expected_publication + timedelta(seconds=int(profile["fast_window_after_sec"]))
    return {
        "run_time_utc": iso_utc(run_time),
        "expected_publication_at_utc": iso_utc(expected_publication),
        "fast_window_start_utc": iso_utc(start),
        "fast_window_end_utc": iso_utc(end),
        "learned_delay_sec": learned_delay,
    }


def source_publication_schedule(source: str, source_state: dict[str, Any], now: datetime) -> dict[str, Any]:
    profile = PUBLICATION_PROFILES[source]
    delay = learned_publication_delay_sec(source, source_state)
    horizon_start = now - timedelta(days=2)
    horizon_end = now + timedelta(days=2)
    runs = scheduled_runs_between(source, horizon_start, horizon_end)
    expected_ready_runs = [run for run in runs if run + timedelta(seconds=delay) <= now]
    latest_expected = expected_ready_runs[-1] if expected_ready_runs else None
    future_runs = [run for run in runs if run + timedelta(seconds=delay) > now]
    next_expected = future_runs[0] if future_runs else None
    latest_window = publication_window(source, source_state, latest_expected) if latest_expected else None
    next_window = publication_window(source, source_state, next_expected) if next_expected else None
    last_seen = utc_datetime_or_none(source_state.get("last_seen_run_time_utc"))
    missing_expected = bool(latest_expected and (last_seen is None or last_seen < latest_expected))
    delayed = False
    if missing_expected and latest_window:
        delayed = now > parse_utc_datetime(latest_window["fast_window_end_utc"])
    return {
        "profile": {
            "run_hours_utc": list(profile["run_hours_utc"]),
            "default_delay_sec": profile["default_delay_sec"],
            "delayed_poll_interval_sec": profile["delayed_poll_interval_sec"],
        },
        "last_seen_run_time_utc": iso_utc(last_seen) if last_seen else None,
        "latest_expected_run_time_utc": iso_utc(latest_expected) if latest_expected else None,
        "next_expected_run_time_utc": iso_utc(next_expected) if next_expected else None,
        "latest_window": latest_window,
        "next_window": next_window,
        "missing_expected_run": missing_expected,
        "publication_status": "delayed" if delayed else ("waiting_for_expected_run" if missing_expected else "on_time"),
    }


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
    run_time, _ = latest_complete_run(coverage_ids(response.text), "001")
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


def git_value(args: tuple[str, ...]) -> str | None:
    try:
        proc = subprocess.run(
            ("git", *args),
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def runtime_metadata() -> dict[str, Any]:
    dirty = git_value(("status", "--porcelain"))
    return {
        "root": str(ROOT),
        "container_root": os.getenv("CORSEWIND_CONTAINER_ROOT"),
        "host_root": os.getenv("CORSEWIND_HOST_ROOT"),
        "git_branch": git_value(("rev-parse", "--abbrev-ref", "HEAD")),
        "git_commit": git_value(("rev-parse", "HEAD")),
        "git_dirty": bool(dirty),
    }


def run_command(args: tuple[str, ...], dry_run: bool, timeout_sec: int | None = None) -> dict[str, Any]:
    global ACTIVE_PROCESS
    cmd = command_line(args)
    printable = printable_command(cmd)
    started = time.time()
    if dry_run:
        print(f"dry-run: {printable}", flush=True)
        return {"cmd": printable, "status": "dry_run", "elapsed_s": 0.0}

    print(f"running: {printable}", flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ACTIVE_PROCESS = proc
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate()
    finally:
        if ACTIVE_PROCESS is proc:
            ACTIVE_PROCESS = None
    elapsed = round(time.time() - started, 3)
    stdout = stdout or ""
    stderr = stderr or ""
    if stdout:
        print(stdout, end="", flush=True)
    if stderr:
        print(stderr, end="", file=sys.stderr, flush=True)
    result = {
        "cmd": printable,
        "status": "timeout" if timed_out else ("pass" if proc.returncode == 0 else "fail"),
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }
    if timed_out:
        result["timeout_sec"] = timeout_sec
        if SHUTDOWN_REQUESTED:
            raise RuntimeError(f"command stopped during shutdown: {printable}")
        raise CommandFailed(f"command timed out after {timeout_sec}s: {printable}", result)
    if proc.returncode != 0:
        if SHUTDOWN_REQUESTED:
            raise RuntimeError(f"command stopped during shutdown: {printable}")
        raise CommandFailed(f"command failed: {printable}", result)
    return result


def cleanup_raw_args(args: argparse.Namespace) -> tuple[str, ...]:
    return ("--cleanup-raw",) if args.cleanup_raw else ("--no-cleanup-raw",)


def arome_refresh_command(lead_hours: tuple[str, ...], request_sleep_sec: float, cleanup_raw: bool) -> tuple[str, ...]:
    return (
        "scripts/build_arome_corsica_wind_layer.py",
        "--lead-hours",
        *lead_hours,
        "--request-sleep-sec",
        str(request_sleep_sec),
        "--cleanup-raw" if cleanup_raw else "--no-cleanup-raw",
    )


def arome_pi_refresh_command(args: argparse.Namespace) -> tuple[str, ...]:
    return (
        "scripts/build_aromepi_corsica_wind_layer.py",
        "--horizon-hours",
        str(args.aromepi_horizon_hours),
        "--request-sleep-sec",
        str(args.aromepi_request_sleep_sec),
        *cleanup_raw_args(args),
    )


def moloch_refresh_command(source: str | None, dataset: str, lead_hours: tuple[str, ...], cleanup_raw: bool) -> tuple[str, ...]:
    source_args = ("--input", source) if source else ()
    lead_args = ("--lead-hours", *lead_hours) if lead_hours else ()
    return (
        "scripts/build_moloch_corsica_wind_layer.py",
        *source_args,
        "--dataset",
        dataset,
        *lead_args,
        "--cleanup-raw" if cleanup_raw else "--no-cleanup-raw",
    )


def icon2i_refresh_command(source: str | None, dataset: str, lead_hours: tuple[str, ...], cleanup_raw: bool) -> tuple[str, ...]:
    source_args = ("--input", source) if source else ()
    lead_args = ("--lead-hours", *lead_hours) if lead_hours else ()
    return (
        "scripts/build_icon2i_corsica_wind_layer.py",
        *source_args,
        "--dataset",
        dataset,
        *lead_args,
        "--cleanup-raw" if cleanup_raw else "--no-cleanup-raw",
    )


def compress_wind2d_json_command() -> tuple[str, ...]:
    return ("scripts/compress_wind2d_json.py",)


def publish_wind2d_json(args: argparse.Namespace, commands: list[dict[str, Any]], reason: str) -> None:
    result = run_command(compress_wind2d_json_command(), args.dry_run)
    result["reason"] = reason
    commands.append(result)


def archive_model_layer_command(source: str, output_root: Path) -> tuple[str, ...]:
    return (
        "scripts/ml_dataset/archive_model_layer_snapshot.py",
        "--source",
        source,
        "--input",
        str(SOURCE_LAYER_PATHS[source].relative_to(ROOT)),
        "--output-root",
        str(output_root),
    )


def sample_model_layer_command(source: str, output_root: Path) -> tuple[str, ...]:
    return (
        "scripts/ml_dataset/sample_model_layers_at_spots.py",
        "--source",
        source,
        "--input",
        str(SOURCE_LAYER_PATHS[source].relative_to(ROOT)),
        "--output-root",
        str(output_root),
    )


def collect_nwp_extra_fields_command(source: str, args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "scripts/ml_dataset/collect_meteo_france_nwp_spot_features.py",
        "--source",
        source,
        "--input",
        str(SOURCE_LAYER_PATHS[source].relative_to(ROOT)),
        "--raw-root",
        str(args.ml_nwp_extra_fields_raw_root),
        "--output-root",
        str(args.ml_nwp_extra_fields_samples_root),
        "--max-steps",
        str(args.ml_nwp_extra_fields_max_steps),
        "--request-sleep-sec",
        str(args.ml_nwp_extra_fields_request_sleep_sec),
    ]
    if args.ml_nwp_extra_fields_include_context_spots:
        command.append("--include-context-spots")
    return tuple(command)


def collect_nwp_extra_fields_if_needed(
    source: str,
    args: argparse.Namespace,
    status: dict[str, Any],
    commands: list[dict[str, Any]],
) -> None:
    source_status = status.get("sources", {}).get(source, {})
    if not args.enable_ml_nwp_extra_fields or source not in {"arome", "aromepi"}:
        return
    if source_status.get("status") not in {"updated", "unchanged"}:
        return
    try:
        result = run_command(collect_nwp_extra_fields_command(source, args), args.dry_run)
        commands.append(result)
    except CommandFailed as exc:
        commands.append(exc.result)
        source_status["ml_nwp_extra_fields"] = {
            "enabled": True,
            "status": "failed",
            "raw_root": str(args.ml_nwp_extra_fields_raw_root),
            "samples_root": str(args.ml_nwp_extra_fields_samples_root),
            "command": {
                "cmd": exc.result.get("cmd"),
                "elapsed_s": exc.result.get("elapsed_s"),
                "returncode": exc.result.get("returncode"),
            },
            "error": str(exc),
        }
        return
    parsed = parse_command_json(result) or {}
    source_status["ml_nwp_extra_fields"] = {
        "enabled": True,
        "status": result.get("status"),
        "raw_root": str(args.ml_nwp_extra_fields_raw_root),
        "samples_root": str(args.ml_nwp_extra_fields_samples_root),
        "row_count": parsed.get("row_count"),
        "step_count": parsed.get("step_count"),
        "features": parsed.get("features"),
        "command": {
            "cmd": result.get("cmd"),
            "elapsed_s": result.get("elapsed_s"),
            "returncode": result.get("returncode"),
        },
    }


def collect_nwp_vertical_profiles_command(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "scripts/ml_dataset/collect_meteo_france_vertical_profiles.py",
        "--input",
        str(AROME_LAYER.relative_to(ROOT)),
        "--raw-root",
        str(args.ml_nwp_vertical_profiles_raw_root),
        "--output-root",
        str(args.ml_nwp_vertical_profiles_samples_root),
        "--max-steps",
        str(args.ml_nwp_vertical_profiles_max_steps),
        "--request-sleep-sec",
        str(args.ml_nwp_vertical_profiles_request_sleep_sec),
    ]
    for level in args.ml_nwp_vertical_profiles_pressure_levels_hpa:
        command.extend(["--pressure-level-hpa", str(level)])
    if args.ml_nwp_vertical_profiles_include_context_spots:
        command.append("--include-context-spots")
    return tuple(command)


def collect_nwp_vertical_profiles_if_needed(
    args: argparse.Namespace,
    status: dict[str, Any],
    commands: list[dict[str, Any]],
) -> None:
    source_status = status.get("sources", {}).get("arome", {})
    if not args.enable_ml_nwp_vertical_profiles:
        return
    if source_status.get("status") not in {"updated", "unchanged"}:
        return
    try:
        result = run_command(collect_nwp_vertical_profiles_command(args), args.dry_run)
        commands.append(result)
    except CommandFailed as exc:
        commands.append(exc.result)
        source_status["ml_nwp_vertical_profiles"] = {
            "enabled": True,
            "status": "failed",
            "raw_root": str(args.ml_nwp_vertical_profiles_raw_root),
            "samples_root": str(args.ml_nwp_vertical_profiles_samples_root),
            "pressure_levels_hpa": list(args.ml_nwp_vertical_profiles_pressure_levels_hpa),
            "command": {
                "cmd": exc.result.get("cmd"),
                "elapsed_s": exc.result.get("elapsed_s"),
                "returncode": exc.result.get("returncode"),
            },
            "error": str(exc),
        }
        return
    parsed = parse_command_json(result) or {}
    source_status["ml_nwp_vertical_profiles"] = {
        "enabled": True,
        "status": result.get("status"),
        "raw_root": str(args.ml_nwp_vertical_profiles_raw_root),
        "samples_root": str(args.ml_nwp_vertical_profiles_samples_root),
        "row_count": parsed.get("row_count"),
        "step_count": parsed.get("step_count"),
        "features": parsed.get("features"),
        "pressure_levels_hpa": parsed.get("pressure_levels_hpa"),
        "command": {
            "cmd": result.get("cmd"),
            "elapsed_s": result.get("elapsed_s"),
            "returncode": result.get("returncode"),
        },
    }


def build_ml_feature_store_command(args: argparse.Namespace) -> tuple[str, ...]:
    return (
        "scripts/ml_dataset/build_spot_feature_store.py",
        "--ml-root",
        str(args.ml_root),
        "--registry",
        str(args.ml_registry),
        "--output-root",
        str(args.ml_feature_store_root),
    )


def build_ml_feature_store_if_needed(
    args: argparse.Namespace,
    status: dict[str, Any],
    commands: list[dict[str, Any]],
) -> None:
    status["ml_feature_store"] = {
        "enabled": bool(args.enable_ml_feature_store),
        "root": str(args.ml_feature_store_root),
    }
    if not args.enable_ml_feature_store:
        status["ml_feature_store"]["status"] = "disabled"
        return
    try:
        result = run_command(build_ml_feature_store_command(args), args.dry_run)
        commands.append(result)
    except CommandFailed as exc:
        commands.append(exc.result)
        status["ml_feature_store"].update({
            "status": "failed",
            "command": {
                "cmd": exc.result.get("cmd"),
                "elapsed_s": exc.result.get("elapsed_s"),
                "returncode": exc.result.get("returncode"),
            },
            "error": str(exc),
        })
        return
    parsed = parse_command_json(result) or {}
    status["ml_feature_store"].update({
        "status": result.get("status"),
        "row_count": parsed.get("row_count"),
        "spot_count": parsed.get("spots"),
        "first_target_time_utc": parsed.get("first_target_time_utc"),
        "last_target_time_utc": parsed.get("last_target_time_utc"),
        "outputs": parsed.get("outputs"),
        "command": {
            "cmd": result.get("cmd"),
            "elapsed_s": result.get("elapsed_s"),
            "returncode": result.get("returncode"),
        },
    })


def copernicus_sst_window(args: argparse.Namespace, now: datetime) -> tuple[datetime, datetime]:
    if args.ml_copernicus_sst_start_datetime and args.ml_copernicus_sst_end_datetime:
        return (
            parse_utc_datetime(args.ml_copernicus_sst_start_datetime),
            parse_utc_datetime(args.ml_copernicus_sst_end_datetime),
        )
    end = now - timedelta(hours=args.ml_copernicus_sst_end_lag_hours)
    end = end.replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=args.ml_copernicus_sst_window_hours - 1)
    return start, end


def copernicus_cli_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def copernicus_sst_filename(start: datetime, end: datetime) -> str:
    return f"sst_corse_{start:%Y%m%dT%H}_{end:%Y%m%dT%H}.nc"


def collect_copernicus_sst_command(args: argparse.Namespace, now: datetime) -> tuple[str, ...]:
    start, end = copernicus_sst_window(args, now)
    command = [
        "scripts/ml_dataset/collect_copernicus_marine_sst.py",
        "--start-datetime",
        copernicus_cli_datetime(start),
        "--end-datetime",
        copernicus_cli_datetime(end),
        "--output-filename",
        copernicus_sst_filename(start, end),
        "--raw-root",
        str(args.ml_copernicus_sst_raw_root),
        "--output-root",
        str(args.ml_copernicus_sst_samples_root),
        "--log-level",
        args.ml_copernicus_sst_log_level,
    ]
    if args.ml_copernicus_sst_include_context_spots:
        command.append("--include-context-spots")
    return tuple(command)


def collect_copernicus_sst_if_needed(
    args: argparse.Namespace,
    status: dict[str, Any],
    commands: list[dict[str, Any]],
    now: datetime,
) -> None:
    status["copernicus_marine_sst"] = {
        "enabled": bool(args.enable_ml_copernicus_sst),
        "raw_root": str(args.ml_copernicus_sst_raw_root),
        "samples_root": str(args.ml_copernicus_sst_samples_root),
    }
    if not args.enable_ml_copernicus_sst:
        status["copernicus_marine_sst"]["status"] = "disabled"
        return
    start, end = copernicus_sst_window(args, now)
    status["copernicus_marine_sst"].update({
        "status": "pending",
        "start_datetime_utc": iso_utc(start),
        "end_datetime_utc": iso_utc(end),
    })
    try:
        result = run_command(collect_copernicus_sst_command(args, now), args.dry_run)
        commands.append(result)
    except CommandFailed as exc:
        commands.append(exc.result)
        status["copernicus_marine_sst"].update({
            "status": "failed",
            "command": {
                "cmd": exc.result.get("cmd"),
                "elapsed_s": exc.result.get("elapsed_s"),
                "returncode": exc.result.get("returncode"),
            },
            "error": str(exc),
        })
        return
    status["copernicus_marine_sst"].update({
        "status": result.get("status"),
        "command": {
            "cmd": result.get("cmd"),
            "elapsed_s": result.get("elapsed_s"),
            "returncode": result.get("returncode"),
        },
    })


def eumetsat_cloud_mask_window(args: argparse.Namespace, now: datetime) -> tuple[datetime, datetime]:
    if args.ml_eumetsat_cloud_mask_start_datetime and args.ml_eumetsat_cloud_mask_end_datetime:
        return (
            parse_utc_datetime(args.ml_eumetsat_cloud_mask_start_datetime),
            parse_utc_datetime(args.ml_eumetsat_cloud_mask_end_datetime),
        )
    end = now - timedelta(minutes=args.ml_eumetsat_cloud_mask_end_lag_minutes)
    end = end.replace(second=0, microsecond=0)
    start = end - timedelta(minutes=args.ml_eumetsat_cloud_mask_window_minutes)
    return start, end


def eumetsat_cli_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def collect_eumetsat_cloud_mask_command(args: argparse.Namespace, now: datetime) -> tuple[str, ...]:
    start, end = eumetsat_cloud_mask_window(args, now)
    command = [
        "scripts/ml_dataset/collect_eumetsat_cloud_mask.py",
        "--collection-id",
        args.ml_eumetsat_cloud_mask_collection_id,
        "--start-datetime",
        eumetsat_cli_datetime(start),
        "--end-datetime",
        eumetsat_cli_datetime(end),
        "--bbox",
        args.ml_eumetsat_cloud_mask_bbox,
        "--raw-root",
        str(args.ml_eumetsat_cloud_mask_raw_root),
        "--output-root",
        str(args.ml_eumetsat_cloud_mask_samples_root),
        "--max-products",
        str(args.ml_eumetsat_cloud_mask_max_products),
        "--radius-cells",
        str(args.ml_eumetsat_cloud_mask_radius_cells),
    ]
    if args.ml_eumetsat_cloud_mask_include_context_spots:
        command.append("--include-context-spots")
    if not args.ml_eumetsat_cloud_mask_quality_flags:
        command.append("--no-quality-flags")
    return tuple(command)


def eumetsat_thermal_window(args: argparse.Namespace, now: datetime) -> tuple[datetime, datetime]:
    if args.ml_eumetsat_thermal_start_datetime and args.ml_eumetsat_thermal_end_datetime:
        return (
            parse_utc_datetime(args.ml_eumetsat_thermal_start_datetime),
            parse_utc_datetime(args.ml_eumetsat_thermal_end_datetime),
        )
    end = now - timedelta(minutes=args.ml_eumetsat_thermal_end_lag_minutes)
    end = end.replace(second=0, microsecond=0)
    start = end - timedelta(minutes=args.ml_eumetsat_thermal_window_minutes)
    return start, end


def eumetsat_thermal_product_enabled(product: str, args: argparse.Namespace) -> bool:
    config = EUMETSAT_THERMAL_PRODUCTS[product]
    return bool(args.enable_ml_eumetsat_thermal_products or getattr(args, config["enable_attr"]))


def eumetsat_product_collection_id(product: str, args: argparse.Namespace) -> str:
    return str(getattr(args, f"ml_eumetsat_{product}_collection_id"))


def eumetsat_product_root(product: str, args: argparse.Namespace, kind: str) -> Path:
    return getattr(args, f"ml_eumetsat_{product}_{kind}_root")


def collect_eumetsat_spot_product_command(product: str, args: argparse.Namespace, now: datetime) -> tuple[str, ...]:
    start, end = eumetsat_thermal_window(args, now)
    command = [
        "scripts/ml_dataset/collect_eumetsat_spot_product.py",
        "--product",
        product,
        "--collection-id",
        eumetsat_product_collection_id(product, args),
        "--start-datetime",
        eumetsat_cli_datetime(start),
        "--end-datetime",
        eumetsat_cli_datetime(end),
        "--bbox",
        args.ml_eumetsat_thermal_bbox,
        "--raw-root",
        str(eumetsat_product_root(product, args, "raw")),
        "--output-root",
        str(eumetsat_product_root(product, args, "samples")),
        "--max-products",
        str(args.ml_eumetsat_thermal_max_products),
        "--max-variables",
        str(args.ml_eumetsat_thermal_max_variables),
        "--radius-cells",
        str(args.ml_eumetsat_thermal_radius_cells),
    ]
    if args.ml_eumetsat_thermal_include_context_spots:
        command.append("--include-context-spots")
    return tuple(command)


def default_external_data_state() -> dict[str, Any]:
    return {
        "last_poll_at_utc": None,
        "last_success_at_utc": None,
        "last_failure_at_utc": None,
        "last_window_start_utc": None,
        "last_window_end_utc": None,
        "last_error": None,
        "consecutive_failures": 0,
    }


def ensure_external_data_states(state: dict[str, Any]) -> None:
    external = state.setdefault("external_data", {})
    for source in ("eumetsat_cloud_mask", *[config["status_key"] for config in EUMETSAT_THERMAL_PRODUCTS.values()]):
        current = external.setdefault(source, default_external_data_state())
        for key, value in default_external_data_state().items():
            current.setdefault(key, value)


def external_data_state(state: dict[str, Any], source: str) -> dict[str, Any]:
    ensure_external_data_states(state)
    return state["external_data"][source]


def external_data_poll_decision(
    source_state: dict[str, Any],
    interval_sec: int,
    args: argparse.Namespace,
    now: datetime,
) -> dict[str, Any]:
    if args.dry_run:
        return {"due": True, "reason": "dry_run", "next_due_at_utc": iso_utc(now), "interval_sec": interval_sec}
    backoff = source_backoff_sec(source_state, args)
    last_failure = utc_datetime_or_none(source_state.get("last_failure_at_utc"))
    if backoff and last_failure:
        next_retry = last_failure + timedelta(seconds=backoff)
        if now < next_retry:
            return {
                "due": False,
                "reason": "error_backoff",
                "backoff_sec": backoff,
                "next_due_at_utc": iso_utc(next_retry),
                "interval_sec": interval_sec,
            }
    last_poll = utc_datetime_or_none(source_state.get("last_poll_at_utc"))
    if last_poll is None:
        return {"due": True, "reason": "never_polled", "interval_sec": interval_sec}
    next_poll = last_poll + timedelta(seconds=interval_sec)
    if now >= next_poll:
        return {"due": True, "reason": "interval_elapsed", "next_due_at_utc": iso_utc(next_poll), "interval_sec": interval_sec}
    return {"due": False, "reason": "not_due", "next_due_at_utc": iso_utc(next_poll), "interval_sec": interval_sec}


def parse_command_json(result: dict[str, Any]) -> dict[str, Any] | None:
    text = str(result.get("stdout_tail") or "").strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def collect_eumetsat_cloud_mask_if_needed(
    args: argparse.Namespace,
    state: dict[str, Any],
    status: dict[str, Any],
    commands: list[dict[str, Any]],
    now: datetime,
) -> None:
    source_state = external_data_state(state, "eumetsat_cloud_mask")
    start, end = eumetsat_cloud_mask_window(args, now)
    decision = external_data_poll_decision(
        source_state,
        args.ml_eumetsat_cloud_mask_poll_interval_sec,
        args,
        now,
    )
    status["eumetsat_cloud_mask"] = {
        "enabled": bool(args.enable_ml_eumetsat_cloud_mask),
        "decision": decision,
        "collection_id": args.ml_eumetsat_cloud_mask_collection_id,
        "bbox": args.ml_eumetsat_cloud_mask_bbox,
        "raw_root": str(args.ml_eumetsat_cloud_mask_raw_root),
        "samples_root": str(args.ml_eumetsat_cloud_mask_samples_root),
        "start_datetime_utc": iso_utc(start),
        "end_datetime_utc": iso_utc(end),
        "max_products": args.ml_eumetsat_cloud_mask_max_products,
        "last_success_at_utc": source_state.get("last_success_at_utc"),
    }
    if not args.enable_ml_eumetsat_cloud_mask:
        status["eumetsat_cloud_mask"]["status"] = "disabled"
        return
    if not decision.get("due"):
        status["eumetsat_cloud_mask"]["status"] = "skipped"
        return
    status["eumetsat_cloud_mask"]["status"] = "pending"
    if not args.dry_run:
        source_state["last_poll_at_utc"] = utc_now()
        source_state["last_window_start_utc"] = iso_utc(start)
        source_state["last_window_end_utc"] = iso_utc(end)
    try:
        result = run_command(collect_eumetsat_cloud_mask_command(args, now), args.dry_run)
        commands.append(result)
    except CommandFailed as exc:
        commands.append(exc.result)
        if not args.dry_run:
            source_state["last_failure_at_utc"] = utc_now()
            source_state["last_error"] = str(exc)
            source_state["consecutive_failures"] = int(source_state.get("consecutive_failures") or 0) + 1
        status["eumetsat_cloud_mask"].update({
            "status": "failed",
            "command": {
                "cmd": exc.result.get("cmd"),
                "elapsed_s": exc.result.get("elapsed_s"),
                "returncode": exc.result.get("returncode"),
            },
            "error": str(exc),
        })
        return
    parsed = parse_command_json(result) or {}
    if not args.dry_run:
        source_state["last_success_at_utc"] = utc_now()
        source_state["last_failure_at_utc"] = None
        source_state["last_error"] = None
        source_state["consecutive_failures"] = 0
    status["eumetsat_cloud_mask"].update({
        "status": result.get("status"),
        "command": {
            "cmd": result.get("cmd"),
            "elapsed_s": result.get("elapsed_s"),
            "returncode": result.get("returncode"),
        },
        "product_count": parsed.get("product_count"),
        "spot_count": parsed.get("spot_count"),
        "row_count": parsed.get("row_count"),
        "written": parsed.get("written"),
    })


def collect_eumetsat_spot_product_if_needed(
    product: str,
    args: argparse.Namespace,
    state: dict[str, Any],
    status: dict[str, Any],
    commands: list[dict[str, Any]],
    now: datetime,
) -> None:
    product_config = EUMETSAT_THERMAL_PRODUCTS[product]
    status_key = product_config["status_key"]
    source_state = external_data_state(state, status_key)
    start, end = eumetsat_thermal_window(args, now)
    decision = external_data_poll_decision(
        source_state,
        args.ml_eumetsat_thermal_poll_interval_sec,
        args,
        now,
    )
    enabled = eumetsat_thermal_product_enabled(product, args)
    status[status_key] = {
        "enabled": enabled,
        "decision": decision,
        "product": product,
        "collection_id": eumetsat_product_collection_id(product, args),
        "bbox": args.ml_eumetsat_thermal_bbox,
        "raw_root": str(eumetsat_product_root(product, args, "raw")),
        "samples_root": str(eumetsat_product_root(product, args, "samples")),
        "start_datetime_utc": iso_utc(start),
        "end_datetime_utc": iso_utc(end),
        "max_products": args.ml_eumetsat_thermal_max_products,
        "last_success_at_utc": source_state.get("last_success_at_utc"),
    }
    if not enabled:
        status[status_key]["status"] = "disabled"
        return
    if not decision.get("due"):
        status[status_key]["status"] = "skipped"
        return
    status[status_key]["status"] = "pending"
    if not args.dry_run:
        source_state["last_poll_at_utc"] = utc_now()
        source_state["last_window_start_utc"] = iso_utc(start)
        source_state["last_window_end_utc"] = iso_utc(end)
    try:
        result = run_command(collect_eumetsat_spot_product_command(product, args, now), args.dry_run)
        commands.append(result)
    except CommandFailed as exc:
        commands.append(exc.result)
        if not args.dry_run:
            source_state["last_failure_at_utc"] = utc_now()
            source_state["last_error"] = str(exc)
            source_state["consecutive_failures"] = int(source_state.get("consecutive_failures") or 0) + 1
        status[status_key].update({
            "status": "failed",
            "command": {
                "cmd": exc.result.get("cmd"),
                "elapsed_s": exc.result.get("elapsed_s"),
                "returncode": exc.result.get("returncode"),
            },
            "error": str(exc),
        })
        return
    parsed = parse_command_json(result) or {}
    if not args.dry_run:
        source_state["last_success_at_utc"] = utc_now()
        source_state["last_failure_at_utc"] = None
        source_state["last_error"] = None
        source_state["consecutive_failures"] = 0
    status[status_key].update({
        "status": result.get("status"),
        "command": {
            "cmd": result.get("cmd"),
            "elapsed_s": result.get("elapsed_s"),
            "returncode": result.get("returncode"),
        },
        "product_count": parsed.get("product_count"),
        "spot_count": parsed.get("spot_count"),
        "row_count": parsed.get("row_count"),
        "sampled_variables": parsed.get("sampled_variables"),
        "written": parsed.get("written"),
    })


def archive_source_snapshot_if_needed(
    source: str,
    args: argparse.Namespace,
    status: dict[str, Any],
    commands: list[dict[str, Any]],
) -> None:
    source_status = status.get("sources", {}).get(source, {})
    if not args.enable_ml_dataset_archive:
        return
    if source_status.get("status") not in {"updated", "unchanged"}:
        return
    try:
        result = run_command(archive_model_layer_command(source, args.ml_dataset_root), args.dry_run)
        commands.append(result)
    except CommandFailed as exc:
        commands.append(exc.result)
        source_status["ml_dataset_archive"] = {
            "enabled": True,
            "status": "failed",
            "output_root": str(args.ml_dataset_root),
            "command": {
                "cmd": exc.result.get("cmd"),
                "elapsed_s": exc.result.get("elapsed_s"),
                "returncode": exc.result.get("returncode"),
            },
            "error": str(exc),
        }
        return
    source_status["ml_dataset_archive"] = {
        "enabled": True,
        "status": result.get("status"),
        "output_root": str(args.ml_dataset_root),
        "command": {
            "cmd": result.get("cmd"),
            "elapsed_s": result.get("elapsed_s"),
            "returncode": result.get("returncode"),
        },
    }
    samples_root = args.ml_dataset_samples_root
    try:
        sample_result = run_command(sample_model_layer_command(source, samples_root), args.dry_run)
        commands.append(sample_result)
    except CommandFailed as exc:
        commands.append(exc.result)
        source_status["ml_dataset_samples"] = {
            "enabled": True,
            "status": "failed",
            "output_root": str(samples_root),
            "command": {
                "cmd": exc.result.get("cmd"),
                "elapsed_s": exc.result.get("elapsed_s"),
                "returncode": exc.result.get("returncode"),
            },
            "error": str(exc),
        }
        return
    source_status["ml_dataset_samples"] = {
        "enabled": True,
        "status": sample_result.get("status"),
        "output_root": str(samples_root),
        "command": {
            "cmd": sample_result.get("cmd"),
            "elapsed_s": sample_result.get("elapsed_s"),
            "returncode": sample_result.get("returncode"),
        },
    }


# Models whose colour overlay is served as pre-baked raster tiles (Google-Maps style) so the
# Wind2D client renders instantly instead of computing the field per pixel in JavaScript.
RASTER_TILE_MODELS = ("arome", "aromepi", "moloch", "icon2i")
RASTER_TILE_EXPECTED_FORMAT = "corsewind.model.raster_tiles.v2"
RASTER_TILE_EXPECTED_ZOOMS = [8, 9, 10]
RASTER_TILE_EXPECTED_RENDER_SCALE = 1
RASTER_TILE_EXPECTED_WEBP_METHOD = 2
RASTER_TILE_QUEUE: queue.Queue[str] = queue.Queue(maxsize=len(RASTER_TILE_MODELS))
RASTER_TILE_QUEUE_LOCK = threading.Lock()
RASTER_TILE_QUEUED_MODELS: set[str] = set()
RASTER_TILE_RUNNING_MODELS: set[str] = set()
RASTER_TILE_RERUN_MODELS: set[str] = set()
RASTER_TILE_WORKER_STARTED = False


def model_raster_tiles_command(model: str) -> tuple[str, ...]:
    return ("scripts/build_model_raster_tiles.py", "--model", model)


def source_layer_has_cloud_rain(layer_path: Path | None) -> bool:
    if not layer_path or not layer_path.exists():
        return False
    try:
        payload = json.loads(layer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return any(
        step.get("cloud_cover_pct") is not None or step.get("precipitation_mm") is not None
        for step in payload.get("forecast_steps") or []
    )


def raster_manifest_needs_rebuild(manifest_path: Path, layer_path: Path | None = None) -> bool:
    if not manifest_path.exists():
        return True
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    # Rebuild older manifests on deploy: v2 adds per-step source hashes (incremental rebuilds),
    # gust_data tiles, and set pruning — older formats miss those.
    if manifest.get("format") != RASTER_TILE_EXPECTED_FORMAT:
        return True
    # Rebuild older PNG tiles into the smaller WebP format on deploy (old manifests have no
    # tileFormat field, or "png").
    if manifest.get("tileFormat") != "webp":
        return True
    if manifest.get("zooms") != RASTER_TILE_EXPECTED_ZOOMS:
        return True
    if manifest.get("renderScale") != RASTER_TILE_EXPECTED_RENDER_SCALE:
        return True
    if manifest.get("webpMethod") != RASTER_TILE_EXPECTED_WEBP_METHOD:
        return True
    if source_layer_has_cloud_rain(layer_path) and "cloud_rain" not in set(manifest.get("modes") or []):
        return True
    steps = manifest.get("steps") or []
    keys = [step.get("key") for step in steps if step.get("key")]
    return not keys or len(keys) != len(set(keys))


def raster_tile_queue_worker() -> None:
    while True:
        model = RASTER_TILE_QUEUE.get()
        with RASTER_TILE_QUEUE_LOCK:
            RASTER_TILE_QUEUED_MODELS.discard(model)
            RASTER_TILE_RUNNING_MODELS.add(model)
        print(f"bootstrapping raster tiles for {model}", flush=True)
        cmd = command_line(model_raster_tiles_command(model))
        try:
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
        except Exception as exc:
            print(f"failed to bootstrap raster tiles for {model}: {exc}", flush=True)
        else:
            if proc.stdout:
                print(proc.stdout, end="", flush=True)
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr, flush=True)
            if proc.returncode != 0:
                print(f"failed to bootstrap raster tiles for {model}: exit {proc.returncode}", flush=True)
        finally:
            with RASTER_TILE_QUEUE_LOCK:
                RASTER_TILE_RUNNING_MODELS.discard(model)
                rerun = model in RASTER_TILE_RERUN_MODELS
                RASTER_TILE_RERUN_MODELS.discard(model)
                if rerun:
                    try:
                        RASTER_TILE_QUEUE.put_nowait(model)
                    except queue.Full:
                        print(f"raster tile queue full; dropped rerun for {model}", flush=True)
                    else:
                        RASTER_TILE_QUEUED_MODELS.add(model)
                        print(f"re-queued raster tiles for {model} after in-flight update", flush=True)
            RASTER_TILE_QUEUE.task_done()


def ensure_raster_tile_worker_started() -> None:
    global RASTER_TILE_WORKER_STARTED
    if RASTER_TILE_WORKER_STARTED:
        return
    threading.Thread(target=raster_tile_queue_worker, name="raster-tile-worker", daemon=True).start()
    RASTER_TILE_WORKER_STARTED = True


def enqueue_raster_tile_build(model: str, reason: str) -> dict[str, Any]:
    if model not in RASTER_TILE_MODELS:
        raise ValueError(f"Unknown raster tile model: {model}")
    with RASTER_TILE_QUEUE_LOCK:
        ensure_raster_tile_worker_started()
        if model in RASTER_TILE_QUEUED_MODELS:
            print(f"raster tiles for {model} already queued ({reason})", flush=True)
            return {"cmd": printable_command(command_line(model_raster_tiles_command(model))), "status": "already_queued", "model": model}
        if model in RASTER_TILE_RUNNING_MODELS:
            RASTER_TILE_RERUN_MODELS.add(model)
            print(f"raster tiles for {model} already running; queued rerun ({reason})", flush=True)
            return {"cmd": printable_command(command_line(model_raster_tiles_command(model))), "status": "queued_after_running", "model": model}
        try:
            RASTER_TILE_QUEUE.put_nowait(model)
        except queue.Full as exc:
            raise RuntimeError(f"raster tile queue is full; cannot queue {model}") from exc
        RASTER_TILE_QUEUED_MODELS.add(model)
    print(f"queued raster tiles for {model} ({reason})", flush=True)
    return {"cmd": printable_command(command_line(model_raster_tiles_command(model))), "status": "queued", "model": model}


def enqueue_raster_tiles_if_source_changed(
    source: str,
    args: argparse.Namespace,
    status: dict[str, Any],
    commands: list[dict[str, Any]],
    reason: str,
) -> None:
    if source not in RASTER_TILE_MODELS:
        return
    if not (args.dry_run or status["sources"].get(source, {}).get("changed")):
        return
    if args.dry_run:
        commands.append(run_command(model_raster_tiles_command(source), args.dry_run))
    else:
        commands.append(enqueue_raster_tile_build(source, reason))


def wait_for_raster_tile_queue() -> None:
    if RASTER_TILE_QUEUE.unfinished_tasks:
        print("waiting for queued raster tile builds", flush=True)
    RASTER_TILE_QUEUE.join()


def bootstrap_raster_tiles() -> None:
    """On startup, pre-bake colour tiles for any model whose forecast JSON already exists but
    whose tiles are missing or whose manifest is invalid (e.g. a fresh deploy or an older
    AROME-PI manifest with duplicate sub-hourly step keys). Runs each as a detached background
    process so it never blocks the poll loop or the container healthcheck — the web server serves
    each model's manifest as soon as it is written, and the client picks it up on its next poll.
    Steady-state regeneration on run changes is handled by the same bounded queue."""
    rebuild_models: list[str] = []
    for model in RASTER_TILE_MODELS:
        layer_path = SOURCE_LAYER_PATHS.get(model)
        manifest_path = ROOT / "visualizations/wind2d/tiles" / model / "manifest.json"
        if layer_path and layer_path.exists() and raster_manifest_needs_rebuild(manifest_path, layer_path):
            reason = "missing" if not manifest_path.exists() else "invalid"
            rebuild_models.append(model)
    for model in rebuild_models:
        enqueue_raster_tile_build(model, "bootstrap")


def is_aromepi_waiting_result(result: dict[str, Any]) -> bool:
    output = f"{result.get('stdout_tail') or ''}\n{result.get('stderr_tail') or ''}"
    waiting_markers = (
        "No complete AROME-PI hybrid run found",
        "No AROME-PI valid times available inside requested forecast horizon",
    )
    return any(marker in output for marker in waiting_markers)


def refresh_source(
    source: str,
    command: tuple[str, ...],
    args: argparse.Namespace,
    state: dict[str, Any],
    status: dict[str, Any],
    commands: list[dict[str, Any]],
    command_timeout_sec: int | None = None,
) -> dict[str, Any]:
    source_state = model_state(state, source)
    previous_completed = source_state.get("last_completed_run_time_utc")
    previous_layer_run = read_run_time(SOURCE_LAYER_PATHS[source])
    source_status = status.setdefault("sources", {}).setdefault(source, {})
    source_status.update(
        {
            "enabled": True,
            "status": "running",
            "previous_completed_run_time_utc": previous_completed,
            "previous_layer_run_time_utc": previous_layer_run,
        }
    )
    source_state["last_poll_at_utc"] = utc_now()
    try:
        command_result = run_command(command, args.dry_run, timeout_sec=command_timeout_sec)
        commands.append(command_result)
        current_run = previous_layer_run if args.dry_run else read_run_time(SOURCE_LAYER_PATHS[source])
        changed = bool(current_run and current_run != previous_completed)
        seen_at_utc = utc_now()
        source_state["last_seen_run_time_utc"] = current_run
        source_state["last_completed_run_time_utc"] = current_run
        source_state["last_success_at_utc"] = seen_at_utc
        source_state["last_error"] = None
        source_state["consecutive_failures"] = 0
        publication_observation = record_publication_observation(source, source_state, current_run, seen_at_utc)
        if changed:
            source_state["last_changed_at_utc"] = seen_at_utc
        source_status.update(
            {
                "status": "updated" if changed else "unchanged",
                "run_time_utc": current_run,
                "changed": changed,
                "command": command_result,
                "publication_observation": publication_observation,
                "publication_schedule": source_publication_schedule(source, source_state, datetime.now(timezone.utc)),
            }
        )
        return source_status
    except CommandFailed as exc:
        if source == "aromepi" and is_aromepi_waiting_result(exc.result):
            commands.append(exc.result)
            waited_at_utc = utc_now()
            current_run = read_run_time(SOURCE_LAYER_PATHS[source])
            source_state["last_waiting_at_utc"] = waited_at_utc
            source_state["last_error"] = str(exc)
            source_state["last_failure_at_utc"] = None
            source_state["consecutive_failures"] = 0
            source_state["last_seen_run_time_utc"] = current_run
            source_state["last_completed_run_time_utc"] = current_run
            source_status.update(
                {
                    "status": "waiting_for_complete_run",
                    "run_time_utc": current_run,
                    "changed": False,
                    "command": exc.result,
                    "reason": "aromepi_run_not_complete_yet",
                    "next_retry_hint_sec": args.aromepi_stale_poll_interval_sec,
                    "publication_schedule": source_publication_schedule(source, source_state, datetime.now(timezone.utc)),
                }
            )
            return source_status
        source_state["last_failure_at_utc"] = utc_now()
        source_state["last_error"] = str(exc)
        source_state["consecutive_failures"] = int(source_state.get("consecutive_failures") or 0) + 1
        current_run = read_run_time(SOURCE_LAYER_PATHS[source])
        source_status.update(
            {
                "status": "failed",
                "run_time_utc": current_run,
                "changed": False,
                "command": exc.result,
                "error": str(exc),
                "consecutive_failures": source_state["consecutive_failures"],
            }
        )
        if source == "arome" and not current_run:
            raise
        return source_status
    except Exception as exc:
        source_state["last_failure_at_utc"] = utc_now()
        source_state["last_error"] = str(exc)
        source_state["consecutive_failures"] = int(source_state.get("consecutive_failures") or 0) + 1
        current_run = read_run_time(SOURCE_LAYER_PATHS[source])
        source_status.update(
            {
                "status": "failed",
                "run_time_utc": current_run,
                "changed": False,
                "error": str(exc),
                "consecutive_failures": source_state["consecutive_failures"],
            }
        )
        if source == "arome" and not current_run:
            raise
        return source_status


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
    state = read_json(path) or {
        "format": "corsewind.forecast_update_engine.state.v1",
        "created_at_utc": utc_now(),
        "last_seen_run_time_utc": None,
        "last_completed_run_time_utc": None,
        "last_success_at_utc": None,
        "last_failure_at_utc": None,
        "consecutive_failures": 0,
        "history": [],
    }
    ensure_model_states(state)
    ensure_external_data_states(state)
    return state


def default_model_state() -> dict[str, Any]:
    return {
        "last_seen_run_time_utc": None,
        "last_completed_run_time_utc": None,
        "last_success_at_utc": None,
        "last_failure_at_utc": None,
        "last_poll_at_utc": None,
        "last_changed_at_utc": None,
        "last_error": None,
        "consecutive_failures": 0,
        "publication_history": [],
    }


def ensure_model_states(state: dict[str, Any]) -> None:
    models = state.setdefault("models", {})
    for source in SOURCE_LAYER_PATHS:
        current = models.setdefault(source, default_model_state())
        for key, value in default_model_state().items():
            current.setdefault(key, value)
    if state.get("last_seen_run_time_utc") and not models["arome"].get("last_seen_run_time_utc"):
        models["arome"]["last_seen_run_time_utc"] = state.get("last_seen_run_time_utc")
    if state.get("last_success_at_utc") and not models["arome"].get("last_success_at_utc"):
        models["arome"]["last_success_at_utc"] = state.get("last_success_at_utc")


def model_state(state: dict[str, Any], source: str) -> dict[str, Any]:
    ensure_model_states(state)
    return state["models"][source]


def source_payload_metadata(source: str) -> dict[str, Any]:
    payload = read_json(SOURCE_LAYER_PATHS[source]) or {}
    bundle = payload.get("source_bundle") or {}
    return {
        "source_file": payload.get("source_file"),
        "source_filename": bundle.get("filename") or Path(str(payload.get("source_file") or "")).name or None,
        "dataset_id": payload.get("dataset_id") or bundle.get("dataset_id"),
        "bundle_run_time_utc": bundle.get("run_time_utc"),
    }


def record_publication_observation(source: str, source_state: dict[str, Any], run_time_utc: str | None, seen_at_utc: str) -> dict[str, Any] | None:
    if not run_time_utc:
        return None
    seen_at = utc_datetime_or_none(seen_at_utc)
    run_time = utc_datetime_or_none(run_time_utc)
    if not seen_at or not run_time:
        return None
    history = list(source_state.get("publication_history") or [])
    existing = next((item for item in history if item.get("run_time_utc") == run_time_utc), None)
    if existing:
        return existing
    delay = int((seen_at - run_time).total_seconds())
    profile = PUBLICATION_PROFILES[source]
    metadata = source_payload_metadata(source)
    record = {
        "run_time_utc": run_time_utc,
        "first_seen_at_utc": seen_at_utc,
        "delay_after_run_sec": delay,
        "usable_for_schedule": 0 <= delay <= int(profile["max_usable_delay_sec"]),
        **{key: value for key, value in metadata.items() if value},
    }
    history.append(record)
    source_state["publication_history"] = history[-PUBLICATION_HISTORY_LIMIT:]
    return record


def source_enabled(source: str, args: argparse.Namespace) -> bool:
    if source == "arome":
        return True
    if source == "aromepi":
        return bool(args.enable_aromepi)
    if source == "moloch":
        return bool(args.enable_moloch)
    if source == "icon2i":
        return bool(args.enable_icon2i)
    return False


def source_poll_interval_sec(source: str, source_state: dict[str, Any], args: argparse.Namespace, now: datetime) -> int:
    if source == "arome":
        base_interval = args.arome_poll_interval_sec
    elif source == "aromepi":
        last_seen = utc_datetime_or_none(source_state.get("last_seen_run_time_utc"))
        if last_seen is None:
            base_interval = args.aromepi_stale_poll_interval_sec
        else:
            age_sec = (now - last_seen).total_seconds()
            base_interval = args.aromepi_stale_poll_interval_sec if age_sec >= args.aromepi_freshness_target_sec else args.aromepi_poll_interval_sec
    elif source == "moloch":
        base_interval = args.moloch_poll_interval_sec
    elif source == "icon2i":
        base_interval = args.icon2i_poll_interval_sec
    else:
        base_interval = args.poll_interval_sec

    schedule = source_publication_schedule(source, source_state, now)
    windows = [schedule.get("latest_window"), schedule.get("next_window")]
    last_seen = utc_datetime_or_none(source_state.get("last_seen_run_time_utc"))
    for window in windows:
        if not window:
            continue
        window_run = parse_utc_datetime(window["run_time_utc"])
        if last_seen is not None and last_seen >= window_run:
            continue
        start = parse_utc_datetime(window["fast_window_start_utc"])
        end = parse_utc_datetime(window["fast_window_end_utc"])
        if start <= now <= end:
            if source == "aromepi":
                return min(base_interval, args.aromepi_stale_poll_interval_sec)
            return min(base_interval, args.fast_window_poll_interval_sec)
    if schedule.get("publication_status") == "delayed":
        return min(base_interval, int(PUBLICATION_PROFILES[source]["delayed_poll_interval_sec"]))
    return base_interval


def source_backoff_sec(source_state: dict[str, Any], args: argparse.Namespace) -> int:
    failures = max(0, int(source_state.get("consecutive_failures") or 0))
    if failures <= 0:
        return 0
    return min(args.source_error_backoff_max_sec, args.source_error_backoff_sec * (2 ** (failures - 1)))


def source_poll_decision(source: str, source_state: dict[str, Any], args: argparse.Namespace, now: datetime) -> dict[str, Any]:
    if args.dry_run:
        schedule = source_publication_schedule(source, source_state, now)
        return {"due": True, "reason": "dry_run", "next_due_at_utc": iso_utc(now), "publication_schedule": schedule}
    schedule = source_publication_schedule(source, source_state, now)
    backoff = source_backoff_sec(source_state, args)
    last_failure = utc_datetime_or_none(source_state.get("last_failure_at_utc"))
    if backoff and last_failure:
        next_retry = last_failure + timedelta(seconds=backoff)
        if now < next_retry:
            return {
                "due": False,
                "reason": "error_backoff",
                "backoff_sec": backoff,
                "next_due_at_utc": iso_utc(next_retry),
                "publication_schedule": schedule,
            }
    interval = source_poll_interval_sec(source, source_state, args, now)
    last_poll = utc_datetime_or_none(source_state.get("last_poll_at_utc"))
    if last_poll is None:
        return {"due": True, "reason": "never_polled", "interval_sec": interval, "publication_schedule": schedule}
    next_poll = last_poll + timedelta(seconds=interval)
    if now >= next_poll:
        return {
            "due": True,
            "reason": "interval_elapsed",
            "interval_sec": interval,
            "next_due_at_utc": iso_utc(next_poll),
            "publication_schedule": schedule,
        }
    return {
        "due": False,
        "reason": "not_due",
        "interval_sec": interval,
        "next_due_at_utc": iso_utc(next_poll),
        "publication_schedule": schedule,
    }


def next_source_sleep_sec(state: dict[str, Any], args: argparse.Namespace) -> int:
    now = datetime.now(timezone.utc)
    waits = []
    for source in SOURCE_LAYER_PATHS:
        if not source_enabled(source, args):
            continue
        decision = source_poll_decision(source, model_state(state, source), args, now)
        waits.append(seconds_until(utc_datetime_or_none(decision.get("next_due_at_utc")), now))
    if args.enable_ml_eumetsat_cloud_mask:
        decision = external_data_poll_decision(
            external_data_state(state, "eumetsat_cloud_mask"),
            args.ml_eumetsat_cloud_mask_poll_interval_sec,
            args,
            now,
        )
        waits.append(seconds_until(utc_datetime_or_none(decision.get("next_due_at_utc")), now))
    for product, config in EUMETSAT_THERMAL_PRODUCTS.items():
        if not eumetsat_thermal_product_enabled(product, args):
            continue
        decision = external_data_poll_decision(
            external_data_state(state, config["status_key"]),
            args.ml_eumetsat_thermal_poll_interval_sec,
            args,
            now,
        )
        waits.append(seconds_until(utc_datetime_or_none(decision.get("next_due_at_utc")), now))
    if not waits:
        return max(30, args.poll_interval_sec)
    return max(30, min(max(0, item) for item in waits))


def sleep_interruptibly(seconds: int) -> None:
    deadline = time.time() + max(0, seconds)
    while not SHUTDOWN_REQUESTED and time.time() < deadline:
        time.sleep(min(5, max(0.1, deadline - time.time())))


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
    ensure_model_states(state)
    commands: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    source_decisions = {
        source: source_poll_decision(source, model_state(state, source), args, now)
        for source in SOURCE_LAYER_PATHS
        if source_enabled(source, args)
    }
    arome_due = bool(source_decisions.get("arome", {}).get("due"))
    arome_lead_hours = resolve_arome_lead_hours(args) if arome_due or args.force else tuple(str(item) for item in (args.lead_hours or ()))
    moloch_lead_hours = tuple(str(item) for item in (args.moloch_lead_hours or ()))
    icon2i_lead_hours = tuple(str(item) for item in (args.icon2i_lead_hours or ()))
    status: dict[str, Any] = {
        "format": "corsewind.forecast_update_engine.status.v1",
        "generated_at_utc": utc_now(),
        "result": "running",
        "dry_run": bool(args.dry_run),
        "runtime": runtime_metadata(),
        "poll_interval_sec": args.poll_interval_sec,
        "previous_completed_run_time_utc": state.get("last_completed_run_time_utc"),
        "current_run_time_utc": None,
        "source_polling": {
            "arome_poll_interval_sec": args.arome_poll_interval_sec,
            "aromepi_poll_interval_sec": args.aromepi_poll_interval_sec,
            "aromepi_stale_poll_interval_sec": args.aromepi_stale_poll_interval_sec,
            "aromepi_freshness_target_sec": args.aromepi_freshness_target_sec,
            "aromepi_horizon_hours": args.aromepi_horizon_hours,
            "moloch_poll_interval_sec": args.moloch_poll_interval_sec,
            "moloch_command_timeout_sec": args.moloch_command_timeout_sec,
            "icon2i_poll_interval_sec": args.icon2i_poll_interval_sec,
            "icon2i_command_timeout_sec": args.icon2i_command_timeout_sec,
            "source_error_backoff_sec": args.source_error_backoff_sec,
            "source_error_backoff_max_sec": args.source_error_backoff_max_sec,
        },
        "sources": {},
        "changed": False,
        "forced": bool(args.force),
        "cleanup_raw_enabled": bool(args.cleanup_raw),
        "ml_root": str(args.ml_root),
        "ml_registry": str(args.ml_registry),
        "ml_dataset_archive_enabled": bool(args.enable_ml_dataset_archive),
        "ml_dataset_root": str(args.ml_dataset_root),
        "ml_dataset_samples_root": str(args.ml_dataset_samples_root),
        "ml_feature_store_enabled": bool(args.enable_ml_feature_store),
        "ml_feature_store_root": str(args.ml_feature_store_root),
        "ml_nwp_extra_fields_enabled": bool(args.enable_ml_nwp_extra_fields),
        "ml_nwp_extra_fields_raw_root": str(args.ml_nwp_extra_fields_raw_root),
        "ml_nwp_extra_fields_samples_root": str(args.ml_nwp_extra_fields_samples_root),
        "ml_nwp_vertical_profiles_enabled": bool(args.enable_ml_nwp_vertical_profiles),
        "ml_nwp_vertical_profiles_raw_root": str(args.ml_nwp_vertical_profiles_raw_root),
        "ml_nwp_vertical_profiles_samples_root": str(args.ml_nwp_vertical_profiles_samples_root),
        "ml_nwp_vertical_profiles_pressure_levels_hpa": list(args.ml_nwp_vertical_profiles_pressure_levels_hpa),
        "ml_copernicus_sst_enabled": bool(args.enable_ml_copernicus_sst),
        "ml_copernicus_sst_raw_root": str(args.ml_copernicus_sst_raw_root),
        "ml_copernicus_sst_samples_root": str(args.ml_copernicus_sst_samples_root),
        "ml_eumetsat_cloud_mask_enabled": bool(args.enable_ml_eumetsat_cloud_mask),
        "ml_eumetsat_cloud_mask_raw_root": str(args.ml_eumetsat_cloud_mask_raw_root),
        "ml_eumetsat_cloud_mask_samples_root": str(args.ml_eumetsat_cloud_mask_samples_root),
        "ml_eumetsat_thermal_products_enabled": bool(args.enable_ml_eumetsat_thermal_products),
        "arome_lead_hours": list(arome_lead_hours),
        "aromepi_enabled": bool(args.enable_aromepi),
        "moloch_enabled": bool(args.enable_moloch),
        "moloch_dataset": args.moloch_dataset,
        "moloch_lead_hours": list(moloch_lead_hours) if moloch_lead_hours else "all_available",
        "icon2i_enabled": bool(args.enable_icon2i),
        "icon2i_dataset": args.icon2i_dataset,
        "icon2i_lead_hours": list(icon2i_lead_hours) if icon2i_lead_hours else "all_available",
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
        "windninja_enabled": bool(args.enable_windninja),
        "windninja_steps": [],
        "published_windninja_steps": [],
        "commands": commands,
        "artifacts": WINDNINJA_50M_ARTIFACTS,
        "export_manifest": str(args.export_manifest),
    }
    for source, decision in source_decisions.items():
        status["sources"][source] = {
            "enabled": True,
            "due": bool(decision.get("due")),
            "decision": decision,
            "publication_schedule": decision.get("publication_schedule"),
            "run_time_utc": read_run_time(SOURCE_LAYER_PATHS[source]),
            "changed": False,
            "status": "pending" if decision.get("due") else "skipped",
        }
    for source in SOURCE_LAYER_PATHS:
        if source not in status["sources"]:
            status["sources"][source] = {"enabled": False, "status": "disabled", "changed": False}
    write_json(args.status_file, status)

    any_source_ran = False
    if source_decisions.get("aromepi", {}).get("due"):
        refresh_source("aromepi", arome_pi_refresh_command(args), args, state, status, commands)
        publish_wind2d_json(args, commands, "aromepi source refreshed")
        archive_source_snapshot_if_needed("aromepi", args, status, commands)
        collect_nwp_extra_fields_if_needed("aromepi", args, status, commands)
        enqueue_raster_tiles_if_source_changed("aromepi", args, status, commands, "aromepi source changed")
        any_source_ran = True
    if source_decisions.get("arome", {}).get("due"):
        refresh_source(
            "arome",
            arome_refresh_command(arome_lead_hours, args.arome_request_sleep_sec, args.cleanup_raw),
            args,
            state,
            status,
            commands,
        )
        publish_wind2d_json(args, commands, "arome source refreshed")
        archive_source_snapshot_if_needed("arome", args, status, commands)
        collect_nwp_extra_fields_if_needed("arome", args, status, commands)
        collect_nwp_vertical_profiles_if_needed(args, status, commands)
        enqueue_raster_tiles_if_source_changed("arome", args, status, commands, "arome source changed")
        any_source_ran = True
    if source_decisions.get("moloch", {}).get("due"):
        source = args.moloch_input or os.getenv("MOLOCH_SOURCE") or os.getenv("MOLOCH_SOURCE_URL")
        refresh_source(
            "moloch",
            moloch_refresh_command(source, args.moloch_dataset, moloch_lead_hours, args.cleanup_raw),
            args,
            state,
            status,
            commands,
            command_timeout_sec=args.moloch_command_timeout_sec,
        )
        publish_wind2d_json(args, commands, "moloch source refreshed")
        archive_source_snapshot_if_needed("moloch", args, status, commands)
        enqueue_raster_tiles_if_source_changed("moloch", args, status, commands, "moloch source changed")
        any_source_ran = True
    if source_decisions.get("icon2i", {}).get("due"):
        source = args.icon2i_input or os.getenv("ICON2I_SOURCE") or os.getenv("ICON2I_SOURCE_URL")
        refresh_source(
            "icon2i",
            icon2i_refresh_command(source, args.icon2i_dataset, icon2i_lead_hours, args.cleanup_raw),
            args,
            state,
            status,
            commands,
            command_timeout_sec=args.icon2i_command_timeout_sec,
        )
        publish_wind2d_json(args, commands, "icon2i source refreshed")
        archive_source_snapshot_if_needed("icon2i", args, status, commands)
        enqueue_raster_tiles_if_source_changed("icon2i", args, status, commands, "icon2i source changed")
        any_source_ran = True
    any_source_changed = any(bool(item.get("changed")) for item in status["sources"].values())
    collect_copernicus_sst_if_needed(args, status, commands, now)
    collect_eumetsat_cloud_mask_if_needed(args, state, status, commands, now)
    for product in EUMETSAT_THERMAL_PRODUCTS:
        collect_eumetsat_spot_product_if_needed(product, args, state, status, commands, now)
    build_ml_feature_store_if_needed(args, status, commands)
    current_run_time = read_run_time()
    status["current_run_time_utc"] = current_run_time
    state["last_seen_run_time_utc"] = current_run_time
    state["last_poll_at_utc"] = utc_now()
    selected_steps = select_windninja_steps(args)
    if args.dry_run and not selected_steps:
        selected_steps = synthetic_dry_run_steps(arome_lead_hours)
    status["windninja_steps"] = selected_step_summary(selected_steps, args.session_timezone)
    failed_sources = [
        source
        for source, source_status in status["sources"].items()
        if source_status.get("status") == "failed"
    ]
    status["weather_sources_changed"] = any_source_changed
    status["failed_sources"] = failed_sources
    status["next_source_poll_sec"] = next_source_sleep_sec(state, args)

    if args.dry_run:
        for index, step in enumerate(selected_steps):
            if not args.enable_windninja:
                break
            lead_hour = int(step["lead_hour"])
            for command in windninja_50m_commands(
                lead_hour,
                args.windninja_runtime_min,
                args.windninja_parallel,
                force_batch=True,
                append_tiles=index > 0,
            ):
                commands.append(run_command(command, dry_run=True))
        status.update(
            {
                "result": "dry_run",
                "changed": True,
                "windninja_generation_skipped": not bool(args.enable_windninja),
                "elapsed_s": round(time.time() - started, 3),
            }
        )
        write_json(args.status_file, status)
        return status

    if not current_run_time:
        raise RuntimeError(f"AROME refresh did not write run_time_utc to {AROME_LAYER.relative_to(ROOT)}")

    changed = bool(args.force or current_run_time != state.get("last_completed_run_time_utc"))
    status["changed"] = changed
    status["windninja_forcing_changed"] = changed
    if changed and not args.enable_windninja:
        result = "windninja_disabled"
        status.update(
            {
                "result": result,
                "elapsed_s": round(time.time() - started, 3),
                "windninja_generation_skipped": True,
                "windninja_skip_reason": "WINDNINJA_ENABLED=false or --no-enable-windninja",
            }
        )
        write_json(args.status_file, status)
        append_history(
            state,
            {
                "at_utc": utc_now(),
                "result": result,
                "run_time_utc": current_run_time,
                "weather_sources_changed": any_source_changed,
                "failed_sources": failed_sources,
            },
        )
        return status
    if not changed:
        if any_source_changed:
            result = "model_layers_updated"
        elif failed_sources:
            result = "source_failed"
        else:
            result = "unchanged"
        status.update({"result": result, "elapsed_s": round(time.time() - started, 3)})
        write_json(args.status_file, status)
        append_history(
            state,
            {
                "at_utc": utc_now(),
                "result": result,
                "run_time_utc": current_run_time,
                "weather_sources_changed": any_source_changed,
                "failed_sources": failed_sources,
            },
        )
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
    parser.add_argument("--fast-window-poll-interval-sec", type=int, default=60, help="Polling interval used inside learned fast publication windows.")
    parser.add_argument("--once", action="store_true", help="Run one poll cycle and exit.")
    parser.add_argument("--force", action="store_true", help="Run the 50 m pipeline even if the run was already completed.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cleanup-raw", action=argparse.BooleanOptionalAction, default=True, help="Delete raw weather downloads after derived Wind2D artifacts are published.")
    parser.add_argument("--ml-root", type=Path, default=DEFAULT_ML_ROOT, help="Root directory for ML dataset inputs and derived tables.")
    parser.add_argument("--ml-registry", type=Path, default=DEFAULT_ML_REGISTRY, help="ML spot registry JSON.")
    parser.add_argument("--enable-ml-dataset-archive", action=argparse.BooleanOptionalAction, default=env_bool("ML_DATASET_ARCHIVE_ENABLED", False), help="Archive derived model-layer JSON snapshots and sample them at ML spots for dataset construction.")
    parser.add_argument("--ml-dataset-root", type=Path, default=DEFAULT_ML_DATASET_ROOT, help="Root directory for archived ML dataset model-run snapshots.")
    parser.add_argument("--ml-dataset-samples-root", type=Path, default=DEFAULT_ML_MODEL_SAMPLES_ROOT, help="Root directory for ML dataset model samples at spots.")
    parser.add_argument("--enable-ml-feature-store", action=argparse.BooleanOptionalAction, default=env_bool("ML_FEATURE_STORE_ENABLED", False), help="Rebuild the canonical 15-minute ML feature store at the end of each cycle.")
    parser.add_argument("--ml-feature-store-root", type=Path, default=DEFAULT_ML_FEATURE_STORE_ROOT)
    parser.add_argument("--enable-ml-nwp-extra-fields", action=argparse.BooleanOptionalAction, default=env_bool("ML_NWP_EXTRA_FIELDS_ENABLED", False), help="Collect extra AROME/AROME-PI thermal/context fields at ML spots.")
    parser.add_argument("--ml-nwp-extra-fields-raw-root", type=Path, default=DEFAULT_ML_NWP_EXTRA_FIELDS_RAW_ROOT)
    parser.add_argument("--ml-nwp-extra-fields-samples-root", type=Path, default=DEFAULT_ML_NWP_EXTRA_FIELDS_SAMPLES_ROOT)
    parser.add_argument("--ml-nwp-extra-fields-max-steps", type=int, default=env_int("ML_NWP_EXTRA_FIELDS_MAX_STEPS", 24), help="Maximum forecast steps sampled for extra NWP fields per source cycle.")
    parser.add_argument("--ml-nwp-extra-fields-request-sleep-sec", type=float, default=float(os.getenv("ML_NWP_EXTRA_FIELDS_REQUEST_SLEEP_SEC", "0.2")))
    parser.add_argument("--ml-nwp-extra-fields-include-context-spots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-ml-nwp-vertical-profiles", action=argparse.BooleanOptionalAction, default=env_bool("ML_NWP_VERTICAL_PROFILES_ENABLED", False), help="Collect AROME 0.025 isobaric vertical profiles at ML spots.")
    parser.add_argument("--ml-nwp-vertical-profiles-raw-root", type=Path, default=DEFAULT_ML_NWP_VERTICAL_PROFILES_RAW_ROOT)
    parser.add_argument("--ml-nwp-vertical-profiles-samples-root", type=Path, default=DEFAULT_ML_NWP_VERTICAL_PROFILES_SAMPLES_ROOT)
    parser.add_argument("--ml-nwp-vertical-profiles-max-steps", type=int, default=env_int("ML_NWP_VERTICAL_PROFILES_MAX_STEPS", 5), help="Maximum AROME forecast steps sampled for vertical profiles per cycle.")
    parser.add_argument("--ml-nwp-vertical-profiles-pressure-levels-hpa", nargs="+", type=int, default=env_int_list("ML_NWP_VERTICAL_PROFILES_PRESSURE_LEVELS_HPA", [1000, 925, 850]))
    parser.add_argument("--ml-nwp-vertical-profiles-request-sleep-sec", type=float, default=float(os.getenv("ML_NWP_VERTICAL_PROFILES_REQUEST_SLEEP_SEC", "0.2")))
    parser.add_argument("--ml-nwp-vertical-profiles-include-context-spots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-ml-copernicus-sst", action=argparse.BooleanOptionalAction, default=env_bool("ML_COPERNICUS_SST_ENABLED", False), help="Collect Copernicus Marine SST for ML dataset construction.")
    parser.add_argument("--ml-copernicus-sst-raw-root", type=Path, default=DEFAULT_ML_COPERNICUS_SST_RAW_ROOT, help="Root directory for Copernicus Marine SST NetCDF cache.")
    parser.add_argument("--ml-copernicus-sst-samples-root", type=Path, default=DEFAULT_ML_COPERNICUS_SST_SAMPLES_ROOT, help="Root directory for Copernicus Marine SST spot samples.")
    parser.add_argument("--ml-copernicus-sst-window-hours", type=int, default=12, help="Hourly SST collection window length when explicit start/end are not provided.")
    parser.add_argument("--ml-copernicus-sst-end-lag-hours", type=int, default=18, help="Lag between current UTC hour and the default SST window end.")
    parser.add_argument("--ml-copernicus-sst-start-datetime", default=None, help="Explicit Copernicus SST collection start datetime in UTC.")
    parser.add_argument("--ml-copernicus-sst-end-datetime", default=None, help="Explicit Copernicus SST collection end datetime in UTC.")
    parser.add_argument("--ml-copernicus-sst-include-context-spots", action=argparse.BooleanOptionalAction, default=True, help="Include context spots when sampling Copernicus SST.")
    parser.add_argument("--ml-copernicus-sst-log-level", default="INFO", choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL", "QUIET"])
    parser.add_argument("--enable-ml-eumetsat-cloud-mask", action=argparse.BooleanOptionalAction, default=env_bool("ML_EUMETSAT_CLOUD_MASK_ENABLED", False), help="Collect EUMETSAT MTG Cloud Mask products for ML dataset construction.")
    parser.add_argument("--ml-eumetsat-cloud-mask-collection-id", default=os.getenv("ML_EUMETSAT_CLOUD_MASK_COLLECTION_ID", DEFAULT_ML_EUMETSAT_CLOUD_MASK_COLLECTION_ID))
    parser.add_argument("--ml-eumetsat-cloud-mask-bbox", default=os.getenv("ML_EUMETSAT_CLOUD_MASK_BBOX", DEFAULT_ML_EUMETSAT_BBOX), help="EUMDAC bbox as west,south,east,north.")
    parser.add_argument("--ml-eumetsat-cloud-mask-raw-root", type=Path, default=DEFAULT_ML_EUMETSAT_CLOUD_MASK_RAW_ROOT, help="Root directory for EUMETSAT Cloud Mask NetCDF cache.")
    parser.add_argument("--ml-eumetsat-cloud-mask-samples-root", type=Path, default=DEFAULT_ML_EUMETSAT_CLOUD_MASK_SAMPLES_ROOT, help="Root directory for EUMETSAT Cloud Mask spot samples.")
    parser.add_argument("--ml-eumetsat-cloud-mask-poll-interval-sec", type=int, default=env_int("ML_EUMETSAT_CLOUD_MASK_POLL_INTERVAL_SEC", 480), help="Minimum interval between EUMETSAT Cloud Mask collections.")
    parser.add_argument("--ml-eumetsat-cloud-mask-window-minutes", type=int, default=env_int("ML_EUMETSAT_CLOUD_MASK_WINDOW_MINUTES", 120), help="Rolling EUMETSAT Cloud Mask search window length.")
    parser.add_argument("--ml-eumetsat-cloud-mask-end-lag-minutes", type=int, default=env_int("ML_EUMETSAT_CLOUD_MASK_END_LAG_MINUTES", 5), help="Lag between now and the default EUMETSAT search window end.")
    parser.add_argument("--ml-eumetsat-cloud-mask-start-datetime", default=None, help="Explicit EUMETSAT Cloud Mask collection start datetime in UTC.")
    parser.add_argument("--ml-eumetsat-cloud-mask-end-datetime", default=None, help="Explicit EUMETSAT Cloud Mask collection end datetime in UTC.")
    parser.add_argument("--ml-eumetsat-cloud-mask-max-products", type=int, default=env_int("ML_EUMETSAT_CLOUD_MASK_MAX_PRODUCTS", 18))
    parser.add_argument("--ml-eumetsat-cloud-mask-include-context-spots", action=argparse.BooleanOptionalAction, default=True, help="Include context spots when sampling EUMETSAT Cloud Mask.")
    parser.add_argument("--ml-eumetsat-cloud-mask-radius-cells", type=int, default=env_int("ML_EUMETSAT_CLOUD_MASK_RADIUS_CELLS", 3), help="Cloud-state neighbourhood radius around the nearest satellite pixel.")
    parser.add_argument("--ml-eumetsat-cloud-mask-quality-flags", action=argparse.BooleanOptionalAction, default=True, help="Sample EUMETSAT Cloud Mask quality variables when available.")
    parser.add_argument("--enable-ml-eumetsat-thermal-products", action=argparse.BooleanOptionalAction, default=env_bool("ML_EUMETSAT_THERMAL_PRODUCTS_ENABLED", False), help="Collect Cloud Type, Land Surface Temperature, and Global Instability Indices.")
    parser.add_argument("--enable-ml-eumetsat-cloud-type", action=argparse.BooleanOptionalAction, default=env_bool("ML_EUMETSAT_CLOUD_TYPE_ENABLED", False), help="Collect EUMETSAT MTG Cloud Type.")
    parser.add_argument("--enable-ml-eumetsat-land-surface-temperature", action=argparse.BooleanOptionalAction, default=env_bool("ML_EUMETSAT_LAND_SURFACE_TEMPERATURE_ENABLED", False), help="Collect EUMETSAT MTG Land Surface Temperature.")
    parser.add_argument("--enable-ml-eumetsat-global-instability-indices", action=argparse.BooleanOptionalAction, default=env_bool("ML_EUMETSAT_GLOBAL_INSTABILITY_INDICES_ENABLED", False), help="Collect EUMETSAT MTG Global Instability Indices.")
    parser.add_argument("--ml-eumetsat-thermal-bbox", default=os.getenv("ML_EUMETSAT_THERMAL_BBOX", DEFAULT_ML_EUMETSAT_BBOX), help="EUMDAC bbox for EUMETSAT thermal/context products.")
    parser.add_argument("--ml-eumetsat-thermal-poll-interval-sec", type=int, default=env_int("ML_EUMETSAT_THERMAL_POLL_INTERVAL_SEC", 900), help="Minimum interval between EUMETSAT thermal/context collections.")
    parser.add_argument("--ml-eumetsat-thermal-window-minutes", type=int, default=env_int("ML_EUMETSAT_THERMAL_WINDOW_MINUTES", 180), help="Rolling EUMETSAT thermal/context search window length.")
    parser.add_argument("--ml-eumetsat-thermal-end-lag-minutes", type=int, default=env_int("ML_EUMETSAT_THERMAL_END_LAG_MINUTES", 10), help="Lag between now and the default EUMETSAT thermal/context search window end.")
    parser.add_argument("--ml-eumetsat-thermal-start-datetime", default=None, help="Explicit EUMETSAT thermal/context start datetime in UTC.")
    parser.add_argument("--ml-eumetsat-thermal-end-datetime", default=None, help="Explicit EUMETSAT thermal/context end datetime in UTC.")
    parser.add_argument("--ml-eumetsat-thermal-max-products", type=int, default=env_int("ML_EUMETSAT_THERMAL_MAX_PRODUCTS", 12))
    parser.add_argument("--ml-eumetsat-thermal-max-variables", type=int, default=env_int("ML_EUMETSAT_THERMAL_MAX_VARIABLES", 24))
    parser.add_argument("--ml-eumetsat-thermal-radius-cells", type=int, default=env_int("ML_EUMETSAT_THERMAL_RADIUS_CELLS", 3))
    parser.add_argument("--ml-eumetsat-thermal-include-context-spots", action=argparse.BooleanOptionalAction, default=True, help="Include context spots when sampling EUMETSAT thermal/context products.")
    parser.add_argument("--ml-eumetsat-cloud-type-collection-id", default=os.getenv("ML_EUMETSAT_CLOUD_TYPE_COLLECTION_ID", EUMETSAT_THERMAL_PRODUCTS["cloud_type"]["collection_id"]))
    parser.add_argument("--ml-eumetsat-cloud-type-raw-root", type=Path, default=EUMETSAT_THERMAL_PRODUCTS["cloud_type"]["raw_root"])
    parser.add_argument("--ml-eumetsat-cloud-type-samples-root", type=Path, default=EUMETSAT_THERMAL_PRODUCTS["cloud_type"]["samples_root"])
    parser.add_argument("--ml-eumetsat-land-surface-temperature-collection-id", default=os.getenv("ML_EUMETSAT_LAND_SURFACE_TEMPERATURE_COLLECTION_ID", EUMETSAT_THERMAL_PRODUCTS["land_surface_temperature"]["collection_id"]))
    parser.add_argument("--ml-eumetsat-land-surface-temperature-raw-root", type=Path, default=EUMETSAT_THERMAL_PRODUCTS["land_surface_temperature"]["raw_root"])
    parser.add_argument("--ml-eumetsat-land-surface-temperature-samples-root", type=Path, default=EUMETSAT_THERMAL_PRODUCTS["land_surface_temperature"]["samples_root"])
    parser.add_argument("--ml-eumetsat-global-instability-indices-collection-id", default=os.getenv("ML_EUMETSAT_GLOBAL_INSTABILITY_INDICES_COLLECTION_ID", EUMETSAT_THERMAL_PRODUCTS["global_instability_indices"]["collection_id"]))
    parser.add_argument("--ml-eumetsat-global-instability-indices-raw-root", type=Path, default=EUMETSAT_THERMAL_PRODUCTS["global_instability_indices"]["raw_root"])
    parser.add_argument("--ml-eumetsat-global-instability-indices-samples-root", type=Path, default=EUMETSAT_THERMAL_PRODUCTS["global_instability_indices"]["samples_root"])
    parser.add_argument("--lead-hours", nargs="+", default=None)
    parser.add_argument("--arome-lead-hour-policy", choices=["session", "all-48"], default="all-48")
    parser.add_argument("--arome-poll-interval-sec", type=int, default=900, help="Normal polling interval for the main AROME source.")
    parser.add_argument("--arome-request-sleep-sec", type=float, default=1.3)
    parser.add_argument("--enable-aromepi", action=argparse.BooleanOptionalAction, default=True, help="Build the AROME-PI hybrid 15 min nowcast viewer layer.")
    parser.add_argument("--aromepi-poll-interval-sec", type=int, default=300, help="Normal polling interval for AROME-PI when the source is fresh.")
    parser.add_argument("--aromepi-stale-poll-interval-sec", type=int, default=60, help="Fast AROME-PI polling interval once the last seen run is older than the freshness target.")
    parser.add_argument("--aromepi-freshness-target-sec", type=int, default=900, help="AROME-PI freshness target before switching to fast polling.")
    parser.add_argument("--aromepi-horizon-hours", type=int, default=24, help="Rolling AROME-PI forecast horizon to publish.")
    parser.add_argument("--aromepi-request-sleep-sec", type=float, default=1.3)
    parser.add_argument("--enable-moloch", action="store_true", help="Build the optional MOLOCH 1.2 km viewer layer.")
    parser.add_argument("--moloch-input", default=None, help="Local GRIB/NetCDF/JSON file or direct URL. Defaults to MOLOCH_SOURCE_URL.")
    parser.add_argument("--moloch-dataset", default="MOLOCH")
    parser.add_argument("--moloch-lead-hours", nargs="+", default=None)
    parser.add_argument("--moloch-poll-interval-sec", type=int, default=1800)
    parser.add_argument(
        "--moloch-command-timeout-sec",
        type=int,
        default=int(os.getenv("MOLOCH_COMMAND_TIMEOUT_SEC", "1800")),
        help="Maximum MOLOCH builder runtime before the cycle keeps the previous layer.",
    )
    parser.add_argument("--moloch-skip-if-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-icon2i", action="store_true", help="Build the optional ICON-2I 2.2 km viewer layer.")
    parser.add_argument("--icon2i-input", default=None, help="Local GRIB/NetCDF/JSON file or direct URL. Defaults to latest MeteoHub ICON-2I bundle.")
    parser.add_argument("--icon2i-dataset", default="ICON_2I_SURFACE_PRESSURE_LEVELS")
    parser.add_argument("--icon2i-lead-hours", nargs="+", default=None)
    parser.add_argument("--icon2i-poll-interval-sec", type=int, default=1800)
    parser.add_argument(
        "--icon2i-command-timeout-sec",
        type=int,
        default=int(os.getenv("ICON2I_COMMAND_TIMEOUT_SEC", "1800")),
        help="Maximum ICON-2I builder runtime before the cycle keeps the previous layer.",
    )
    parser.add_argument(
        "--enable-windninja",
        action=argparse.BooleanOptionalAction,
        default=env_bool("WINDNINJA_ENABLED", True),
        help="Run WindNinja 50 m generation. Defaults to WINDNINJA_ENABLED, true when unset.",
    )
    parser.add_argument("--windninja-lead-hours", nargs="+", type=int, default=None)
    parser.add_argument("--windninja-parallel", type=int, default=6)
    parser.add_argument("--windninja-runtime-min", type=float, default=60.0)
    parser.add_argument("--session-timezone", default=DEFAULT_SESSION_TIMEZONE)
    parser.add_argument("--session-start-hour", type=int, default=11)
    parser.add_argument("--session-end-hour", type=int, default=18)
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
    parser.add_argument("--source-error-backoff-sec", type=int, default=300)
    parser.add_argument("--source-error-backoff-max-sec", type=int, default=1800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    args.state_file = resolve_path(args.state_file)
    args.status_file = resolve_path(args.status_file)
    args.export_manifest = resolve_path(args.export_manifest)
    args.ml_root = resolve_path(args.ml_root)
    args.ml_registry = resolve_path(args.ml_registry)
    args.ml_dataset_root = resolve_path(args.ml_dataset_root)
    args.ml_dataset_samples_root = resolve_path(args.ml_dataset_samples_root)
    args.ml_feature_store_root = resolve_path(args.ml_feature_store_root)
    args.ml_nwp_extra_fields_raw_root = resolve_path(args.ml_nwp_extra_fields_raw_root)
    args.ml_nwp_extra_fields_samples_root = resolve_path(args.ml_nwp_extra_fields_samples_root)
    args.ml_nwp_vertical_profiles_raw_root = resolve_path(args.ml_nwp_vertical_profiles_raw_root)
    args.ml_nwp_vertical_profiles_samples_root = resolve_path(args.ml_nwp_vertical_profiles_samples_root)
    args.ml_copernicus_sst_raw_root = resolve_path(args.ml_copernicus_sst_raw_root)
    args.ml_copernicus_sst_samples_root = resolve_path(args.ml_copernicus_sst_samples_root)
    args.ml_eumetsat_cloud_mask_raw_root = resolve_path(args.ml_eumetsat_cloud_mask_raw_root)
    args.ml_eumetsat_cloud_mask_samples_root = resolve_path(args.ml_eumetsat_cloud_mask_samples_root)
    for product in EUMETSAT_THERMAL_PRODUCTS:
        setattr(args, f"ml_eumetsat_{product}_raw_root", resolve_path(getattr(args, f"ml_eumetsat_{product}_raw_root")))
        setattr(args, f"ml_eumetsat_{product}_samples_root", resolve_path(getattr(args, f"ml_eumetsat_{product}_samples_root")))
    args.lock_file = resolve_path(args.lock_file)

    acquire_lock(args.lock_file, args.lock_stale_after_sec)
    if not args.dry_run:
        bootstrap_raster_tiles()
    try:
        while not SHUTDOWN_REQUESTED:
            state = load_state(args.state_file)
            try:
                status = poll_once(args, state)
                write_json(args.state_file, state)
                print(
                    f"engine cycle {status['result']} run={status.get('current_run_time_utc')} "
                    f"elapsed={status.get('elapsed_s')}s",
                    flush=True,
                )
                if args.once:
                    wait_for_raster_tile_queue()
            except Exception as exc:
                state["in_progress_run_time_utc"] = None
                result = "stopping" if SHUTDOWN_REQUESTED else "failed"
                if SHUTDOWN_REQUESTED:
                    consecutive_failures = int(state.get("consecutive_failures") or 0)
                else:
                    state["last_failure_at_utc"] = utc_now()
                    consecutive_failures = int(state.get("consecutive_failures") or 0) + 1
                    state["consecutive_failures"] = consecutive_failures
                append_history(state, {"at_utc": utc_now(), "result": result, "error": str(exc)})
                write_json(args.state_file, state)
                failure_status = {
                    "format": "corsewind.forecast_update_engine.status.v1",
                    "generated_at_utc": utc_now(),
                    "result": result,
                    "error": str(exc),
                    "runtime": runtime_metadata(),
                    "current_run_time_utc": read_run_time(),
                    "consecutive_failures": consecutive_failures,
                }
                write_json(args.status_file, failure_status)
                print(f"engine cycle {result}: {exc}", file=sys.stderr, flush=True)
                if args.once and not SHUTDOWN_REQUESTED:
                    raise
                if SHUTDOWN_REQUESTED:
                    break
                sleep_interruptibly(max(1, args.sleep_on_error_sec))
            if args.once or SHUTDOWN_REQUESTED:
                break
            sleep_sec = next_source_sleep_sec(state, args)
            print(f"next source poll in {sleep_sec}s", flush=True)
            sleep_interruptibly(sleep_sec)
    finally:
        release_lock(args.lock_file)


if __name__ == "__main__":
    main()
