#!/usr/bin/env python3
"""Build a hybrid AROME-PI nowcast layer for the CorseWind 2D map.

The viewer uses the 0.025 deg AROME-PI mean wind for direction, particles,
and color raster. The 0.01 deg 15-minute gust field is kept for inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime, time as day_time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from build_arome_corsica_wind_layer import finite_stats, round_grid, write_layer_atomic
from meteo_france_client import coverage_ids, endpoint, load_dotenv, request_api
from raw_cache_cleanup import cleanup_message, cleanup_raw_dir
from sample_arome_tiff_at_stations import read_float64_tiff


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BBOX = (8.45, 41.25, 9.75, 43.1)
DEFAULT_SESSION_START_HOUR = 11
DEFAULT_SESSION_END_HOUR = 18
DEFAULT_TIMEZONE = "Europe/Paris"

MEAN_VARIABLES = {
    "mean_speed": "WIND_SPEED__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
    "mean_u": "U_COMPONENT_OF_WIND__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
    "mean_v": "V_COMPONENT_OF_WIND__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
}
GUST_VARIABLES = {
    "gust_speed": "WIND_SPEED_GUST_15MIN__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_coverage_run(coverage_id: str, prefix: str, suffix: str = "") -> datetime | None:
    pattern = re.compile(
        rf"^{re.escape(prefix)}___(\d{{4}}-\d{{2}}-\d{{2}})T(\d{{2}})\.00\.00Z{re.escape(suffix)}$"
    )
    match = pattern.match(coverage_id)
    if not match:
        return None
    return datetime.fromisoformat(f"{match.group(1)}T{match.group(2)}:00:00+00:00")


def latest_common_run(mean_ids: list[str], gust_ids: list[str]) -> tuple[datetime, dict[str, str]]:
    by_run: dict[datetime, dict[str, str]] = {}
    for coverage_id in mean_ids:
        for variable_name, prefix in MEAN_VARIABLES.items():
            run_time = parse_coverage_run(coverage_id, prefix)
            if run_time:
                by_run.setdefault(run_time, {})[variable_name] = coverage_id
    for coverage_id in gust_ids:
        run_time = parse_coverage_run(coverage_id, GUST_VARIABLES["gust_speed"], "_PT15M")
        if run_time:
            by_run.setdefault(run_time, {})["gust_speed"] = coverage_id

    required = set(MEAN_VARIABLES) | set(GUST_VARIABLES)
    complete = {run_time: values for run_time, values in by_run.items() if required <= set(values)}
    if not complete:
        raise SystemExit("No complete AROME-PI hybrid run found for 0.025 mean wind and 0.01 gust PT15M.")
    run_time = max(complete)
    return run_time, complete[run_time]


def load_capabilities(product: str, resolution: str, auth_header: str) -> list[str]:
    response = request_api(
        endpoint(product, resolution, "GetCapabilities"),
        [("service", "WCS"), ("version", "2.0.1"), ("language", "eng")],
        auth_header,
    )
    return coverage_ids(response.text)


def session_valid_times(
    run_time: datetime,
    session_date: datetime,
    session_start_hour: int,
    session_end_hour: int,
    timezone_name: str,
) -> list[datetime]:
    try:
        from zoneinfo import ZoneInfo
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Python zoneinfo is required for local session filtering.") from exc

    tz = ZoneInfo(timezone_name)
    local_day = session_date.astimezone(tz).date()
    start_local = datetime.combine(local_day, day_time(session_start_hour, 0), tzinfo=tz)
    end_local = datetime.combine(local_day, day_time(session_end_hour, 0), tzinfo=tz)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)
    horizon_end = run_time + timedelta(hours=6)

    first = max(run_time + timedelta(minutes=15), start_utc)
    first_minutes = math.ceil((first - run_time).total_seconds() / 900) * 15
    times: list[datetime] = []
    offset = first_minutes
    while offset <= 360:
      valid_time = run_time + timedelta(minutes=offset)
      if valid_time > horizon_end or valid_time > end_utc:
          break
      if valid_time >= start_utc:
          times.append(valid_time)
      offset += 15
    return times


def download_tiff(
    product: str,
    resolution: str,
    coverage_id: str,
    output: Path,
    bbox: tuple[float, float, float, float],
    valid_time: datetime,
    auth_header: str,
    height_m: int = 10,
) -> bool:
    if output.exists() and output.stat().st_size > 0:
        return False
    min_lon, min_lat, max_lon, max_lat = bbox
    response = request_api(
        endpoint(product, resolution, "GetCoverage"),
        [
            ("service", "WCS"),
            ("version", "2.0.1"),
            ("coverageid", coverage_id),
            ("format", "image/tiff"),
            ("subset", f"long({min_lon},{max_lon})"),
            ("subset", f"lat({min_lat},{max_lat})"),
            ("subset", f"height({height_m})"),
            ("subset", f"time({valid_time.isoformat().replace('+00:00', 'Z')})"),
        ],
        auth_header,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_bytes(response.content)
    tmp.replace(output)
    return True


def resample_regular(source: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if source.shape == target_shape:
        return source.astype(float, copy=False)
    target_rows, target_cols = target_shape
    source_rows, source_cols = source.shape
    row_positions = np.linspace(0, source_rows - 1, target_rows)
    col_positions = np.linspace(0, source_cols - 1, target_cols)
    output = np.empty((target_rows, target_cols), dtype=float)
    for out_r, src_r in enumerate(row_positions):
        r0 = int(math.floor(src_r))
        r1 = min(source_rows - 1, r0 + 1)
        tr = src_r - r0
        for out_c, src_c in enumerate(col_positions):
            c0 = int(math.floor(src_c))
            c1 = min(source_cols - 1, c0 + 1)
            tc = src_c - c0
            values = (source[r0, c0], source[r1, c0], source[r0, c1], source[r1, c1])
            if not all(math.isfinite(float(value)) for value in values):
                output[out_r, out_c] = np.nan
                continue
            output[out_r, out_c] = (
                values[0] * (1 - tr) * (1 - tc)
                + values[1] * tr * (1 - tc)
                + values[2] * (1 - tr) * tc
                + values[3] * tr * tc
            )
    return output


def build_payload(
    run_time: datetime,
    coverages: dict[str, str],
    valid_times: list[datetime],
    bbox: tuple[float, float, float, float],
    raw_dir: Path,
    auth_header: str,
    request_sleep_sec: float,
    session_start_hour: int,
    session_end_hour: int,
    timezone_name: str,
) -> dict[str, Any]:
    if not valid_times:
        raise SystemExit("No AROME-PI valid times available inside requested session window.")

    steps = []
    slug = run_time.strftime("%Y%m%dT%H")
    for valid_time in valid_times:
        minute_offset = int((valid_time - run_time).total_seconds() // 60)
        rasters: dict[str, np.ndarray] = {}
        for variable_name in ("gust_speed",):
            output = raw_dir / f"aromepi_001_corsica_{slug}_m{minute_offset:03d}_{variable_name}_10m.tiff"
            downloaded = download_tiff("aromepi", "001", coverages[variable_name], output, bbox, valid_time, auth_header)
            if downloaded and request_sleep_sec > 0:
                time.sleep(request_sleep_sec)
            rasters[variable_name] = read_float64_tiff(output)

        for variable_name in ("mean_speed", "mean_u", "mean_v"):
            output = raw_dir / f"aromepi_0025_corsica_{slug}_m{minute_offset:03d}_{variable_name}_10m.tiff"
            downloaded = download_tiff("aromepi", "0025", coverages[variable_name], output, bbox, valid_time, auth_header)
            if downloaded and request_sleep_sec > 0:
                time.sleep(request_sleep_sec)
            rasters[variable_name] = read_float64_tiff(output)

        gust_speed = rasters["gust_speed"]
        target_shape = gust_speed.shape
        mean_speed = resample_regular(rasters["mean_speed"], target_shape)
        mean_u = resample_regular(rasters["mean_u"], target_shape)
        mean_v = resample_regular(rasters["mean_v"], target_shape)
        lead_minutes = minute_offset
        lead_hour = round(lead_minutes / 60, 4)
        steps.append(
            {
                "lead_hour": lead_hour,
                "lead_minutes": lead_minutes,
                "valid_time_utc": valid_time.isoformat().replace("+00:00", "Z"),
                "shape": list(target_shape),
                "stats_ms": finite_stats(mean_speed),
                "mean_stats_ms": finite_stats(mean_speed),
                "gust_stats_ms": finite_stats(gust_speed),
                "speed_ms": round_grid(mean_speed),
                "gust_speed_ms": round_grid(gust_speed),
                "mean_speed_ms": round_grid(mean_speed),
                "u_ms": round_grid(mean_u),
                "v_ms": round_grid(mean_v),
                "mean_u_ms": round_grid(mean_u),
                "mean_v_ms": round_grid(mean_v),
            }
        )

    shape = steps[0]["shape"]
    return {
        "format": "corsewind_aromepi_hybrid_corsica_wind_layer_v0",
        "generated_at_utc": utc_now(),
        "source": "Meteo-France public AROME-PI WCS",
        "product": "aromepi",
        "resolution": "0.025 mean wind / 0.01 gust",
        "model_label": "AROME-PI immediat",
        "height_agl_m": 10,
        "run_time_utc": run_time.isoformat().replace("+00:00", "Z"),
        "bbox_wgs84": list(bbox),
        "grid": {
            "orientation": "rows north-to-south, columns west-to-east",
            "lat_step_deg": round((bbox[3] - bbox[1]) / (shape[0] - 1), 6),
            "lon_step_deg": round((bbox[2] - bbox[0]) / (shape[1] - 1), 6),
            "render_field": "0.025 deg WIND_SPEED resampled to 0.01 deg grid",
            "gust_field": "0.01 deg WIND_SPEED_GUST_15MIN PT15M",
            "vector_field": "0.025 deg U/V wind resampled to 0.01 deg grid",
        },
        "timeline": {
            "step_minutes": 15,
            "session_start_local_hour": session_start_hour,
            "session_end_local_hour": session_end_hour,
            "timezone": timezone_name,
            "only_available_steps": True,
        },
        "coverages": coverages,
        "forecast_steps": steps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth-header", choices=["apikey", "bearer"], default="apikey")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/aromepi_corsica_latest"))
    parser.add_argument("--output", type=Path, default=Path("visualizations/wind2d/aromepi-corsica-latest.json"))
    parser.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BBOX, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--session-date", help="Local session date YYYY-MM-DD. Defaults to today in --timezone.")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--session-start-hour", type=int, default=DEFAULT_SESSION_START_HOUR)
    parser.add_argument("--session-end-hour", type=int, default=DEFAULT_SESSION_END_HOUR)
    parser.add_argument("--request-sleep-sec", type=float, default=0.0)
    parser.add_argument("--cleanup-raw", action=argparse.BooleanOptionalAction, default=False, help="Delete raw downloaded rasters after the Wind2D JSON has been published.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    mean_ids = load_capabilities("aromepi", "0025", args.auth_header)
    gust_ids = load_capabilities("aromepi", "001", args.auth_header)
    run_time, coverages = latest_common_run(mean_ids, gust_ids)
    if args.session_date:
        try:
            from zoneinfo import ZoneInfo
            session_date = datetime.fromisoformat(args.session_date).replace(tzinfo=ZoneInfo(args.timezone))
        except ValueError as exc:
            raise SystemExit("--session-date must be YYYY-MM-DD") from exc
    else:
        session_date = datetime.now(timezone.utc)

    valid_times = session_valid_times(
        run_time,
        session_date,
        args.session_start_hour,
        args.session_end_hour,
        args.timezone,
    )
    payload = build_payload(
        run_time=run_time,
        coverages=coverages,
        valid_times=valid_times,
        bbox=tuple(args.bbox),
        raw_dir=args.raw_dir,
        auth_header=args.auth_header,
        request_sleep_sec=args.request_sleep_sec,
        session_start_hour=args.session_start_hour,
        session_end_hour=args.session_end_hour,
        timezone_name=args.timezone,
    )
    write_layer_atomic(args.output, payload)
    print(
        f"wrote {args.output} run={payload['run_time_utc']} "
        f"steps={len(payload['forecast_steps'])} "
        f"first={payload['forecast_steps'][0]['valid_time_utc']} "
        f"last={payload['forecast_steps'][-1]['valid_time_utc']} "
        f"shape={payload['forecast_steps'][0]['shape']}"
    )
    if args.cleanup_raw:
        print(cleanup_message(cleanup_raw_dir(args.raw_dir, ROOT)))


if __name__ == "__main__":
    main()
