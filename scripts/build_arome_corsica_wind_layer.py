#!/usr/bin/env python3
"""Build a lightweight latest AROME wind layer for the CorseWind 2D map."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from meteo_france_client import coverage_ids, endpoint, load_dotenv, request_api
from raw_cache_cleanup import cleanup_message, cleanup_raw_dir
from sample_arome_tiff_at_stations import read_float64_tiff


VARIABLES = {
    "speed": "WIND_SPEED__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
    "u": "U_COMPONENT_OF_WIND__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
    "v": "V_COMPONENT_OF_WIND__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
}

DEFAULT_BBOX = (8.45, 41.25, 9.75, 43.1)
DEFAULT_LEAD_HOURS = (0, 1, 3, 6, 9, 12, 24)


def coverage_run_time(coverage_id: str, variable_prefix: str) -> datetime | None:
    pattern = re.compile(rf"^{re.escape(variable_prefix)}___(\d{{4}}-\d{{2}}-\d{{2}})T(\d{{2}})\.00\.00Z$")
    match = pattern.match(coverage_id)
    if not match:
        return None
    return datetime.fromisoformat(f"{match.group(1)}T{match.group(2)}:00:00+00:00")


def latest_complete_run(coverage_list: list[str]) -> tuple[datetime, dict[str, str]]:
    by_run: dict[datetime, dict[str, str]] = {}
    for coverage_id in coverage_list:
        for variable_name, prefix in VARIABLES.items():
            run_time = coverage_run_time(coverage_id, prefix)
            if run_time is not None:
                by_run.setdefault(run_time, {})[variable_name] = coverage_id

    complete = {run_time: values for run_time, values in by_run.items() if set(values) == set(VARIABLES)}
    if not complete:
        raise SystemExit("No complete AROME wind run found for speed/U/V.")
    run_time = max(complete)
    return run_time, complete[run_time]


def download_tiff(
    coverage_id: str,
    output: Path,
    bbox: tuple[float, float, float, float],
    valid_time: datetime,
    product: str,
    resolution: str,
    auth_header: str,
) -> bool:
    if output.exists():
        return False
    url = endpoint(product, resolution, "GetCoverage")
    min_lon, min_lat, max_lon, max_lat = bbox
    params = [
        ("service", "WCS"),
        ("version", "2.0.1"),
        ("coverageid", coverage_id),
        ("format", "image/tiff"),
        ("subset", f"long({min_lon},{max_lon})"),
        ("subset", f"lat({min_lat},{max_lat})"),
        ("subset", "height(10)"),
        ("subset", f"time({valid_time.isoformat().replace('+00:00', 'Z')})"),
    ]
    response = request_api(url, params, auth_header)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.content)
    return True


def finite_stats(array: np.ndarray) -> dict[str, float]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": round(float(finite.min()), 3),
        "mean": round(float(finite.mean()), 3),
        "max": round(float(finite.max()), 3),
    }


def round_grid(array: np.ndarray) -> list[list[float | None]]:
    rounded = np.round(array.astype(float), 3)
    rows: list[list[float | None]] = []
    for row in rounded:
        rows.append([None if not math.isfinite(float(value)) else float(value) for value in row])
    return rows


def write_layer_atomic(path: Path, payload: dict[str, Any]) -> None:
    if not payload.get("run_time_utc") or not payload.get("forecast_steps"):
        raise SystemExit("Refusing to publish incomplete AROME layer payload.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    parsed = json.loads(tmp.read_text(encoding="utf-8"))
    if not parsed.get("run_time_utc") or not parsed.get("forecast_steps"):
        tmp.unlink(missing_ok=True)
        raise SystemExit("Refusing to publish invalid AROME layer JSON.")
    tmp.replace(path)


def build_payload(
    run_time: datetime,
    coverages: dict[str, str],
    lead_hours: tuple[int, ...],
    bbox: tuple[float, float, float, float],
    raw_dir: Path,
    product: str,
    resolution: str,
    auth_header: str,
    request_sleep_sec: float,
) -> dict[str, Any]:
    steps = []
    slug = run_time.strftime("%Y%m%dT%H")
    for lead_hour in lead_hours:
        valid_time = run_time + timedelta(hours=lead_hour)
        rasters: dict[str, np.ndarray] = {}
        for variable_name, coverage_id in coverages.items():
            output = raw_dir / f"arome_{resolution}_corsica_{slug}_h{lead_hour:02d}_{variable_name}_10m.tiff"
            downloaded = download_tiff(coverage_id, output, bbox, valid_time, product, resolution, auth_header)
            if downloaded and request_sleep_sec > 0:
                time.sleep(request_sleep_sec)
            rasters[variable_name] = read_float64_tiff(output)

        speed = rasters["speed"]
        u = rasters["u"]
        v = rasters["v"]
        if speed.shape != u.shape or speed.shape != v.shape:
            raise SystemExit(f"Raster shape mismatch at H+{lead_hour}: speed={speed.shape}, u={u.shape}, v={v.shape}")

        steps.append(
            {
                "lead_hour": lead_hour,
                "valid_time_utc": valid_time.isoformat().replace("+00:00", "Z"),
                "shape": list(speed.shape),
                "stats_ms": finite_stats(speed),
                "speed_ms": round_grid(speed),
                "u_ms": round_grid(u),
                "v_ms": round_grid(v),
            }
        )

    return {
        "format": "corsewind_arome_corsica_wind_layer_v0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "Meteo-France public AROME WCS",
        "product": product,
        "resolution": resolution,
        "model_label": "AROME 0.01 deg",
        "height_agl_m": 10,
        "run_time_utc": run_time.isoformat().replace("+00:00", "Z"),
        "bbox_wgs84": list(bbox),
        "grid": {
            "orientation": "rows north-to-south, columns west-to-east",
            "lat_step_deg": round((bbox[3] - bbox[1]) / (steps[0]["shape"][0] - 1), 6),
            "lon_step_deg": round((bbox[2] - bbox[0]) / (steps[0]["shape"][1] - 1), 6),
        },
        "coverages": coverages,
        "forecast_steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", choices=["arome", "aromepi"], default="arome")
    parser.add_argument("--resolution", choices=["001", "0025"], default="001")
    parser.add_argument("--auth-header", choices=["apikey", "bearer"], default="apikey")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/arome_corsica_latest"))
    parser.add_argument("--output", type=Path, default=Path("visualizations/wind2d/arome-corsica-latest.json"))
    parser.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BBOX, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--lead-hours", nargs="+", type=int, default=list(DEFAULT_LEAD_HOURS))
    parser.add_argument("--request-sleep-sec", type=float, default=0.0, help="Pause after each downloaded WCS raster; useful for API quotas.")
    parser.add_argument("--cleanup-raw", action=argparse.BooleanOptionalAction, default=False, help="Delete raw downloaded rasters after the Wind2D JSON has been published.")
    args = parser.parse_args()

    load_dotenv(args.env_file)
    capabilities_url = endpoint(args.product, args.resolution, "GetCapabilities")
    response = request_api(
        capabilities_url,
        [("service", "WCS"), ("version", "2.0.1"), ("language", "eng")],
        args.auth_header,
    )
    run_time, coverages = latest_complete_run(coverage_ids(response.text))
    payload = build_payload(
        run_time=run_time,
        coverages=coverages,
        lead_hours=tuple(args.lead_hours),
        bbox=tuple(args.bbox),
        raw_dir=args.raw_dir,
        product=args.product,
        resolution=args.resolution,
        auth_header=args.auth_header,
        request_sleep_sec=args.request_sleep_sec,
    )
    write_layer_atomic(args.output, payload)
    print(
        f"wrote {args.output} run={payload['run_time_utc']} "
        f"steps={len(payload['forecast_steps'])} shape={payload['forecast_steps'][0]['shape']}"
    )
    if args.cleanup_raw:
        print(cleanup_message(cleanup_raw_dir(args.raw_dir, Path(__file__).resolve().parents[1])))


if __name__ == "__main__":
    main()
