#!/usr/bin/env python3
"""Build a lightweight ICON-2I wind layer for the CorseWind 2D map."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from build_moloch_corsica_wind_layer import (
    CLOUD_CANDIDATES,
    PRECIPITATION_CANDIDATES,
    U_CANDIDATES,
    V_CANDIDATES,
    array_lat_lon,
    common_lead_hours,
    ensure_source_file,
    finite_stats,
    find_data_array,
    nearest_resample_curvilinear,
    open_datasets,
    optional_resampled_grid,
    round_grid,
    run_time_from_arrays,
    select_lead,
)
from meteohub_opendata_client import OpenDataBundle, latest_opendata_bundle
from meteo_france_client import load_dotenv
from raw_cache_cleanup import cleanup_message, cleanup_raw_dir


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BBOX = (8.45, 41.25, 9.75, 43.1)
DEFAULT_DATASET_ID = "ICON_2I_SURFACE_PRESSURE_LEVELS"
DEFAULT_GRID_STEP_DEG = 0.02
DEFAULT_SOURCE_LABEL = "ItaliaMeteo/ARPAE ICON-2I via MeteoHub GRIB2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def source_from_args(args: argparse.Namespace) -> tuple[str, OpenDataBundle | None]:
    source = args.input or os.getenv("ICON2I_SOURCE") or os.getenv("ICON2I_SOURCE_URL")
    if source:
        return source, None
    bundle = latest_opendata_bundle(
        args.dataset,
        required_vars=("u-component of wind", "v-component of wind"),
        timeout_sec=args.discovery_timeout_sec,
    )
    return bundle.download_url, bundle


def write_layer_atomic(path: Path, payload: dict[str, Any]) -> None:
    if not payload.get("run_time_utc") or not payload.get("forecast_steps") or not payload.get("bbox_wgs84"):
        raise SystemExit("Refusing to publish incomplete ICON-2I layer payload.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    parsed = json.loads(tmp.read_text(encoding="utf-8"))
    if not parsed.get("run_time_utc") or not parsed.get("forecast_steps") or not parsed.get("bbox_wgs84"):
        tmp.unlink(missing_ok=True)
        raise SystemExit("Refusing to publish invalid ICON-2I layer JSON.")
    tmp.replace(path)


def optional_copy_json(input_path: Path, output: Path) -> bool:
    if input_path.suffix.lower() != ".json":
        return False
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not payload.get("forecast_steps") or not payload.get("bbox_wgs84"):
        raise SystemExit(f"{input_path} is JSON but not a CorseWind wind layer payload.")
    write_layer_atomic(output, payload)
    return True


def speed_grid_from_components(u_grid: np.ndarray, v_grid: np.ndarray) -> np.ndarray:
    return np.sqrt(u_grid * u_grid + v_grid * v_grid)


def build_payload(input_path: Path, args: argparse.Namespace, bundle: OpenDataBundle | None) -> dict[str, Any]:
    datasets = open_datasets(input_path)
    u_array = find_data_array(datasets, U_CANDIDATES)
    v_array = find_data_array(datasets, V_CANDIDATES)
    cloud_array = find_data_array(datasets, CLOUD_CANDIDATES)
    precipitation_array = find_data_array(datasets, PRECIPITATION_CANDIDATES)
    if u_array is None or v_array is None:
        available = sorted({name for dataset in datasets for name in dataset.data_vars})
        raise SystemExit(f"Cannot find 10 m U/V wind variables in {input_path}. Available: {available[:80]}")

    run_time = bundle.run_time_utc if bundle else run_time_from_arrays(u_array, v_array)
    bbox = tuple(args.bbox)
    lead_hours = tuple(args.lead_hours) if args.lead_hours else common_lead_hours(u_array, v_array)
    if not lead_hours:
        raise SystemExit(f"Cannot determine available ICON-2I lead hours in {input_path}.")
    steps = []
    for lead_hour in lead_hours:
        u_selected = select_lead(u_array, lead_hour)
        v_selected = select_lead(v_array, lead_hour)
        lat, lon = array_lat_lon(u_selected)
        u_values = np.asarray(u_selected.values, dtype=float)
        v_values = np.asarray(v_selected.values, dtype=float)
        if u_values.shape != v_values.shape:
            raise SystemExit(f"ICON-2I U/V shape mismatch at H+{lead_hour}: {u_values.shape} vs {v_values.shape}")

        u_grid = nearest_resample_curvilinear(u_values, lat, lon, bbox, args.grid_step_deg)
        v_grid = nearest_resample_curvilinear(v_values, lat, lon, bbox, args.grid_step_deg)
        speed_grid = speed_grid_from_components(u_grid, v_grid)
        cloud_grid = optional_resampled_grid(cloud_array, lead_hour, bbox, args.grid_step_deg)
        precipitation_grid = optional_resampled_grid(precipitation_array, lead_hour, bbox, args.grid_step_deg)
        valid_time = run_time + timedelta(hours=int(lead_hour))
        step = {
            "lead_hour": int(lead_hour),
            "valid_time_utc": valid_time.isoformat().replace("+00:00", "Z"),
            "shape": list(speed_grid.shape),
            "stats_ms": finite_stats(speed_grid),
            "speed_ms": round_grid(speed_grid),
            "u_ms": round_grid(u_grid),
            "v_ms": round_grid(v_grid),
        }
        if cloud_grid is not None and cloud_grid.shape == speed_grid.shape:
            step["cloud_cover_stats_pct"] = finite_stats(cloud_grid)
            step["cloud_cover_pct"] = round_grid(cloud_grid)
        if precipitation_grid is not None and precipitation_grid.shape == speed_grid.shape:
            step["precipitation_stats_mm"] = finite_stats(precipitation_grid)
            step["precipitation_mm"] = round_grid(precipitation_grid)
        steps.append(step)

    shape = steps[0]["shape"]
    return {
        "format": "corsewind_icon2i_corsica_wind_layer_v0",
        "generated_at_utc": utc_now(),
        "source": args.source_label,
        "product": "ICON-2I",
        "dataset_id": args.dataset,
        "resolution": "2.2 km",
        "model_label": "ICON-2I 2.2 km",
        "height_agl_m": 10,
        "run_time_utc": run_time.isoformat().replace("+00:00", "Z"),
        "bbox_wgs84": list(bbox),
        "grid": {
            "orientation": "rows north-to-south, columns west-to-east",
            "lat_step_deg": round((bbox[3] - bbox[1]) / (shape[0] - 1), 6),
            "lon_step_deg": round((bbox[2] - bbox[0]) / (shape[1] - 1), 6),
            "resampling": "nearest source-grid point to regular WGS84 Corsica grid",
            "weather_fields": "optional cloud/precipitation when present in source bundle",
        },
        "source_file": str(input_path),
        "source_bundle": bundle.to_dict() if bundle else None,
        "forecast_steps": steps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Local GRIB/NetCDF/CorseWind JSON file, or direct URL. Defaults to latest MeteoHub ICON-2I bundle.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ID)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/icon2i_corsica_latest"))
    parser.add_argument("--output", type=Path, default=Path("visualizations/wind2d/icon2i-corsica-latest.json"))
    parser.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BBOX, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--lead-hours", nargs="+", type=int, default=None, help="Lead hours to publish. Defaults to every lead hour available in the source bundle.")
    parser.add_argument("--grid-step-deg", type=float, default=DEFAULT_GRID_STEP_DEG)
    parser.add_argument("--download-timeout-sec", type=int, default=600)
    parser.add_argument("--discovery-timeout-sec", type=int, default=30)
    parser.add_argument("--source-label", default=DEFAULT_SOURCE_LABEL)
    parser.add_argument("--cleanup-raw", action=argparse.BooleanOptionalAction, default=False, help="Delete raw downloaded source files after the Wind2D JSON has been published.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    args.raw_dir = resolve_path(args.raw_dir)
    args.output = resolve_path(args.output)
    source, bundle = source_from_args(args)
    input_path = ensure_source_file(source, args.raw_dir, args.download_timeout_sec)
    if not optional_copy_json(input_path, args.output):
        payload = build_payload(input_path, args, bundle)
        write_layer_atomic(args.output, payload)
        print(
            f"wrote {args.output} run={payload['run_time_utc']} "
            f"steps={len(payload['forecast_steps'])} shape={payload['forecast_steps'][0]['shape']}"
        )
    if args.cleanup_raw:
        print(cleanup_message(cleanup_raw_dir(args.raw_dir, ROOT)))


if __name__ == "__main__":
    main()
