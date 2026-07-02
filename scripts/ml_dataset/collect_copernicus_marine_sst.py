#!/usr/bin/env python3
"""Download Copernicus Marine SST and sample it at ML spot coordinates."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_RAW_ROOT = DEFAULT_ML_ROOT / "copernicus_marine/raw/sst"
DEFAULT_SAMPLE_ROOT = DEFAULT_ML_ROOT / "copernicus_marine/sst_samples"
DEFAULT_TMP_PYTHONPATH = ROOT / "tmp/copernicusmarine_test_pkgs"
DEFAULT_DATASET_ID = "cmems_obs-sst_med_phy-sst_nrt_diurnal-oi-0.0625deg_PT1H-m"
DEFAULT_VARIABLE = "analysed_sst"
DEFAULT_SERVICE = "geoseries"
DEFAULT_BBOX = (7.5, 41.0, 10.2, 43.3)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def import_xarray() -> Any:
    try:
        import xarray as xr

        return xr
    except ModuleNotFoundError:
        if DEFAULT_TMP_PYTHONPATH.exists():
            sys.path.insert(0, str(DEFAULT_TMP_PYTHONPATH))
            import xarray as xr

            return xr
        raise SystemExit(
            "xarray is required to read Copernicus NetCDF files. "
            "Install the ML data dependencies or keep tmp/copernicusmarine_test_pkgs available."
        )


def finite_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_spots(path: Path, include_context: bool, selected_ids: set[str]) -> list[dict[str, Any]]:
    payload = read_json(path)
    spots = payload.get("spots", []) if isinstance(payload, dict) else payload
    if not isinstance(spots, list):
        raise SystemExit(f"Registry has no spots list: {path}")
    selected = []
    for spot in spots:
        if not isinstance(spot, dict):
            continue
        if selected_ids and spot.get("spot_id") not in selected_ids:
            continue
        if not include_context and not spot.get("use_for_ml", False):
            continue
        if finite_float(spot.get("latitude")) is None or finite_float(spot.get("longitude")) is None:
            continue
        selected.append(spot)
    return selected


def sanitize_datetime(value: str) -> str:
    return (
        value.replace(":", "")
        .replace("-", "")
        .replace("+", "")
        .replace("Z", "")
        .replace("T", "T")
    )


def default_filename(start_datetime: str, end_datetime: str) -> str:
    return f"sst_corse_{sanitize_datetime(start_datetime)}_{sanitize_datetime(end_datetime)}.nc"


def find_copernicusmarine_bin(explicit_path: Path | None) -> Path:
    if explicit_path:
        candidate = resolve_path(explicit_path)
        if candidate.exists():
            return candidate
        raise SystemExit(f"copernicusmarine binary not found: {candidate}")
    from_path = shutil.which("copernicusmarine")
    if from_path:
        return Path(from_path)
    local_bin = DEFAULT_TMP_PYTHONPATH / "bin/copernicusmarine"
    if local_bin.exists():
        return local_bin
    raise SystemExit(
        "copernicusmarine CLI not found. Install copernicusmarine or keep "
        "tmp/copernicusmarine_test_pkgs/bin/copernicusmarine available."
    )


def copernicus_env(binary: Path) -> dict[str, str]:
    env = os.environ.copy()
    if str(binary).startswith(str(DEFAULT_TMP_PYTHONPATH)) and DEFAULT_TMP_PYTHONPATH.exists():
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(DEFAULT_TMP_PYTHONPATH)
            if not existing
            else f"{DEFAULT_TMP_PYTHONPATH}{os.pathsep}{existing}"
        )
    return env


def download_subset(args: argparse.Namespace, output_path: Path) -> None:
    if output_path.exists() and not args.overwrite:
        return
    if not os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME") or not os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD"):
        raise SystemExit(
            "Copernicus credentials are required for download. Set "
            "COPERNICUSMARINE_SERVICE_USERNAME and COPERNICUSMARINE_SERVICE_PASSWORD, "
            "or pass --input-netcdf to sample an existing file."
        )
    binary = find_copernicusmarine_bin(args.copernicusmarine_bin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary),
        "subset",
        "--dataset-id",
        args.dataset_id,
        "--variable",
        args.variable,
        "--minimum-longitude",
        str(args.minimum_longitude),
        "--maximum-longitude",
        str(args.maximum_longitude),
        "--minimum-latitude",
        str(args.minimum_latitude),
        "--maximum-latitude",
        str(args.maximum_latitude),
        "--start-datetime",
        args.start_datetime,
        "--end-datetime",
        args.end_datetime,
        "--output-directory",
        str(output_path.parent),
        "--output-filename",
        output_path.name,
        "--file-format",
        "netcdf",
        "--service",
        args.service,
        "--disable-progress-bar",
        "--log-level",
        args.log_level,
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True, env=copernicus_env(binary))


def haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius_km = 6371.0088
    phi_a = math.radians(lat_a)
    phi_b = math.radians(lat_b)
    d_phi = math.radians(lat_b - lat_a)
    d_lam = math.radians(lon_b - lon_a)
    value = math.sin(d_phi / 2) ** 2 + math.cos(phi_a) * math.cos(phi_b) * math.sin(d_lam / 2) ** 2
    return radius_km * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def nearest_index(values: Any, target: float) -> int:
    best_idx = 0
    best_delta = float("inf")
    for idx, value in enumerate(values):
        delta = abs(float(value) - target)
        if delta < best_delta:
            best_idx = idx
            best_delta = delta
    return best_idx


def finite_cell(data: Any, time_idx: int, lat_idx: int, lon_idx: int) -> float | None:
    if lat_idx < 0 or lon_idx < 0 or lat_idx >= data.shape[1] or lon_idx >= data.shape[2]:
        return None
    value = finite_float(data[time_idx, lat_idx, lon_idx].item())
    return value


def sample_nearest_finite(
    data: Any,
    latitudes: Any,
    longitudes: Any,
    time_idx: int,
    target_lat: float,
    target_lon: float,
    search_radius_cells: int,
) -> dict[str, Any]:
    lat_idx = nearest_index(latitudes, target_lat)
    lon_idx = nearest_index(longitudes, target_lon)
    best: dict[str, Any] | None = None
    for radius in range(search_radius_cells + 1):
        for y in range(lat_idx - radius, lat_idx + radius + 1):
            for x in range(lon_idx - radius, lon_idx + radius + 1):
                if radius and abs(y - lat_idx) != radius and abs(x - lon_idx) != radius:
                    continue
                value = finite_cell(data, time_idx, y, x)
                if value is None:
                    continue
                pixel_lat = float(latitudes[y])
                pixel_lon = float(longitudes[x])
                distance_km = haversine_km(target_lat, target_lon, pixel_lat, pixel_lon)
                candidate = {
                    "sst_k": round(value, 4),
                    "sst_c": round(value - 273.15, 4),
                    "sst_pixel_latitude": round(pixel_lat, 6),
                    "sst_pixel_longitude": round(pixel_lon, 6),
                    "sst_sample_distance_km": round(distance_km, 4),
                    "sst_search_radius_cells": radius,
                }
                if best is None or candidate["sst_sample_distance_km"] < best["sst_sample_distance_km"]:
                    best = candidate
        if best is not None:
            return best
    return {
        "sst_k": None,
        "sst_c": None,
        "sst_pixel_latitude": None,
        "sst_pixel_longitude": None,
        "sst_sample_distance_km": None,
        "sst_search_radius_cells": search_radius_cells,
    }


def iso_time(value: Any) -> str:
    try:
        import numpy as np

        if isinstance(value, np.datetime64):
            return str(value.astype("datetime64[s]")) + "Z"
    except ModuleNotFoundError:
        pass
    if hasattr(value, "isoformat"):
        text = value.isoformat()
        return text.replace("+00:00", "Z")
    return str(value)


def output_path(output_root: Path, timestamp_utc: str | None) -> Path:
    day = (timestamp_utc or utc_now())[:10]
    return output_root / f"date={day}" / "sst_samples.jsonl"


def write_jsonl_by_day(output_root: Path, rows: list[dict[str, Any]]) -> dict[str, int]:
    by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_path[output_path(output_root, row.get("timestamp_utc"))].append(row)
    written: dict[str, int] = {}
    for path, path_rows in by_path.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if path.exists():
            existing = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        deduped = {
            (row.get("dataset_id"), row.get("timestamp_utc"), row.get("spot_id")): row
            for row in [*existing, *path_rows]
        }
        ordered = sorted(deduped.values(), key=lambda row: (row.get("timestamp_utc") or "", row.get("spot_id") or ""))
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")
        tmp.replace(path)
        written[str(path)] = len(ordered)
    return written


def sample_netcdf(args: argparse.Namespace, netcdf_path: Path, spots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    xr = import_xarray()
    dataset = xr.open_dataset(netcdf_path)
    try:
        if args.variable not in dataset:
            raise SystemExit(f"Variable {args.variable!r} not found in {netcdf_path}")
        field = dataset[args.variable]
        if not {"time", "latitude", "longitude"}.issubset(field.dims):
            raise SystemExit(f"Variable {args.variable!r} must have time, latitude, longitude dimensions.")
        field = field.transpose("time", "latitude", "longitude")
        data = field.values
        latitudes = dataset["latitude"].values
        longitudes = dataset["longitude"].values
        times = dataset["time"].values
        rows: list[dict[str, Any]] = []
        for time_idx, time_value in enumerate(times):
            timestamp_utc = iso_time(time_value)
            for spot in spots:
                lat = float(spot["latitude"])
                lon = float(spot["longitude"])
                sampled = sample_nearest_finite(
                    data,
                    latitudes,
                    longitudes,
                    time_idx,
                    lat,
                    lon,
                    args.search_radius_cells,
                )
                rows.append({
                    "format": "corsewind.copernicus_marine_sst_spot_sample.v1",
                    "source": "copernicus_marine",
                    "product": "SST_MED_PHY_SUBSKIN_L4_NRT_010_036",
                    "dataset_id": args.dataset_id,
                    "variable": args.variable,
                    "service": args.service,
                    "timestamp_utc": timestamp_utc,
                    "spot_id": spot.get("spot_id"),
                    "spot_name": spot.get("name"),
                    "spot_kind": spot.get("kind"),
                    "spot_source_type": spot.get("source_type"),
                    "station_id": spot.get("station_id"),
                    "latitude": lat,
                    "longitude": lon,
                    "use_for_ml": bool(spot.get("use_for_ml", False)),
                    "sample_method": "nearest_finite_grid_cell",
                    "netcdf_path": str(netcdf_path),
                    "sampled_at_utc": utc_now(),
                    **sampled,
                })
        return rows
    finally:
        dataset.close()


def delete_raw_file(path: Path) -> dict[str, Any]:
    try:
        size_bytes = path.stat().st_size
    except FileNotFoundError:
        return {"path": str(path), "status": "already_missing"}
    path.unlink()
    return {"path": str(path), "status": "deleted", "size_bytes": size_bytes}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--variable", default=DEFAULT_VARIABLE)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--start-datetime", help="Subset start, for example 2026-06-22T12:00:00.")
    parser.add_argument("--end-datetime", help="Subset end, for example 2026-06-22T15:00:00.")
    parser.add_argument("--minimum-longitude", type=float, default=DEFAULT_BBOX[0])
    parser.add_argument("--minimum-latitude", type=float, default=DEFAULT_BBOX[1])
    parser.add_argument("--maximum-longitude", type=float, default=DEFAULT_BBOX[2])
    parser.add_argument("--maximum-latitude", type=float, default=DEFAULT_BBOX[3])
    parser.add_argument("--input-netcdf", type=Path, help="Use an existing NetCDF file instead of downloading.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--copernicusmarine-bin", type=Path)
    parser.add_argument("--output-filename")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--delete-raw-after-sample",
        action="store_true",
        help="Delete downloaded NetCDF file after successful sampling to limit backfill storage.",
    )
    parser.add_argument("--include-context-spots", action="store_true", help="Include spots with use_for_ml=false.")
    parser.add_argument("--spot-id", action="append", default=[], help="Sample only specific spot ids. Repeatable.")
    parser.add_argument("--search-radius-cells", type=int, default=4)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL", "QUIET"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry_path = resolve_path(args.registry)
    raw_root = resolve_path(args.raw_root)
    output_root = resolve_path(args.output_root)
    if args.input_netcdf:
        netcdf_path = resolve_path(args.input_netcdf)
    else:
        if not args.start_datetime or not args.end_datetime:
            raise SystemExit("--start-datetime and --end-datetime are required when downloading.")
        filename = args.output_filename or default_filename(args.start_datetime, args.end_datetime)
        netcdf_path = raw_root / filename
        download_subset(args, netcdf_path)
    if not netcdf_path.exists():
        raise SystemExit(f"NetCDF file not found: {netcdf_path}")
    spots = load_spots(registry_path, args.include_context_spots, set(args.spot_id))
    rows = sample_netcdf(args, netcdf_path, spots)
    written = write_jsonl_by_day(output_root, rows)
    deleted_raw = None
    if args.delete_raw_after_sample and not args.input_netcdf:
        deleted_raw = delete_raw_file(netcdf_path)
    valid_rows = sum(1 for row in rows if row.get("sst_c") is not None)
    print(json.dumps({
        "generated_at_utc": utc_now(),
        "dataset_id": args.dataset_id,
        "variable": args.variable,
        "netcdf_path": str(netcdf_path),
        "registry": str(registry_path),
        "output_root": str(output_root),
        "spot_count": len(spots),
        "row_count": len(rows),
        "valid_sst_rows": valid_rows,
        "deleted_raw": deleted_raw,
        "written": written,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
