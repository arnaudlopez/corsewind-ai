#!/usr/bin/env python3
"""Build a lightweight MOLOCH wind layer for the CorseWind 2D map.

The public CNR-ISAC maps confirm the MOLOCH 10 m wind product, while MeteoHub
exposes GRIB bundles/downloads with filenames that can change by product. This
script therefore accepts either a local GRIB/NetCDF file or an explicit URL and
normalizes it into the same compact JSON shape used by the AROME viewer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import requests

from meteo_france_client import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BBOX = (8.45, 41.25, 9.75, 43.1)
DEFAULT_GRID_STEP_DEG = 0.0113
DEFAULT_SOURCE_LABEL = "CNR-ISAC MOLOCH-ISAC via MeteoHub GRIB2"

U_CANDIDATES = (
    "u10",
    "10u",
    "u_component_of_wind_10m",
    "u_component_of_wind_height_above_ground",
    "U10",
)
V_CANDIDATES = (
    "v10",
    "10v",
    "v_component_of_wind_10m",
    "v_component_of_wind_height_above_ground",
    "V10",
)
SPEED_CANDIDATES = (
    "si10",
    "wind_speed",
    "wind_speed_10m",
    "wind_speed_height_above_ground",
    "WIND_SPEED",
)
LAT_CANDIDATES = ("latitude", "lat", "LAT", "XLAT")
LON_CANDIDATES = ("longitude", "lon", "LON", "XLONG")
TIME_DIMS = ("step", "valid_time", "time", "forecast_time", "forecastTime")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def source_from_args(args: argparse.Namespace) -> str:
    source = args.input or os.getenv("MOLOCH_SOURCE") or os.getenv("MOLOCH_SOURCE_URL")
    if not source:
        raise SystemExit(
            "No MOLOCH source configured. Pass --input, or set MOLOCH_SOURCE_URL to a "
            "MeteoHub GRIB/NetCDF bundle URL after downloading/selecting the product."
        )
    return source


def filename_from_url(url: str) -> str:
    name = Path(unquote(urlparse(url).path)).name
    return name or "moloch_source.grib2"


def ensure_source_file(source: str, raw_dir: Path, timeout_sec: int) -> Path:
    if is_url(source):
        raw_dir.mkdir(parents=True, exist_ok=True)
        output = raw_dir / filename_from_url(source)
        if output.exists() and output.stat().st_size > 0:
            return output
        with requests.get(source, stream=True, timeout=timeout_sec) as response:
            response.raise_for_status()
            tmp = output.with_suffix(output.suffix + ".tmp")
            with tmp.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
            tmp.replace(output)
        return output

    path = resolve_path(Path(source))
    if not path.exists():
        raise SystemExit(f"MOLOCH source file does not exist: {path}")
    return path


def write_layer_atomic(path: Path, payload: dict[str, Any]) -> None:
    if not payload.get("run_time_utc") or not payload.get("forecast_steps") or not payload.get("bbox_wgs84"):
        raise SystemExit("Refusing to publish incomplete MOLOCH layer payload.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    parsed = json.loads(tmp.read_text(encoding="utf-8"))
    if not parsed.get("run_time_utc") or not parsed.get("forecast_steps") or not parsed.get("bbox_wgs84"):
        tmp.unlink(missing_ok=True)
        raise SystemExit("Refusing to publish invalid MOLOCH layer JSON.")
    tmp.replace(path)


def optional_copy_json(input_path: Path, output: Path) -> bool:
    if input_path.suffix.lower() != ".json":
        return False
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not payload.get("forecast_steps") or not payload.get("bbox_wgs84"):
        raise SystemExit(f"{input_path} is JSON but not a CorseWind wind layer payload.")
    write_layer_atomic(output, payload)
    return True


def import_xarray() -> Any:
    try:
        import xarray as xr
    except ImportError as exc:
        raise SystemExit(
            "MOLOCH decoding requires optional dependencies. Install them with: "
            "pip install -r requirements-moloch.txt"
        ) from exc
    return xr


def open_datasets(path: Path) -> list[Any]:
    xr = import_xarray()
    suffixes = "".join(path.suffixes).lower()
    if any(token in suffixes for token in (".grib", ".grb")):
        try:
            import cfgrib
        except ImportError as exc:
            raise SystemExit(
                "GRIB decoding requires cfgrib/eccodes. Install with: "
                "pip install -r requirements-moloch.txt"
            ) from exc
        return list(cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""}))
    return [xr.open_dataset(path)]


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def find_data_array(datasets: list[Any], candidates: tuple[str, ...]) -> Any | None:
    normalized = {normalize_name(item) for item in candidates}
    best: tuple[int, Any] | None = None
    for dataset in datasets:
        for name in dataset.data_vars:
            data_array = dataset[name]
            aliases = {
                normalize_name(name),
                normalize_name(str(data_array.attrs.get("standard_name", ""))),
                normalize_name(str(data_array.attrs.get("long_name", ""))),
                normalize_name(str(data_array.attrs.get("GRIB_shortName", ""))),
            }
            if aliases & normalized:
                score = 100
                type_of_level = normalize_name(str(data_array.attrs.get("GRIB_typeOfLevel", "")))
                level = data_array.attrs.get("GRIB_level")
                if type_of_level in {"heightaboveground", "heightabovegroundlayer"}:
                    score += 40
                if type_of_level in {"isobaricinhpa", "isobaricpa"}:
                    score -= 40
                try:
                    if float(level) == 10:
                        score += 30
                except (TypeError, ValueError):
                    pass
                if any(normalize_name(dim) in {"isobaricinhpa", "isobaricpa"} for dim in data_array.dims):
                    score -= 20
                if best is None or score > best[0]:
                    best = (score, data_array)
    return best[1] if best else None


def find_coord(dataset_or_array: Any, candidates: tuple[str, ...]) -> Any | None:
    normalized = {normalize_name(item) for item in candidates}
    for name in dataset_or_array.coords:
        coord = dataset_or_array.coords[name]
        aliases = {
            normalize_name(name),
            normalize_name(str(coord.attrs.get("standard_name", ""))),
            normalize_name(str(coord.attrs.get("long_name", ""))),
        }
        if aliases & normalized:
            return coord
    return None


def scalar_datetime(value: Any) -> datetime | None:
    try:
        if isinstance(value, np.ndarray):
            value = value.item()
        if isinstance(value, np.datetime64):
            seconds = value.astype("datetime64[s]").astype(int)
            return datetime.fromtimestamp(int(seconds), tz=timezone.utc)
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc)
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None
    return None


def run_time_from_arrays(*arrays: Any) -> datetime:
    for array in arrays:
        for attr_name in ("GRIB_dataDate", "GRIB_refTime", "reference_time", "analysis_time"):
            value = array.attrs.get(attr_name)
            if not value:
                continue
            if attr_name == "GRIB_dataDate":
                hour = int(array.attrs.get("GRIB_dataTime") or 0) // 100
                try:
                    return datetime.strptime(f"{value}{hour:02d}", "%Y%m%d%H").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            parsed = scalar_datetime(value)
            if parsed:
                return parsed
        if "time" in array.coords and array.coords["time"].size:
            parsed = scalar_datetime(array.coords["time"].values)
            if parsed:
                return parsed
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


def lead_values_hours(array: Any) -> tuple[str | None, np.ndarray | None]:
    for dim in TIME_DIMS:
        if dim not in array.dims:
            continue
        coord = array.coords.get(dim)
        if coord is None:
            return dim, np.arange(array.sizes[dim], dtype=float)
        values = np.asarray(coord.values)
        if values.size == 0:
            return dim, None
        if np.issubdtype(values.dtype, np.timedelta64):
            return dim, values.astype("timedelta64[h]").astype(float)
        if np.issubdtype(values.dtype, np.datetime64):
            first = values[0].astype("datetime64[h]")
            return dim, (values.astype("datetime64[h]") - first).astype("timedelta64[h]").astype(float)
        return dim, values.astype(float)
    return None, None


def available_lead_hours(array: Any) -> tuple[int, ...]:
    dim, leads = lead_values_hours(array)
    if dim is None or leads is None:
        return (0,)
    rounded: list[int] = []
    for value in leads:
        number = float(value)
        nearest = int(round(number))
        if abs(number - nearest) > 1e-6:
            continue
        rounded.append(nearest)
    return tuple(dict.fromkeys(rounded))


def common_lead_hours(*arrays: Any) -> tuple[int, ...]:
    lead_sets = [set(available_lead_hours(array)) for array in arrays if array is not None]
    if not lead_sets:
        return ()
    return tuple(sorted(set.intersection(*lead_sets)))


def select_lead(array: Any, lead_hour: int) -> Any:
    dim, leads = lead_values_hours(array)
    selected = array
    if dim and leads is not None and array.sizes.get(dim, 1) > 1:
        diffs = np.abs(leads - lead_hour)
        index = int(np.argmin(diffs))
        if float(diffs[index]) > 1e-6:
            available = ", ".join(str(item) for item in available_lead_hours(array))
            raise SystemExit(f"Requested H+{lead_hour} is not available for {array.name}. Available lead hours: {available}")
        selected = selected.isel({dim: index})
    for dim_name, size in list(selected.sizes.items()):
        if size == 1 and dim_name not in ("latitude", "longitude", "lat", "lon", "y", "x"):
            selected = selected.isel({dim_name: 0})
    return selected.squeeze(drop=True)


def array_lat_lon(data_array: Any) -> tuple[np.ndarray, np.ndarray]:
    lat = find_coord(data_array, LAT_CANDIDATES)
    lon = find_coord(data_array, LON_CANDIDATES)
    if lat is None or lon is None:
        raise SystemExit(f"Cannot find latitude/longitude coordinates for variable {data_array.name}.")
    return np.asarray(lat.values, dtype=float), np.asarray(lon.values, dtype=float)


def regular_target_grid(bbox: tuple[float, float, float, float], step_deg: float) -> tuple[np.ndarray, np.ndarray]:
    min_lon, min_lat, max_lon, max_lat = bbox
    rows = max(2, int(math.ceil((max_lat - min_lat) / step_deg)) + 1)
    cols = max(2, int(math.ceil((max_lon - min_lon) / step_deg)) + 1)
    lats = np.linspace(max_lat, min_lat, rows)
    lons = np.linspace(min_lon, max_lon, cols)
    return lats, lons


def crop_regular_1d(values: np.ndarray, lats: np.ndarray, lons: np.ndarray, bbox: tuple[float, float, float, float]) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    min_lon, min_lat, max_lon, max_lat = bbox
    lat_mask = (lats >= min_lat) & (lats <= max_lat)
    lon_mask = (lons >= min_lon) & (lons <= max_lon)
    if not lat_mask.any() or not lon_mask.any():
        raise SystemExit("Requested Corsica bbox is outside the MOLOCH source grid.")
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    subset = values[np.ix_(lat_idx, lon_idx)]
    subset_lats = lats[lat_idx]
    subset_lons = lons[lon_idx]
    if subset_lats[0] < subset_lats[-1]:
        subset = subset[::-1, :]
        subset_lats = subset_lats[::-1]
    return subset, (float(subset_lons.min()), float(subset_lats.min()), float(subset_lons.max()), float(subset_lats.max()))


def nearest_resample_curvilinear(
    values: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    bbox: tuple[float, float, float, float],
    target_step_deg: float,
) -> np.ndarray:
    min_lon, min_lat, max_lon, max_lat = bbox
    lat2 = np.asarray(lats, dtype=float)
    lon2 = np.asarray(lons, dtype=float)
    if lat2.ndim == 1 and lon2.ndim == 1:
        return crop_regular_1d(values, lat2, lon2, bbox)[0]

    mask = (
        np.isfinite(values)
        & np.isfinite(lat2)
        & np.isfinite(lon2)
        & (lat2 >= min_lat - 0.08)
        & (lat2 <= max_lat + 0.08)
        & (lon2 >= min_lon - 0.08)
        & (lon2 <= max_lon + 0.08)
    )
    if not mask.any():
        raise SystemExit("Requested Corsica bbox is outside the MOLOCH source grid.")

    src_lat = lat2[mask].reshape(-1)
    src_lon = lon2[mask].reshape(-1)
    src_val = values[mask].reshape(-1)
    target_lats, target_lons = regular_target_grid(bbox, target_step_deg)
    lon_mesh, lat_mesh = np.meshgrid(target_lons, target_lats)
    target_lat = lat_mesh.reshape(-1)
    target_lon = lon_mesh.reshape(-1)
    output = np.full(target_lat.shape, np.nan, dtype=float)
    cos_lat = math.cos(math.radians((min_lat + max_lat) / 2))

    chunk_size = 512
    for start in range(0, target_lat.size, chunk_size):
        stop = min(start + chunk_size, target_lat.size)
        dlat = src_lat[None, :] - target_lat[start:stop, None]
        dlon = (src_lon[None, :] - target_lon[start:stop, None]) * cos_lat
        nearest = np.argmin(dlat * dlat + dlon * dlon, axis=1)
        output[start:stop] = src_val[nearest]
    return output.reshape(len(target_lats), len(target_lons))


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
    return [
        [None if not math.isfinite(float(value)) else float(value) for value in row]
        for row in rounded
    ]


def build_payload(input_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    datasets = open_datasets(input_path)
    u_array = find_data_array(datasets, U_CANDIDATES)
    v_array = find_data_array(datasets, V_CANDIDATES)
    speed_array = find_data_array(datasets, SPEED_CANDIDATES)
    if u_array is None or v_array is None:
        available = sorted({name for dataset in datasets for name in dataset.data_vars})
        raise SystemExit(f"Cannot find 10 m U/V wind variables in {input_path}. Available: {available[:80]}")

    run_time = run_time_from_arrays(u_array, v_array)
    bbox = tuple(args.bbox)
    lead_hours = tuple(args.lead_hours) if args.lead_hours else common_lead_hours(u_array, v_array, speed_array)
    if not lead_hours:
        raise SystemExit(f"Cannot determine available MOLOCH lead hours in {input_path}.")

    steps = []
    for lead_hour in lead_hours:
        u_selected = select_lead(u_array, lead_hour)
        v_selected = select_lead(v_array, lead_hour)
        lat, lon = array_lat_lon(u_selected)
        u_values = np.asarray(u_selected.values, dtype=float)
        v_values = np.asarray(v_selected.values, dtype=float)
        if u_values.shape != v_values.shape:
            raise SystemExit(f"MOLOCH U/V shape mismatch at H+{lead_hour}: {u_values.shape} vs {v_values.shape}")

        u_grid = nearest_resample_curvilinear(u_values, lat, lon, bbox, args.grid_step_deg)
        v_grid = nearest_resample_curvilinear(v_values, lat, lon, bbox, args.grid_step_deg)
        if speed_array is not None:
            speed_selected = select_lead(speed_array, lead_hour)
            speed_values = np.asarray(speed_selected.values, dtype=float)
            speed_grid = nearest_resample_curvilinear(speed_values, *array_lat_lon(speed_selected), bbox, args.grid_step_deg)
        else:
            speed_grid = np.sqrt(u_grid * u_grid + v_grid * v_grid)
        valid_time = run_time + timedelta(hours=int(lead_hour))
        steps.append(
            {
                "lead_hour": int(lead_hour),
                "valid_time_utc": valid_time.isoformat().replace("+00:00", "Z"),
                "shape": list(speed_grid.shape),
                "stats_ms": finite_stats(speed_grid),
                "speed_ms": round_grid(speed_grid),
                "u_ms": round_grid(u_grid),
                "v_ms": round_grid(v_grid),
            }
        )

    shape = steps[0]["shape"]
    return {
        "format": "corsewind_moloch_corsica_wind_layer_v0",
        "generated_at_utc": utc_now(),
        "source": args.source_label,
        "product": "MOLOCH-ISAC",
        "resolution": "0.0113 deg / 1.2 km",
        "model_label": "MOLOCH 1.2 km",
        "height_agl_m": 10,
        "run_time_utc": run_time.isoformat().replace("+00:00", "Z"),
        "bbox_wgs84": list(bbox),
        "grid": {
            "orientation": "rows north-to-south, columns west-to-east",
            "lat_step_deg": round((bbox[3] - bbox[1]) / (shape[0] - 1), 6),
            "lon_step_deg": round((bbox[2] - bbox[0]) / (shape[1] - 1), 6),
            "resampling": "nearest source-grid point to regular WGS84 Corsica grid",
        },
        "source_file": str(input_path),
        "forecast_steps": steps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Local GRIB/NetCDF/CorseWind JSON file, or direct URL. Defaults to MOLOCH_SOURCE_URL.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/moloch_corsica_latest"))
    parser.add_argument("--output", type=Path, default=Path("visualizations/wind2d/moloch-corsica-latest.json"))
    parser.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BBOX, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--lead-hours", nargs="+", type=int, default=None, help="Lead hours to publish. Defaults to every lead hour available in the source bundle.")
    parser.add_argument("--grid-step-deg", type=float, default=DEFAULT_GRID_STEP_DEG)
    parser.add_argument("--download-timeout-sec", type=int, default=180)
    parser.add_argument("--source-label", default=DEFAULT_SOURCE_LABEL)
    parser.add_argument("--copy-source-to-raw", action="store_true", help="Copy a local source into raw-dir for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    args.raw_dir = resolve_path(args.raw_dir)
    args.output = resolve_path(args.output)
    source = source_from_args(args)
    input_path = ensure_source_file(source, args.raw_dir, args.download_timeout_sec)
    if args.copy_source_to_raw and not is_url(source):
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        copied = args.raw_dir / input_path.name
        if input_path.resolve() != copied.resolve():
            shutil.copy2(input_path, copied)
            input_path = copied
    if not optional_copy_json(input_path, args.output):
        payload = build_payload(input_path, args)
        write_layer_atomic(args.output, payload)
        print(
            f"wrote {args.output} run={payload['run_time_utc']} "
            f"steps={len(payload['forecast_steps'])} shape={payload['forecast_steps'][0]['shape']}"
        )


if __name__ == "__main__":
    main()
