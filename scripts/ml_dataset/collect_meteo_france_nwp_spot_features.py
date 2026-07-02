#!/usr/bin/env python3
"""Collect extra Meteo-France NWP fields and sample them at ML spots."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from meteo_france_client import coverage_ids, endpoint, load_dotenv, request_api  # noqa: E402
from sample_arome_tiff_at_stations import read_float64_tiff  # noqa: E402


DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_RAW_ROOT = DEFAULT_ML_ROOT / "meteo_france_nwp/raw/extra_fields"
DEFAULT_OUTPUT_ROOT = DEFAULT_ML_ROOT / "meteo_france_nwp/extra_field_samples"

FEATURES: dict[str, list[dict[str, Any]]] = {
    "arome": [
        {"name": "temperature_2m_c", "prefix": "TEMPERATURE__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", "height": 2, "transform": "k_to_c", "unit": "degC"},
        {"name": "dewpoint_2m_c", "prefix": "DEW_POINT_TEMPERATURE__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", "height": 2, "transform": "k_to_c", "unit": "degC"},
        {"name": "relative_humidity_2m_pct", "prefix": "RELATIVE_HUMIDITY__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", "height": 2, "unit": "%"},
        {"name": "pressure_msl_hpa", "prefix": "PRESSURE__MEAN_SEA_LEVEL", "transform": "pa_to_hpa", "unit": "hPa"},
        {"name": "pressure_surface_hpa", "prefix": "PRESSURE__GROUND_OR_WATER_SURFACE", "transform": "pa_to_hpa", "unit": "hPa"},
        {"name": "low_cloud_cover_pct", "prefix": "LOW_CLOUD_COVER__GROUND_OR_WATER_SURFACE", "unit": "%"},
        {"name": "total_cloud_cover_pct", "prefix": "TOTAL_CLOUD_COVER__GROUND_OR_WATER_SURFACE", "unit": "%"},
        {"name": "pbl_height_m", "prefix": "PLANETARY_BOUNDARY_LAYER_HEIGHT__GROUND_OR_WATER_SURFACE", "unit": "m"},
        {"name": "cape_j_kg", "prefix": "MEAN_LAYER_CAPE__GROUND_OR_WATER_SURFACE", "unit": "J/kg"},
        {"name": "downward_shortwave_flux_w_m2", "prefix": "DOWNWARD_SHORT_WAVE_RADIATION_FLUX__GROUND_OR_WATER_SURFACE", "suffix": "_PT1H", "transform": "j_m2_to_w_m2", "period_seconds": 3600, "unit": "W/m2"},
    ],
    "aromepi": [
        {"name": "wind_speed_10m_ms", "prefix": "WIND_SPEED__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", "height": 10, "unit": "m/s"},
        {"name": "wind_u_10m_ms", "prefix": "U_COMPONENT_OF_WIND__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", "height": 10, "unit": "m/s"},
        {"name": "wind_v_10m_ms", "prefix": "V_COMPONENT_OF_WIND__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", "height": 10, "unit": "m/s"},
        {"name": "temperature_2m_c", "prefix": "TEMPERATURE__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", "height": 2, "transform": "k_to_c", "unit": "degC"},
        {"name": "dewpoint_10m_c", "prefix": "DEW_POINT_TEMPERATURE__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", "height": 10, "transform": "k_to_c", "unit": "degC"},
        {"name": "relative_humidity_10m_pct", "prefix": "RELATIVE_HUMIDITY__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", "height": 10, "unit": "%"},
        {"name": "pressure_msl_hpa", "prefix": "PRESSURE__SEA_SURFACE", "transform": "pa_to_hpa", "unit": "hPa"},
        {"name": "cloud_cover_pct", "prefix": "NEBUL__GROUND_OR_WATER_SURFACE", "unit": "%"},
        {"name": "downward_shortwave_flux_w_m2", "prefix": "DOWNWARD_SHORT_WAVE_RADIATION_FLUX__GROUND_OR_WATER_SURFACE", "suffix": "_PT15M", "transform": "j_m2_to_w_m2", "period_seconds": 900, "unit": "W/m2"},
        {"name": "direct_downward_shortwave_flux_w_m2", "prefix": "DIRECT_DOWNWARD_SHORT_WAVE_RADIATION_FLUX__GROUND_OR_WATER_SURFACE", "suffix": "_PT15M", "transform": "j_m2_to_w_m2", "period_seconds": 900, "unit": "W/m2"},
        {"name": "net_shortwave_clear_sky_w_m2", "prefix": "NET_SHORT_WAVE_RADIATION_CLEAR_SKY__GROUND_OR_WATER_SURFACE", "suffix": "_PT15M", "transform": "j_m2_to_w_m2", "period_seconds": 900, "unit": "W/m2"},
    ],
}
SOURCE_DEFAULT_RESOLUTION = {
    "arome": "001",
    "aromepi": "0025",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def finite_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
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


def parse_utc_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def coverage_run_time(coverage_id: str, prefix: str, suffix: str = "") -> datetime | None:
    pattern = re.compile(
        rf"^{re.escape(prefix)}___(\d{{4}}-\d{{2}}-\d{{2}})T(\d{{2}})\.00\.00Z{re.escape(suffix)}$"
    )
    match = pattern.match(coverage_id)
    if not match:
        return None
    return datetime.fromisoformat(f"{match.group(1)}T{match.group(2)}:00:00+00:00")


def coverage_for_run(coverage_list: list[str], run_time: datetime, feature: dict[str, Any]) -> str | None:
    suffix = str(feature.get("suffix") or "")
    for coverage_id in coverage_list:
        parsed = coverage_run_time(coverage_id, feature["prefix"], suffix)
        if parsed == run_time:
            return coverage_id
    return None


def selected_features(source: str, names: set[str]) -> list[dict[str, Any]]:
    features = FEATURES[source]
    if not names:
        return features
    selected = [feature for feature in features if feature["name"] in names]
    missing = sorted(names - {feature["name"] for feature in selected})
    if missing:
        raise SystemExit(f"Unknown {source} extra feature(s): {', '.join(missing)}")
    return selected


def grid_position(lat: float, lon: float, bbox: list[float], rows: int, cols: int) -> tuple[float, float, bool]:
    west, south, east, north = bbox
    x = (lon - west) / (east - west) * (cols - 1)
    y = (north - lat) / (north - south) * (rows - 1)
    return x, y, 0 <= x <= cols - 1 and 0 <= y <= rows - 1


def grid_value(grid: np.ndarray, row: int, col: int) -> float | None:
    if row < 0 or col < 0 or row >= grid.shape[0] or col >= grid.shape[1]:
        return None
    return finite_float(grid[row, col])


def sample_bilinear(grid: np.ndarray, x: float, y: float) -> float | None:
    x0 = math.floor(x)
    y0 = math.floor(y)
    x1 = math.ceil(x)
    y1 = math.ceil(y)
    q11 = grid_value(grid, y0, x0)
    q21 = grid_value(grid, y0, x1)
    q12 = grid_value(grid, y1, x0)
    q22 = grid_value(grid, y1, x1)
    if None in {q11, q21, q12, q22}:
        return grid_value(grid, int(round(y)), int(round(x)))
    if x0 == x1 and y0 == y1:
        return q11
    if x0 == x1:
        return q11 * (y1 - y) + q12 * (y - y0)
    if y0 == y1:
        return q11 * (x1 - x) + q21 * (x - x0)
    return (
        q11 * (x1 - x) * (y1 - y)
        + q21 * (x - x0) * (y1 - y)
        + q12 * (x1 - x) * (y - y0)
        + q22 * (x - x0) * (y - y0)
    )


def transform_value(value: float | None, feature: dict[str, Any]) -> float | None:
    if value is None:
        return None
    transform = feature.get("transform")
    if transform == "k_to_c" and value > 150:
        value -= 273.15
    elif transform == "pa_to_hpa":
        value *= 0.01
    elif transform == "j_m2_to_w_m2":
        value /= float(feature.get("period_seconds") or 1)
    return round(value, 4)


def finite_stats(array: np.ndarray, feature: dict[str, Any]) -> dict[str, float | None]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"min": None, "mean": None, "max": None}
    values = [transform_value(float(value), feature) for value in finite]
    values = [value for value in values if value is not None]
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": round(min(values), 4),
        "mean": round(sum(values) / len(values), 4),
        "max": round(max(values), 4),
    }


def download_tiff(
    product: str,
    resolution: str,
    coverage_id: str,
    output: Path,
    bbox: list[float],
    valid_time: datetime,
    auth_header: str,
    height_m: int | None,
) -> bool:
    if output.exists() and output.stat().st_size > 0:
        return False
    min_lon, min_lat, max_lon, max_lat = bbox
    params: list[tuple[str, str]] = [
        ("service", "WCS"),
        ("version", "2.0.1"),
        ("coverageid", coverage_id),
        ("format", "image/tiff"),
        ("subset", f"long({min_lon},{max_lon})"),
        ("subset", f"lat({min_lat},{max_lat})"),
        ("subset", f"time({valid_time.isoformat().replace('+00:00', 'Z')})"),
    ]
    if height_m is not None:
        params.insert(-1, ("subset", f"height({height_m})"))
    response = request_api(endpoint(product, resolution, "GetCoverage"), params, auth_header)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    tmp.write_bytes(response.content)
    tmp.replace(output)
    return True


def is_optional_unavailable(exc: BaseException) -> bool:
    text = str(exc)
    return "InvalidSubsetting" in text or "time must be" in text or "NoApplicableCode" in text


def sample_feature(
    source: str,
    resolution: str,
    run_time: datetime,
    valid_time: datetime,
    feature: dict[str, Any],
    coverage_id: str,
    bbox: list[float],
    raw_root: Path,
    auth_header: str,
    request_sleep_sec: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    lead_minutes = int((valid_time - run_time).total_seconds() // 60)
    suffix = str(feature.get("suffix") or "").strip("_").lower()
    suffix_part = f"_{suffix}" if suffix else ""
    output = raw_root / source / f"run={run_time:%Y%m%dT%H}" / f"{source}_{resolution}_m{lead_minutes:04d}_{feature['name']}{suffix_part}.tiff"
    try:
        downloaded = download_tiff(
            source,
            resolution,
            coverage_id,
            output,
            bbox,
            valid_time,
            auth_header,
            feature.get("height"),
        )
        if downloaded and request_sleep_sec > 0:
            time.sleep(request_sleep_sec)
        raster = read_float64_tiff(output)
    except SystemExit as exc:
        if is_optional_unavailable(exc):
            return None, {"status": "unavailable", "error": str(exc)}
        raise
    return raster, {
        "status": "ok",
        "coverage_id": coverage_id,
        "raw_path": str(output),
        "stats": finite_stats(raster, feature),
    }


def output_path(output_root: Path, source: str, valid_time: str | None) -> Path:
    day = (valid_time or utc_now())[:10]
    return output_root / f"source={source}" / f"date={day}" / "extra_fields.jsonl"


def write_jsonl_by_valid_day(output_root: Path, source: str, rows: list[dict[str, Any]]) -> dict[str, int]:
    by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_path[output_path(output_root, source, row.get("valid_time_utc"))].append(row)
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
            (row.get("source"), row.get("run_time_utc"), row.get("valid_time_utc"), row.get("spot_id")): row
            for row in [*existing, *path_rows]
        }
        ordered = sorted(deduped.values(), key=lambda row: (row.get("valid_time_utc") or "", row.get("spot_id") or ""))
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")
        tmp.replace(path)
        written[str(path)] = len(ordered)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["arome", "aromepi"], required=True)
    parser.add_argument("--input", required=True, type=Path, help="Latest Wind2D model JSON used for run and valid-time metadata.")
    parser.add_argument("--resolution", choices=["001", "0025"])
    parser.add_argument("--auth-header", choices=["apikey", "bearer"], default="apikey")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--feature", action="append", default=[], help="Feature name to collect. Repeatable.")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--request-sleep-sec", type=float, default=0.0)
    parser.add_argument("--include-context-spots", action="store_true")
    parser.add_argument("--spot-id", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    input_path = resolve_path(args.input)
    registry_path = resolve_path(args.registry)
    raw_root = resolve_path(args.raw_root)
    output_root = resolve_path(args.output_root)
    source = args.source
    resolution = args.resolution or SOURCE_DEFAULT_RESOLUTION[source]
    layer = read_json(input_path)
    run_time = parse_utc_datetime(str(layer.get("run_time_utc")))
    bbox = layer.get("bbox_wgs84")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise SystemExit("Input layer has no valid bbox_wgs84.")
    spots = load_spots(registry_path, args.include_context_spots, set(args.spot_id))
    features = selected_features(source, set(args.feature))
    capabilities = coverage_ids(request_api(endpoint(source, resolution, "GetCapabilities"), [("service", "WCS"), ("version", "2.0.1"), ("language", "eng")], args.auth_header).text)
    coverage_by_feature = {
        feature["name"]: coverage_for_run(capabilities, run_time, feature)
        for feature in features
    }
    steps = list(layer.get("forecast_steps") or [])
    if args.max_steps > 0:
        steps = steps[: args.max_steps]
    rows: list[dict[str, Any]] = []
    feature_status_by_step: dict[str, dict[str, Any]] = {}
    for step in steps:
        valid_time = parse_utc_datetime(str(step.get("valid_time_utc")))
        shape = step.get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            continue
        sampled_rasters: dict[str, np.ndarray] = {}
        feature_metadata: dict[str, Any] = {}
        for feature in features:
            coverage_id = coverage_by_feature.get(feature["name"])
            if not coverage_id:
                feature_metadata[feature["name"]] = {"status": "missing_coverage"}
                continue
            raster, metadata = sample_feature(
                source,
                resolution,
                run_time,
                valid_time,
                feature,
                coverage_id,
                bbox,
                raw_root,
                args.auth_header,
                args.request_sleep_sec,
            )
            feature_metadata[feature["name"]] = metadata
            if raster is not None:
                sampled_rasters[feature["name"]] = raster
        feature_status_by_step[step.get("valid_time_utc")] = feature_metadata
        for spot in spots:
            lat = float(spot["latitude"])
            lon = float(spot["longitude"])
            values: dict[str, float | None] = {}
            units: dict[str, str] = {}
            inside_grid = None
            grid_x = None
            grid_y = None
            for feature in features:
                raster = sampled_rasters.get(feature["name"])
                if raster is None:
                    values[feature["name"]] = None
                    units[feature["name"]] = feature.get("unit")
                    continue
                x, y, inside = grid_position(lat, lon, bbox, raster.shape[0], raster.shape[1])
                inside_grid = inside if inside_grid is None else inside_grid and inside
                grid_x = x if grid_x is None else grid_x
                grid_y = y if grid_y is None else grid_y
                sampled = sample_bilinear(raster, x, y) if inside else None
                values[feature["name"]] = transform_value(sampled, feature)
                units[feature["name"]] = feature.get("unit")
            rows.append({
                "format": "corsewind.ml_meteo_france_nwp_extra_spot_sample.v1",
                "source": source,
                "product": source,
                "resolution": resolution,
                "model_label": layer.get("model_label"),
                "run_time_utc": layer.get("run_time_utc"),
                "generated_at_utc": layer.get("generated_at_utc"),
                "valid_time_utc": step.get("valid_time_utc"),
                "lead_hour": step.get("lead_hour"),
                "lead_minutes": step.get("lead_minutes"),
                "spot_id": spot.get("spot_id"),
                "spot_name": spot.get("name"),
                "spot_kind": spot.get("kind"),
                "spot_source_type": spot.get("source_type"),
                "station_id": spot.get("station_id"),
                "latitude": lat,
                "longitude": lon,
                "use_for_ml": bool(spot.get("use_for_ml", False)),
                "sample_method": "bilinear",
                "grid_x": round(grid_x, 4) if grid_x is not None else None,
                "grid_y": round(grid_y, 4) if grid_y is not None else None,
                "inside_grid": inside_grid,
                "features": values,
                "units": units,
                "feature_metadata": feature_metadata,
                "bbox_wgs84": bbox,
                "sampled_at_utc": utc_now(),
            })
    written = write_jsonl_by_valid_day(output_root, source, rows)
    print(json.dumps({
        "generated_at_utc": utc_now(),
        "source": source,
        "resolution": resolution,
        "input": str(input_path),
        "registry": str(registry_path),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "run_time_utc": layer.get("run_time_utc"),
        "step_count": len(steps),
        "spot_count": len(spots),
        "row_count": len(rows),
        "features": [feature["name"] for feature in features],
        "coverage_by_feature": coverage_by_feature,
        "feature_status_by_step": feature_status_by_step,
        "written": written,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
