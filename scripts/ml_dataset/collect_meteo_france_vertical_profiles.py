#!/usr/bin/env python3
"""Collect AROME isobaric vertical profiles and sample them at ML spots."""

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
DEFAULT_INPUT = ROOT / "visualizations/wind2d/arome-corsica-latest.json"
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_RAW_ROOT = DEFAULT_ML_ROOT / "meteo_france_nwp/raw/vertical_profiles"
DEFAULT_OUTPUT_ROOT = DEFAULT_ML_ROOT / "meteo_france_nwp/vertical_profiles"
DEFAULT_PRESSURE_LEVELS_HPA = [1000, 950, 925, 900, 850, 800, 750, 700]
G0 = 9.80665

FEATURES: dict[str, dict[str, Any]] = {
    "temperature_c": {
        "prefix": "TEMPERATURE__ISOBARIC_SURFACE",
        "transform": "k_to_c",
        "unit": "degC",
    },
    "relative_humidity_pct": {
        "prefix": "RELATIVE_HUMIDITY__ISOBARIC_SURFACE",
        "unit": "%",
    },
    "vertical_velocity_pressure_pa_s": {
        "prefix": "VERTICAL_VELOCITY_PRESSURE__ISOBARIC_SURFACE",
        "unit": "Pa/s",
    },
    "geopotential_height_m": {
        "prefix": "GEOPOTENTIAL__ISOBARIC_SURFACE",
        "transform": "geopotential_to_m",
        "unit": "m",
    },
    "pseudo_adiabatic_potential_temperature_c": {
        "prefix": "PSEUDO_ADIABATIC_POTENTIAL_TEMPERATURE__ISOBARIC_SURFACE",
        "transform": "k_to_c",
        "unit": "degC",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_utc_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def coverage_run_time(coverage_id: str, prefix: str) -> datetime | None:
    pattern = re.compile(rf"^{re.escape(prefix)}___(\d{{4}}-\d{{2}}-\d{{2}})T(\d{{2}})\.00\.00Z$")
    match = pattern.match(coverage_id)
    if not match:
        return None
    return datetime.fromisoformat(f"{match.group(1)}T{match.group(2)}:00:00+00:00")


def coverage_for_run(coverage_list: list[str], run_time: datetime, prefix: str) -> str | None:
    for coverage_id in coverage_list:
        if coverage_run_time(coverage_id, prefix) == run_time:
            return coverage_id
    return None


def selected_features(names: set[str]) -> dict[str, dict[str, Any]]:
    if not names:
        return FEATURES
    missing = sorted(names - set(FEATURES))
    if missing:
        raise SystemExit(f"Unknown vertical profile feature(s): {', '.join(missing)}")
    return {name: FEATURES[name] for name in FEATURES if name in names}


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
    elif transform == "geopotential_to_m":
        value /= G0
    return round(value, 4)


def download_tiff(
    coverage_id: str,
    valid_time: datetime,
    pressure_hpa: int,
    bbox: list[float],
    output: Path,
    auth_header: str,
) -> bool:
    if output.exists() and output.stat().st_size > 0:
        return False
    min_lon, min_lat, max_lon, max_lat = bbox
    params = [
        ("service", "WCS"),
        ("version", "2.0.1"),
        ("coverageid", coverage_id),
        ("format", "image/tiff"),
        ("subset", f"long({min_lon},{max_lon})"),
        ("subset", f"lat({min_lat},{max_lat})"),
        ("subset", f"pressure({pressure_hpa})"),
        ("subset", f"time({iso_z(valid_time)})"),
    ]
    response = request_api(endpoint("arome", "0025", "GetCoverage"), params, auth_header)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    tmp.write_bytes(response.content)
    tmp.replace(output)
    return True


def optional_unavailable(exc: BaseException) -> bool:
    text = str(exc)
    return "InvalidSubsetting" in text or "time must be" in text or "NoApplicableCode" in text


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def sample_slice(
    coverage_id: str,
    feature_name: str,
    feature: dict[str, Any],
    run_time: datetime,
    valid_time: datetime,
    pressure_hpa: int,
    bbox: list[float],
    raw_root: Path,
    auth_header: str,
    request_sleep_sec: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    lead_minutes = int((valid_time - run_time).total_seconds() // 60)
    output = (
        raw_root
        / "arome"
        / "resolution=0025"
        / f"run={run_time:%Y%m%dT%H}"
        / f"m{lead_minutes:04d}"
        / f"{feature_name}_p{pressure_hpa}.tiff"
    )
    try:
        downloaded = download_tiff(coverage_id, valid_time, pressure_hpa, bbox, output, auth_header)
        if downloaded and request_sleep_sec > 0:
            time.sleep(request_sleep_sec)
        raster = read_float64_tiff(output)
    except SystemExit as exc:
        if optional_unavailable(exc):
            return None, {"status": "unavailable", "error": str(exc)}
        raise
    finite = raster[np.isfinite(raster)]
    stats = {"min": None, "mean": None, "max": None}
    if finite.size:
        values = [transform_value(float(value), feature) for value in finite]
        values = [value for value in values if value is not None]
        if values:
            stats = {"min": round(min(values), 4), "mean": round(sum(values) / len(values), 4), "max": round(max(values), 4)}
    return raster, {
        "status": "ok",
        "coverage_id": coverage_id,
        "pressure_hpa": pressure_hpa,
        "raw_path": str(output),
        "stats": stats,
    }


def first_value(profile: dict[str, dict[str, float | None]], feature: str, level: int) -> float | None:
    values = profile.get(feature)
    if not values:
        return None
    return values.get(str(level))


def level_range_values(profile: dict[str, dict[str, float | None]], feature: str, levels: list[int]) -> list[float]:
    values = []
    for level in levels:
        value = first_value(profile, feature, level)
        if value is not None:
            values.append(value)
    return values


def derive_features(profile: dict[str, dict[str, float | None]], levels: list[int]) -> dict[str, float | None]:
    low_levels = [level for level in levels if 850 <= level <= 1000]
    derived: dict[str, float | None] = {
        "geopotential_thickness_1000_850_m": None,
        "temperature_lapse_rate_1000_850_c_per_km": None,
        "relative_humidity_mean_1000_850_pct": None,
        "vertical_velocity_pressure_850_pa_s": first_value(profile, "vertical_velocity_pressure_pa_s", 850),
        "low_level_inversion_strength_c": None,
    }
    z1000 = first_value(profile, "geopotential_height_m", 1000)
    z850 = first_value(profile, "geopotential_height_m", 850)
    t1000 = first_value(profile, "temperature_c", 1000)
    t850 = first_value(profile, "temperature_c", 850)
    if z1000 is not None and z850 is not None:
        derived["geopotential_thickness_1000_850_m"] = round(z850 - z1000, 4)
    if z1000 is not None and z850 is not None and t1000 is not None and t850 is not None and z850 != z1000:
        derived["temperature_lapse_rate_1000_850_c_per_km"] = round((t1000 - t850) / ((z850 - z1000) / 1000.0), 4)
    rh_values = level_range_values(profile, "relative_humidity_pct", low_levels)
    if rh_values:
        derived["relative_humidity_mean_1000_850_pct"] = round(sum(rh_values) / len(rh_values), 4)
    inversion_strengths = []
    for lower, upper in zip(levels, levels[1:]):
        if lower < upper:
            continue
        lower_temp = first_value(profile, "temperature_c", lower)
        upper_temp = first_value(profile, "temperature_c", upper)
        if lower_temp is not None and upper_temp is not None and upper_temp > lower_temp:
            inversion_strengths.append(upper_temp - lower_temp)
    if inversion_strengths:
        derived["low_level_inversion_strength_c"] = round(max(inversion_strengths), 4)
    return derived


def output_path(output_root: Path, valid_time: str | None) -> Path:
    day = (valid_time or utc_now())[:10]
    return output_root / "source=arome" / "resolution=0025" / f"date={day}" / "vertical_profiles.jsonl"


def write_jsonl_by_valid_day(output_root: Path, rows: list[dict[str, Any]]) -> dict[str, int]:
    by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_path[output_path(output_root, row.get("valid_time_utc"))].append(row)
    written: dict[str, int] = {}
    for path, path_rows in by_path.items():
        existing = []
        if path.exists():
            existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        deduped = {
            (row.get("source"), row.get("resolution"), row.get("run_time_utc"), row.get("valid_time_utc"), row.get("spot_id")): row
            for row in [*existing, *path_rows]
        }
        ordered = sorted(deduped.values(), key=lambda row: (row.get("valid_time_utc") or "", row.get("spot_id") or ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")
        tmp.replace(path)
        written[str(path)] = len(ordered)
    return written


def parse_bbox(values: list[float] | None, layer: dict[str, Any]) -> list[float]:
    if values:
        return [float(value) for value in values]
    bbox = layer.get("bbox_wgs84")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise SystemExit("No valid bbox. Pass --bbox or use an input layer with bbox_wgs84.")
    return [float(value) for value in bbox]


def parse_steps(args: argparse.Namespace, layer: dict[str, Any], run_time: datetime) -> list[dict[str, Any]]:
    if args.valid_time_utc:
        return [
            {
                "valid_time_utc": iso_z(parse_utc_datetime(value)),
                "lead_minutes": int((parse_utc_datetime(value) - run_time).total_seconds() // 60),
            }
            for value in args.valid_time_utc
        ]
    steps = list(layer.get("forecast_steps") or [])
    if args.max_steps > 0:
        steps = steps[: args.max_steps]
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="AROME layer JSON used for run/valid-time metadata.")
    parser.add_argument("--run-time-utc", help="Override run time instead of reading --input.")
    parser.add_argument("--valid-time-utc", action="append", default=[], help="Valid time to collect. Repeatable. Defaults to input forecast steps.")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--auth-header", choices=["apikey", "bearer"], default="apikey")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--feature", action="append", default=[], help="Profile feature name to collect. Repeatable.")
    parser.add_argument("--pressure-level-hpa", type=int, action="append", default=[], help="Pressure level in hPa. Repeatable.")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--request-sleep-sec", type=float, default=0.2)
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
    layer = read_json(input_path) if input_path.exists() else {}
    run_time = parse_utc_datetime(args.run_time_utc or str(layer.get("run_time_utc")))
    bbox = parse_bbox(args.bbox, layer)
    steps = parse_steps(args, layer, run_time)
    levels = sorted(args.pressure_level_hpa or DEFAULT_PRESSURE_LEVELS_HPA, reverse=True)
    features = selected_features(set(args.feature))
    spots = load_spots(registry_path, args.include_context_spots, set(args.spot_id))

    capabilities = coverage_ids(
        request_api(
            endpoint("arome", "0025", "GetCapabilities"),
            [("service", "WCS"), ("version", "2.0.1"), ("language", "eng")],
            args.auth_header,
        ).text
    )
    coverage_by_feature = {
        name: coverage_for_run(capabilities, run_time, feature["prefix"])
        for name, feature in features.items()
    }

    rows: list[dict[str, Any]] = []
    status_by_step: dict[str, Any] = {}
    for step in steps:
        valid_text = str(step.get("valid_time_utc"))
        valid_time = parse_utc_datetime(valid_text)
        rasters: dict[tuple[str, int], np.ndarray] = {}
        metadata: dict[str, Any] = {}
        for feature_name, feature in features.items():
            coverage_id = coverage_by_feature.get(feature_name)
            metadata[feature_name] = {}
            if not coverage_id:
                metadata[feature_name]["coverage"] = {"status": "missing_coverage"}
                continue
            for level in levels:
                raster, slice_meta = sample_slice(
                    coverage_id,
                    feature_name,
                    feature,
                    run_time,
                    valid_time,
                    level,
                    bbox,
                    raw_root,
                    args.auth_header,
                    args.request_sleep_sec,
                )
                metadata[feature_name][str(level)] = slice_meta
                if raster is not None:
                    rasters[(feature_name, level)] = raster
        status_by_step[valid_text] = metadata

        for spot in spots:
            lat = float(spot["latitude"])
            lon = float(spot["longitude"])
            profile: dict[str, dict[str, float | None]] = {name: {} for name in features}
            units = {name: feature.get("unit") for name, feature in features.items()}
            inside_grid = None
            grid_x = None
            grid_y = None
            for feature_name, feature in features.items():
                for level in levels:
                    raster = rasters.get((feature_name, level))
                    if raster is None:
                        profile[feature_name][str(level)] = None
                        continue
                    x, y, inside = grid_position(lat, lon, bbox, raster.shape[0], raster.shape[1])
                    inside_grid = inside if inside_grid is None else inside_grid and inside
                    grid_x = x if grid_x is None else grid_x
                    grid_y = y if grid_y is None else grid_y
                    sampled = sample_bilinear(raster, x, y) if inside else None
                    profile[feature_name][str(level)] = transform_value(sampled, feature)
            rows.append({
                "format": "corsewind.ml_meteo_france_nwp_vertical_profile_spot_sample.v1",
                "source": "arome",
                "product": "arome",
                "resolution": "0025",
                "model_label": layer.get("model_label") or "AROME 0.025 isobaric profile",
                "run_time_utc": iso_z(run_time),
                "generated_at_utc": layer.get("generated_at_utc"),
                "valid_time_utc": valid_text,
                "lead_minutes": step.get("lead_minutes"),
                "spot_id": spot.get("spot_id"),
                "spot_name": spot.get("name"),
                "spot_kind": spot.get("kind"),
                "spot_source_type": spot.get("source_type"),
                "station_id": spot.get("station_id"),
                "latitude": lat,
                "longitude": lon,
                "use_for_ml": bool(spot.get("use_for_ml", False)),
                "pressure_levels_hpa": levels,
                "profile": profile,
                "derived_features": derive_features(profile, levels),
                "units": units,
                "feature_metadata": metadata,
                "sample_method": "bilinear",
                "grid_x": round(grid_x, 4) if grid_x is not None else None,
                "grid_y": round(grid_y, 4) if grid_y is not None else None,
                "inside_grid": inside_grid,
                "bbox_wgs84": bbox,
                "sampled_at_utc": utc_now(),
            })
    written = write_jsonl_by_valid_day(output_root, rows)
    print(json.dumps({
        "generated_at_utc": utc_now(),
        "source": "arome",
        "resolution": "0025",
        "input": str(input_path),
        "registry": str(registry_path),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "run_time_utc": iso_z(run_time),
        "step_count": len(steps),
        "spot_count": len(spots),
        "row_count": len(rows),
        "pressure_levels_hpa": levels,
        "features": list(features),
        "coverage_by_feature": coverage_by_feature,
        "feature_status_by_step": status_by_step,
        "written": written,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
