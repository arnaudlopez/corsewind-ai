#!/usr/bin/env python3
"""Generate DEM-derived static spot features for CorseWind ML spots."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_DEM_DIR = ROOT / "data/raw/dem/copernicus_glo30"
DEFAULT_OUTPUT = ROOT / "configs/ml_spot_static_features.json"
SECTOR_BEARINGS = {
    "n": 0.0,
    "ne": 45.0,
    "e": 90.0,
    "se": 135.0,
    "s": 180.0,
    "sw": 225.0,
    "w": 270.0,
    "nw": 315.0,
}


@dataclass(frozen=True)
class Sample:
    distance_km: float
    bearing_deg: float
    elevation_m: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def import_deps() -> dict[str, Any]:
    try:
        import numpy as np
        import rasterio
    except ImportError as exc:
        raise SystemExit(
            "Missing raster dependency. Install with: python -m pip install rasterio"
        ) from exc
    return {"np": np, "rasterio": rasterio}


def load_spots(path: Path, include_context: bool) -> list[dict[str, Any]]:
    payload = read_json(path)
    spots = payload.get("spots", []) if isinstance(payload, dict) else payload
    out = []
    for spot in spots:
        if not isinstance(spot, dict) or not spot.get("spot_id"):
            continue
        if not include_context and not spot.get("use_for_ml", False):
            continue
        if finite_float(spot.get("latitude")) is None or finite_float(spot.get("longitude")) is None:
            continue
        out.append(spot)
    return out


def destination_point(latitude: float, longitude: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    radius_km = 6371.0088
    angular = distance_km / radius_km
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(latitude)
    lon1 = math.radians(longitude)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular)
        + math.cos(lat1) * math.sin(angular) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(lat1),
        math.cos(angular) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), ((math.degrees(lon2) + 540.0) % 360.0) - 180.0


def angle_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def open_dem_tiles(dem_dir: Path, pattern: str, rasterio: Any) -> list[Any]:
    paths = sorted(dem_dir.glob(pattern))
    if not paths:
        raise SystemExit(f"No DEM tiles found in {dem_dir} matching {pattern!r}.")
    return [rasterio.open(path) for path in paths]


def tile_contains(dataset: Any, latitude: float, longitude: float) -> bool:
    bounds = dataset.bounds
    return bounds.left <= longitude <= bounds.right and bounds.bottom <= latitude <= bounds.top


def sample_elevation(datasets: list[Any], latitude: float, longitude: float) -> float | None:
    for dataset in datasets:
        if not tile_contains(dataset, latitude, longitude):
            continue
        try:
            value = next(dataset.sample([(longitude, latitude)]))[0]
        except (StopIteration, IndexError, ValueError):
            continue
        nodata = dataset.nodata
        number = finite_float(value)
        if number is None:
            continue
        if nodata is not None and math.isclose(number, float(nodata)):
            continue
        return number
    return None


def percentile(values: list[float], q: float, np: Any) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values, dtype="float64"), q)), 6)


def aggregate(values: list[float], np: Any) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "p90": None}
    array = np.asarray(values, dtype="float64")
    return {
        "count": int(len(values)),
        "mean": round(float(np.mean(array)), 6),
        "min": round(float(np.min(array)), 6),
        "max": round(float(np.max(array)), 6),
        "p90": round(float(np.percentile(array, 90)), 6),
    }


def collect_radial_samples(
    datasets: list[Any],
    latitude: float,
    longitude: float,
    *,
    max_radius_km: float,
    radial_step_km: float,
    bearing_step_deg: float,
) -> list[Sample]:
    samples = []
    bearing_count = int(round(360.0 / bearing_step_deg))
    distance_count = int(math.floor(max_radius_km / radial_step_km))
    for bearing_index in range(bearing_count):
        bearing = (bearing_index * bearing_step_deg) % 360.0
        for distance_index in range(1, distance_count + 1):
            distance = distance_index * radial_step_km
            lat, lon = destination_point(latitude, longitude, bearing, distance)
            elevation = sample_elevation(datasets, lat, lon)
            if elevation is not None:
                samples.append(Sample(distance, bearing, elevation))
    return samples


def add_radius_features(
    features: dict[str, Any],
    samples: list[Sample],
    spot_elevation: float | None,
    radii_km: list[float],
    np: Any,
) -> None:
    for radius in radii_km:
        values = [sample.elevation_m for sample in samples if sample.distance_km <= radius]
        stats = aggregate(values, np)
        suffix = str(radius).replace(".", "p")
        for key, value in stats.items():
            features[f"dem_radius_{suffix}km_elevation_{key}"] = value
        if spot_elevation is not None and values:
            rel = [value - spot_elevation for value in values]
            rel_stats = aggregate(rel, np)
            for key, value in rel_stats.items():
                features[f"dem_radius_{suffix}km_relief_{key}"] = value
            features[f"dem_radius_{suffix}km_relief_range_m"] = round(max(rel) - min(rel), 6)
            features[f"dem_radius_{suffix}km_uphill_share"] = round(sum(value > 25.0 for value in rel) / len(rel), 6)


def add_sector_features(
    features: dict[str, Any],
    samples: list[Sample],
    spot_elevation: float | None,
    *,
    sector_half_width_deg: float,
    sector_max_km: float,
    np: Any,
) -> None:
    for name, bearing in SECTOR_BEARINGS.items():
        sector_samples = [
            sample
            for sample in samples
            if sample.distance_km <= sector_max_km and angle_diff_deg(sample.bearing_deg, bearing) <= sector_half_width_deg
        ]
        values = [sample.elevation_m for sample in sector_samples]
        stats = aggregate(values, np)
        for key, value in stats.items():
            features[f"dem_sector_{name}_{int(sector_max_km)}km_elevation_{key}"] = value
        if spot_elevation is not None and sector_samples:
            relief = [sample.elevation_m - spot_elevation for sample in sector_samples]
            relief_stats = aggregate(relief, np)
            for key, value in relief_stats.items():
                features[f"dem_sector_{name}_{int(sector_max_km)}km_relief_{key}"] = value
            max_relief = max(relief)
            p90_relief = percentile(relief, 90, np)
            low_or_sea = [sample for sample in sector_samples if sample.elevation_m <= 5.0]
            high_barrier = [sample for sample in sector_samples if sample.elevation_m - spot_elevation >= 150.0]
            mountain_barrier = [sample for sample in sector_samples if sample.elevation_m - spot_elevation >= 500.0]
            features[f"dem_sector_{name}_{int(sector_max_km)}km_barrier_max_m"] = round(max_relief, 6)
            features[f"dem_sector_{name}_{int(sector_max_km)}km_barrier_p90_m"] = p90_relief
            features[f"dem_sector_{name}_{int(sector_max_km)}km_barrier_sample_share"] = round(len(high_barrier) / len(sector_samples), 6)
            features[f"dem_sector_{name}_{int(sector_max_km)}km_low_or_sea_sample_share"] = round(
                len(low_or_sea) / len(sector_samples),
                6,
            )
            features[f"dem_sector_{name}_{int(sector_max_km)}km_open_exposure_score"] = round(
                1.0 - (len(high_barrier) / len(sector_samples)),
                6,
            )
            if high_barrier:
                features[f"dem_sector_{name}_{int(sector_max_km)}km_nearest_barrier_distance_km"] = round(
                    min(sample.distance_km for sample in high_barrier),
                    6,
                )
            else:
                features[f"dem_sector_{name}_{int(sector_max_km)}km_nearest_barrier_distance_km"] = None
            if mountain_barrier:
                features[f"dem_sector_{name}_{int(sector_max_km)}km_nearest_mountain_500m_distance_km"] = round(
                    min(sample.distance_km for sample in mountain_barrier),
                    6,
                )
            else:
                features[f"dem_sector_{name}_{int(sector_max_km)}km_nearest_mountain_500m_distance_km"] = None


def add_gradient_features(features: dict[str, Any], samples: list[Sample], spot_elevation: float | None, np: Any) -> None:
    if spot_elevation is None:
        return
    pairs = [("n", "s"), ("e", "w"), ("ne", "sw"), ("se", "nw")]
    sector_mean = {}
    for name, bearing in SECTOR_BEARINGS.items():
        values = [
            sample.elevation_m - spot_elevation
            for sample in samples
            if sample.distance_km <= 10.0 and angle_diff_deg(sample.bearing_deg, bearing) <= 22.5
        ]
        sector_mean[name] = None if not values else float(np.mean(np.asarray(values, dtype="float64")))
    for a, b in pairs:
        va = sector_mean.get(a)
        vb = sector_mean.get(b)
        features[f"dem_relief_gradient_{a}_minus_{b}_m"] = None if va is None or vb is None else round(va - vb, 6)


def build_features_for_spot(spot: dict[str, Any], datasets: list[Any], args: argparse.Namespace, np: Any) -> dict[str, Any]:
    lat = float(spot["latitude"])
    lon = float(spot["longitude"])
    spot_elevation = sample_elevation(datasets, lat, lon)
    samples = collect_radial_samples(
        datasets,
        lat,
        lon,
        max_radius_km=args.max_radius_km,
        radial_step_km=args.radial_step_km,
        bearing_step_deg=args.bearing_step_deg,
    )
    nearest_land_sample = min(samples, key=lambda item: item.distance_km, default=None)
    reference_elevation = spot_elevation
    if reference_elevation is None and nearest_land_sample is not None and nearest_land_sample.distance_km <= args.nearest_land_max_km:
        reference_elevation = nearest_land_sample.elevation_m
    features: dict[str, Any] = {
        "latitude": round(lat, 7),
        "longitude": round(lon, 7),
        "source_resolution_minutes": finite_float(spot.get("source_resolution_minutes")),
        "is_ml_target": 1.0 if spot.get("use_for_ml", False) else 0.0,
        "dem_spot_elevation_m": None if spot_elevation is None else round(float(spot_elevation), 6),
        "dem_reference_elevation_m": None if reference_elevation is None else round(float(reference_elevation), 6),
        "dem_nearest_land_distance_km": None if nearest_land_sample is None else round(nearest_land_sample.distance_km, 6),
        "dem_nearest_land_elevation_m": None if nearest_land_sample is None else round(nearest_land_sample.elevation_m, 6),
        "dem_radial_sample_count": len(samples),
        "dem_max_radius_km": args.max_radius_km,
        "dem_radial_step_km": args.radial_step_km,
        "dem_bearing_step_deg": args.bearing_step_deg,
    }
    if samples:
        elevations = [sample.elevation_m for sample in samples]
        stats = aggregate(elevations, np)
        for key, value in stats.items():
            features[f"dem_all_{int(args.max_radius_km)}km_elevation_{key}"] = value
    add_radius_features(features, samples, reference_elevation, args.radius_km, np)
    add_sector_features(
        features,
        samples,
        reference_elevation,
        sector_half_width_deg=args.sector_half_width_deg,
        sector_max_km=args.sector_max_km,
        np=np,
    )
    add_gradient_features(features, samples, reference_elevation, np)
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--dem-dir", type=Path, default=DEFAULT_DEM_DIR)
    parser.add_argument("--dem-glob", default="*.tif")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-context-spots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-radius-km", type=float, default=20.0)
    parser.add_argument("--sector-max-km", type=float, default=20.0)
    parser.add_argument("--radial-step-km", type=float, default=0.25)
    parser.add_argument("--bearing-step-deg", type=float, default=5.0)
    parser.add_argument("--sector-half-width-deg", type=float, default=22.5)
    parser.add_argument("--nearest-land-max-km", type=float, default=2.0)
    parser.add_argument("--radius-km", type=float, action="append", default=[1.0, 2.0, 5.0, 10.0, 20.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = import_deps()
    np = deps["np"]
    rasterio = deps["rasterio"]
    registry = resolve_path(args.registry)
    dem_dir = resolve_path(args.dem_dir)
    output = resolve_path(args.output)
    spots = load_spots(registry, args.include_context_spots)
    datasets = open_dem_tiles(dem_dir, args.dem_glob, rasterio)
    try:
        rows = []
        for spot in spots:
            features = build_features_for_spot(spot, datasets, args, np)
            rows.append({
                "spot_id": spot["spot_id"],
                "name": spot.get("name"),
                "features": features,
            })
    finally:
        for dataset in datasets:
            dataset.close()

    payload = {
        "format": "corsewind.ml_spot_static_features.dem_v1",
        "generated_at_utc": utc_now(),
        "source_registry": str(registry),
        "dem_dir": str(dem_dir),
        "dem_glob": args.dem_glob,
        "spot_count": len(rows),
        "parameters": {
            "include_context_spots": args.include_context_spots,
            "max_radius_km": args.max_radius_km,
            "sector_max_km": args.sector_max_km,
            "radial_step_km": args.radial_step_km,
            "bearing_step_deg": args.bearing_step_deg,
            "sector_half_width_deg": args.sector_half_width_deg,
            "radius_km": args.radius_km,
        },
        "spots": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "spot_count": len(rows),
        "dem_dir": str(dem_dir),
        "feature_count_first_spot": len(rows[0]["features"]) if rows else 0,
    }, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
