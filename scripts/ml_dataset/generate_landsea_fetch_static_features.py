#!/usr/bin/env python3
"""Generate true maritime fetch features from ESA WorldCover land/sea rasters."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_LANDCOVER_DIR = ROOT / "data/raw/landcover/esa_worldcover_v200_2021"
DEFAULT_BASE_STATIC = ROOT / "configs/ml_spot_static_features.json"
DEFAULT_OUTPUT = ROOT / "configs/ml_spot_static_features.fetch_v1.json"
WATER_CLASSES = {80}
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


def load_static_features(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    payload = read_json(path)
    rows = payload.get("spots", []) if isinstance(payload, dict) else payload
    out = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("spot_id"):
            continue
        features = row.get("features") if isinstance(row.get("features"), dict) else row
        out[str(row["spot_id"])] = {
            str(key): value
            for key, value in features.items()
            if key not in {"spot_id", "name"}
        }
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


def open_tiles(landcover_dir: Path, pattern: str, rasterio: Any) -> list[Any]:
    paths = sorted(landcover_dir.glob(pattern))
    if not paths:
        raise SystemExit(f"No landcover tiles found in {landcover_dir} matching {pattern!r}.")
    return [rasterio.open(path) for path in paths]


def tile_contains(dataset: Any, latitude: float, longitude: float) -> bool:
    bounds = dataset.bounds
    return bounds.left <= longitude <= bounds.right and bounds.bottom <= latitude <= bounds.top


def sample_class(datasets: list[Any], latitude: float, longitude: float) -> int | None:
    for dataset in datasets:
        if not tile_contains(dataset, latitude, longitude):
            continue
        try:
            value = next(dataset.sample([(longitude, latitude)]))[0]
        except (StopIteration, IndexError, ValueError):
            continue
        number = finite_float(value)
        if number is None:
            continue
        nodata = dataset.nodata
        if nodata is not None and math.isclose(number, float(nodata)):
            continue
        return int(round(number))
    return None


def classify(value: int | None) -> str:
    if value is None:
        return "missing"
    if value in WATER_CLASSES:
        return "water"
    return "land"


def run_length_at_start(values: list[str], wanted: str, step_km: float) -> float:
    length = 0.0
    for value in values:
        if value != wanted:
            break
        length += step_km
    return round(length, 6)


def longest_run(values: list[str], wanted: str, step_km: float) -> float:
    best = 0
    current = 0
    for value in values:
        if value == wanted:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return round(best * step_km, 6)


def first_distance(values: list[str], wanted: str, step_km: float) -> float | None:
    for index, value in enumerate(values, start=1):
        if value == wanted:
            return round(index * step_km, 6)
    return None


def fetch_for_bearing(
    datasets: list[Any],
    latitude: float,
    longitude: float,
    bearing_deg: float,
    *,
    max_fetch_km: float,
    step_km: float,
    coastal_snap_km: float,
) -> dict[str, Any]:
    classes = []
    distances = []
    steps = int(math.floor(max_fetch_km / step_km))
    for index in range(1, steps + 1):
        distance = index * step_km
        lat, lon = destination_point(latitude, longitude, bearing_deg, distance)
        cls = classify(sample_class(datasets, lat, lon))
        classes.append(cls)
        distances.append(distance)

    observed = [value for value in classes if value != "missing"]
    water_count = sum(value == "water" for value in observed)
    land_count = sum(value == "land" for value in observed)
    missing_count = sum(value == "missing" for value in classes)
    first_water = first_distance(classes, "water", step_km)
    first_land = first_distance(classes, "land", step_km)

    # For stations just inland, skip a short land segment before the sea.
    start_index = 0
    if first_water is not None and first_water <= coastal_snap_km:
        start_index = max(0, int(round(first_water / step_km)) - 1)
    snapped = classes[start_index:]
    snapped_fetch = run_length_at_start(snapped, "water", step_km)
    direct_fetch = run_length_at_start(classes, "water", step_km)

    return {
        "sample_count": len(classes),
        "observed_count": len(observed),
        "missing_count": missing_count,
        "water_share": None if not observed else round(water_count / len(observed), 6),
        "land_share": None if not observed else round(land_count / len(observed), 6),
        "first_water_distance_km": first_water,
        "first_land_distance_km": first_land,
        "direct_water_fetch_km": direct_fetch,
        "coastal_snapped_water_fetch_km": snapped_fetch,
        "longest_water_run_km": longest_run(classes, "water", step_km),
        "longest_land_run_km": longest_run(classes, "land", step_km),
    }


def add_cross_shore_features(features: dict[str, Any], prefix: str = "fetch_sector") -> None:
    opposite_pairs = [("n", "s"), ("ne", "sw"), ("e", "w"), ("se", "nw")]
    for a, b in opposite_pairs:
        va = finite_float(features.get(f"{prefix}_{a}_coastal_snapped_water_fetch_km"))
        vb = finite_float(features.get(f"{prefix}_{b}_coastal_snapped_water_fetch_km"))
        features[f"{prefix}_{a}_minus_{b}_fetch_km"] = None if va is None or vb is None else round(va - vb, 6)


def build_features_for_spot(spot: dict[str, Any], datasets: list[Any], args: argparse.Namespace) -> dict[str, Any]:
    lat = float(spot["latitude"])
    lon = float(spot["longitude"])
    features: dict[str, Any] = {
        "fetch_max_km": args.max_fetch_km,
        "fetch_step_km": args.step_km,
        "fetch_coastal_snap_km": args.coastal_snap_km,
    }
    for name, bearing in SECTOR_BEARINGS.items():
        stats = fetch_for_bearing(
            datasets,
            lat,
            lon,
            bearing,
            max_fetch_km=args.max_fetch_km,
            step_km=args.step_km,
            coastal_snap_km=args.coastal_snap_km,
        )
        for key, value in stats.items():
            features[f"fetch_sector_{name}_{key}"] = value
    add_cross_shore_features(features)
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--landcover-dir", type=Path, default=DEFAULT_LANDCOVER_DIR)
    parser.add_argument("--landcover-glob", default="ESA_WorldCover_10m_2021_v200_*_Map.tif")
    parser.add_argument("--base-static-features", type=Path, default=DEFAULT_BASE_STATIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-context-spots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-fetch-km", type=float, default=80.0)
    parser.add_argument("--step-km", type=float, default=0.25)
    parser.add_argument("--coastal-snap-km", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = import_deps()
    rasterio = deps["rasterio"]
    registry = resolve_path(args.registry)
    landcover_dir = resolve_path(args.landcover_dir)
    output = resolve_path(args.output)
    spots = load_spots(registry, args.include_context_spots)
    base_static = load_static_features(resolve_path(args.base_static_features) if args.base_static_features else None)
    datasets = open_tiles(landcover_dir, args.landcover_glob, rasterio)
    try:
        rows = []
        for spot in spots:
            spot_id = str(spot["spot_id"])
            features = dict(base_static.get(spot_id, {}))
            features.update(build_features_for_spot(spot, datasets, args))
            rows.append({"spot_id": spot_id, "name": spot.get("name"), "features": features})
    finally:
        for dataset in datasets:
            dataset.close()

    payload = {
        "format": "corsewind.ml_spot_static_features.fetch_v1",
        "generated_at_utc": utc_now(),
        "source_registry": str(registry),
        "landcover_dir": str(landcover_dir),
        "landcover_glob": args.landcover_glob,
        "base_static_features": str(resolve_path(args.base_static_features)) if args.base_static_features else None,
        "spot_count": len(rows),
        "parameters": {
            "include_context_spots": args.include_context_spots,
            "max_fetch_km": args.max_fetch_km,
            "step_km": args.step_km,
            "coastal_snap_km": args.coastal_snap_km,
            "water_classes": sorted(WATER_CLASSES),
        },
        "spots": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "spot_count": len(rows),
        "feature_count_first_spot": len(rows[0]["features"]) if rows else 0,
    }, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
