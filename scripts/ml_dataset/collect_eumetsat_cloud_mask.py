#!/usr/bin/env python3
"""Download EUMETSAT MTG Cloud Mask products and sample them at ML spots."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_RAW_ROOT = DEFAULT_ML_ROOT / "eumetsat/raw/cloud_mask"
DEFAULT_SAMPLE_ROOT = DEFAULT_ML_ROOT / "eumetsat/cloud_mask_samples"
DEFAULT_TMP_PATHS = [
    ROOT / "tmp/eumdac_test_pkgs",
    ROOT / "tmp/copernicusmarine_test_pkgs",
]
DEFAULT_COLLECTION_ID = "EO:EUM:DAT:0678"
DEFAULT_BBOX = "7.5,41.0,10.2,43.3"
QUALITY_VARIABLES = [
    "quality_illumination",
    "quality_nwp_parameters",
    "quality_MTG_parameters",
    "quality_overall_processing",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def add_tmp_paths() -> None:
    for path in reversed(DEFAULT_TMP_PATHS):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def import_eumdac() -> Any:
    try:
        import eumdac

        return eumdac
    except ModuleNotFoundError:
        add_tmp_paths()
        try:
            import eumdac

            return eumdac
        except ModuleNotFoundError as exc:
            raise SystemExit("eumdac is required; install requirements-ml-dataset.txt.") from exc


def import_netcdf4() -> Any:
    try:
        import netCDF4

        return netCDF4
    except ModuleNotFoundError:
        add_tmp_paths()
        try:
            import netCDF4

            return netCDF4
        except ModuleNotFoundError as exc:
            raise SystemExit("netCDF4 is required; install requirements-ml-dataset.txt.") from exc


def import_pyproj() -> tuple[Any, Any]:
    try:
        from pyproj import CRS, Transformer

        return CRS, Transformer
    except ModuleNotFoundError:
        add_tmp_paths()
        try:
            from pyproj import CRS, Transformer

            return CRS, Transformer
        except ModuleNotFoundError as exc:
            raise SystemExit("pyproj is required; install requirements-ml-dataset.txt.") from exc


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


def parse_utc_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def iso_utc(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    text = str(value)
    if text.endswith("Z"):
        return text
    try:
        return parse_utc_datetime(text).isoformat().replace("+00:00", "Z")
    except ValueError:
        return text


def connect_datastore() -> Any:
    key = os.environ.get("EUMETSAT_CONSUMER_KEY")
    secret = os.environ.get("EUMETSAT_CONSUMER_SECRET")
    if not key or not secret:
        raise SystemExit("Set EUMETSAT_CONSUMER_KEY and EUMETSAT_CONSUMER_SECRET.")
    eumdac = import_eumdac()
    token = eumdac.AccessToken((key, secret), cache=False)
    return eumdac.DataStore(token)


def product_entry_name(product: Any) -> str:
    entries = list(product.entries)
    try:
        return next(entry for entry in entries if str(entry).endswith(".nc"))
    except StopIteration as exc:
        raise RuntimeError(f"No NetCDF entry found in product {product}") from exc


def product_identifier(product: Any) -> str:
    return str(product)


def product_filename(product: Any, entry: str) -> str:
    return Path(entry).name or f"{product_identifier(product)}.nc"


def download_product(product: Any, raw_root: Path, overwrite: bool) -> Path:
    entry = product_entry_name(product)
    output_path = raw_root / product_filename(product, entry)
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.tmp")
    with product.open(entry) as source, tmp.open("wb") as target:
        shutil.copyfileobj(source, target)
    tmp.replace(output_path)
    return output_path


def search_products(args: argparse.Namespace) -> list[Any]:
    datastore = connect_datastore()
    collection = datastore.get_collection(args.collection_id)
    products = collection.search(
        dtstart=parse_utc_datetime(args.start_datetime),
        dtend=parse_utc_datetime(args.end_datetime),
        bbox=args.bbox,
    )
    selected = []
    for product in products:
        selected.append(product)
        if args.max_products and len(selected) >= args.max_products:
            break
    return selected


def product_metadata(product: Any) -> dict[str, Any]:
    metadata = getattr(product, "metadata", {}) or {}
    properties = metadata.get("properties", {}) if isinstance(metadata, dict) else {}
    info = properties.get("productInformation", {}) if isinstance(properties, dict) else {}
    return {
        "product_id": product_identifier(product),
        "sensing_start_utc": iso_utc(getattr(product, "sensing_start", None)),
        "sensing_end_utc": iso_utc(getattr(product, "sensing_end", None)),
        "ingested_utc": iso_utc(getattr(product, "ingested", None)),
        "processing_time_utc": iso_utc(getattr(product, "processingTime", None)),
        "timeliness": getattr(product, "timeliness", None) or info.get("timeliness"),
        "quality_status": getattr(product, "qualityStatus", None),
        "size_bytes": getattr(product, "size", None) or info.get("size"),
    }


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


def int_cell(variable: Any, row: int, col: int) -> int | None:
    if row < 0 or col < 0 or row >= variable.shape[0] or col >= variable.shape[1]:
        return None
    value = variable[row, col]
    if hasattr(value, "item"):
        value = value.item()
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def square_counts(variable: Any, row: int, col: int, radius_cells: int) -> dict[str, Any]:
    counter: Counter[int] = Counter()
    valid = 0
    for y in range(row - radius_cells, row + radius_cells + 1):
        for x in range(col - radius_cells, col + radius_cells + 1):
            value = int_cell(variable, y, x)
            if value is None:
                continue
            counter[value] += 1
            valid += 1
    fractions = {
        str(key): round(count / valid, 6)
        for key, count in sorted(counter.items())
        if valid
    }
    mode = counter.most_common(1)[0][0] if counter else None
    return {
        "valid_count": valid,
        "counts": {str(key): count for key, count in sorted(counter.items())},
        "fractions": fractions,
        "mode": mode,
    }


def scalar_value(nc: Any, name: str) -> float | None:
    if name not in nc.variables:
        return None
    variable = nc.variables[name]
    try:
        value = variable[()]
    except Exception:
        return None
    if hasattr(value, "item"):
        value = value.item()
    return finite_float(value)


def projection_transformers(nc: Any) -> tuple[Any, Any, float]:
    CRS, Transformer = import_pyproj()
    projection = nc.variables["mtg_geos_projection"]
    cf_attrs = {attr: getattr(projection, attr) for attr in projection.ncattrs()}
    crs_geos = CRS.from_cf(cf_attrs)
    crs_wgs84 = CRS.from_epsg(4326)
    forward = Transformer.from_crs(crs_wgs84, crs_geos, always_xy=True)
    inverse = Transformer.from_crs(crs_geos, crs_wgs84, always_xy=True)
    height = float(getattr(projection, "perspective_point_height"))
    return forward, inverse, height


def sample_file(
    nc_path: Path,
    metadata: dict[str, Any],
    spots: list[dict[str, Any]],
    collection_id: str,
    radius_cells: int,
    include_quality: bool,
) -> list[dict[str, Any]]:
    netCDF4 = import_netcdf4()
    rows: list[dict[str, Any]] = []
    nc = netCDF4.Dataset(nc_path)
    try:
        forward, inverse, height = projection_transformers(nc)
        x_values = [float(value) * height for value in nc.variables["x"][:]]
        y_values = [float(value) * height for value in nc.variables["y"][:]]
        cloud = nc.variables["cloud_state"]
        quality_vars = {
            name: nc.variables[name]
            for name in QUALITY_VARIABLES
            if include_quality and name in nc.variables
        }
        for spot in spots:
            lat = float(spot["latitude"])
            lon = float(spot["longitude"])
            projected_x, projected_y = forward.transform(lon, lat)
            col = nearest_index(x_values, projected_x)
            row = nearest_index(y_values, projected_y)
            pixel_lon, pixel_lat = inverse.transform(x_values[col], y_values[row])
            center = int_cell(cloud, row, col)
            counts = square_counts(cloud, row, col, radius_cells)
            quality = {
                name: int_cell(variable, row, col)
                for name, variable in quality_vars.items()
            }
            rows.append({
                "format": "corsewind.eumetsat_cloud_mask_spot_sample.v1",
                "source": "eumetsat",
                "collection_id": collection_id,
                "product_id": metadata.get("product_id"),
                "sensing_start_utc": metadata.get("sensing_start_utc"),
                "sensing_end_utc": metadata.get("sensing_end_utc"),
                "ingested_utc": metadata.get("ingested_utc"),
                "processing_time_utc": metadata.get("processing_time_utc"),
                "timeliness": metadata.get("timeliness"),
                "quality_status": metadata.get("quality_status"),
                "spot_id": spot.get("spot_id"),
                "spot_name": spot.get("name"),
                "spot_kind": spot.get("kind"),
                "spot_source_type": spot.get("source_type"),
                "station_id": spot.get("station_id"),
                "latitude": lat,
                "longitude": lon,
                "use_for_ml": bool(spot.get("use_for_ml", False)),
                "sample_method": "nearest_mtg_geos_pixel",
                "grid_row": row,
                "grid_col": col,
                "pixel_latitude": round(float(pixel_lat), 6),
                "pixel_longitude": round(float(pixel_lon), 6),
                "sample_distance_km": round(haversine_km(lat, lon, float(pixel_lat), float(pixel_lon)), 4),
                "cloud_state": center,
                "cloud_state_radius_cells": radius_cells,
                "cloud_state_valid_count": counts["valid_count"],
                "cloud_state_counts": counts["counts"],
                "cloud_state_fractions": counts["fractions"],
                "cloud_state_mode": counts["mode"],
                "quality_flags": quality,
                "product_quality": scalar_value(nc, "product_quality"),
                "product_completeness": scalar_value(nc, "product_completeness"),
                "product_timeliness": scalar_value(nc, "product_timeliness"),
                "netcdf_path": str(nc_path),
                "sampled_at_utc": utc_now(),
            })
    finally:
        nc.close()
    return rows


def output_path(output_root: Path, timestamp_utc: str | None) -> Path:
    day = (timestamp_utc or utc_now())[:10]
    return output_root / f"date={day}" / "cloud_mask_samples.jsonl"


def write_jsonl_by_day(output_root: Path, rows: list[dict[str, Any]]) -> dict[str, int]:
    by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_path[output_path(output_root, row.get("sensing_start_utc"))].append(row)
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
            (row.get("product_id"), row.get("spot_id")): row
            for row in [*existing, *path_rows]
        }
        ordered = sorted(
            deduped.values(),
            key=lambda row: (row.get("sensing_start_utc") or "", row.get("spot_id") or ""),
        )
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")
        tmp.replace(path)
        written[str(path)] = len(ordered)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID)
    parser.add_argument("--start-datetime", required=True)
    parser.add_argument("--end-datetime", required=True)
    parser.add_argument("--bbox", default=DEFAULT_BBOX, help="EUMDAC bbox as west,south,east,north.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--max-products", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-context-spots", action="store_true")
    parser.add_argument("--spot-id", action="append", default=[])
    parser.add_argument("--radius-cells", type=int, default=3)
    parser.add_argument("--no-quality-flags", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry_path = resolve_path(args.registry)
    raw_root = resolve_path(args.raw_root)
    output_root = resolve_path(args.output_root)
    spots = load_spots(registry_path, args.include_context_spots, set(args.spot_id))
    products = search_products(args)
    all_rows: list[dict[str, Any]] = []
    downloaded = []
    for product in products:
        metadata = product_metadata(product)
        nc_path = download_product(product, raw_root, args.overwrite)
        downloaded.append({"product_id": metadata.get("product_id"), "path": str(nc_path), "size_bytes": nc_path.stat().st_size})
        all_rows.extend(sample_file(nc_path, metadata, spots, args.collection_id, args.radius_cells, not args.no_quality_flags))
    written = write_jsonl_by_day(output_root, all_rows)
    print(json.dumps({
        "generated_at_utc": utc_now(),
        "collection_id": args.collection_id,
        "start_datetime": args.start_datetime,
        "end_datetime": args.end_datetime,
        "bbox": args.bbox,
        "registry": str(registry_path),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "product_count": len(products),
        "spot_count": len(spots),
        "row_count": len(all_rows),
        "downloaded": downloaded,
        "written": written,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
