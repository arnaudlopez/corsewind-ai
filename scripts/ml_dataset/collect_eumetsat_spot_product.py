#!/usr/bin/env python3
"""Download EUMETSAT NetCDF products and sample grid variables at ML spots."""

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
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_BBOX = "7.5,41.0,10.2,43.3"
DEFAULT_TMP_PATHS = [
    ROOT / "tmp/eumdac_test_pkgs",
    ROOT / "tmp/copernicusmarine_test_pkgs",
]
PRODUCTS: dict[str, dict[str, Any]] = {
    "cloud_type": {
        "collection_id": "EO:EUM:DAT:0680",
        "output_name": "cloud_type",
        "target_features": ["cloud_type_dominant", "low_cloud_fraction", "high_cloud_fraction"],
        "prefer_variables": ["cloud_type", "ct", "quality_overall_processing"],
        "categorical_keywords": ["type", "quality", "flag", "class"],
    },
    "land_surface_temperature": {
        "collection_id": "EO:EUM:DAT:1088",
        "output_name": "land_surface_temperature",
        "target_features": ["land_surface_temperature_c", "land_minus_sea_surface_temperature_c"],
        "prefer_variables": ["land_surface_temperature", "lst", "temperature", "quality"],
        "categorical_keywords": ["quality", "flag", "mask"],
        "temperature_keywords": ["temperature", "lst"],
    },
    "global_instability_indices": {
        "collection_id": "EO:EUM:DAT:0683",
        "output_name": "global_instability_indices",
        "target_features": ["satellite_instability_index", "convective_potential_flag"],
        "prefer_variables": ["instability", "index", "ki", "li", "showalter", "total_totals", "quality"],
        "categorical_keywords": ["quality", "flag", "mask"],
    },
}
SKIP_VARIABLES = {
    "x",
    "y",
    "lat",
    "latitude",
    "lon",
    "longitude",
    "mtg_geos_projection",
}


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
    if value is None or (isinstance(value, str) and value == ""):
        return None
    try:
        if hasattr(value, "mask"):
            mask = value.mask
            if bool(mask.all() if hasattr(mask, "all") else mask):
                return None
    except (TypeError, ValueError):
        pass
    try:
        if hasattr(value, "filled"):
            value = value.filled(float("nan"))
        if hasattr(value, "item"):
            value = value.item()
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


def delete_raw_file(path: Path) -> dict[str, Any]:
    try:
        size_bytes = path.stat().st_size
    except FileNotFoundError:
        return {"path": str(path), "status": "already_missing"}
    path.unlink()
    return {"path": str(path), "status": "deleted", "size_bytes": size_bytes}


def search_products(args: argparse.Namespace, collection_id: str) -> list[Any]:
    datastore = connect_datastore()
    collection = datastore.get_collection(collection_id)
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


def projection_variable(nc: Any) -> Any:
    if "mtg_geos_projection" in nc.variables:
        return nc.variables["mtg_geos_projection"]
    for variable in nc.variables.values():
        attrs = {attr: getattr(variable, attr) for attr in variable.ncattrs()}
        if attrs.get("grid_mapping_name") == "geostationary":
            return variable
    raise RuntimeError("No geostationary projection variable found.")


def latlon_grid(nc: Any) -> tuple[Any, Any] | None:
    if "latitude" not in nc.variables or "longitude" not in nc.variables:
        return None
    latitude = nc.variables["latitude"]
    longitude = nc.variables["longitude"]
    if len(getattr(latitude, "shape", ())) != 2 or latitude.shape != longitude.shape:
        return None
    return latitude[:], longitude[:]


def nearest_latlon_cell(latitudes: Any, longitudes: Any, target_lat: float, target_lon: float) -> tuple[int, int, float, float]:
    try:
        import numpy as np
    except ModuleNotFoundError:
        add_tmp_paths()
        import numpy as np

    lat_values = np.ma.filled(latitudes, np.nan).astype(float)
    lon_values = np.ma.filled(longitudes, np.nan).astype(float)
    lon_scale = math.cos(math.radians(target_lat))
    distance = (lat_values - target_lat) ** 2 + ((lon_values - target_lon) * lon_scale) ** 2
    flat_index = int(np.nanargmin(distance))
    row, col = np.unravel_index(flat_index, lat_values.shape)
    return int(row), int(col), float(lat_values[row, col]), float(lon_values[row, col])


def projection_transformers(nc: Any) -> tuple[Any, Any, float]:
    CRS, Transformer = import_pyproj()
    projection = projection_variable(nc)
    cf_attrs = {attr: getattr(projection, attr) for attr in projection.ncattrs()}
    crs_geos = CRS.from_cf(cf_attrs)
    crs_wgs84 = CRS.from_epsg(4326)
    forward = Transformer.from_crs(crs_wgs84, crs_geos, always_xy=True)
    inverse = Transformer.from_crs(crs_geos, crs_wgs84, always_xy=True)
    height = float(getattr(projection, "perspective_point_height"))
    return forward, inverse, height


def coord_values(nc: Any, name: str, height: float) -> list[float]:
    if name not in nc.variables:
        raise RuntimeError(f"Missing coordinate variable {name!r}")
    values = [float(value) for value in nc.variables[name][:]]
    units = str(getattr(nc.variables[name], "units", "")).lower()
    if units in {"rad", "radian", "radians"} or max(abs(value) for value in values) < 1:
        return [value * height for value in values]
    return values


def numeric_dtype(variable: Any) -> bool:
    kind = getattr(getattr(variable, "dtype", None), "kind", "")
    return kind in {"i", "u", "f"}


def variable_2d_axes(variable: Any, y_len: int, x_len: int) -> tuple[int, int] | None:
    shape = tuple(int(item) for item in getattr(variable, "shape", ()))
    if not shape:
        return None
    dims = tuple(str(item).lower() for item in getattr(variable, "dimensions", ()))
    y_axis = next((idx for idx, dim in enumerate(dims) if dim in {"y", "number_of_rows", "rows", "row"}), None)
    x_axis = next((idx for idx, dim in enumerate(dims) if dim in {"x", "number_of_columns", "columns", "column"}), None)
    if y_axis is not None and x_axis is not None and shape[y_axis] == y_len and shape[x_axis] == x_len:
        return y_axis, x_axis
    if len(shape) >= 2 and shape[-2:] == (y_len, x_len):
        return len(shape) - 2, len(shape) - 1
    return None


def choose_variables(nc: Any, args: argparse.Namespace, config: dict[str, Any], y_len: int, x_len: int) -> list[str]:
    explicit = [item for item in args.variable if item]
    if explicit:
        missing = [name for name in explicit if name not in nc.variables]
        if missing:
            raise SystemExit(f"Variables not found in NetCDF: {', '.join(missing)}")
        return explicit
    candidates = []
    for name, variable in nc.variables.items():
        lower = name.lower()
        if lower in SKIP_VARIABLES:
            continue
        if not numeric_dtype(variable):
            continue
        if variable_2d_axes(variable, y_len, x_len) is None:
            continue
        candidates.append(name)
    preferred = []
    for token in config.get("prefer_variables", []):
        token_lower = str(token).lower()
        preferred.extend(name for name in candidates if token_lower in name.lower() and name not in preferred)
    ordered = [*preferred, *[name for name in candidates if name not in preferred]]
    return ordered[: args.max_variables] if args.max_variables else ordered


def read_cell(variable: Any, row: int, col: int, y_axis: int, x_axis: int) -> float | int | None:
    if row < 0 or col < 0 or row >= variable.shape[y_axis] or col >= variable.shape[x_axis]:
        return None
    index: list[Any] = []
    for axis, size in enumerate(variable.shape):
        if axis == y_axis:
            index.append(row)
        elif axis == x_axis:
            index.append(col)
        else:
            index.append(0 if size else slice(None))
    value = variable[tuple(index)]
    number = finite_float(value)
    if number is None:
        return None
    kind = getattr(getattr(variable, "dtype", None), "kind", "")
    if kind in {"i", "u"}:
        return int(round(number))
    return round(number, 6)


def is_categorical(variable_name: str, variable: Any, config: dict[str, Any]) -> bool:
    lower = variable_name.lower()
    return any(str(token).lower() in lower for token in config.get("categorical_keywords", []))


def is_temperature(variable_name: str, config: dict[str, Any]) -> bool:
    lower = variable_name.lower()
    return any(str(token).lower() in lower for token in config.get("temperature_keywords", []))


def square_summary(variable: Any, row: int, col: int, y_axis: int, x_axis: int, radius_cells: int, categorical: bool) -> dict[str, Any]:
    values: list[float | int] = []
    counter: Counter[str] = Counter()
    for y in range(row - radius_cells, row + radius_cells + 1):
        for x in range(col - radius_cells, col + radius_cells + 1):
            value = read_cell(variable, y, x, y_axis, x_axis)
            if value is None:
                continue
            values.append(value)
            if categorical:
                counter[str(value)] += 1
    if not values:
        return {"valid_count": 0}
    if categorical:
        total = sum(counter.values())
        return {
            "valid_count": len(values),
            "counts": dict(sorted(counter.items())),
            "fractions": {key: round(count / total, 6) for key, count in sorted(counter.items())},
            "mode": counter.most_common(1)[0][0],
        }
    numeric = [float(value) for value in values]
    return {
        "valid_count": len(numeric),
        "mean": round(sum(numeric) / len(numeric), 6),
        "min": round(min(numeric), 6),
        "max": round(max(numeric), 6),
    }


def scalar_value(nc: Any, name: str) -> float | None:
    if name not in nc.variables:
        return None
    variable = nc.variables[name]
    try:
        value = variable[()]
    except Exception:
        return None
    return finite_float(value)


def variable_attrs(variable: Any) -> dict[str, Any]:
    attrs = {}
    for name in variable.ncattrs():
        value = getattr(variable, name)
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, (str, int, float)) or value is None:
            attrs[name] = value
    return attrs


def sample_file(
    nc_path: Path,
    metadata: dict[str, Any],
    spots: list[dict[str, Any]],
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    netCDF4 = import_netcdf4()
    rows: list[dict[str, Any]] = []
    nc = netCDF4.Dataset(nc_path)
    try:
        forward = inverse = None
        x_values = y_values = None
        latlon_values = None
        sample_method = "nearest_mtg_geos_pixel"
        try:
            forward, inverse, height = projection_transformers(nc)
            x_values = coord_values(nc, "x", height)
            y_values = coord_values(nc, "y", height)
        except RuntimeError:
            latlon_values = latlon_grid(nc)
            if latlon_values is None:
                raise
            y_values = list(range(int(latlon_values[0].shape[0])))
            x_values = list(range(int(latlon_values[0].shape[1])))
            sample_method = "nearest_latlon_pixel"
        variable_names = choose_variables(nc, args, config, len(y_values), len(x_values))
        variable_meta = {
            name: variable_attrs(nc.variables[name])
            for name in variable_names
        }
        for spot in spots:
            lat = float(spot["latitude"])
            lon = float(spot["longitude"])
            if latlon_values is not None:
                row, col, pixel_lat, pixel_lon = nearest_latlon_cell(latlon_values[0], latlon_values[1], lat, lon)
            else:
                projected_x, projected_y = forward.transform(lon, lat)
                col = nearest_index(x_values, projected_x)
                row = nearest_index(y_values, projected_y)
                pixel_lon, pixel_lat = inverse.transform(x_values[col], y_values[row])
            sampled_values: dict[str, Any] = {}
            sampled_values_c: dict[str, Any] = {}
            neighbourhoods: dict[str, Any] = {}
            for name in variable_names:
                variable = nc.variables[name]
                axes = variable_2d_axes(variable, len(y_values), len(x_values))
                if axes is None:
                    continue
                y_axis, x_axis = axes
                value = read_cell(variable, row, col, y_axis, x_axis)
                sampled_values[name] = value
                if value is not None and is_temperature(name, config) and float(value) > 150:
                    sampled_values_c[f"{name}_c"] = round(float(value) - 273.15, 6)
                neighbourhoods[name] = square_summary(
                    variable,
                    row,
                    col,
                    y_axis,
                    x_axis,
                    args.radius_cells,
                    is_categorical(name, variable, config),
                )
            rows.append({
                "format": "corsewind.eumetsat_spot_product_sample.v1",
                "source": "eumetsat",
                "product_key": args.product,
                "collection_id": args.collection_id,
                "target_features": config.get("target_features", []),
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
                "sample_method": sample_method,
                "grid_row": row,
                "grid_col": col,
                "pixel_latitude": round(float(pixel_lat), 6),
                "pixel_longitude": round(float(pixel_lon), 6),
                "sample_distance_km": round(haversine_km(lat, lon, float(pixel_lat), float(pixel_lon)), 4),
                "sampled_values": sampled_values,
                "sampled_values_c": sampled_values_c,
                "neighborhood_radius_cells": args.radius_cells,
                "neighborhoods": neighbourhoods,
                "variable_attrs": variable_meta,
                "product_quality": scalar_value(nc, "product_quality"),
                "product_completeness": scalar_value(nc, "product_completeness"),
                "product_timeliness": scalar_value(nc, "product_timeliness"),
                "netcdf_path": str(nc_path),
                "sampled_at_utc": utc_now(),
            })
    finally:
        nc.close()
    return rows, variable_names


def output_path(output_root: Path, output_name: str, timestamp_utc: str | None) -> Path:
    day = (timestamp_utc or utc_now())[:10]
    return output_root / f"date={day}" / f"{output_name}_samples.jsonl"


def write_jsonl_by_day(output_root: Path, output_name: str, rows: list[dict[str, Any]]) -> dict[str, int]:
    by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_path[output_path(output_root, output_name, row.get("sensing_start_utc"))].append(row)
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
    parser.add_argument("--product", choices=sorted(PRODUCTS), required=True)
    parser.add_argument("--collection-id")
    parser.add_argument("--start-datetime", required=True)
    parser.add_argument("--end-datetime", required=True)
    parser.add_argument("--bbox", default=DEFAULT_BBOX, help="EUMDAC bbox as west,south,east,north.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-products", type=int, default=6)
    parser.add_argument("--max-variables", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--delete-raw-after-sample",
        action="store_true",
        help="Delete downloaded NetCDF files after successful sampling to limit backfill storage.",
    )
    parser.add_argument("--include-context-spots", action="store_true")
    parser.add_argument("--spot-id", action="append", default=[])
    parser.add_argument("--radius-cells", type=int, default=3)
    parser.add_argument("--variable", action="append", default=[], help="Specific NetCDF variable to sample. Repeatable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PRODUCTS[args.product]
    output_name = config["output_name"]
    args.collection_id = args.collection_id or config["collection_id"]
    registry_path = resolve_path(args.registry)
    raw_root = resolve_path(args.raw_root or DEFAULT_ML_ROOT / f"eumetsat/raw/{output_name}")
    output_root = resolve_path(args.output_root or DEFAULT_ML_ROOT / f"eumetsat/{output_name}_samples")
    spots = load_spots(registry_path, args.include_context_spots, set(args.spot_id))
    products = search_products(args, args.collection_id)
    all_rows: list[dict[str, Any]] = []
    sampled_variables: set[str] = set()
    downloaded = []
    deleted_raw = []
    for product in products:
        metadata = product_metadata(product)
        nc_path = download_product(product, raw_root, args.overwrite)
        downloaded.append({"product_id": metadata.get("product_id"), "path": str(nc_path), "size_bytes": nc_path.stat().st_size})
        rows, variables = sample_file(nc_path, metadata, spots, args, config)
        all_rows.extend(rows)
        sampled_variables.update(variables)
        if args.delete_raw_after_sample:
            deleted_raw.append(delete_raw_file(nc_path))
    written = write_jsonl_by_day(output_root, output_name, all_rows)
    print(json.dumps({
        "generated_at_utc": utc_now(),
        "product": args.product,
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
        "sampled_variables": sorted(sampled_variables),
        "downloaded": downloaded,
        "deleted_raw": deleted_raw,
        "written": written,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
