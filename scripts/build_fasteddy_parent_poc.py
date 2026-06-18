#!/usr/bin/env python3
"""Build a FastEddy parent NetCDF POC from downloaded AROME inputs.

The production target is still a full FastEddy IC/BC generator. This POC stops one
step earlier: it decodes the AROME isobaric slices, stacks them into a compact
parent-state NetCDF, and writes an explicit readiness manifest.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sample_arome_tiff_at_stations import read_float64_tiff
from prepare_corsica_windninja_tiles import load_dem_tiles, sample_dem


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "data/processed/benchmarks/fasteddy/arome_poc/arome_fasteddy_poc_download_plan.json"
DEFAULT_OUTPUT = ROOT / "data/processed/benchmarks/fasteddy/arome_poc/fasteddy_parent_poc.nc"
DEFAULT_MANIFEST = ROOT / "data/processed/benchmarks/fasteddy/arome_poc/fasteddy_parent_poc_manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


FASTEDDY_VAR_NAMES = {
    "u_wind_3d": "u",
    "v_wind_3d": "v",
    "w_wind_3d": "w",
    "temperature_3d": "temperature",
    "humidity_3d": "relative_humidity",
    "pressure_or_height_3d": "geopotential_or_height",
    "surface_temperature": "surface_temperature",
}


def grib_available() -> tuple[bool, str | None]:
    try:
        import cfgrib  # noqa: F401
        import xarray  # noqa: F401
    except ImportError as exc:
        return False, str(exc)
    return True, None


def read_grib_field(path: Path) -> tuple[Any | None, str | None]:
    try:
        import xarray as xr
    except ImportError as exc:
        return None, str(exc)
    try:
        dataset = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    except Exception as exc:  # cfgrib surfaces ecCodes errors with several exception classes.
        return None, str(exc)
    data_vars = list(dataset.data_vars)
    if not data_vars:
        return None, f"no data variables in {display_path(path)}"
    return dataset[data_vars[0]].load(), None


def collect_inputs(plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path], list[str], list[str], list[dict[str, str]]]:
    grib_groups: dict[str, list[tuple[int | None, Path]]] = {}
    surface_gribs: dict[str, Path] = {}
    tiff_arrays: dict[str, Any] = {}
    missing: list[str] = []
    unsupported: list[str] = []
    decode_errors: list[dict[str, str]] = []
    for step in plan["steps"]:
        for item in step["downloads"]:
            if item.get("status") != "planned":
                missing.append(item["requirement_id"])
                continue
            output_candidates = []
            if item.get("output_template") and item.get("pressure_levels_hpa"):
                output_candidates.extend(
                    ROOT / item["output_template"].format(pressure_hpa=pressure_hpa)
                    for pressure_hpa in item["pressure_levels_hpa"]
                )
            else:
                output_candidates.append(ROOT / item["output"])
            existing = [path for path in output_candidates if path.exists()]
            if not existing:
                missing.append(item["requirement_id"])
                continue
            for path in existing:
                if path.suffix.lower() in {".tiff", ".tif"}:
                    tiff_arrays[item["requirement_id"]] = read_float64_tiff(path)
                elif path.suffix.lower() in {".grib", ".grib2"}:
                    pressure_hpa = None
                    for level in item.get("pressure_levels_hpa") or []:
                        if f"__p{level}." in path.name:
                            pressure_hpa = int(level)
                    if pressure_hpa is None:
                        surface_gribs[item["requirement_id"]] = path
                    else:
                        grib_groups.setdefault(item["requirement_id"], []).append((pressure_hpa, path))
                else:
                    unsupported.append(item["requirement_id"])
    return grib_groups, tiff_arrays, surface_gribs, sorted(set(missing)), sorted(set(unsupported)), decode_errors


def write_parent_nc(path: Path, grib_groups: dict[str, Any], tiff_arrays: dict[str, Any], surface_gribs: dict[str, Path], plan: dict[str, Any]) -> tuple[bool, str | None, list[dict[str, str]], list[str]]:
    available, import_error = grib_available()
    if not available:
        return False, f"GRIB decoder unavailable: {import_error}; run pip install -r requirements-benchmark.txt", [], []
    try:
        import numpy as np
        import xarray as xr
    except ImportError as exc:
        return False, str(exc), [], []

    variables: dict[str, Any] = {}
    decoded_files: list[str] = []
    decode_errors: list[dict[str, str]] = []
    for requirement_id, entries in sorted(grib_groups.items()):
        target_name = FASTEDDY_VAR_NAMES.get(requirement_id, requirement_id)
        levels = []
        arrays = []
        for pressure_hpa, grib_path in sorted(entries, key=lambda item: (item[0] is None, item[0] or 0), reverse=True):
            field, error = read_grib_field(grib_path)
            if error:
                decode_errors.append({"requirement_id": requirement_id, "path": display_path(grib_path), "error": error})
                continue
            levels.append(float(pressure_hpa) if pressure_hpa is not None else np.nan)
            arrays.append(field)
            decoded_files.append(display_path(grib_path))
        if arrays:
            stacked = xr.concat(arrays, dim=xr.DataArray(levels, dims="pressure_hpa", name="pressure_hpa"))
            variables[target_name] = stacked

    for requirement_id, grib_path in sorted(surface_gribs.items()):
        target_name = FASTEDDY_VAR_NAMES.get(requirement_id, requirement_id)
        field, error = read_grib_field(grib_path)
        if error:
            decode_errors.append({"requirement_id": requirement_id, "path": display_path(grib_path), "error": error})
            continue
        variables[target_name] = field
        decoded_files.append(display_path(grib_path))

    for requirement_id, values in tiff_arrays.items():
        target_name = FASTEDDY_VAR_NAMES.get(requirement_id, requirement_id)
        variables[target_name] = xr.DataArray(values, dims=("y", "x"))

    if not variables:
        return False, "no decoded variables available", decode_errors, decoded_files

    dataset = xr.Dataset(variables)
    if "geopotential_or_height" in dataset:
        dataset["height_m"] = dataset["geopotential_or_height"] / 9.80665
        dataset["height_m"].attrs.update({"units": "m", "long_name": "geopotential height estimate from Z / g0"})
    if "temperature" in dataset and "pressure_hpa" in dataset.coords:
        dataset["potential_temperature"] = dataset["temperature"] * (1000.0 / dataset["pressure_hpa"]) ** 0.2854
        dataset["potential_temperature"].attrs.update({"units": "K", "long_name": "potential temperature estimate"})
    dataset.attrs.update(
        {
            "format": "corsewind.fasteddy_parent_poc.v1",
            "warning": "Parent-state POC only; not yet a final FastEddy IC/BC file.",
            "source_product": plan["source"]["product"],
            "source_resolution": plan["source"]["resolution"],
            "source_run_time_utc": plan["source"]["run_time_utc"],
            "bbox_wgs84": json.dumps(plan["target"]["bbox_wgs84"]),
        }
    )
    if "pressure_hpa" in dataset.coords:
        dataset["pressure_pa"] = dataset["pressure_hpa"] * 100.0
        dataset["pressure_pa"].attrs["units"] = "Pa"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dataset.to_netcdf(path)
    except Exception as exc:
        return False, str(exc), decode_errors, decoded_files
    return True, None, decode_errors, decoded_files


def add_surface_fields(nc_path: Path) -> dict[str, Any]:
    try:
        import numpy as np
        import xarray as xr
    except ImportError as exc:
        return {"added": False, "error": str(exc)}
    try:
        dataset = xr.open_dataset(nc_path).load()
    except Exception as exc:
        return {"added": False, "error": str(exc)}
    if "latitude" not in dataset.coords or "longitude" not in dataset.coords:
        return {"added": False, "error": "parent NetCDF has no latitude/longitude coordinates"}
    try:
        dem_tiles = load_dem_tiles()
    except Exception as exc:
        return {"added": False, "error": str(exc)}

    lon2d, lat2d = np.meshgrid(dataset["longitude"].values, dataset["latitude"].values)
    topography = sample_dem(dem_tiles, lon2d, lat2d)
    if not np.isfinite(topography).any():
        return {"added": False, "error": "DEM sampling produced no finite values"}
    topography = np.where(np.isfinite(topography), topography, 0.0).astype(np.float32)
    landmask = (topography > 1.0).astype(np.float32)
    z0m = np.where(landmask > 0.5, 0.03, 0.0002).astype(np.float32)
    z0m = np.where(topography > 200.0, 0.08, z0m).astype(np.float32)

    dataset["topography_m"] = xr.DataArray(topography, dims=("latitude", "longitude"))
    dataset["landmask"] = xr.DataArray(landmask, dims=("latitude", "longitude"))
    dataset["z0m"] = xr.DataArray(z0m, dims=("latitude", "longitude"))
    dataset["topography_m"].attrs.update({"units": "m", "source": "Copernicus GLO-30 sampled to AROME parent grid"})
    dataset["landmask"].attrs.update({"units": "1", "description": "1 land, 0 sea; derived from DEM elevation > 1 m"})
    dataset["z0m"].attrs.update({"units": "m", "description": "POC roughness: sea 0.0002, low land 0.03, rough terrain 0.08"})
    dataset.attrs["surface_fields"] = "Copernicus DEM derived topography_m, landmask, z0m"
    tmp_path = nc_path.with_suffix(nc_path.suffix + ".tmp")
    dataset.to_netcdf(tmp_path)
    dataset.close()
    tmp_path.replace(nc_path)
    return {
        "added": True,
        "topography_min_m": float(np.nanmin(topography)),
        "topography_max_m": float(np.nanmax(topography)),
        "land_fraction": float(np.nanmean(landmask)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--skip-surface-fields", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    manifest_path = args.manifest_output if args.manifest_output.is_absolute() else ROOT / args.manifest_output
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    grib_groups, tiff_arrays, surface_gribs, missing, unsupported, collect_errors = collect_inputs(plan)
    written, write_error, decode_errors, decoded_files = write_parent_nc(output_path, grib_groups, tiff_arrays, surface_gribs, plan)
    surface_status = {"added": False, "error": "parent NetCDF was not written"}
    if written and not args.skip_surface_fields:
        surface_status = add_surface_fields(output_path)
    unresolved_inputs = list(missing)
    derived_inputs: list[str] = []
    if surface_status.get("added"):
        derived_inputs.append("land_sea_and_roughness")
        unresolved_inputs = [item for item in unresolved_inputs if item != "land_sea_and_roughness"]
    expected_grib_inputs = sum(len(entries) for entries in grib_groups.values())
    expected_grib_inputs += len(surface_gribs)
    manifest = {
        "format": "corsewind.fasteddy_parent_poc_manifest.v1",
        "generated_at_utc": utc_now(),
        "plan": display_path(plan_path),
        "output": display_path(output_path) if written else None,
        "fasteddy_parent_test_ready": written and not decode_errors and surface_status.get("added", False),
        "production_fasteddy_ready": False,
        "poc_parent_written": written,
        "surface_fields": surface_status,
        "expected_grib_inputs": expected_grib_inputs,
        "decoded_grib_inputs": len(decoded_files),
        "decoded_files": decoded_files,
        "readable_tiff_arrays": sorted(tiff_arrays),
        "surface_grib_inputs": {key: display_path(value) for key, value in surface_gribs.items()},
        "derived_inputs": derived_inputs,
        "missing_inputs": unresolved_inputs,
        "missing_source_inputs": missing,
        "unsupported_inputs": unsupported,
        "decode_errors": collect_errors + decode_errors,
        "error": write_error,
        "next_step": "Generate FastEddy IC/BC files from this parent NetCDF, then run the Ajaccio/Bonifacio GPU benchmark against the WindNinja baseline.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {display_path(manifest_path)}")
    if write_error:
        print(f"parent_poc_not_written={write_error}")
    else:
        print(f"wrote {display_path(output_path)}")


if __name__ == "__main__":
    main()
