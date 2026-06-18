#!/usr/bin/env python3
"""Prepare prod-like FastEddy real-case inputs from the validated AROME parent dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from apply_terrain_wind_correction import inverse_lambert93, lambert93
from build_fasteddy_parent_poc import load_worldcover_tiles, sample_worldcover
from prepare_corsica_windninja_tiles import load_dem_tiles, sample_dem
from validate_fasteddy_parent_inputs import validate_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "benchmarks/fasteddy/prod_like_config.json"
DEFAULT_PARENT = ROOT / "data/processed/benchmarks/fasteddy/arome_poc/fasteddy_parent_poc.nc"
DEFAULT_MANIFEST = ROOT / "data/processed/benchmarks/fasteddy/arome_poc/fasteddy_parent_poc_manifest.json"
DEFAULT_OUTPUT_ROOT = ROOT / "data/processed/benchmarks/fasteddy/prod_like"
DEFAULT_REPORT = ROOT / "reports/fasteddy_prod_like_plan.md"
DEFAULT_STATUS = ROOT / "data/processed/benchmarks/fasteddy/prod_like_status.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def projected_domain_from_center(center_wgs84: list[float], size_km: list[float], padding_km: float) -> tuple[float, float, float, float]:
    center_x, center_y = lambert93(float(center_wgs84[0]), float(center_wgs84[1]))
    width_m = (float(size_km[0]) + 2.0 * float(padding_km)) * 1000.0
    height_m = (float(size_km[1]) + 2.0 * float(padding_km)) * 1000.0
    return center_x - width_m / 2.0, center_y - height_m / 2.0, center_x + width_m / 2.0, center_y + height_m / 2.0


def xy_axes(min_x: float, min_y: float, max_x: float, max_y: float, cellsize_m: float) -> tuple[np.ndarray, np.ndarray]:
    ncols = int(math.ceil((max_x - min_x) / cellsize_m))
    nrows = int(math.ceil((max_y - min_y) / cellsize_m))
    xs = min_x + (np.arange(ncols, dtype=np.float64) + 0.5) * cellsize_m
    ys = min_y + (np.arange(nrows, dtype=np.float64) + 0.5) * cellsize_m
    return xs, ys


def bounds_wgs84_from_grid(lons: np.ndarray, lats: np.ndarray) -> list[float]:
    return [
        round(float(np.nanmin(lons)), 6),
        round(float(np.nanmin(lats)), 6),
        round(float(np.nanmax(lons)), 6),
        round(float(np.nanmax(lats)), 6),
    ]


def load_roughness_lookup(path: Path) -> dict[int, float]:
    lookup: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lookup[int(row["class"])] = float(row["z0m"])
    return lookup


def z0m_from_lookup(landcover: np.ndarray, lookup: dict[int, float], landmask: np.ndarray) -> np.ndarray:
    z0m = np.full(landcover.shape, np.nan, dtype=np.float32)
    for class_id, roughness in lookup.items():
        z0m[landcover == class_id] = roughness
    return np.where(np.isfinite(z0m), z0m, np.where(landmask > 0.5, 0.03, 0.0002)).astype(np.float32)


def write_geospec_input(
    path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    elevation: np.ndarray,
    landcover: np.ndarray,
    cellsize_m: float,
) -> None:
    from netCDF4 import Dataset

    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("x", len(xs))
        dataset.createDimension("y", len(ys))
        dataset.createVariable("x", "f4", ("x",))[:] = xs.astype(np.float32)
        dataset.createVariable("y", "f4", ("y",))[:] = ys.astype(np.float32)
        dataset.createVariable("cellsize", "f4").assignValue(float(cellsize_m))
        dataset.createVariable("elevation", "f4", ("y", "x"))[:, :] = elevation.astype(np.float32)
        dataset.createVariable("lat", "f8", ("y", "x"))[:, :] = lats.astype(np.float64)
        dataset.createVariable("lon", "f8", ("y", "x"))[:, :] = lons.astype(np.float64)
        dataset.createVariable("LandCover", "i4", ("y", "x"))[:, :] = landcover.astype(np.int32)
        dataset.setncattr("format", "corsewind.fasteddy.geospec_gis_input.v1")
        dataset.setncattr("projection", "Lambert-93 EPSG:2154")
        dataset.setncattr("orientation", "south_to_north_west_to_east")


def saturation_vapor_pressure_pa(temperature_k: Any) -> Any:
    return 611.2 * np.exp(17.67 * (temperature_k - 273.15) / (temperature_k - 29.65))


def write_arome_bridge(parent_path: Path, output_path: Path) -> dict[str, Any]:
    import xarray as xr

    ds = xr.open_dataset(parent_path).load()
    pressure = ds["pressure_pa"]
    temperature = ds["temperature"]
    rh = ds["relative_humidity"].clip(0.0, 100.0)
    vapor_pressure = (rh / 100.0) * saturation_vapor_pressure_pa(temperature)
    qv = 0.622 * vapor_pressure / (pressure - 0.378 * vapor_pressure)
    virtual_temperature = temperature * (1.0 + 0.61 * qv)
    density = pressure / (287.05 * virtual_temperature)
    bridge = xr.Dataset(
        {
            "U": ds["u"],
            "V": ds["v"],
            "W": ds["w"],
            "T": ds["temperature"],
            "THETA": ds["potential_temperature"],
            "RH": ds["relative_humidity"],
            "QVAPOR": qv,
            "P": pressure,
            "HEIGHT": ds["height_m"],
            "RHO": density,
            "ALT": 1.0 / density,
            "TSK": ds["surface_temperature"],
            "PSFC": ds["surface_pressure"],
            "HGT": ds["topography_m"],
            "XLAT": xr.DataArray(
                np.broadcast_to(ds["latitude"].values[:, None], (ds.sizes["latitude"], ds.sizes["longitude"])),
                dims=("latitude", "longitude"),
            ),
            "XLONG": xr.DataArray(
                np.broadcast_to(ds["longitude"].values[None, :], (ds.sizes["latitude"], ds.sizes["longitude"])),
                dims=("latitude", "longitude"),
            ),
        }
    )
    bridge.attrs.update(
        {
            "format": "corsewind.arome_fasteddy_bridge.v1",
            "source": display_path(parent_path),
            "compatibility": "Normalized AROME parent forcing; requires CorseWind AROME-to-FastEddy IC/BC adapter before stock FastEddy run.",
        }
    )
    bridge["QVAPOR"].attrs.update({"units": "kg kg-1", "long_name": "water vapor mixing ratio approximation from RH/T/P"})
    bridge["RHO"].attrs.update({"units": "kg m-3", "long_name": "moist air density approximation"})
    bridge["ALT"].attrs.update({"units": "m3 kg-1", "long_name": "inverse density"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bridge.to_netcdf(output_path)
    return {
        "path": display_path(output_path),
        "fields": list(bridge.data_vars),
        "qv_range_kgkg": [float(bridge["QVAPOR"].min()), float(bridge["QVAPOR"].max())],
        "rho_range_kgm3": [float(bridge["RHO"].min()), float(bridge["RHO"].max())],
    }


def write_landcover_table(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def fasteddy_params_text(zone_id: str, nx: int, ny: int, config: dict[str, Any], latitude: float) -> str:
    defaults = config["defaults"]
    return f"""Description = CorseWind prod-like real-case FastEddy template for {zone_id}.
#--MPI_AALES
numProcsX = {int(defaults["mpi_ranks"])}
numProcsY = 1
#--CUDA_AALES
tBx = 1
tBy = 8
tBz = 32
#--IO
inPath = ./
inFile = FE_interp_INITIAL.0
outPath = ./output/
outFileBase = FE_{zone_id}
frqOutput = {int(defaults["output_frequency_steps"])}
ioOutputMode = 0
#--GRID
Nx = {nx}
Ny = {ny}
Nz = {int(defaults["vertical_levels"])}
Nh = 3
d_xi = {float(defaults["solver_horizontal_resolution_m"]):.3f}
d_eta = {float(defaults["solver_horizontal_resolution_m"]):.3f}
d_zeta = {float(defaults["vertical_cell_m"]):.3f}
coordHorizHalos = 1
topoFile = ./TO_BE_FILLED_BY_SIMGRID
verticalDeformSwitch = 1
verticalDeformFactor = 0.50
verticalDeformQuadCoeff = 0.0
#--TIME_INTEGRATION
timeMethod = 0
Nt = {int(defaults["timesteps"])}
dt = {float(defaults["dt_s"]):.5f}
NtBatch = {int(defaults["output_frequency_steps"])}
#--HYDRO_CORE
hydroBCs = 1
hydroForcingWrite = 0
hydroSubGridWrite = 0
hydroForcingLog = 0
advectionSelector = 3
b_hyb = 0.0
moistureSelector = 1
moistureNvars = 1
coriolisSelector = 1
coriolisLatitude = {latitude:.5f}
turbulenceSelector = 1
TKESelector = 1
TKEAdvSelector = 3
TKEAdvSelector_b_hyb = 0.0
c_s = 0.18
c_k = 0.10
filterSelector = 1
dampingLayerSelector = 1
dampingLayerDepth = 200.0
surflayerSelector = 2
surflayer_z0 = 0.03
surflayer_z0t = 0.003
surflayer_offshore = 1
surflayer_offshore_opt = 4
cellpertSelector = 1
stabilityScheme = 2
lsfSelector = 0
"""


def prepare_zone(
    zone: dict[str, Any],
    config: dict[str, Any],
    parent_path: Path,
    output_root: Path,
    roughness_table: Path,
) -> dict[str, Any]:
    defaults = config["defaults"]
    min_x, min_y, max_x, max_y = projected_domain_from_center(
        zone["center_wgs84"],
        zone["size_km"],
        float(defaults["domain_padding_km"]),
    )
    xs, ys = xy_axes(min_x, min_y, max_x, max_y, float(defaults["surface_resolution_m"]))
    x_grid, y_grid = np.meshgrid(xs, ys)
    lons, lats = inverse_lambert93(x_grid, y_grid)

    dem_tiles = load_dem_tiles()
    worldcover_tiles = load_worldcover_tiles()
    elevation_raw = sample_dem(dem_tiles, lons, lats)
    missing_dem_fraction = float(np.mean(~np.isfinite(elevation_raw)))
    elevation = np.where(np.isfinite(elevation_raw), np.maximum(elevation_raw, 0.0), 0.0).astype(np.float32)
    landcover = sample_worldcover(worldcover_tiles, lons, lats)
    unknown_landcover_fraction = float(np.mean(landcover == 0))
    landmask = (landcover != 80).astype(np.float32)
    lookup = load_roughness_lookup(roughness_table)
    z0m = z0m_from_lookup(landcover, lookup, landmask)

    run_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    case_dir = output_root / f"{zone['id']}_{run_label}"
    gis_dir = case_dir / "gis"
    geospec_dir = case_dir / "geospec"
    simgrid_dir = case_dir / "simgrid"
    icbc_dir = case_dir / "icbc"
    output_dir = case_dir / "output"
    for directory in [gis_dir, geospec_dir, simgrid_dir, icbc_dir, output_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    geospec_input = gis_dir / "input_gis.nc"
    write_geospec_input(
        geospec_input,
        xs,
        ys,
        lons,
        lats,
        elevation,
        landcover,
        float(defaults["surface_resolution_m"]),
    )
    landcover_table = gis_dir / roughness_table.name
    write_landcover_table(roughness_table, landcover_table)
    bridge = write_arome_bridge(parent_path, icbc_dir / "arome_fasteddy_bridge.nc")

    nx = int(math.ceil((max_x - min_x) / float(defaults["solver_horizontal_resolution_m"])))
    ny = int(math.ceil((max_y - min_y) / float(defaults["solver_horizontal_resolution_m"])))
    params_path = case_dir / "fasteddy_real.in"
    params_path.write_text(fasteddy_params_text(zone["id"], nx, ny, config, float(zone["center_wgs84"][1])), encoding="utf-8")

    geospec = {
        "name_dom": zone["id"],
        "gis_root": str(gis_dir.resolve()) + "/",
        "gis_file": geospec_input.name,
        "nlcd_name": str(landcover_table.resolve()),
        "water_cats": [80],
        "urban_opt": 0,
        "FE_dataset_path": str(geospec_dir.resolve()) + "/",
        "name_dom_add": "",
        "gis_opt": 0,
        "save_plot_opt": 0,
    }
    simgrid = {
        "name_dom": zone["id"],
        "FE_ref_GIS_nc": str((geospec_dir / f"{zone['id']}.nc").resolve()),
        "FE_params_file": str(params_path.resolve()),
        "center_lat": float(zone["center_wgs84"][1]),
        "center_lon": float(zone["center_wgs84"][0]),
        "urban_opt": 0,
        "FE_new_nc_path": str(simgrid_dir.resolve()) + "/",
        "name_dom_add": "",
        "urban_heatRedis_opt": 0,
        "landcover_table": str(landcover_table.resolve()),
        "topo_average_opt": 0,
        "save_plot_opt": 0,
    }
    genicbcs = {
        "name_dom": zone["id"],
        "parent_model": "corsewind_arome_bridge",
        "stock_genicbcs_compatible": False,
        "reason": "Stock GenICBCs expects WRF/FastEddy parent files. This package provides a normalized AROME bridge for the CorseWind adapter/direct ICBC writer.",
        "simgrid_output": str((simgrid_dir / f"{zone['id']}.0").resolve()),
        "arome_bridge": str((icbc_dir / "arome_fasteddy_bridge.nc").resolve()),
        "FE_ICBC_path": str(icbc_dir.resolve()) + "/",
        "secInc": 3600,
    }
    (case_dir / "geospec.json").write_text(json.dumps(geospec, indent=2), encoding="utf-8")
    (case_dir / "simgrid.json").write_text(json.dumps(simgrid, indent=2), encoding="utf-8")
    (case_dir / "genicbcs_arome_adapter.json").write_text(json.dumps(genicbcs, indent=2), encoding="utf-8")

    return {
        "zone": zone,
        "case_dir": display_path(case_dir),
        "bounds_wgs84": bounds_wgs84_from_grid(lons, lats),
        "grid": {
            "surface_shape_yx": [int(len(ys)), int(len(xs))],
            "surface_resolution_m": float(defaults["surface_resolution_m"]),
            "solver_shape_xyz": [nx, ny, int(defaults["vertical_levels"])],
            "solver_horizontal_resolution_m": float(defaults["solver_horizontal_resolution_m"]),
            "vertical_cell_m": float(defaults["vertical_cell_m"]),
        },
        "surface": {
            "geospec_input": display_path(geospec_input),
            "landcover_table": display_path(landcover_table),
            "missing_dem_fraction": round(missing_dem_fraction, 6),
            "unknown_landcover_fraction": round(unknown_landcover_fraction, 6),
            "worldcover_classes": sorted(int(value) for value in np.unique(landcover)),
            "z0m_range_m": [float(np.nanmin(z0m)), float(np.nanmax(z0m))],
        },
        "forcing": bridge,
        "configs": {
            "fasteddy_params": display_path(params_path),
            "geospec": display_path(case_dir / "geospec.json"),
            "simgrid": display_path(case_dir / "simgrid.json"),
            "genicbcs_adapter": display_path(case_dir / "genicbcs_arome_adapter.json"),
        },
        "commands": {
            "geospec": "python $FASTEDDY_COUPLER_DIR/GeoSpec.py -f geospec.json",
            "simgrid": "python $FASTEDDY_COUPLER_DIR/SimGrid.py -f simgrid.json",
            "genicbcs": "python $CORSEWIND_FASTEDDY_ADAPTER -f genicbcs_arome_adapter.json",
            "fasteddy": "$FASTEDDY_BIN fasteddy_real.in",
        },
    }


def write_report(status: dict[str, Any], path: Path) -> None:
    lines = [
        "# FastEddy Prod-Like Preparation",
        "",
        f"- Generated: `{status['generated_at_utc']}`",
        f"- Parent dataset: `{status['parent_dataset']}`",
        f"- Parent validation blockers: `{status['parent_validation']['blockers']}`",
        f"- Stock GenICBCs compatible now: `{status['readiness']['stock_genicbcs_compatible_now']}`",
        f"- Prod-like package ready: `{status['readiness']['prod_like_package_ready']}`",
        "",
        "## Cases",
        "",
    ]
    for case in status["cases"]:
        lines.extend(
            [
                f"### {case['zone']['label']}",
                "",
                f"- Case dir: `{case['case_dir']}`",
                f"- Bounds: `{case['bounds_wgs84']}`",
                f"- Surface grid y/x: `{case['grid']['surface_shape_yx']}`",
                f"- Solver grid x/y/z: `{case['grid']['solver_shape_xyz']}`",
                f"- WorldCover classes: `{case['surface']['worldcover_classes']}`",
                f"- z0m range: `{case['surface']['z0m_range_m']}`",
                "",
            ]
        )
    lines.extend(["## Remaining Production Work", ""])
    for item in status["readiness"]["remaining_work"]:
        lines.append(f"- {item}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--parent-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--allow-parent-warnings", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    parent_path = args.parent if args.parent.is_absolute() else ROOT / args.parent
    parent_manifest = args.parent_manifest if args.parent_manifest.is_absolute() else ROOT / args.parent_manifest
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    report_output = args.report_output if args.report_output.is_absolute() else ROOT / args.report_output
    status_output = args.status_output if args.status_output.is_absolute() else ROOT / args.status_output
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validation = validate_dataset(parent_path, parent_manifest)
    blockers = validation["verdict"]["blockers"]
    if blockers:
        raise SystemExit(f"Parent dataset is not usable: {blockers}")
    if validation["verdict"]["warnings"] and not args.allow_parent_warnings:
        raise SystemExit(f"Parent dataset has warnings; pass --allow-parent-warnings to continue: {validation['verdict']['warnings']}")
    roughness_table = ROOT / config["surface"]["roughness_lookup"]
    cases = [
        prepare_zone(zone, config, parent_path, output_root, roughness_table)
        for zone in config["zones"]
        if zone.get("enabled", True)
    ]
    status = {
        "format": "corsewind.fasteddy.prod_like_status.v1",
        "generated_at_utc": utc_now(),
        "config": display_path(config_path),
        "parent_dataset": display_path(parent_path),
        "parent_validation": validation["verdict"],
        "cases": cases,
        "readiness": {
            "prod_like_package_ready": True,
            "stock_geospec_simgrid_ready": True,
            "stock_genicbcs_compatible_now": False,
            "remaining_work": [
                "Implement or install the CorseWind AROME-to-FastEddy IC/BC adapter/direct writer.",
                "Run GeoSpec.py and SimGrid.py from the FastEddy coupler utilities on a GPU/Linux machine.",
                "Calibrate the WorldCover class-to-z0m lookup against local observations and coastline QA.",
                "Run FastEddy and convert FE outputs into the Wind2D raster contract.",
            ],
        },
        "sources": {
            "fasteddy_real_case_docs": "https://fasteddy-model.readthedocs.io/en/latest/Tutorials/cases_real/WRF_coupling_case0.html",
        },
    }
    status_output.parent.mkdir(parents=True, exist_ok=True)
    status_output.write_text(json.dumps(status, indent=2), encoding="utf-8")
    write_report(status, report_output)
    print(f"prepared {len(cases)} prod-like FastEddy case(s)")
    print(f"wrote {display_path(status_output)}")
    print(f"wrote {display_path(report_output)}")


if __name__ == "__main__":
    main()
