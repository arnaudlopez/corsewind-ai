#!/usr/bin/env python3
"""Prepare FastEddy smoke benchmark cases for Ajaccio and Bonifacio."""

from __future__ import annotations

import argparse
import json
import math
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from apply_terrain_wind_correction import inverse_lambert93, lambert93
from prepare_corsica_windninja_tiles import (
    ascii_axes,
    bounds_wgs84,
    header_for_bbox,
    load_dem_tiles,
    load_forecast_step,
    sample_dem,
)
from windninja_grid_utils import bilinear_sample, direction_mean_deg, meteorological_direction_from_uv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "benchmarks/fasteddy/benchmark_config.json"
DEFAULT_PLAN = ROOT / "data/processed/benchmarks/fasteddy/benchmark_plan.json"
DEFAULT_REPORT = ROOT / "reports/fasteddy_benchmark_plan.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def projected_domain_from_center(center_wgs84: list[float], size_km: list[float]) -> tuple[float, float, float, float]:
    center_x, center_y = lambert93(float(center_wgs84[0]), float(center_wgs84[1]))
    width_m = float(size_km[0]) * 1000.0
    height_m = float(size_km[1]) * 1000.0
    return (
        center_x - width_m / 2.0,
        center_y - height_m / 2.0,
        center_x + width_m / 2.0,
        center_y + height_m / 2.0,
    )


def write_fasteddy_topography(path: Path, dem: np.ndarray) -> None:
    rows, cols = dem.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("i", cols))
        handle.write(struct.pack("i", rows))
        handle.write(struct.pack(f"{cols * rows}f", *dem.astype(np.float32).flatten()))


def write_fasteddy_gis(path: Path, lons: np.ndarray, lats: np.ndarray, dem: np.ndarray, landcover: np.ndarray, cellsize_m: float) -> str | None:
    try:
        from netCDF4 import Dataset
    except ImportError:
        return "netCDF4 is not installed; install requirements-benchmark.txt before running FastEddy preprocessing."

    path.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = dem.shape
    with Dataset(path, "w") as dataset:
        dataset.createDimension("y", rows)
        dataset.createDimension("x", cols)
        dataset.createVariable("cellsize", "f4").assignValue(float(cellsize_m))
        dataset.createVariable("lat", "f8", ("y", "x"))[:, :] = lats
        dataset.createVariable("lon", "f8", ("y", "x"))[:, :] = lons
        dataset.createVariable("elevation", "f4", ("y", "x"))[:, :] = dem.astype(np.float32)
        dataset.createVariable("LandCover", "i4", ("y", "x"))[:, :] = landcover.astype(np.int32)
    return None


def write_landcover_table(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "class,name,z0m",
                "11,water,0.0002",
                "21,open_land,0.03",
                "41,forest_or_rough,0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def fasteddy_input_text(
    zone_id: str,
    nx: int,
    ny: int,
    args: argparse.Namespace,
    terrain_name: str,
    mean_u_ms: float,
    mean_v_ms: float,
    latitude: float,
) -> str:
    return f"""Description = CorseWind FastEddy smoke benchmark for {zone_id}.
#--MPI_AALES
numProcsX = {int(args.mpi_ranks)}
numProcsY = 1
#--CUDA_AALES
tBx = 1
tBy = 8
tBz = 32
#--IO
inPath =
inFile =
outPath = ./output/
outFileBase = FE_{zone_id}
frqOutput = {int(args.output_frequency_steps)}
ioOutputMode = 0
#--GRID
Nx = {nx}
Ny = {ny}
Nz = {int(args.vertical_levels)}
Nh = 3
d_xi = {args.horizontal_resolution_m:.3f}
d_eta = {args.horizontal_resolution_m:.3f}
d_zeta = {args.vertical_cell_m:.3f}
coordHorizHalos = 1
topoFile = ./{terrain_name}
verticalDeformSwitch = 1
verticalDeformFactor = 0.50
verticalDeformQuadCoeff = 0.0
#--TIME_INTEGRATION
timeMethod = 0
Nt = {int(args.timesteps)}
dt = {args.dt_s:.5f}
NtBatch = {int(args.output_frequency_steps)}
#--HYDRO_CORE
hydroBCs = 2
hydroForcingWrite = 0
hydroSubGridWrite = 0
hydroForcingLog = 0
advectionSelector = 3
b_hyb = 0.0
moistureSelector = 0
moistureNvars = 0
coriolisSelector = 1
coriolisLatitude = {latitude:.5f}
turbulenceSelector = 1
TKESelector = 1
TKEAdvSelector = 3
TKEAdvSelector_b_hyb = 0.0
c_s = 0.18
c_k = 0.10
diffusionSelector = 0
nu_0 = 0.0
filterSelector = 1
filter_6thdiff_vert = 1
filter_6thdiff_vert_coeff = 0.03
dampingLayerSelector = 1
dampingLayerDepth = 200.0
surflayerSelector = 2
surflayer_z0 = 0.03
surflayer_z0t = 0.003
surflayer_wth = 0.0
surflayer_tr = 0.0
surflayer_wq = 0.0
surflayer_qr = 0.0
surflayer_idealsine = 0
surflayer_ideal_ts = 0.0
surflayer_ideal_te = 43200
surflayer_ideal_amp = 0.0
surflayer_offshore = 1
surflayer_offshore_opt = 4
cellpertSelector = 0
stabilityScheme = 2
temp_grnd = 293.0
pres_grnd = 100000.0
zStableBottom = 500.0
stableGradient = 0.003
zStableBottom2 = 650.0
stableGradient2 = 0.003
zStableBottom3 = 50000.0
stableGradient3 = 0.003
thetaPerturbationSwitch = 0
thetaHeight = 300.0
thetaAmplitude = 0.0
U_g = {mean_u_ms:.4f}
V_g = {mean_v_ms:.4f}
z_Ug = 10000.0
z_Vg = 10000.0
Ug_grad = 0.0
Vg_grad = 0.0
lsfSelector = 0
"""


def prepare_case(
    zone: dict[str, Any],
    dem_tiles: list[Any],
    arome_payload: dict[str, Any],
    step: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    min_x, min_y, max_x, max_y = projected_domain_from_center(zone["center_wgs84"], zone["size_km"])
    header = header_for_bbox(min_x, min_y, max_x, max_y, args.horizontal_resolution_m)
    xs, ys = ascii_axes(header)
    x_grid = xs[None, :]
    y_grid = ys[:, None]
    lons, lats = inverse_lambert93(x_grid, y_grid)

    dem_raw = sample_dem(dem_tiles, lons, lats)
    missing_dem_fraction = float(np.mean(~np.isfinite(dem_raw)))
    dem = np.where(np.isfinite(dem_raw), np.maximum(dem_raw, 0.0), 0.0).astype(np.float32)
    u_grid = bilinear_sample(np.array(step["u_ms"], dtype=np.float32), lons, lats, arome_payload["bbox_wgs84"])
    v_grid = bilinear_sample(np.array(step["v_ms"], dtype=np.float32), lons, lats, arome_payload["bbox_wgs84"])
    speed_grid = np.hypot(u_grid, v_grid).astype(np.float32)
    direction_grid = meteorological_direction_from_uv(u_grid, v_grid)
    mean_u = float(np.nanmean(u_grid))
    mean_v = float(np.nanmean(v_grid))
    mean_speed = float(np.nanmean(speed_grid))
    mean_direction = direction_mean_deg(direction_grid)

    landcover = np.where(dem > 2.0, 21, 11).astype(np.int32)
    landcover[(dem > 250.0)] = 41

    lead_key = f"h{int(args.lead_hour):02d}"
    case_dir = ROOT / "data/processed/benchmarks/fasteddy" / f"{zone['id']}_{lead_key}_{int(args.horizontal_resolution_m)}m"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "output").mkdir(exist_ok=True)

    terrain_path = case_dir / f"{zone['id']}_Topography_{int(header['ncols'])}x{int(header['nrows'])}.dat"
    write_fasteddy_topography(terrain_path, dem)
    gis_warning = write_fasteddy_gis(case_dir / "gis" / "input_gis.nc", lons, lats, dem, landcover, args.horizontal_resolution_m)
    write_landcover_table(case_dir / "gis" / "landcover_table.csv")

    input_path = case_dir / "fasteddy_smoke.in"
    input_path.write_text(
        fasteddy_input_text(
            zone["id"],
            int(header["ncols"]),
            int(header["nrows"]),
            args,
            terrain_path.name,
            mean_u,
            mean_v,
            float(zone["center_wgs84"][1]),
        ),
        encoding="utf-8",
    )

    geospec = {
        "name_dom": zone["id"],
        "gis_root": str((case_dir / "gis").resolve()) + "/",
        "gis_file": "input_gis.nc",
        "landcover_table": "landcover_table.csv",
        "water_cats": [11],
        "urban_opt": 0,
        "FE_dataset_path": str((case_dir / "geospec").resolve()) + "/",
        "name_dom_add": "",
        "gis_opt": 0,
        "save_plot_opt": 0,
    }
    simgrid = {
        "name_dom": zone["id"],
        "FE_ref_GIS_nc": str((case_dir / "geospec" / f"{zone['id']}.nc").resolve()),
        "FE_params_file": str(input_path.resolve()),
        "center_lat": float(zone["center_wgs84"][1]),
        "center_lon": float(zone["center_wgs84"][0]),
        "urban_opt": 0,
        "FE_new_nc_path": str((case_dir / "simgrid").resolve()) + "/",
        "name_dom_add": "",
        "urban_heatRedis_opt": 0,
        "landcover_table": str((case_dir / "gis" / "landcover_table.csv").resolve()),
        "topo_average_opt": 0,
        "save_plot_opt": 0,
    }
    (case_dir / "geospec.json").write_text(json.dumps(geospec, indent=2), encoding="utf-8")
    (case_dir / "simgrid.json").write_text(json.dumps(simgrid, indent=2), encoding="utf-8")

    metadata = {
        "format": "corsewind.fasteddy_benchmark.case.v1",
        "zone": zone,
        "case_dir": str(case_dir.relative_to(ROOT)),
        "lead_hour": int(args.lead_hour),
        "valid_time_utc": step["valid_time_utc"],
        "domain": {
            "bounds_wgs84": bounds_wgs84(header),
            "shape": [int(header["nrows"]), int(header["ncols"]), int(args.vertical_levels)],
            "cellsize_m": args.horizontal_resolution_m,
            "vertical_cell_m": args.vertical_cell_m,
            "output_height_m": args.output_height_m,
            "simulated_seconds": round(float(args.timesteps) * float(args.dt_s), 3),
            "missing_dem_fraction_filled": round(missing_dem_fraction, 6),
        },
        "forcing": {
            "source": f"AROME {arome_payload['run_time_utc']} H+{args.lead_hour}",
            "mean_u_ms": round(mean_u, 4),
            "mean_v_ms": round(mean_v, 4),
            "mean_speed_ms": round(mean_speed, 4),
            "mean_direction_from_deg": round(mean_direction, 3),
            "mode": "smoke_benchmark_geostrophic_mean_from_arome",
        },
        "fasteddy": {
            "input_file": str(input_path.relative_to(ROOT)),
            "topography_file": str(terrain_path.relative_to(ROOT)),
            "gis_file": str((case_dir / "gis" / "input_gis.nc").relative_to(ROOT)),
            "geospec_config": str((case_dir / "geospec.json").relative_to(ROOT)),
            "simgrid_config": str((case_dir / "simgrid.json").relative_to(ROOT)),
            "output_glob": "output/FE_*",
            "mpi_ranks": int(args.mpi_ranks),
            "command": [
                "{FASTEDDY_BIN}",
                str(input_path.relative_to(case_dir)),
            ],
            "mpi_command": [
                "mpirun",
                "-np",
                str(int(args.mpi_ranks)),
                "{FASTEDDY_BIN}",
                str(input_path.relative_to(case_dir)),
            ],
            "warnings": [gis_warning] if gis_warning else [],
        },
    }
    (case_dir / "case_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def write_plan(cases: list[dict[str, Any]], config: dict[str, Any], args: argparse.Namespace) -> None:
    payload = {
        "format": "corsewind.fasteddy_benchmark.plan.v1",
        "generated_at_utc": utc_now(),
        "objective": "Reveal early whether FastEddy GPU LES is promising for CorseWind coastal microscale wind.",
        "sources": config.get("fasteddy", {}),
        "settings": {
            "lead_hour": args.lead_hour,
            "horizontal_resolution_m": args.horizontal_resolution_m,
            "vertical_cell_m": args.vertical_cell_m,
            "vertical_levels": args.vertical_levels,
            "timesteps": args.timesteps,
            "dt_s": args.dt_s,
            "mpi_ranks": args.mpi_ranks,
        },
        "cases": cases,
    }
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# FastEddy Benchmark Plan",
        "",
        f"- Cases: `{len(cases)}`",
        f"- Lead hour: `H+{args.lead_hour}`",
        f"- Horizontal resolution: `{args.horizontal_resolution_m:g} m`",
        f"- Simulated duration: `{float(args.timesteps) * float(args.dt_s):g} s`",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                f"## {case['zone']['label']}",
                "",
                f"- Bounds: `{case['domain']['bounds_wgs84']}`",
                f"- Shape y/x/z: `{case['domain']['shape']}`",
                f"- AROME mean u/v/speed: `{case['forcing']['mean_u_ms']} / {case['forcing']['mean_v_ms']} / {case['forcing']['mean_speed_ms']} m/s`",
                f"- Command: `{' '.join(case['fasteddy']['command'])}`",
                "",
            ]
        )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--lead-hour", type=int, default=None)
    parser.add_argument("--horizontal-resolution-m", type=float, default=None)
    parser.add_argument("--vertical-cell-m", type=float, default=None)
    parser.add_argument("--vertical-levels", type=int, default=None)
    parser.add_argument("--output-height-m", type=float, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--dt-s", type=float, default=None)
    parser.add_argument("--output-frequency-steps", type=int, default=None)
    parser.add_argument("--mpi-ranks", type=int, default=None)
    parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def apply_defaults(args: argparse.Namespace, defaults: dict[str, Any]) -> argparse.Namespace:
    for attr, key in [
        ("lead_hour", "lead_hour"),
        ("horizontal_resolution_m", "horizontal_resolution_m"),
        ("vertical_cell_m", "vertical_cell_m"),
        ("vertical_levels", "vertical_levels"),
        ("output_height_m", "output_height_m"),
        ("timesteps", "timesteps"),
        ("dt_s", "dt_s"),
        ("output_frequency_steps", "output_frequency_steps"),
        ("mpi_ranks", "mpi_ranks"),
    ]:
        if getattr(args, attr) is None:
            setattr(args, attr, defaults[key])
    args.plan_output = args.plan_output if args.plan_output.is_absolute() else ROOT / args.plan_output
    args.report_output = args.report_output if args.report_output.is_absolute() else ROOT / args.report_output
    return args


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = apply_defaults(args, config["defaults"])
    dem_tiles = load_dem_tiles()
    arome_payload, step = load_forecast_step(args.lead_hour)
    cases = [prepare_case(zone, dem_tiles, arome_payload, step, args) for zone in config["zones"]]
    write_plan(cases, config, args)
    print(f"prepared {len(cases)} FastEddy benchmark case(s)")
    print(f"wrote {args.plan_output.relative_to(ROOT)}")
    print(f"wrote {args.report_output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
