#!/usr/bin/env python3
"""Prepare coarse Corsica-wide WindNinja tiles from the current AROME layer.

This is the operational overview tier: coarse enough to fit an update budget,
but terrain-aware enough to improve the island-scale AROME context.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from apply_terrain_wind_correction import inverse_lambert93, lambert93
from windninja_grid_utils import (
    LAMBERT93_PRJ,
    bilinear_sample,
    direction_mean_deg,
    meteorological_direction_from_uv,
    write_ascii_grid,
)


ROOT = Path(__file__).resolve().parents[1]
DEM_DIR = ROOT / "data/raw/dem/copernicus_glo30"
AROME_LAYER = ROOT / "visualizations/wind2d/arome-corsica-latest.json"
OUT_ROOT = ROOT / "data/processed/physics/windninja_corsica"
PLAN_PATH = ROOT / "data/processed/physics/corsica_windninja_tile_plan.json"
REPORT_PATH = ROOT / "reports/corsica_windninja_automatic_process.md"
BATCH_STATUS_PATH = ROOT / "data/processed/diagnostics/corsica_windninja_batch_status.json"
BATCH_STATUS_1M_PATH = ROOT / "data/processed/diagnostics/corsica_windninja_1m_batch_status.json"


@dataclass
class DemTile:
    path: Path
    lon_min: int
    lat_min: int
    values: np.ndarray

    @property
    def lon_max(self) -> int:
        return self.lon_min + 1

    @property
    def lat_max(self) -> int:
        return self.lat_min + 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_dem_tile(path: Path) -> DemTile:
    match = re.search(r"_N(\d{2})_00_E(\d{3})_00_DEM\.tif$", path.name)
    if not match:
        raise ValueError(f"Unsupported Copernicus DEM tile name: {path.name}")
    lat_min = int(match.group(1))
    lon_min = int(match.group(2))
    values = np.array(Image.open(path), dtype=np.float32)
    values[values < -1000] = np.nan
    return DemTile(path=path, lon_min=lon_min, lat_min=lat_min, values=values)


def load_dem_tiles() -> list[DemTile]:
    tiles = [parse_dem_tile(path) for path in sorted(DEM_DIR.glob("Copernicus_DSM_COG_10_*_DEM.tif"))]
    if not tiles:
        raise FileNotFoundError(f"No Copernicus GLO-30 DEM tiles found in {DEM_DIR}")
    return tiles


def sample_dem(tiles: list[DemTile], lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    out = np.full(lon.shape, np.nan, dtype=np.float32)
    for tile in tiles:
        mask = (
            (lon >= tile.lon_min)
            & (lon < tile.lon_max)
            & (lat >= tile.lat_min)
            & (lat < tile.lat_max)
        )
        if not np.any(mask):
            continue
        rows, cols = tile.values.shape
        x = (lon[mask] - tile.lon_min) * cols - 0.5
        y = (tile.lat_max - lat[mask]) * rows - 0.5
        x = np.clip(x, 0, cols - 1)
        y = np.clip(y, 0, rows - 1)
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, cols - 1)
        y1 = np.clip(y0 + 1, 0, rows - 1)
        wx = x - x0
        wy = y - y0
        top = (1 - wx) * tile.values[y0, x0] + wx * tile.values[y0, x1]
        bottom = (1 - wx) * tile.values[y1, x0] + wx * tile.values[y1, x1]
        out[mask] = ((1 - wy) * top + wy * bottom).astype(np.float32)
    return out


def load_forecast_step(lead_hour: int) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(AROME_LAYER.read_text(encoding="utf-8"))
    for step in payload["forecast_steps"]:
        if int(step["lead_hour"]) == int(lead_hour):
            return payload, step
    raise ValueError(f"Lead hour H+{lead_hour} not found in {AROME_LAYER}")


def projected_bbox(wgs84_bbox: list[float]) -> tuple[float, float, float, float]:
    min_lon, min_lat, max_lon, max_lat = wgs84_bbox
    points = [lambert93(lon, lat) for lon in (min_lon, max_lon) for lat in (min_lat, max_lat)]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def header_for_bbox(min_x: float, min_y: float, max_x: float, max_y: float, cellsize_m: float) -> dict[str, float]:
    ncols = int(math.ceil((max_x - min_x) / cellsize_m))
    nrows = int(math.ceil((max_y - min_y) / cellsize_m))
    return {
        "ncols": float(ncols),
        "nrows": float(nrows),
        "xllcorner": float(min_x),
        "yllcorner": float(min_y),
        "cellsize": float(cellsize_m),
    }


def ascii_axes(header: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    cell = float(header["cellsize"])
    xll = float(header["xllcorner"])
    yll = float(header["yllcorner"])
    xs = xll + (np.arange(ncols, dtype=np.float64) + 0.5) * cell
    ys = yll + (nrows - np.arange(nrows, dtype=np.float64) - 0.5) * cell
    return xs, ys


def bounds_wgs84(header: dict[str, float]) -> list[float]:
    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    cell = float(header["cellsize"])
    xll = float(header["xllcorner"])
    yll = float(header["yllcorner"])
    xs = np.array([[xll, xll + ncols * cell]], dtype=np.float64)
    ys = np.array([[yll, yll + nrows * cell]], dtype=np.float64)
    lon, lat = inverse_lambert93(xs, ys)
    return [
        round(float(np.min(lon)), 6),
        round(float(np.min(lat)), 6),
        round(float(np.max(lon)), 6),
        round(float(np.max(lat)), 6),
    ]


def union_bounds_wgs84(bounds: list[list[float]]) -> list[float] | None:
    if not bounds:
        return None
    return [
        round(min(item[0] for item in bounds), 6),
        round(min(item[1] for item in bounds), 6),
        round(max(item[2] for item in bounds), 6),
        round(max(item[3] for item in bounds), 6),
    ]


def dem_tile_name(lat_floor: int, lon_floor: int) -> str:
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00_DEM.tif"


def required_dem_tiles_for_bounds(bbox: list[float]) -> list[str]:
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_start = math.floor(min_lon)
    lat_start = math.floor(min_lat)
    lon_stop = math.ceil(max_lon)
    lat_stop = math.ceil(max_lat)
    return [
        dem_tile_name(lat_floor, lon_floor)
        for lat_floor in range(lat_start, lat_stop)
        for lon_floor in range(lon_start, lon_stop)
    ]


def dem_audit(
    dem_tiles: list[DemTile],
    arome_bbox: list[float],
    windninja_bounds: list[float] | None,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    loaded = sorted(item.path.name for item in dem_tiles)
    loaded_set = set(loaded)
    arome_required = required_dem_tiles_for_bounds(arome_bbox)
    windninja_required = required_dem_tiles_for_bounds(windninja_bounds) if windninja_bounds else []
    missing_fractions = [float(case["domain"]["missing_dem_fraction_filled"]) for case in cases]
    weighted_cells = [
        int(case["domain"]["shape"][0]) * int(case["domain"]["shape"][1])
        for case in cases
    ]
    weighted_missing = (
        sum(frac * cells for frac, cells in zip(missing_fractions, weighted_cells)) / sum(weighted_cells)
        if weighted_cells and sum(weighted_cells) > 0
        else None
    )
    return {
        "source": "Copernicus GLO-30 Public DSM COG",
        "source_registry_url": "https://registry.opendata.aws/copernicus-dem/",
        "tile_dir": str(DEM_DIR.relative_to(ROOT)),
        "loaded_tile_count": len(loaded),
        "loaded_tiles": loaded,
        "arome_bbox_wgs84": arome_bbox,
        "arome_required_tiles": arome_required,
        "arome_missing_tiles": [name for name in arome_required if name not in loaded_set],
        "arome_coverage_complete": all(name in loaded_set for name in arome_required),
        "windninja_bounds_wgs84": windninja_bounds,
        "windninja_required_tiles": windninja_required,
        "windninja_missing_tiles": [name for name in windninja_required if name not in loaded_set],
        "windninja_coverage_complete": all(name in loaded_set for name in windninja_required),
        "prepared_tiles_missing_dem_fraction": {
            "min": round(min(missing_fractions), 6) if missing_fractions else None,
            "mean": round(sum(missing_fractions) / len(missing_fractions), 6) if missing_fractions else None,
            "max": round(max(missing_fractions), 6) if missing_fractions else None,
            "weighted_by_cell_count": round(weighted_missing, 6) if weighted_missing is not None else None,
        },
    }


def build_tile_headers(args: argparse.Namespace, arome_bbox: list[float]) -> list[dict[str, Any]]:
    min_x, min_y, max_x, max_y = projected_bbox(arome_bbox)
    tile_m = args.tile_size_km * 1000.0
    stride_m = max(args.cellsize_m, tile_m - args.overlap_km * 1000.0)
    tile_headers: list[dict[str, Any]] = []
    tile_id = 0
    y = min_y
    while y < max_y:
        x = min_x
        while x < max_x:
            tile_id += 1
            header = header_for_bbox(x, y, min(x + tile_m, max_x), min(y + tile_m, max_y), args.cellsize_m)
            tile_headers.append({"id": f"corsica_wn_{tile_id:03d}", "header": header})
            x += stride_m
        y += stride_m
    return tile_headers


def write_windninja_config(case_dir: Path, mesh_resolution_m: float, output_height_m: float, output_path: str) -> Path:
    config = f"""# CorseWind.ai Corsica-wide WindNinja overview tile.

elevation_file = dem_lambert93.asc
initialization_method = griddedInitialization
input_speed_grid = arome_speed_grid.asc
input_dir_grid = arome_dir_grid.asc
input_speed_units = mps
input_wind_height = 10.0
units_input_wind_height = m
output_speed_units = mps
output_wind_height = {output_height_m:.1f}
units_output_wind_height = m
mesh_resolution = {mesh_resolution_m:.1f}
units_mesh_resolution = m
vegetation = grass
diurnal_winds = false
write_goog_output = false
write_shapefile_output = false
write_ascii_output = true
output_path = {output_path}
"""
    path = case_dir / "windninja_corsica_tile.cfg"
    path.write_text(config, encoding="utf-8")
    return path


def prepare_tile(
    tile: dict[str, Any],
    dem_tiles: list[DemTile],
    arome_payload: dict[str, Any],
    step: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    header = tile["header"]
    xs, ys = ascii_axes(header)
    x_grid = xs[None, :]
    y_grid = ys[:, None]
    lons, lats = inverse_lambert93(x_grid, y_grid)
    dem = sample_dem(dem_tiles, lons, lats)
    missing_dem_fraction = float(np.mean(~np.isfinite(dem)))
    land_mask = np.isfinite(dem) & (dem > args.land_elevation_threshold_m)
    land_fraction = float(np.mean(land_mask))
    if args.min_land_fraction > 0 and land_fraction < args.min_land_fraction:
        return None
    dem = np.where(np.isfinite(dem), dem, 0.0).astype(np.float32)

    u_grid = bilinear_sample(np.array(step["u_ms"], dtype=np.float32), lons, lats, arome_payload["bbox_wgs84"])
    v_grid = bilinear_sample(np.array(step["v_ms"], dtype=np.float32), lons, lats, arome_payload["bbox_wgs84"])
    speed_grid = np.hypot(u_grid, v_grid).astype(np.float32)
    direction_grid = meteorological_direction_from_uv(u_grid, v_grid)

    case_dir = OUT_ROOT / f"{tile['id']}_{int(args.cellsize_m)}m_h{int(args.output_height_m):02d}_h{int(args.lead_hour):02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    write_ascii_grid(case_dir / "dem_lambert93.asc", dem, header)
    write_ascii_grid(case_dir / "arome_speed_grid.asc", speed_grid, header)
    write_ascii_grid(case_dir / "arome_dir_grid.asc", direction_grid, header)
    (case_dir / "dem_lambert93.prj").write_text(LAMBERT93_PRJ, encoding="utf-8")
    config_path = write_windninja_config(case_dir, args.mesh_resolution_m, args.output_height_m, "windninja_corsica_output")

    metadata = {
        "format": "corsewind.windninja.corsica.tile.case.v1",
        "case_dir": str(case_dir.relative_to(ROOT)),
        "tile_id": tile["id"],
        "domain": {
            "name": f"Corsica WindNinja overview {tile['id']}",
            "bounds_wgs84": bounds_wgs84(header),
            "shape": [int(header["nrows"]), int(header["ncols"])],
            "cellsize_m": args.cellsize_m,
            "land_fraction": round(land_fraction, 4),
            "missing_dem_fraction_filled": round(missing_dem_fraction, 4),
            "xllcorner": header["xllcorner"],
            "yllcorner": header["yllcorner"],
        },
        "terrain": {
            "source": "Copernicus GLO-30 DSM sampled to Lambert-93",
            "dem_tiles": [str(item.path.relative_to(ROOT)) for item in dem_tiles],
            "min_m": round(float(np.nanmin(dem)), 3),
            "mean_m": round(float(np.nanmean(dem)), 3),
            "max_m": round(float(np.nanmax(dem)), 3),
        },
        "forcing": {
            "source": f"AROME {arome_payload['run_time_utc']} H+{args.lead_hour} 10 m U/V",
            "run_time_utc": arome_payload["run_time_utc"],
            "valid_time_utc": step["valid_time_utc"],
            "lead_hour": args.lead_hour,
            "mean_speed_ms": round(float(np.nanmean(speed_grid)), 3),
            "max_speed_ms": round(float(np.nanmax(speed_grid)), 3),
            "mean_direction_from_deg": round(direction_mean_deg(direction_grid), 1),
        },
        "windninja": {
            "config": config_path.name,
            "output_dir": "windninja_corsica_output",
            "mesh_resolution_m": args.mesh_resolution_m,
            "output_height_m": args.output_height_m,
            "docker_image": "usdaarsnwrc/katana:latest",
            "run_command": (
                "python3 scripts/run_windninja_cases_docker.py "
                f"--case-dir {case_dir.relative_to(ROOT)} "
                "--config-name windninja_corsica_tile.cfg "
                "--output-dir-name windninja_corsica_output"
            ),
        },
    }
    (case_dir / "case_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (case_dir / "README.md").write_text(
        f"# {metadata['domain']['name']}\n\nRun:\n\n```bash\n{metadata['windninja']['run_command']}\n```\n",
        encoding="utf-8",
    )
    return metadata


def write_plan(cases: list[dict[str, Any]], skipped_count: int, args: argparse.Namespace, arome_payload: dict[str, Any], dem: dict[str, Any]) -> Path:
    payload = {
        "format": "corsewind.windninja.corsica.tile_plan.v1",
        "generated_at_utc": utc_now(),
        "objective": "Corsica-wide terrain-aware AROME correction under a 30 min operational budget.",
        "strategy": {
            "overview_resolution_m": args.cellsize_m,
            "tile_size_km": args.tile_size_km,
            "overlap_km": args.overlap_km,
            "output_height_m": args.output_height_m,
            "lead_hour": args.lead_hour,
            "runtime_policy": "Run all prepared tiles in parallel batches for production; pilot one tile before enabling full automatic runs.",
        },
        "arome": {
            "run_time_utc": arome_payload["run_time_utc"],
            "bbox_wgs84": arome_payload["bbox_wgs84"],
        },
        "bounds_wgs84": dem["windninja_bounds_wgs84"],
        "dem": dem,
        "summary": {
            "prepared_tile_count": len(cases),
            "skipped_tile_count": skipped_count,
            "estimated_cells": int(sum(case["domain"]["shape"][0] * case["domain"]["shape"][1] for case in cases)),
        },
        "tiles": [
            {
                "tile_id": case["tile_id"],
                "case_dir": case["case_dir"],
                "bounds_wgs84": case["domain"]["bounds_wgs84"],
                "shape": case["domain"]["shape"],
                "cellsize_m": case["domain"]["cellsize_m"],
                "land_fraction": case["domain"]["land_fraction"],
                "run_command": case["windninja"]["run_command"],
            }
            for case in cases
        ],
    }
    plan_path = args.plan_output if args.plan_output.is_absolute() else ROOT / args.plan_output
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return plan_path


def write_report(cases: list[dict[str, Any]], skipped_count: int, args: argparse.Namespace, dem: dict[str, Any]) -> Path:
    sample_commands = "\n".join(f"- `{case['windninja']['run_command']}`" for case in cases[:3])
    batch_status_path = args.batch_status_output or (BATCH_STATUS_1M_PATH if math.isclose(args.output_height_m, 1.0) else BATCH_STATUS_PATH)
    batch_status_path = batch_status_path if batch_status_path.is_absolute() else ROOT / batch_status_path
    batch_status = json.loads(batch_status_path.read_text(encoding="utf-8")) if batch_status_path.exists() else None
    recommended_parallel = 2
    recommended_runtime_min = 60 if args.cellsize_m <= 50 else 30
    batch_text = "No batch run recorded yet."
    if batch_status:
        batch_text = (
            f"- Submitted: `{batch_status.get('submitted')}`\n"
            f"- Completed: `{batch_status.get('completed')}`\n"
            f"- Passed: `{batch_status.get('passed')}`\n"
            f"- Failed: `{batch_status.get('failed')}`\n"
            f"- Elapsed: `{batch_status.get('elapsed_s')} s`\n"
            f"- Parallelism: `{batch_status.get('parallel')}`\n"
        )
    text = f"""# Corsica WindNinja Automatic Process

## Purpose

After each AROME update, this tier prepares a Corsica-wide terrain-aware overview. It is not the windsurf micro layer. It is the broad correction layer used to improve the large-scale AROME context before Ajaccio/spot downscales.

## Operational Choice

- Whole island resolution: `{args.cellsize_m:g} m`
- WindNinja mesh resolution: `{args.mesh_resolution_m:g} m`
- Tile size: `{args.tile_size_km:g} km`
- Tile overlap: `{args.overlap_km:g} km`
- Output height: `{args.output_height_m:g} m`
- Prepared tiles: `{len(cases)}`
- Skipped low-land/offshore tiles: `{skipped_count}`
- WindNinja bounds: `{dem.get('windninja_bounds_wgs84')}`

This is the level intended to stay near the 30 min operational budget on the current Docker memory envelope with controlled parallelism. The 20/25 m level remains for selected spots only.

## DEM Coverage Audit

- Source: `{dem['source']}`
- Loaded DEM tiles: `{dem['loaded_tile_count']}`
- AROME required tiles: `{', '.join(dem['arome_required_tiles'])}`
- AROME missing tiles: `{', '.join(dem['arome_missing_tiles']) or 'none'}`
- WindNinja required tiles: `{', '.join(dem['windninja_required_tiles'])}`
- WindNinja missing tiles: `{', '.join(dem['windninja_missing_tiles']) or 'none'}`
- Prepared tile missing DEM fraction min/mean/max: `{dem['prepared_tiles_missing_dem_fraction']['min']}` / `{dem['prepared_tiles_missing_dem_fraction']['mean']}` / `{dem['prepared_tiles_missing_dem_fraction']['max']}`
- Prepared tile missing DEM fraction weighted by cell count: `{dem['prepared_tiles_missing_dem_fraction']['weighted_by_cell_count']}`

## Automatic Update Slot

Recommended sequence after `scripts/build_arome_corsica_wind_layer.py`:

```bash
python3 scripts/prepare_corsica_windninja_tiles.py --cellsize-m {args.cellsize_m:g} --mesh-resolution-m {args.mesh_resolution_m:g} --tile-size-km {args.tile_size_km:g} --overlap-km {args.overlap_km:g} --output-height-m {args.output_height_m:g} --lead-hour {args.lead_hour} --min-land-fraction {args.min_land_fraction:g} --plan-output {args.plan_output} --report-output {args.report_output} --batch-status-output {args.batch_status_output or batch_status_path.relative_to(ROOT)}
python3 scripts/run_corsica_windninja_batch.py --plan {args.plan_output} --status-output {args.batch_status_output or batch_status_path.relative_to(ROOT)} --max-runtime-min {recommended_runtime_min} --parallel {recommended_parallel}
```

Scheduler-friendly full update:

```bash
python3 scripts/update_wind2d_from_meteofrance.py --with-corsica-windninja
```

The scheduled update also builds the Wind2D color and data raster tiles so the layer can be selected directly in the 2D map.

## Last Batch Status

{batch_text}

## Pilot Commands

{sample_commands or "- No tile prepared."}

## Notes

- DEM source is Copernicus GLO-30, sampled to Lambert-93.
- This tier improves terrain exposure, cap effects, valley/channel hints, and broad wind corridors.
- It should not replace 20/25 m local tiles or 5 m windsurf pockets.
- `parallel=4` caused Docker exit `137` on the 50 m / 20 km tiles with the current 8 GB VM memory; use `parallel=2` unless the Docker memory budget is increased and revalidated.
"""
    report_path = args.report_output if args.report_output.is_absolute() else ROOT / args.report_output
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cellsize-m", type=float, default=250.0)
    parser.add_argument("--mesh-resolution-m", type=float, default=250.0)
    parser.add_argument("--tile-size-km", type=float, default=30.0)
    parser.add_argument("--overlap-km", type=float, default=3.0)
    parser.add_argument("--output-height-m", type=float, default=10.0)
    parser.add_argument("--lead-hour", type=int, default=0)
    parser.add_argument("--min-land-fraction", type=float, default=0.03)
    parser.add_argument("--land-elevation-threshold-m", type=float, default=2.0)
    parser.add_argument("--max-tiles", type=int, default=0, help="Limit prepared tiles for pilots; 0 means all.")
    parser.add_argument("--plan-output", type=Path, default=PLAN_PATH)
    parser.add_argument("--report-output", type=Path, default=REPORT_PATH)
    parser.add_argument("--batch-status-output", type=Path, default=None, help="Optional batch status JSON to summarize in the report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dem_tiles = load_dem_tiles()
    arome_payload, step = load_forecast_step(args.lead_hour)
    tile_headers = build_tile_headers(args, arome_payload["bbox_wgs84"])
    cases: list[dict[str, Any]] = []
    skipped_count = 0
    for tile in tile_headers:
        case = prepare_tile(tile, dem_tiles, arome_payload, step, args)
        if case is None:
            skipped_count += 1
            continue
        cases.append(case)
        if args.max_tiles and len(cases) >= args.max_tiles:
            break

    windninja_bounds = union_bounds_wgs84([case["domain"]["bounds_wgs84"] for case in cases])
    dem = dem_audit(dem_tiles, arome_payload["bbox_wgs84"], windninja_bounds, cases)
    plan_path = write_plan(cases, skipped_count, args, arome_payload, dem)
    report_path = write_report(cases, skipped_count, args, dem)
    print(f"prepared {len(cases)} Corsica WindNinja tile case(s); skipped {skipped_count}")
    print(f"wrote {plan_path.relative_to(ROOT)}")
    print(f"wrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
