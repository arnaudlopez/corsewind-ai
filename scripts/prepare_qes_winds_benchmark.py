#!/usr/bin/env python3
"""Prepare matched WindNinja and QES-Winds benchmark cases."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from apply_terrain_wind_correction import inverse_lambert93, lambert93
from prepare_corsica_windninja_tiles import (
    ascii_axes,
    bounds_wgs84,
    header_for_bbox,
    load_dem_tiles,
    load_forecast_step,
    sample_dem,
    write_windninja_config,
)
from windninja_grid_utils import (
    bilinear_sample,
    direction_mean_deg,
    meteorological_direction_from_uv,
    write_ascii_grid,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "benchmarks/qes_winds/benchmark_config.json"
DEFAULT_PLAN = ROOT / "data/processed/benchmarks/qes_winds/benchmark_plan.json"
DEFAULT_REPORT = ROOT / "reports/qes_winds_benchmark_plan.md"


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


def write_qes_xml(
    case_dir: Path,
    dem_name: str,
    header: dict[str, float],
    speed_ms: float,
    direction_deg: float,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> Path:
    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    nz_out = int(math.ceil(args.vertical_extent_m / args.vertical_cell_m))
    # QES visualization output has dimensions nx-1, ny-1, nz-2.
    qes_nx = ncols + 1
    qes_ny = nrows + 1
    qes_nz = nz_out + 2
    sensor_x = ncols * args.horizontal_resolution_m / 2.0
    sensor_y = nrows * args.horizontal_resolution_m / 2.0
    timestamp = config.get("valid_time_utc", "2020-01-01T00:00:00Z").replace("Z", "")
    xml = f"""<QESWindsParameters>
  <simulationParameters>
    <DEM>{dem_name}</DEM>
    <halo_x>0.0</halo_x>
    <halo_y>0.0</halo_y>
    <domain>{qes_nx} {qes_ny} {qes_nz}</domain>
    <cellSize>{args.horizontal_resolution_m:.3f} {args.horizontal_resolution_m:.3f} {args.vertical_cell_m:.3f}</cellSize>
    <verticalStretching>0</verticalStretching>
    <totalTimeIncrements>1</totalTimeIncrements>
    <maxIterations>{args.max_iterations}</maxIterations>
    <tolerance>{args.tolerance}</tolerance>
    <meshTypeFlag>1</meshTypeFlag>
    <domainRotation>0</domainRotation>
    <originFlag>0</originFlag>
    <DEMDistancex>0.0</DEMDistancex>
    <DEMDistancey>0.0</DEMDistancey>
    <UTMx>{header["xllcorner"]:.3f}</UTMx>
    <UTMy>{header["yllcorner"]:.3f}</UTMy>
    <UTMZone>32</UTMZone>
    <UTMZoneLetter>17</UTMZoneLetter>
    <readCoefficientsFlag>0</readCoefficientsFlag>
  </simulationParameters>
  <metParams>
    <z0_domain_flag>0</z0_domain_flag>
    <sensor>
      <site_coord_flag>1</site_coord_flag>
      <site_xcoord>{sensor_x:.3f}</site_xcoord>
      <site_ycoord>{sensor_y:.3f}</site_ycoord>
      <site_UTM_x>0.0</site_UTM_x>
      <site_UTM_y>0.0</site_UTM_y>
      <site_UTM_zone>0</site_UTM_zone>
      <timeSeries>
        <timeStamp>{timestamp}</timeStamp>
        <boundaryLayerFlag>1</boundaryLayerFlag>
        <siteZ0>0.1</siteZ0>
        <reciprocal>0.0</reciprocal>
        <height>10.0</height>
        <speed>{speed_ms:.4f}</speed>
        <direction>{direction_deg:.3f}</direction>
      </timeSeries>
    </sensor>
  </metParams>
  <turbParams>
    <method>0</method>
  </turbParams>
  <fileOptions>
    <outputFlag>1</outputFlag>
    <outputFields>all</outputFields>
    <outputFields>u</outputFields>
    <outputFields>v</outputFields>
    <outputFields>w</outputFields>
    <outputFields>mag</outputFields>
    <outputFields>icell</outputFields>
  </fileOptions>
</QESWindsParameters>
"""
    path = case_dir / "qes_input.xml"
    path.write_text(xml, encoding="utf-8")
    return path


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

    dem = sample_dem(dem_tiles, lons, lats)
    missing_dem_fraction = float(np.mean(~np.isfinite(dem)))
    dem = np.where(np.isfinite(dem), dem, 0.0).astype(np.float32)
    u_grid = bilinear_sample(np.array(step["u_ms"], dtype=np.float32), lons, lats, arome_payload["bbox_wgs84"])
    v_grid = bilinear_sample(np.array(step["v_ms"], dtype=np.float32), lons, lats, arome_payload["bbox_wgs84"])
    speed_grid = np.hypot(u_grid, v_grid).astype(np.float32)
    direction_grid = meteorological_direction_from_uv(u_grid, v_grid)
    mean_speed = float(np.nanmean(speed_grid))
    mean_direction = direction_mean_deg(direction_grid)

    lead_key = f"h{int(args.lead_hour):02d}"
    case_dir = ROOT / "data/processed/benchmarks/qes_winds" / f"{zone['id']}_{lead_key}_{int(args.horizontal_resolution_m)}m"
    windninja_dir = case_dir / "windninja"
    qes_dir = case_dir / "qes"
    windninja_dir.mkdir(parents=True, exist_ok=True)
    qes_dir.mkdir(parents=True, exist_ok=True)

    write_ascii_grid(windninja_dir / "dem_lambert93.asc", dem, header)
    write_ascii_grid(windninja_dir / "arome_speed_grid.asc", speed_grid, header)
    write_ascii_grid(windninja_dir / "arome_dir_grid.asc", direction_grid, header)
    wn_cfg = write_windninja_config(
        windninja_dir,
        args.horizontal_resolution_m,
        args.output_height_m,
        "windninja_corsica_output",
    )

    qes_dem = qes_dir / "dem_lambert93.tif"
    Image.fromarray(dem.astype(np.float32)).save(qes_dem)
    qes_xml = write_qes_xml(
        qes_dir,
        qes_dem.name,
        header,
        mean_speed,
        mean_direction,
        args,
        {"valid_time_utc": step["valid_time_utc"]},
    )
    qes_output_basename = qes_dir / zone["id"]

    metadata = {
        "format": "corsewind.qes_winds_benchmark.case.v1",
        "zone": zone,
        "case_dir": str(case_dir.relative_to(ROOT)),
        "lead_hour": int(args.lead_hour),
        "valid_time_utc": step["valid_time_utc"],
        "domain": {
            "bounds_wgs84": bounds_wgs84(header),
            "shape": [int(header["nrows"]), int(header["ncols"])],
            "cellsize_m": args.horizontal_resolution_m,
            "vertical_cell_m": args.vertical_cell_m,
            "vertical_extent_m": args.vertical_extent_m,
            "output_height_m": args.output_height_m,
            "missing_dem_fraction_filled": round(missing_dem_fraction, 6),
        },
        "forcing": {
            "source": f"AROME {arome_payload['run_time_utc']} H+{args.lead_hour}",
            "mean_speed_ms": round(mean_speed, 4),
            "mean_direction_from_deg": round(mean_direction, 3),
        },
        "windninja": {
            "case_dir": str(windninja_dir.relative_to(ROOT)),
            "config": wn_cfg.name,
            "output_dir": "windninja_corsica_output",
            "command": [
                "python3",
                "scripts/run_windninja_cases_docker.py",
                "--case-dir",
                str(windninja_dir.relative_to(ROOT)),
                "--config-name",
                wn_cfg.name,
                "--output-dir-name",
                "windninja_corsica_output",
            ],
        },
        "qes_winds": {
            "case_dir": str(qes_dir.relative_to(ROOT)),
            "input_xml": str(qes_xml.relative_to(ROOT)),
            "dem_tiff": str(qes_dem.relative_to(ROOT)),
            "output_basename": str(qes_output_basename.relative_to(ROOT)),
            "expected_winds_out": str(qes_output_basename.with_name(qes_output_basename.name + "_windsOut.nc").relative_to(ROOT)),
            "command": [
                "{QES_WINDS_BIN}",
                "-q",
                str(qes_xml.relative_to(ROOT)),
                "-s",
                "2",
                "-w",
                "-o",
                str(qes_output_basename.relative_to(ROOT)),
            ],
        },
    }
    (case_dir / "case_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def write_plan(cases: list[dict[str, Any]], config: dict[str, Any], args: argparse.Namespace) -> None:
    payload = {
        "format": "corsewind.qes_winds_benchmark.plan.v1",
        "generated_at_utc": utc_now(),
        "objective": "Compare QES-Winds GPU against the current WindNinja path on focused coastal Corsica domains.",
        "sources": config.get("qes", {}),
        "settings": {
            "lead_hour": args.lead_hour,
            "horizontal_resolution_m": args.horizontal_resolution_m,
            "vertical_cell_m": args.vertical_cell_m,
            "vertical_extent_m": args.vertical_extent_m,
            "output_height_m": args.output_height_m,
            "max_iterations": args.max_iterations,
            "tolerance": args.tolerance,
        },
        "cases": cases,
    }
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# QES-Winds Benchmark Plan",
        "",
        f"- Cases: `{len(cases)}`",
        f"- Lead hour: `H+{args.lead_hour}`",
        f"- Horizontal resolution: `{args.horizontal_resolution_m:g} m`",
        f"- Output height: `{args.output_height_m:g} m`",
        "",
        "## Cases",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                f"### {case['zone']['label']}",
                "",
                f"- Bounds: `{case['domain']['bounds_wgs84']}`",
                f"- Shape: `{case['domain']['shape']}`",
                f"- AROME mean: `{case['forcing']['mean_speed_ms']} m/s from {case['forcing']['mean_direction_from_deg']} deg`",
                f"- WindNinja command: `{' '.join(case['windninja']['command'])}`",
                f"- QES command: `{' '.join(case['qes_winds']['command'])}`",
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
    parser.add_argument("--vertical-extent-m", type=float, default=None)
    parser.add_argument("--output-height-m", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--tolerance", default=None)
    parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def apply_defaults(args: argparse.Namespace, defaults: dict[str, Any]) -> argparse.Namespace:
    for attr, key in [
        ("lead_hour", "lead_hour"),
        ("horizontal_resolution_m", "horizontal_resolution_m"),
        ("vertical_cell_m", "vertical_cell_m"),
        ("vertical_extent_m", "vertical_extent_m"),
        ("output_height_m", "output_height_m"),
        ("max_iterations", "max_iterations"),
        ("tolerance", "tolerance"),
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
    print(f"prepared {len(cases)} QES/WindNinja benchmark case(s)")
    print(f"wrote {args.plan_output.relative_to(ROOT)}")
    print(f"wrote {args.report_output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
