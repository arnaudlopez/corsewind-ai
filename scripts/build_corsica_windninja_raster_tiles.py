#!/usr/bin/env python3
"""Build Leaflet raster tiles from Corsica-wide WindNinja outputs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "data/processed/physics/corsica_windninja_tile_plan.json"
OUTPUT_ROOT = ROOT / "visualizations/wind2d/windninja-corsica-tiles"
MANIFEST = OUTPUT_ROOT / "manifest.json"
REPORT = ROOT / "reports/corsica_windninja_raster_tiles_report.md"
TILE_SIZE = 256
KNOTS_PER_MPS = 1.943844492
SPEED_SCALE_MAX_KT = 12.0
DEFAULT_ZOOMS = tuple(range(8, 13))
DEFAULT_MODES = ("speed", "acceleration", "devente")
CANDIDATE_BOUNDS_PAD_DEG = 0.04


@dataclass
class WindNinjaTile:
    tile_id: str
    case_dir: Path
    bounds: tuple[float, float, float, float]
    speed_ms: np.ndarray
    parent_speed_ms: np.ndarray
    header: dict[str, float]


def read_ascii_grid(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    header: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for _ in range(6):
            key, value = handle.readline().split()[:2]
            header[key.lower()] = float(value)
        values = np.loadtxt(handle, dtype=np.float32)
    nodata = header.get("nodata_value", -9999.0)
    values[values <= nodata + 1e-3] = np.nan
    return values, header


def deg_to_rad(value: float | np.ndarray) -> float | np.ndarray:
    return value * math.pi / 180.0


def lambert93_vector(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat0 = math.radians(46.5)
    lat1 = math.radians(44.0)
    lat2 = math.radians(49.0)
    lon0 = math.radians(3.0)
    x0 = 700_000.0
    y0 = 6_600_000.0
    a = 6_378_137.0
    e = math.sqrt(0.00669438002290)

    def t(phi: float | np.ndarray) -> float | np.ndarray:
        return np.tan(np.pi / 4 - phi / 2) / (
            ((1 - e * np.sin(phi)) / (1 + e * np.sin(phi))) ** (e / 2)
        )

    def m(phi: float) -> float:
        return math.cos(phi) / math.sqrt(1 - e * e * math.sin(phi) ** 2)

    n = (math.log(m(lat1)) - math.log(m(lat2))) / (math.log(t(lat1)) - math.log(t(lat2)))
    f = m(lat1) / (n * t(lat1) ** n)
    r0 = a * f * t(lat0) ** n
    phi = np.radians(lat)
    lam = np.radians(lon)
    r = a * f * t(phi) ** n
    theta = n * (lam - lon0)
    return x0 + r * np.sin(theta), y0 + r0 - r * np.cos(theta)


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[float, float]:
    lat = max(-85.05112878, min(85.05112878, lat))
    n = 2**z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = deg_to_rad(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_pixel_to_lonlat(z: int, x: int, y: int, px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = 2**z
    xtile = x + px / TILE_SIZE
    ytile = y + py / TILE_SIZE
    lon = xtile / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(math.pi * (1.0 - 2.0 * ytile / n)))
    return lon.astype(np.float32), np.degrees(lat_rad).astype(np.float32)


def tile_ranges(bounds: tuple[float, float, float, float], z: int) -> tuple[range, range]:
    min_lon, min_lat, max_lon, max_lat = bounds
    x0, y1 = lonlat_to_tile(min_lon, min_lat, z)
    x1, y0 = lonlat_to_tile(max_lon, max_lat, z)
    return (
        range(max(0, int(math.floor(min(x0, x1)))), min(2**z - 1, int(math.floor(max(x0, x1)))) + 1),
        range(max(0, int(math.floor(min(y0, y1)))), min(2**z - 1, int(math.floor(max(y0, y1)))) + 1),
    )


def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    t = np.clip((values - edge0) / max(0.0001, edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)


def color_stops(value: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    out = np.zeros((*value.shape, 3), dtype=np.float32)
    for idx, (stop, rgb) in enumerate(stops):
        if idx == 0:
            out[value <= stop] = rgb
            continue
        prev_stop, prev_rgb = stops[idx - 1]
        mask = (value > prev_stop) & (value <= stop)
        t = ((value - prev_stop) / max(0.0001, stop - prev_stop))[..., None]
        out[mask] = (np.array(prev_rgb, dtype=np.float32) + (np.array(rgb, dtype=np.float32) - np.array(prev_rgb, dtype=np.float32)) * t)[mask]
    out[value > stops[-1][0]] = stops[-1][1]
    return out


def mode_color(mode: str, speed_kt: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    if mode == "speed":
        return color_stops(
            np.clip(speed_kt / SPEED_SCALE_MAX_KT, 0, 1),
            [
                (0, (32, 85, 180)),
                (0.2, (37, 137, 210)),
                (0.38, (34, 197, 180)),
                (0.52, (82, 190, 96)),
                (0.68, (245, 202, 66)),
                (0.82, (245, 139, 42)),
                (0.94, (226, 54, 54)),
                (1, (150, 67, 190)),
            ],
        )
    if mode == "acceleration":
        return color_stops(
            np.clip((ratio - 1.0) / 0.34, 0, 1),
            [(0, (18, 54, 96)), (0.28, (34, 197, 180)), (0.62, (245, 202, 66)), (1, (245, 139, 42))],
        )
    return color_stops(
        np.clip((1.0 - ratio) / 0.45, 0, 1),
        [(0, (24, 91, 84)), (0.35, (37, 137, 210)), (0.72, (148, 163, 184)), (1, (248, 250, 252))],
    )


def bilinear_projected(grid: np.ndarray, header: dict[str, float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rows, cols = grid.shape
    cell = float(header["cellsize"])
    xll = float(header["xllcorner"])
    yll = float(header["yllcorner"])
    y_top = yll + rows * cell
    col = (x - (xll + cell * 0.5)) / cell
    row = ((y_top - cell * 0.5) - y) / cell
    outside = (row < 0) | (row > rows - 1) | (col < 0) | (col > cols - 1)
    row = np.clip(row, 0, rows - 1)
    col = np.clip(col, 0, cols - 1)
    r0 = np.floor(row).astype(np.int32)
    c0 = np.floor(col).astype(np.int32)
    r1 = np.clip(r0 + 1, 0, rows - 1)
    c1 = np.clip(c0 + 1, 0, cols - 1)
    tr = row - r0
    tc = col - c0
    values = (
        grid[r0, c0] * (1 - tr) * (1 - tc)
        + grid[r1, c0] * tr * (1 - tc)
        + grid[r0, c1] * (1 - tr) * tc
        + grid[r1, c1] * tr * tc
    )
    values[outside] = np.nan
    return values.astype(np.float32)


def projected_edge_weight(header: dict[str, float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rows = int(header["nrows"])
    cols = int(header["ncols"])
    cell = float(header["cellsize"])
    xll = float(header["xllcorner"])
    yll = float(header["yllcorner"])
    x_norm = (x - xll) / max(cell, cols * cell)
    y_norm = (y - yll) / max(cell, rows * cell)
    edge = np.minimum.reduce([x_norm, y_norm, 1 - x_norm, 1 - y_norm])
    return smoothstep(0.005, 0.08, edge)


def fit_grid_to_shape(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    out = np.full(shape, np.nan, dtype=np.float32)
    copy_rows = min(rows, grid.shape[0])
    copy_cols = min(cols, grid.shape[1])
    out[:copy_rows, :copy_cols] = grid[:copy_rows, :copy_cols]
    return out


def load_tiles(plan: dict[str, Any]) -> list[WindNinjaTile]:
    tiles: list[WindNinjaTile] = []
    for meta in plan["tiles"]:
        case_dir = ROOT / meta["case_dir"]
        output_dir = case_dir / "windninja_corsica_output"
        speed_files = sorted(output_dir.glob("*_vel.asc"))
        if not speed_files:
            continue
        speed, header = read_ascii_grid(speed_files[0])
        parent, _ = read_ascii_grid(case_dir / "arome_speed_grid.asc")
        if parent.shape != speed.shape:
            parent = fit_grid_to_shape(parent, speed.shape)
        tiles.append(
            WindNinjaTile(
                tile_id=meta["tile_id"],
                case_dir=case_dir,
                bounds=tuple(float(value) for value in meta["bounds_wgs84"]),
                speed_ms=speed,
                parent_speed_ms=parent,
                header=header,
            )
        )
    return tiles


def union_bounds(tiles: list[WindNinjaTile]) -> tuple[float, float, float, float]:
    return (
        min(tile.bounds[0] for tile in tiles),
        min(tile.bounds[1] for tile in tiles),
        max(tile.bounds[2] for tile in tiles),
        max(tile.bounds[3] for tile in tiles),
    )


def candidate_tiles(tiles: list[WindNinjaTile], bounds: tuple[float, float, float, float]) -> list[WindNinjaTile]:
    min_lon, min_lat, max_lon, max_lat = bounds
    min_lon -= CANDIDATE_BOUNDS_PAD_DEG
    min_lat -= CANDIDATE_BOUNDS_PAD_DEG
    max_lon += CANDIDATE_BOUNDS_PAD_DEG
    max_lat += CANDIDATE_BOUNDS_PAD_DEG
    return [
        tile
        for tile in tiles
        if not (
            tile.bounds[2] < min_lon
            or tile.bounds[0] > max_lon
            or tile.bounds[3] < min_lat
            or tile.bounds[1] > max_lat
        )
    ]


def compose_tile(tiles: list[WindNinjaTile], z: int, x_tile: int, y_tile: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.arange(TILE_SIZE, dtype=np.float32) + 0.5
    px, py = np.meshgrid(grid, grid)
    lons, lats = tile_pixel_to_lonlat(z, x_tile, y_tile, px, py)
    x_l93, y_l93 = lambert93_vector(lons, lats)

    speed_sum = np.zeros(lons.shape, dtype=np.float32)
    parent_sum = np.zeros(lons.shape, dtype=np.float32)
    weight_sum = np.zeros(lons.shape, dtype=np.float32)
    for tile in tiles:
        speed = bilinear_projected(tile.speed_ms, tile.header, x_l93, y_l93)
        parent = bilinear_projected(tile.parent_speed_ms, tile.header, x_l93, y_l93)
        weight = projected_edge_weight(tile.header, x_l93, y_l93)
        weight[~np.isfinite(speed)] = 0
        speed_sum += np.nan_to_num(speed, nan=0.0) * weight
        parent_sum += np.nan_to_num(parent, nan=0.0) * weight
        weight_sum += weight

    with np.errstate(divide="ignore", invalid="ignore"):
        speed_ms = speed_sum / weight_sum
        parent_ms = parent_sum / weight_sum
        ratio = speed_ms / parent_ms
    speed_ms[weight_sum <= 0] = np.nan
    ratio[(weight_sum <= 0) | ~np.isfinite(ratio)] = np.nan
    coverage = smoothstep(0.18, 0.82, weight_sum)
    return speed_ms, ratio, coverage


def render_tile(tiles: list[WindNinjaTile], mode: str, z: int, x_tile: int, y_tile: int) -> Image.Image:
    speed_ms, ratio, coverage = compose_tile(tiles, z, x_tile, y_tile)
    speed_kt = speed_ms * KNOTS_PER_MPS
    rgb = mode_color(mode, speed_kt, np.nan_to_num(ratio, nan=1.0))
    if mode == "speed":
        alpha = coverage * (0.34 + smoothstep(0.08, 0.42, np.abs(np.nan_to_num(ratio, nan=1.0) - 1.0)) * 0.28)
    elif mode == "acceleration":
        alpha = coverage * smoothstep(0.02, 0.34, np.nan_to_num(ratio, nan=1.0) - 1.0) * 0.72
    else:
        alpha = coverage * smoothstep(0.02, 0.34, 1.0 - np.nan_to_num(ratio, nan=1.0)) * 0.72
    alpha[~np.isfinite(speed_ms)] = 0
    rgba = np.dstack([rgb, np.clip(alpha * 255, 0, 190)]).astype(np.uint8)
    rgba[alpha < 0.025, 3] = 0
    return Image.fromarray(rgba, mode="RGBA")


def render_data_tile(tiles: list[WindNinjaTile], z: int, x_tile: int, y_tile: int) -> Image.Image:
    speed_ms, ratio, coverage = compose_tile(tiles, z, x_tile, y_tile)
    speed_kt = speed_ms * KNOTS_PER_MPS
    valid = np.isfinite(speed_kt) & np.isfinite(ratio) & (coverage > 0.025)
    speed_u16 = np.clip(np.nan_to_num(speed_kt, nan=0.0) * 100.0, 0, 65535).astype(np.uint16)
    ratio_u8 = np.zeros(speed_kt.shape, dtype=np.uint8)
    ratio_norm = np.clip((np.nan_to_num(ratio, nan=1.0) - 0.5) / 1.5, 0, 1)
    ratio_u8[valid] = np.clip(1 + np.round(ratio_norm[valid] * 254), 1, 255).astype(np.uint8)
    alpha = np.zeros(speed_kt.shape, dtype=np.uint8)
    alpha[valid] = np.clip(np.round(coverage[valid] * 255), 8, 255).astype(np.uint8)
    rgba = np.dstack(
        [
            (speed_u16 >> 8).astype(np.uint8),
            (speed_u16 & 255).astype(np.uint8),
            ratio_u8,
            alpha,
        ]
    )
    return Image.fromarray(rgba, mode="RGBA")


def tile_bounds_wgs84(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    lon_a, lat_a = tile_pixel_to_lonlat(z, x, y, np.array([[0]], dtype=np.float32), np.array([[0]], dtype=np.float32))
    lon_b, lat_b = tile_pixel_to_lonlat(z, x, y, np.array([[TILE_SIZE]], dtype=np.float32), np.array([[TILE_SIZE]], dtype=np.float32))
    return (float(min(lon_a[0, 0], lon_b[0, 0])), float(min(lat_a[0, 0], lat_b[0, 0])), float(max(lon_a[0, 0], lon_b[0, 0])), float(max(lat_a[0, 0], lat_b[0, 0])))


def build(
    modes: tuple[str, ...],
    zooms: tuple[int, ...],
    plan_path: Path,
    output_root: Path,
    report_path: Path,
    url_template: str,
    encoding: str,
    append: bool,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tiles = load_tiles(plan)
    if not tiles:
        raise RuntimeError("No solved Corsica WindNinja tiles found")
    bounds = union_bounds(tiles)
    step_key = f"h{int(plan['strategy']['lead_hour']):02d}"
    manifest_path = output_root / "manifest.json"
    previous_manifest = None
    if append and manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if output_root.exists() and not append:
        shutil.rmtree(output_root)
    elif output_root.exists() and append:
        shutil.rmtree(output_root / step_key, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)
    tile_count = 0
    if encoding == "data":
        for z in zooms:
            x_range, y_range = tile_ranges(bounds, z)
            for x_tile in x_range:
                for y_tile in y_range:
                    candidates = candidate_tiles(tiles, tile_bounds_wgs84(z, x_tile, y_tile))
                    if not candidates:
                        continue
                    image = render_data_tile(candidates, z, x_tile, y_tile)
                    if not np.any(np.array(image)[..., 3] > 0):
                        continue
                    path = output_root / step_key / "data" / str(z) / str(x_tile) / f"{y_tile}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(path, compress_level=1)
                    tile_count += 1
    else:
        for mode in modes:
            for z in zooms:
                x_range, y_range = tile_ranges(bounds, z)
                for x_tile in x_range:
                    for y_tile in y_range:
                        candidates = candidate_tiles(tiles, tile_bounds_wgs84(z, x_tile, y_tile))
                        if not candidates:
                            continue
                        image = render_tile(candidates, mode, z, x_tile, y_tile)
                        if not np.any(np.array(image)[..., 3] > 0):
                            continue
                        path = output_root / step_key / mode / str(z) / str(x_tile) / f"{y_tile}.png"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        image.save(path, compress_level=1)
                        tile_count += 1

    steps = [{"key": step_key, "lead_hour": plan["strategy"]["lead_hour"]}]
    tile_count_by_step = {step_key: tile_count}
    if previous_manifest:
        previous_steps = [
            step
            for step in previous_manifest.get("steps", [])
            if int(step.get("lead_hour", -9999)) != int(plan["strategy"]["lead_hour"])
        ]
        steps = sorted([*previous_steps, *steps], key=lambda item: int(item.get("lead_hour", 0)))
        previous_counts = dict(previous_manifest.get("tileCountByStep") or {})
        previous_counts.pop(step_key, None)
        tile_count_by_step = {**previous_counts, step_key: tile_count}

    manifest = {
        "format": "corsewind.windninja_corsica.data_tiles.v1" if encoding == "data" else "corsewind.windninja_corsica.raster_tiles.v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "bounds_wgs84": list(bounds),
        "tileSize": TILE_SIZE,
        "zooms": list(zooms),
        "modes": list(modes),
        "encoding": encoding,
        "steps": steps,
        "urlTemplate": url_template,
        "source": {
            "tile_plan": str(plan_path.relative_to(ROOT)),
            "resolution_m": plan["strategy"]["overview_resolution_m"],
            "output_height_m": plan["strategy"].get("output_height_m"),
            "tile_count": len(tiles),
        },
        "candidateBoundsPadDeg": CANDIDATE_BOUNDS_PAD_DEG,
        "speedScaleMaxKt": SPEED_SCALE_MAX_KT,
        "dataEncoding": {
            "speed_kt": "uint16_be_rg_div_100",
            "ratio_vs_parent": "uint8_b_linear_0p5_to_2p0_zero_is_nodata",
            "coverage": "alpha_0_to_1",
        } if encoding == "data" else None,
        "tileCount": sum(int(value) for value in tile_count_by_step.values()),
        "tileCountByStep": tile_count_by_step,
        "opacity": 0.72,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(manifest), encoding="utf-8")
    return manifest


def render_report(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Corsica WindNinja Raster Tiles Report",
            "",
            f"Generated: `{manifest['generatedAt']}`",
            f"Tiles: `{manifest['tileCount']}`",
            f"Bounds: `{manifest['bounds_wgs84']}`",
            f"Zooms: `{manifest['zooms']}`",
            f"Modes: `{', '.join(manifest['modes'])}`",
            f"Encoding: `{manifest.get('encoding', 'color')}`",
            f"Source resolution: `{manifest['source']['resolution_m']} m`",
            f"Output height: `{manifest['source'].get('output_height_m', '?')} m AGL`",
            f"Source tile count: `{manifest['source']['tile_count']}`",
            f"Candidate bounds padding: `{manifest.get('candidateBoundsPadDeg', 0)}°`",
            f"Speed color scale max: `{manifest['speedScaleMaxKt']} kt`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES), choices=["speed", "acceleration", "devente"])
    parser.add_argument("--zooms", nargs="+", type=int, default=list(DEFAULT_ZOOMS))
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--report-output", type=Path, default=REPORT)
    parser.add_argument("--url-template", default="./windninja-corsica-tiles/{step}/{mode}/{z}/{x}/{y}.png")
    parser.add_argument("--encoding", choices=["color", "data"], default="color")
    parser.add_argument("--append", action="store_true", help="Append this lead hour to an existing multi-step manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    report_output = args.report_output if args.report_output.is_absolute() else ROOT / args.report_output
    manifest = build(tuple(args.modes), tuple(args.zooms), plan_path, output_root, report_output, args.url_template, args.encoding, args.append)
    print(f"Corsica WindNinja raster tiles: {manifest['tileCount']} tiles")
    print(f"wrote {(output_root / 'manifest.json').relative_to(ROOT)}")
    print(f"wrote {report_output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
