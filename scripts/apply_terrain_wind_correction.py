#!/usr/bin/env python3
"""Apply a first terrain correction to an AROME wind-speed raster.

This is a Phase 1 prototype: it resamples a coarse AROME WCS GeoTIFF onto the
30 m terrain grid and applies an uncalibrated west-flow exposure factor.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

from sample_arome_tiff_at_stations import read_float64_tiff


STATIONS = [
    {"station_id": "20092001", "name": "CONCA", "lat": 41.7357, "lon": 9.3402},
    {"station_id": "20247001", "name": "LA CHIAPPA", "lat": 41.5962, "lon": 9.3628},
    {"station_id": "20114002", "name": "FIGARI", "lat": 41.5022, "lon": 9.0978},
    {"station_id": "20272004", "name": "SARTENE", "lat": 41.6210, "lon": 8.9830},
    {"station_id": "20041001", "name": "CAP PERTUSATO", "lat": 41.3672, "lon": 9.1840},
]


def lambert93(lon: float, lat: float) -> tuple[float, float]:
    constants = lambert93_constants()
    lat0 = constants["lat0"]
    lon0 = constants["lon0"]
    x0 = constants["x0"]
    y0 = constants["y0"]
    a = constants["a"]
    n = constants["n"]
    f = constants["f"]
    r0 = constants["r0"]

    phi = math.radians(lat)
    lam = math.radians(lon)
    r = a * f * _t(phi, constants["e"]) ** n
    theta = n * (lam - lon0)
    return x0 + r * math.sin(theta), y0 + r0 - r * math.cos(theta)


def _t(phi: float | np.ndarray, e: float) -> float | np.ndarray:
    return np.tan(np.pi / 4 - phi / 2) / (
        ((1 - e * np.sin(phi)) / (1 + e * np.sin(phi))) ** (e / 2)
    )


def lambert93_constants() -> dict[str, float]:
    lat0 = math.radians(46.5)
    lat1 = math.radians(44.0)
    lat2 = math.radians(49.0)
    lon0 = math.radians(3.0)
    x0 = 700_000.0
    y0 = 6_600_000.0
    a = 6_378_137.0
    e = math.sqrt(0.00669438002290)

    def m(phi: float) -> float:
        return math.cos(phi) / math.sqrt(1 - e * e * math.sin(phi) ** 2)

    n = (math.log(m(lat1)) - math.log(m(lat2))) / (math.log(_t(lat1, e)) - math.log(_t(lat2, e)))
    f = m(lat1) / (n * _t(lat1, e) ** n)
    r0 = a * f * _t(lat0, e) ** n
    return {
        "lat0": lat0,
        "lon0": lon0,
        "x0": x0,
        "y0": y0,
        "a": a,
        "e": e,
        "n": n,
        "f": f,
        "r0": r0,
    }


def inverse_lambert93(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    constants = lambert93_constants()
    dx = x - constants["x0"]
    dy = constants["r0"] - (y - constants["y0"])
    radius = np.hypot(dx, dy)
    gamma = np.arctan2(dx, dy)
    lon = constants["lon0"] + gamma / constants["n"]
    t_value = (radius / (constants["a"] * constants["f"])) ** (1 / constants["n"])
    phi = np.pi / 2 - 2 * np.arctan(t_value)
    e = constants["e"]
    for _ in range(6):
        phi = np.pi / 2 - 2 * np.arctan(
            t_value * ((1 - e * np.sin(phi)) / (1 + e * np.sin(phi))) ** (e / 2)
        )
    return np.degrees(lon), np.degrees(phi)


def projected_bbox(wgs84_bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    min_lon, min_lat, max_lon, max_lat = wgs84_bbox
    points = [
        lambert93(lon, lat)
        for lon in (min_lon, max_lon)
        for lat in (min_lat, max_lat)
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def terrain_lon_lat_axes(terrain: np.lib.npyio.NpzFile, bbox: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    if "lon" in terrain.files and "lat" in terrain.files:
        return terrain["lon"], terrain["lat"]

    if "x_l93" not in terrain.files or "y_l93" not in terrain.files:
        raise ValueError("Terrain bundle must contain lon/lat or x_l93/y_l93 axes")

    raise ValueError("Projected terrain axes require projected resampling, not 1D lon/lat axes")


def bilinear_resample_to_grid(
    source: np.ndarray,
    target_lons: np.ndarray,
    target_lats: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    min_lon, min_lat, max_lon, max_lat = bbox
    src_h, src_w = source.shape

    x = (target_lons - min_lon) / (max_lon - min_lon) * (src_w - 1)
    y = (max_lat - target_lats) / (max_lat - min_lat) * (src_h - 1)
    x = np.clip(x, 0, src_w - 1)
    y = np.clip(y, 0, src_h - 1)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    wx = x - x0
    wy = y - y0

    top = (1.0 - wx)[None, :] * source[y0[:, None], x0[None, :]] + wx[None, :] * source[y0[:, None], x1[None, :]]
    bottom = (1.0 - wx)[None, :] * source[y1[:, None], x0[None, :]] + wx[None, :] * source[y1[:, None], x1[None, :]]
    return ((1.0 - wy)[:, None] * top + wy[:, None] * bottom).astype(np.float32)


def bilinear_resample_projected_to_grid(
    source: np.ndarray,
    target_xs: np.ndarray,
    target_ys: np.ndarray,
    bbox: tuple[float, float, float, float],
    chunk_rows: int = 512,
) -> np.ndarray:
    min_lon, min_lat, max_lon, max_lat = bbox
    src_h, src_w = source.shape
    output = np.empty((len(target_ys), len(target_xs)), dtype=np.float32)
    x_grid = target_xs[None, :]

    for start in range(0, len(target_ys), chunk_rows):
        end = min(start + chunk_rows, len(target_ys))
        y_grid = target_ys[start:end, None]
        lon, lat = inverse_lambert93(x_grid, y_grid)
        x = (lon - min_lon) / (max_lon - min_lon) * (src_w - 1)
        y = (max_lat - lat) / (max_lat - min_lat) * (src_h - 1)
        x = np.clip(x, 0, src_w - 1)
        y = np.clip(y, 0, src_h - 1)

        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, src_w - 1)
        y1 = np.clip(y0 + 1, 0, src_h - 1)
        wx = x - x0
        wy = y - y0

        top = (1.0 - wx) * source[y0, x0] + wx * source[y0, x1]
        bottom = (1.0 - wx) * source[y1, x0] + wx * source[y1, x1]
        output[start:end, :] = ((1.0 - wy) * top + wy * bottom).astype(np.float32)

    return output


def sample_grid(array: np.ndarray, xs: np.ndarray, ys: np.ndarray, x: float, y: float) -> float:
    col = int(np.argmin(np.abs(xs - x)))
    row = int(np.argmin(np.abs(ys - y)))
    return float(array[row, col])


def nearest_cell(xs: np.ndarray, ys: np.ndarray, x: float, y: float) -> tuple[int, int]:
    col = int(np.argmin(np.abs(xs - x)))
    row = int(np.argmin(np.abs(ys - y)))
    return row, col


def nearest_finite_cell(
    elevation: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    x: float,
    y: float,
    max_radius_m: float = 250.0,
) -> tuple[int, int, float]:
    row, col = nearest_cell(xs, ys, x, y)
    if np.isfinite(elevation[row, col]):
        return row, col, 0.0

    if len(xs) < 2:
        return row, col, float("nan")

    cell_m = float(np.median(np.diff(xs)))
    radius_cells = int(math.ceil(max_radius_m / abs(cell_m)))
    r0 = max(0, row - radius_cells)
    r1 = min(elevation.shape[0], row + radius_cells + 1)
    c0 = max(0, col - radius_cells)
    c1 = min(elevation.shape[1], col + radius_cells + 1)
    window = elevation[r0:r1, c0:c1]
    finite_rows, finite_cols = np.where(np.isfinite(window))
    if len(finite_rows) == 0:
        return row, col, float("nan")

    candidate_rows = finite_rows + r0
    candidate_cols = finite_cols + c0
    dx = xs[candidate_cols] - x
    dy = ys[candidate_rows] - y
    distances = np.hypot(dx, dy)
    best = int(np.argmin(distances))
    return int(candidate_rows[best]), int(candidate_cols[best]), float(distances[best])


def colorize_speed(values: np.ndarray, max_speed: float = 12.0) -> Image.Image:
    scaled = np.clip(values / max_speed, 0.0, 1.0)
    r = np.clip((scaled - 0.45) / 0.55, 0, 1)
    g = np.clip(1.0 - np.abs(scaled - 0.45) / 0.45, 0, 1)
    b = np.clip((0.55 - scaled) / 0.55, 0, 1)
    rgb = np.stack([r, g, b], axis=2)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def write_station_samples(
    path: Path,
    sample_xs: np.ndarray,
    sample_ys: np.ndarray,
    raw_speed: np.ndarray,
    corrected_speed: np.ndarray,
    factor: np.ndarray,
    terrain: np.lib.npyio.NpzFile,
) -> None:
    fields = [
        "station_id",
        "name",
        "lat",
        "lon",
        "arome_raw_speed_ms",
        "terrain_factor",
        "corrected_speed_ms",
        "terrain_sample_x",
        "terrain_sample_y",
        "terrain_snap_distance_m",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        terrain_is_projected = "x_l93" in terrain.files and "y_l93" in terrain.files
        for station in STATIONS:
            if terrain_is_projected:
                sample_x, sample_y = lambert93(station["lon"], station["lat"])
            else:
                sample_x, sample_y = station["lon"], station["lat"]
            raw_sample = sample_grid(raw_speed, sample_xs, sample_ys, sample_x, sample_y)
            if terrain_is_projected and "elevation_m" in terrain.files:
                terrain_row, terrain_col, snap_distance = nearest_finite_cell(
                    terrain["elevation_m"], sample_xs, sample_ys, sample_x, sample_y
                )
                factor_sample = float(factor[terrain_row, terrain_col])
                terrain_sample_x = float(sample_xs[terrain_col])
                terrain_sample_y = float(sample_ys[terrain_row])
            else:
                factor_sample = sample_grid(factor, sample_xs, sample_ys, sample_x, sample_y)
                terrain_sample_x = float(sample_x)
                terrain_sample_y = float(sample_y)
                snap_distance = 0.0
            writer.writerow(
                {
                    **station,
                    "arome_raw_speed_ms": f"{raw_sample:.3f}",
                    "terrain_factor": f"{factor_sample:.3f}",
                    "corrected_speed_ms": f"{raw_sample * factor_sample:.3f}",
                    "terrain_sample_x": f"{terrain_sample_x:.2f}",
                    "terrain_sample_y": f"{terrain_sample_y:.2f}",
                    "terrain_snap_distance_m": f"{snap_distance:.2f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arome-tiff", type=Path, required=True)
    parser.add_argument("--terrain", type=Path, default=Path("data/processed/terrain/bonifacio_figari_30m_terrain_features.npz"))
    parser.add_argument("--bbox", nargs=4, type=float, default=(8.90, 41.30, 9.42, 41.76))
    parser.add_argument("--factor", default="west_exposure_factor")
    parser.add_argument("--factor-strength", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("data/processed/terrain/bonifacio_figari_corrected_west_wind_30m.npz"))
    parser.add_argument("--preview-output", type=Path, default=Path("data/processed/terrain/bonifacio_figari_corrected_west_wind_30m.png"))
    parser.add_argument("--station-output", type=Path, default=Path("data/processed/terrain/bonifacio_figari_corrected_west_wind_station_samples.csv"))
    args = parser.parse_args()

    terrain = np.load(args.terrain)
    projected_terrain = "x_l93" in terrain.files and "y_l93" in terrain.files
    sample_xs = terrain["x_l93"] if projected_terrain else terrain["lon"]
    sample_ys = terrain["y_l93"] if projected_terrain else terrain["lat"]
    factor = terrain[args.factor].astype(np.float32)
    if args.factor_strength != 1.0:
        factor = 1.0 + (factor - 1.0) * args.factor_strength

    arome = read_float64_tiff(args.arome_tiff)
    if projected_terrain:
        raw_30m = bilinear_resample_projected_to_grid(arome, terrain["x_l93"], terrain["y_l93"], tuple(args.bbox))
    else:
        lons, lats = terrain_lon_lat_axes(terrain, tuple(args.bbox))
        raw_30m = bilinear_resample_to_grid(arome, lons, lats, tuple(args.bbox))
    corrected_30m = (raw_30m * factor).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        arome_raw_speed_ms=raw_30m,
        terrain_factor=factor,
        corrected_speed_ms=corrected_30m,
    )
    colorize_speed(corrected_30m).save(args.preview_output)
    write_station_samples(args.station_output, sample_xs, sample_ys, raw_30m, corrected_30m, factor, terrain)

    print(f"Wrote {args.output}")
    print(f"Wrote {args.preview_output}")
    print(f"Wrote {args.station_output}")


if __name__ == "__main__":
    main()
