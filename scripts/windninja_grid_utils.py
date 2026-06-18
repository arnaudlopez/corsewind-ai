"""Shared grid helpers for WindNinja case preparation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

LAMBERT93_PRJ = """PROJCS["RGF93 / Lambert-93",
    GEOGCS["RGF93",
        DATUM["Reseau_Geodesique_Francais_1993",
            SPHEROID["GRS 1980",6378137,298.257222101]],
        PRIMEM["Greenwich",0],
        UNIT["degree",0.0174532925199433]],
    PROJECTION["Lambert_Conformal_Conic_2SP"],
    PARAMETER["standard_parallel_1",44],
    PARAMETER["standard_parallel_2",49],
    PARAMETER["latitude_of_origin",46.5],
    PARAMETER["central_meridian",3],
    PARAMETER["false_easting",700000],
    PARAMETER["false_northing",6600000],
    UNIT["metre",1],
    AUTHORITY["EPSG","2154"]]
"""


def write_ascii_grid(path: Path, values: np.ndarray, header: dict[str, float], fmt: str = "%.3f") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"ncols {int(header['ncols'])}\n")
        handle.write(f"nrows {int(header['nrows'])}\n")
        handle.write(f"xllcorner {header['xllcorner']:.6f}\n")
        handle.write(f"yllcorner {header['yllcorner']:.6f}\n")
        handle.write(f"cellsize {header['cellsize']:.6f}\n")
        handle.write("NODATA_value -9999\n")
        np.savetxt(handle, np.where(np.isfinite(values), values, -9999.0), fmt=fmt)
    path.with_suffix(".prj").write_text(LAMBERT93_PRJ, encoding="utf-8")


def bilinear_sample(
    source: np.ndarray,
    target_lons: np.ndarray,
    target_lats: np.ndarray,
    bbox: list[float],
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

    top = (1.0 - wx) * source[y0, x0] + wx * source[y0, x1]
    bottom = (1.0 - wx) * source[y1, x0] + wx * source[y1, x1]
    return ((1.0 - wy) * top + wy * bottom).astype(np.float32)


def meteorological_direction_from_uv(u_ms: np.ndarray, v_ms: np.ndarray) -> np.ndarray:
    return ((np.degrees(np.arctan2(-u_ms, -v_ms)) + 360.0) % 360.0).astype(np.float32)


def direction_mean_deg(direction_deg: np.ndarray) -> float:
    rad = np.radians(direction_deg[np.isfinite(direction_deg)])
    if rad.size == 0:
        return float("nan")
    mean_sin = float(np.mean(np.sin(rad)))
    mean_cos = float(np.mean(np.cos(rad)))
    return (math.degrees(math.atan2(mean_sin, mean_cos)) + 360.0) % 360.0
