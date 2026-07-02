#!/usr/bin/env python3
"""Pre-bake Leaflet raster tiles for a raw wind model (AROME, AROME-PI, MOLOCH, ICON-2I).

The Wind2D client computes the colour overlay pixel-by-pixel in JavaScript at interaction
time, which is the main source of pan/zoom lag. This script renders the same colour field
into a static tile pyramid (Web Mercator / XYZ, the Google-Maps model) so the client can
serve it through a plain L.tileLayer / GridLayer instead — instant, GPU-composited,
progressive.

The colour ramp and per-pixel alpha intentionally mirror the client (colorArray + the speed
branch of renderFieldColor + drawWindHeatTile), so pre-baked tiles match the legend exactly.

Modes per step:
  - speed        colour WebP tiles (fallback path when the client cannot recolor data tiles)
  - speed_data   u16 knots in RG + validity alpha (lossless PNG) — recolored live client-side
  - gust_data    same encoding for gust speed, when the source step has gust_speed_ms
  - cloud_rain   colour WebP tiles, when the source step has cloud/precipitation grids

Incremental rebuilds: each step's source grids are hashed; steps whose hash and render
parameters match the previously published set are hard-linked from it instead of being
re-rendered. Old tile sets and stale staging directories are pruned after each publish so
the tiles volume cannot grow without bound.

Output layout:
    visualizations/wind2d/tiles/<model>/_sets/<tile-set>/<step>/<mode>/<z>/<x>/<y>.<format>
    visualizations/wind2d/tiles/<model>/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WIND2D = ROOT / "visualizations/wind2d"
TILE_SIZE = 256
KNOTS_PER_MPS = 1.943844492
DEFAULT_SCALE_MAX_KT = 14.0
# The raw models are ~1 km resolution; z10 is enough native detail for Corsica and Leaflet
# overzooms beyond maxNativeZoom for closer views. Higher native zooms multiply rebuild time.
DEFAULT_ZOOMS = tuple(range(8, 11))
SPEED_DATA_MODE = "speed_data"
GUST_DATA_MODE = "gust_data"
SPEED_DATA_QUANTUM_KT = 0.1
MANIFEST_FORMAT = "corsewind.model.raster_tiles.v2"

# Per-model source JSON and the render alpha the client applies to each raw layer
# (see rawModelFieldAt defaults in wind2d.js). Baking this keeps tiles visually identical
# to the JS heat overlay, so the client can show raster tiles at full opacity.
MODELS: dict[str, dict[str, Any]] = {
    "arome": {"json": "arome-corsica-latest.json", "render_alpha": 0.62},
    "aromepi": {"json": "aromepi-corsica-latest.json", "render_alpha": 0.66},
    "moloch": {"json": "moloch-corsica-latest.json", "render_alpha": 0.58},
    "icon2i": {"json": "icon2i-corsica-latest.json", "render_alpha": 0.58},
}

# Grid fields that affect the rendered tiles; hashed per step for incremental rebuilds.
STEP_SOURCE_FIELDS = ("speed_ms", "gust_speed_ms", "cloud_cover_pct", "precipitation_mm")

# Identical to colorArray() stops in wind2d.js.
SPEED_STOPS: list[tuple[float, tuple[int, int, int]]] = [
    (0.0, (32, 85, 180)),
    (0.2, (37, 137, 210)),
    (0.38, (34, 197, 180)),
    (0.52, (82, 190, 96)),
    (0.68, (245, 202, 66)),
    (0.82, (245, 139, 42)),
    (0.94, (226, 54, 54)),
    (1.0, (150, 67, 190)),
]
RAIN_STOPS: list[tuple[float, tuple[int, int, int]]] = [
    (0.0, (125, 211, 252)),
    (0.2, (56, 189, 248)),
    (0.5, (14, 116, 224)),
    (0.8, (79, 70, 229)),
    (1.0, (126, 34, 206)),
]


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[float, float]:
    lat = max(-85.05112878, min(85.05112878, lat))
    n = 2**z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def tile_pixel_to_lonlat(z: int, x: int, y: int, px: np.ndarray, py: np.ndarray, dim: int = TILE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    n = 2**z
    xtile = x + px / dim
    ytile = y + py / dim
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


def bilinear_wgs84(grid: np.ndarray, bbox: tuple[float, float, float, float], lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Sample a regular WGS84 grid (row 0 = north / max lat) at lon/lat. Mirrors rawModelFieldAt."""
    min_lon, min_lat, max_lon, max_lat = bbox
    rows, cols = grid.shape
    row = (max_lat - lat) / (max_lat - min_lat) * (rows - 1)
    col = (lon - min_lon) / (max_lon - min_lon) * (cols - 1)
    outside = (lon < min_lon) | (lon > max_lon) | (lat < min_lat) | (lat > max_lat)
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


def gradient_color(intensity: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    out = np.zeros((*intensity.shape, 3), dtype=np.float32)
    out[intensity <= stops[0][0]] = stops[0][1]
    for idx in range(1, len(stops)):
        stop, rgb = stops[idx]
        prev_stop, prev_rgb = stops[idx - 1]
        mask = (intensity > prev_stop) & (intensity <= stop)
        t = ((intensity - prev_stop) / max(0.0001, stop - prev_stop))[..., None]
        blended = np.array(prev_rgb, dtype=np.float32) + (np.array(rgb, dtype=np.float32) - np.array(prev_rgb, dtype=np.float32)) * t
        out[mask] = blended[mask]
    out[intensity > stops[-1][0]] = stops[-1][1]
    return np.round(out)


def speed_color(intensity: np.ndarray) -> np.ndarray:
    """Vectorised equivalent of colorArray() in wind2d.js."""
    return gradient_color(intensity, SPEED_STOPS)


def render_tile(speed_kt: np.ndarray, scale_max_kt: float, render_alpha: float) -> Image.Image | None:
    intensity = np.clip(np.nan_to_num(speed_kt, nan=0.0) / scale_max_kt, 0, 1)
    rgb = speed_color(intensity)
    # renderFieldColor speed branch: alpha = 84 + intensity^0.68 * 146, capped at 220 by
    # drawWindHeatTile, then multiplied by the model render alpha.
    base_alpha = np.minimum(220.0, 84.0 + np.power(intensity, 0.68) * 146.0)
    alpha = np.clip(base_alpha * render_alpha, 0, 255)
    # Only out-of-domain (NaN) pixels are transparent; calm areas stay faintly tinted like the
    # JS overlay (which never zeroes low-speed pixels for raw models).
    alpha[~np.isfinite(speed_kt)] = 0
    if not np.any(alpha > 1):
        return None
    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def render_speed_data_tile(speed_kt: np.ndarray) -> Image.Image | None:
    valid = np.isfinite(speed_kt)
    if not np.any(valid):
        return None
    speed_q = np.clip(np.rint(np.nan_to_num(speed_kt, nan=0.0) / SPEED_DATA_QUANTUM_KT), 0, 65535).astype(np.uint16)
    rgba = np.zeros((*speed_kt.shape, 4), dtype=np.uint8)
    rgba[..., 0] = speed_q & 0xFF
    rgba[..., 1] = speed_q >> 8
    rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def normalise_cloud_pct(cloud: np.ndarray) -> np.ndarray:
    finite = cloud[np.isfinite(cloud)]
    if finite.size and float(np.nanmax(finite)) <= 1.5:
        return cloud * 100.0
    return cloud


def render_cloud_rain_tile(cloud_pct: np.ndarray | None, precipitation_mm: np.ndarray | None) -> Image.Image | None:
    shape = cloud_pct.shape if cloud_pct is not None else precipitation_mm.shape
    cloud = np.zeros(shape, dtype=np.float32) if cloud_pct is None else normalise_cloud_pct(cloud_pct).astype(np.float32)
    rain = np.zeros(shape, dtype=np.float32) if precipitation_mm is None else precipitation_mm.astype(np.float32)
    valid = np.isfinite(cloud) | np.isfinite(rain)
    cloud = np.nan_to_num(cloud, nan=0.0)
    rain = np.nan_to_num(rain, nan=0.0)

    cloud_intensity = np.clip(cloud / 100.0, 0, 1)
    rain_intensity = np.clip(np.log1p(np.maximum(0.0, rain)) / np.log1p(8.0), 0, 1)

    cloud_alpha = np.clip(np.power(np.clip((cloud_intensity - 0.12) / 0.88, 0, 1), 0.8) * 205.0, 0, 205)
    cloud_rgb = np.empty((*shape, 3), dtype=np.float32)
    cloud_rgb[..., 0] = 176 + cloud_intensity * 66
    cloud_rgb[..., 1] = 185 + cloud_intensity * 58
    cloud_rgb[..., 2] = 195 + cloud_intensity * 50

    rain_alpha = np.clip(np.power(rain_intensity, 0.72) * 215.0, 0, 215)
    rain_rgb = gradient_color(rain_intensity, RAIN_STOPS)

    ca = (cloud_alpha / 255.0)[..., None]
    ra = (rain_alpha / 255.0)[..., None]
    out_a = ra + ca * (1.0 - ra)
    out_rgb = np.divide(
        rain_rgb * ra + cloud_rgb * ca * (1.0 - ra),
        np.maximum(out_a, 0.0001),
    )
    alpha = np.clip(out_a[..., 0] * 255.0, 0, 255)
    alpha[~valid] = 0
    if not np.any(valid):
        return None
    rgba = np.dstack([out_rgb, alpha]).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def normalise_lead_hour(step: dict[str, Any]) -> int | float:
    lead_hour = float(step["lead_hour"])
    return int(lead_hour) if lead_hour.is_integer() else round(lead_hour, 4)


def step_key_for_step(step: dict[str, Any]) -> str:
    if step.get("lead_minutes") is not None:
        lead_minutes = int(round(float(step["lead_minutes"])))
    else:
        lead_minutes = int(round(float(step["lead_hour"]) * 60))
    sign = "-" if lead_minutes < 0 else ""
    absolute_minutes = abs(lead_minutes)
    hours, minutes = divmod(absolute_minutes, 60)
    if minutes == 0:
        return f"{sign}h{hours:02d}"
    return f"{sign}h{hours:02d}m{minutes:02d}"


def step_source_hash(step: dict[str, Any]) -> str:
    """Deterministic hash of the grids that affect this step's rendered tiles."""
    digest = hashlib.sha1()
    for field in STEP_SOURCE_FIELDS:
        value = step.get(field)
        digest.update(field.encode())
        digest.update(b"\x00")
        if value is None:
            digest.update(b"null")
        else:
            array = np.asarray(value, dtype=np.float32)
            digest.update(str(array.shape).encode())
            digest.update(np.nan_to_num(array, nan=-99999.0).tobytes())
        digest.update(b"\x01")
    return digest.hexdigest()


def slug_part(value: Any, fallback: str = "unknown") -> str:
    text = str(value or fallback)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return slug or fallback


def render_params_key(zooms: tuple[int, ...], scale: int, tile_format: str, webp_quality: int, webp_method: int, scale_max_kt: float, render_alpha: float) -> str:
    zoom_label = "-".join(str(zoom) for zoom in zooms)
    quality_label = f"-q{webp_quality}-m{webp_method}" if tile_format == "webp" else ""
    return f"{tile_format}-s{scale}{quality_label}-z{zoom_label}-max{scale_max_kt:g}-a{render_alpha:g}"


def tile_set_key(payload: dict[str, Any], zooms: tuple[int, ...], scale: int, tile_format: str, webp_quality: int, webp_method: int) -> str:
    run_time = payload.get("run_time_utc") or payload.get("runTimeUtc") or payload.get("generated_at_utc")
    zoom_label = "-".join(str(zoom) for zoom in zooms)
    quality_label = f"-q{webp_quality}-m{webp_method}" if tile_format == "webp" else ""
    return f"run-{slug_part(run_time)}-{tile_format}-s{scale}{quality_label}-z{zoom_label}"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def publish_tile_set(staging_root: Path, tile_root: Path) -> None:
    tile_root.parent.mkdir(parents=True, exist_ok=True)
    backup_root = None
    if tile_root.exists():
        backup_root = tile_root.with_name(f".previous-{tile_root.name}-{os.getpid()}")
        if backup_root.exists():
            shutil.rmtree(backup_root)
        tile_root.rename(backup_root)
    try:
        staging_root.rename(tile_root)
    except Exception:
        if backup_root and backup_root.exists() and not tile_root.exists():
            backup_root.rename(tile_root)
        raise
    if backup_root and backup_root.exists():
        shutil.rmtree(backup_root)


def prune_output_root(output_root: Path, keep_tile_set: str) -> int:
    """Delete tile sets other than the published one plus stale staging/backup leftovers.

    Without this every model run leaves its previous ~20-80 MB set behind and the tiles
    volume grows without bound (AROME-PI publishes many runs per day)."""
    removed = 0
    sets_root = output_root / "_sets"
    if sets_root.exists():
        for entry in sets_root.iterdir():
            if entry.name != keep_tile_set:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
    staging_parent = output_root / "_staging"
    if staging_parent.exists():
        for entry in staging_parent.iterdir():
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
        try:
            staging_parent.rmdir()
        except OSError:
            pass
    for entry in output_root.iterdir():
        # Anything else that is a directory is either a `.previous-*` backup or a leftover from
        # the pre-`_sets` layout (step dirs written directly under the model root) — remove it.
        if entry.is_dir() and entry.name != "_sets":
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    return removed


def link_or_copy_tree(source: Path, target: Path) -> None:
    """Mirror a published step directory into staging via hard links (fast, no extra disk)."""
    for dirpath, _dirnames, filenames in os.walk(source):
        rel = Path(dirpath).relative_to(source)
        (target / rel).mkdir(parents=True, exist_ok=True)
        for name in filenames:
            src = Path(dirpath) / name
            dst = target / rel / name
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)


def render_step_task(task: dict[str, Any]) -> dict[str, Any]:
    """Render every tile of one forecast step into the staging directory.

    Top-level (picklable) so it can run in a ProcessPoolExecutor worker: steps are
    independent, which makes per-step parallelism safe and coarse-grained."""
    staging_root = Path(task["staging_root"])
    bbox = tuple(task["bbox"])
    zooms = tuple(task["zooms"])
    out_px = int(task["out_px"])
    scale_max_kt = float(task["scale_max_kt"])
    render_alpha = float(task["render_alpha"])
    ext = task["ext"]
    tile_format = task["tile_format"]
    webp_quality = int(task["webp_quality"])
    webp_method = int(task["webp_method"])
    step_key = task["step_key"]

    speed_ms = np.array(task["speed_ms"], dtype=np.float32)
    gust_ms = np.array(task["gust_speed_ms"], dtype=np.float32) if task.get("gust_speed_ms") is not None else None
    cloud_pct = np.array(task["cloud_cover_pct"], dtype=np.float32) if task.get("cloud_cover_pct") is not None else None
    precipitation_mm = np.array(task["precipitation_mm"], dtype=np.float32) if task.get("precipitation_mm") is not None else None
    has_weather = cloud_pct is not None or precipitation_mm is not None

    grid = np.arange(out_px, dtype=np.float32) + 0.5
    px, py = np.meshgrid(grid, grid)

    def save_color(image: Image.Image, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if tile_format == "webp":
            image.save(path, "WEBP", quality=webp_quality, method=webp_method)
        else:
            image.save(path, compress_level=2)

    counts = {"tile_count": 0, "color_tile_count": 0, "data_tile_count": 0}
    render_seconds = 0.0
    encode_seconds = 0.0
    for z in zooms:
        x_range, y_range = tile_ranges(bbox, z)
        for x_tile in x_range:
            for y_tile in y_range:
                render_started = time.perf_counter()
                lons, lats = tile_pixel_to_lonlat(z, x_tile, y_tile, px, py, out_px)
                speed_kt = bilinear_wgs84(speed_ms, bbox, lons, lats) * KNOTS_PER_MPS
                image = render_tile(speed_kt, scale_max_kt, render_alpha)
                data_image = render_speed_data_tile(speed_kt)
                gust_image = None
                if gust_ms is not None:
                    gust_kt = bilinear_wgs84(gust_ms, bbox, lons, lats) * KNOTS_PER_MPS
                    gust_image = render_speed_data_tile(gust_kt)
                render_seconds += time.perf_counter() - render_started
                encode_started = time.perf_counter()
                if image is not None:
                    save_color(image, staging_root / step_key / "speed" / str(z) / str(x_tile) / f"{y_tile}.{ext}")
                    counts["tile_count"] += 1
                    counts["color_tile_count"] += 1
                if data_image is not None:
                    data_path = staging_root / step_key / SPEED_DATA_MODE / str(z) / str(x_tile) / f"{y_tile}.png"
                    data_path.parent.mkdir(parents=True, exist_ok=True)
                    data_image.save(data_path, compress_level=2)
                    counts["tile_count"] += 1
                    counts["data_tile_count"] += 1
                if gust_image is not None:
                    gust_path = staging_root / step_key / GUST_DATA_MODE / str(z) / str(x_tile) / f"{y_tile}.png"
                    gust_path.parent.mkdir(parents=True, exist_ok=True)
                    gust_image.save(gust_path, compress_level=2)
                    counts["tile_count"] += 1
                    counts["data_tile_count"] += 1
                encode_seconds += time.perf_counter() - encode_started
                if has_weather:
                    weather_render_started = time.perf_counter()
                    cloud_tile = bilinear_wgs84(cloud_pct, bbox, lons, lats) if cloud_pct is not None else None
                    rain_tile = bilinear_wgs84(precipitation_mm, bbox, lons, lats) if precipitation_mm is not None else None
                    weather_image = render_cloud_rain_tile(cloud_tile, rain_tile)
                    render_seconds += time.perf_counter() - weather_render_started
                    if weather_image is not None:
                        encode_started = time.perf_counter()
                        save_color(weather_image, staging_root / step_key / "cloud_rain" / str(z) / str(x_tile) / f"{y_tile}.{ext}")
                        encode_seconds += time.perf_counter() - encode_started
                        counts["tile_count"] += 1
    counts["render_s"] = render_seconds
    counts["encode_s"] = encode_seconds
    counts["step_key"] = step_key
    return counts


def load_previous_manifest(output_root: Path) -> dict[str, Any] | None:
    manifest_path = output_root / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def reusable_previous_steps(previous: dict[str, Any] | None, output_root: Path, params_key: str) -> dict[str, dict[str, Any]]:
    """Map step key → previous manifest step entry when its rendered dir can be reused."""
    if not previous:
        return {}
    if previous.get("paramsKey") != params_key:
        return {}
    previous_set = previous.get("tileSet")
    if not previous_set:
        return {}
    previous_root = output_root / "_sets" / previous_set
    if not previous_root.exists():
        return {}
    reusable: dict[str, dict[str, Any]] = {}
    for step in previous.get("steps") or []:
        key = step.get("key")
        if key and step.get("sourceHash") and (previous_root / key).exists():
            reusable[key] = step
    return reusable


def build(
    model: str,
    zooms: tuple[int, ...],
    scale_max_kt: float,
    scale: int = 1,
    tile_format: str = "webp",
    webp_quality: int = 90,
    webp_method: int = 2,
    workers: int = 1,
) -> dict[str, Any]:
    build_started = time.perf_counter()
    spec = MODELS[model]
    json_path = WIND2D / spec["json"]
    if not json_path.exists():
        raise FileNotFoundError(f"Model JSON not found: {json_path.relative_to(ROOT)}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    bbox = tuple(float(value) for value in payload["bbox_wgs84"])
    steps = payload.get("forecast_steps") or []
    if not steps:
        raise RuntimeError(f"No forecast_steps in {json_path.name}")
    render_alpha = float(spec["render_alpha"])

    output_root = WIND2D / "tiles" / model
    output_root.mkdir(parents=True, exist_ok=True)

    # Render each 256-CSS-px tile at `scale`× physical pixels for crisp display on hi-dpi
    # screens. The manifest tileSize stays 256, so the browser fits the larger image into the
    # 256 CSS box — sharp on dpr≥2, fine on dpr 1.
    out_px = TILE_SIZE * scale
    # WebP is ~8× smaller than PNG for these smooth translucent overlays, which is the dominant
    # lever for fast colour-tile serving over the network. Data tiles stay lossless PNG.
    ext = "webp" if tile_format == "webp" else "png"
    params_key = render_params_key(zooms, scale, ext, webp_quality, webp_method, scale_max_kt, render_alpha)
    tile_set = tile_set_key(payload, zooms, scale, ext, webp_quality, webp_method)
    tile_root = output_root / "_sets" / tile_set
    staging_root = output_root / "_staging" / f"{tile_set}-{os.getpid()}"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)

    previous_manifest = load_previous_manifest(output_root)
    reusable_steps = reusable_previous_steps(previous_manifest, output_root, params_key)
    previous_set_root = output_root / "_sets" / str(previous_manifest.get("tileSet")) if previous_manifest else None

    manifest_steps: list[dict[str, Any]] = []
    seen_step_keys: set[str] = set()
    manifest_modes: set[str] = {"speed", SPEED_DATA_MODE}
    pending_tasks: list[dict[str, Any]] = []
    reused_step_count = 0
    reuse_seconds = 0.0

    for step in steps:
        lead_hour = normalise_lead_hour(step)
        step_key = step_key_for_step(step)
        if step_key in seen_step_keys:
            valid_time = step.get("valid_time_utc") or "unknown valid time"
            raise RuntimeError(f"Duplicate raster step key {step_key!r} for {model} at {valid_time}")
        seen_step_keys.add(step_key)
        source_hash = step_source_hash(step)
        has_weather = step.get("cloud_cover_pct") is not None or step.get("precipitation_mm") is not None
        has_gust = step.get("gust_speed_ms") is not None
        step_modes = ["speed", SPEED_DATA_MODE]
        if has_gust:
            step_modes.append(GUST_DATA_MODE)
            manifest_modes.add(GUST_DATA_MODE)
        if has_weather:
            step_modes.append("cloud_rain")
            manifest_modes.add("cloud_rain")
        manifest_steps.append(
            {
                "key": step_key,
                "lead_hour": lead_hour,
                "lead_minutes": step.get("lead_minutes"),
                "valid_time_utc": step.get("valid_time_utc"),
                "modes": step_modes,
                "sourceHash": source_hash,
            }
        )

        previous_step = reusable_steps.get(step_key)
        if previous_step and previous_step.get("sourceHash") == source_hash and previous_set_root:
            # Identical source grids and render params: hard-link the already-rendered step
            # directory instead of re-rendering it. On AROME-PI rolling updates most steps are
            # unchanged between runs, so rebuilds become nearly incremental.
            reuse_started = time.perf_counter()
            link_or_copy_tree(previous_set_root / step_key, staging_root / step_key)
            reuse_seconds += time.perf_counter() - reuse_started
            reused_step_count += 1
            continue
        pending_tasks.append(
            {
                "staging_root": str(staging_root),
                "bbox": list(bbox),
                "zooms": list(zooms),
                "out_px": out_px,
                "scale_max_kt": scale_max_kt,
                "render_alpha": render_alpha,
                "ext": ext,
                "tile_format": tile_format,
                "webp_quality": webp_quality,
                "webp_method": webp_method,
                "step_key": step_key,
                "speed_ms": step["speed_ms"],
                "gust_speed_ms": step.get("gust_speed_ms"),
                "cloud_cover_pct": step.get("cloud_cover_pct"),
                "precipitation_mm": step.get("precipitation_mm"),
            }
        )

    tile_count = 0
    color_tile_count = 0
    data_tile_count = 0
    render_seconds = 0.0
    encode_seconds = 0.0
    try:
        if pending_tasks:
            if workers > 1 and len(pending_tasks) > 1:
                with ProcessPoolExecutor(max_workers=min(workers, len(pending_tasks))) as pool:
                    results = list(pool.map(render_step_task, pending_tasks))
            else:
                results = [render_step_task(task) for task in pending_tasks]
            for result in results:
                tile_count += result["tile_count"]
                color_tile_count += result["color_tile_count"]
                data_tile_count += result["data_tile_count"]
                render_seconds += result["render_s"]
                encode_seconds += result["encode_s"]

        manifest = {
            "format": MANIFEST_FORMAT,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "modelLabel": payload.get("model_label"),
            "runTimeUtc": payload.get("run_time_utc"),
            "bounds_wgs84": list(bbox),
            "tileSize": TILE_SIZE,
            "renderScale": scale,
            "tilePixels": TILE_SIZE * scale,
            "zooms": list(zooms),
            "modes": sorted(manifest_modes),
            "encoding": "color",
            "tileFormat": ext,
            "tileSet": tile_set,
            "paramsKey": params_key,
            "dataTileFormat": "png",
            "dataUrlTemplate": f"./tiles/{model}/_sets/{tile_set}/{{step}}/{SPEED_DATA_MODE}/{{z}}/{{x}}/{{y}}.png",
            "encodings": {
                mode: {
                    "type": "u16_kt_rg_alpha",
                    "quantumKt": SPEED_DATA_QUANTUM_KT,
                    "urlMode": mode,
                    "tileFormat": "png",
                    "urlTemplate": f"./tiles/{model}/_sets/{tile_set}/{{step}}/{mode}/{{z}}/{{x}}/{{y}}.png",
                }
                for mode in sorted(manifest_modes & {SPEED_DATA_MODE, GUST_DATA_MODE})
            },
            "webpQuality": webp_quality if ext == "webp" else None,
            "webpMethod": webp_method if ext == "webp" else None,
            "steps": manifest_steps,
            "urlTemplate": f"./tiles/{model}/_sets/{tile_set}/{{step}}/{{mode}}/{{z}}/{{x}}/{{y}}.{ext}",
            "speedScaleMaxKt": scale_max_kt,
            "renderAlpha": render_alpha,
            "opacity": 1.0,
            "tileCount": tile_count,
            "colorTileCount": color_tile_count,
            "dataTileCount": data_tile_count,
            "reusedStepCount": reused_step_count,
            "renderedStepCount": len(pending_tasks),
            "source": {"json": str(json_path.relative_to(ROOT))},
        }
        publish_started = time.perf_counter()
        publish_tile_set(staging_root, tile_root)
        publish_seconds = time.perf_counter() - publish_started
        total_seconds = time.perf_counter() - build_started
        manifest["timings"] = {
            "total_s": round(total_seconds, 3),
            "render_s": round(render_seconds, 3),
            "encode_s": round(encode_seconds, 3),
            "reuse_s": round(reuse_seconds, 3),
            "publish_s": round(publish_seconds, 3),
            "workers": workers,
            "tiles_per_s": round(tile_count / total_seconds, 3) if total_seconds > 0 and tile_count else None,
        }
        write_json_atomic(output_root / "manifest.json", manifest)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    pruned = prune_output_root(output_root, tile_set)
    manifest["prunedEntries"] = pruned
    return manifest


def default_workers() -> int:
    return max(1, min(3, (os.cpu_count() or 2) - 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="arome")
    parser.add_argument("--all", action="store_true", help="Build tiles for every available model JSON.")
    parser.add_argument("--zooms", nargs="+", type=int, default=list(DEFAULT_ZOOMS))
    parser.add_argument("--scale-max-kt", type=float, default=DEFAULT_SCALE_MAX_KT)
    parser.add_argument("--scale", type=int, default=1, choices=[1, 2, 3], help="Physical-pixel supersampling per 256 CSS tile. 1 keeps server rebuilds responsive; Leaflet overzooms closer views.")
    parser.add_argument("--format", dest="tile_format", choices=["webp", "png"], default="webp", help="Tile image format. WebP is ~8× smaller than PNG for these overlays.")
    parser.add_argument("--webp-quality", type=int, default=90, help="WebP quality (lossy). 90 is visually lossless for these translucent overlays.")
    parser.add_argument("--webp-method", type=int, default=2, choices=range(0, 7), help="WebP encoder effort, 0=fastest and 6=slowest. Lower keeps server rebuilds responsive.")
    parser.add_argument("--workers", type=int, default=default_workers(), help="Parallel step-render workers (steps are independent). 1 disables multiprocessing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = sorted(MODELS) if args.all else [args.model]
    for model in models:
        if not (WIND2D / MODELS[model]["json"]).exists():
            if args.all:
                print(f"skip {model}: source JSON missing")
                continue
            raise FileNotFoundError(f"Model JSON not found for {model}")
        manifest = build(
            model,
            tuple(args.zooms),
            args.scale_max_kt,
            args.scale,
            args.tile_format,
            args.webp_quality,
            args.webp_method,
            max(1, args.workers),
        )
        timings = manifest.get("timings") or {}
        print(
            f"{model}: {manifest['tileCount']} tiles rendered · {manifest['reusedStepCount']} steps reused · "
            f"{manifest['renderedStepCount']} steps rendered · zooms {manifest['zooms']} · {manifest['tilePixels']}px · "
            f"{manifest['tileFormat']} · {timings.get('total_s', '?')}s total (render {timings.get('render_s', '?')}s, "
            f"encode {timings.get('encode_s', '?')}s, reuse {timings.get('reuse_s', '?')}s, workers {timings.get('workers', '?')}) · "
            f"pruned {manifest.get('prunedEntries', 0)} old entries"
        )
        print(f"  wrote {(WIND2D / 'tiles' / model / 'manifest.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
