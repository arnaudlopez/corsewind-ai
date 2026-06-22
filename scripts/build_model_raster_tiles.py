#!/usr/bin/env python3
"""Pre-bake Leaflet raster tiles for a raw wind model (AROME, AROME-PI, MOLOCH, ICON-2I).

The Wind2D client computes the colour overlay pixel-by-pixel in JavaScript at interaction
time, which is the main source of pan/zoom lag. This script renders the same colour field
into a static PNG tile pyramid (Web Mercator / XYZ, the Google-Maps model) so the client can
serve it through a plain L.tileLayer instead — instant, GPU-composited, progressive.

The colour ramp and per-pixel alpha intentionally mirror the client (colorArray + the speed
branch of renderFieldColor + drawWindHeatTile), so pre-baked tiles match the legend exactly.

Output layout:
    visualizations/wind2d/tiles/<model>/<step>/<mode>/<z>/<x>/<y>.png
    visualizations/wind2d/tiles/<model>/manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
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
# The raw models are ~1 km resolution, so z11 already heavily oversamples the field; Leaflet
# overzooms beyond maxNativeZoom for closer views. Generating z12+ only multiplies tiles.
DEFAULT_ZOOMS = tuple(range(8, 12))

# Per-model source JSON and the render alpha the client applies to each raw layer
# (see rawModelFieldAt defaults in wind2d.js). Baking this keeps tiles visually identical
# to the JS heat overlay, so the client can show raster tiles at full opacity.
MODELS: dict[str, dict[str, Any]] = {
    "arome": {"json": "arome-corsica-latest.json", "render_alpha": 0.62},
    "aromepi": {"json": "aromepi-corsica-latest.json", "render_alpha": 0.66},
    "moloch": {"json": "moloch-corsica-latest.json", "render_alpha": 0.58},
    "icon2i": {"json": "icon2i-corsica-latest.json", "render_alpha": 0.58},
}

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


def speed_color(intensity: np.ndarray) -> np.ndarray:
    """Vectorised equivalent of colorArray() in wind2d.js."""
    out = np.zeros((*intensity.shape, 3), dtype=np.float32)
    out[intensity <= SPEED_STOPS[0][0]] = SPEED_STOPS[0][1]
    for idx in range(1, len(SPEED_STOPS)):
        stop, rgb = SPEED_STOPS[idx]
        prev_stop, prev_rgb = SPEED_STOPS[idx - 1]
        mask = (intensity > prev_stop) & (intensity <= stop)
        t = ((intensity - prev_stop) / max(0.0001, stop - prev_stop))[..., None]
        blended = np.array(prev_rgb, dtype=np.float32) + (np.array(rgb, dtype=np.float32) - np.array(prev_rgb, dtype=np.float32)) * t
        out[mask] = blended[mask]
    out[intensity > SPEED_STOPS[-1][0]] = SPEED_STOPS[-1][1]
    return np.round(out)


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


def build(model: str, zooms: tuple[int, ...], scale_max_kt: float, scale: int = 2) -> dict[str, Any]:
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
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Render each 256-CSS-px tile at `scale`× physical pixels (e.g. 512) for crisp display on
    # retina/hi-dpi screens. The manifest tileSize stays 256, so the browser fits the larger
    # image into the 256 CSS box — sharp on dpr≥2, fine on dpr 1.
    out_px = TILE_SIZE * scale
    grid = np.arange(out_px, dtype=np.float32) + 0.5
    px, py = np.meshgrid(grid, grid)

    manifest_steps: list[dict[str, Any]] = []
    seen_step_keys: set[str] = set()
    tile_count = 0
    for step in steps:
        lead_hour = normalise_lead_hour(step)
        step_key = step_key_for_step(step)
        if step_key in seen_step_keys:
            valid_time = step.get("valid_time_utc") or "unknown valid time"
            raise RuntimeError(f"Duplicate raster step key {step_key!r} for {model} at {valid_time}")
        seen_step_keys.add(step_key)
        speed_ms = np.array(step["speed_ms"], dtype=np.float32)
        for z in zooms:
            x_range, y_range = tile_ranges(bbox, z)
            for x_tile in x_range:
                for y_tile in y_range:
                    lons, lats = tile_pixel_to_lonlat(z, x_tile, y_tile, px, py, out_px)
                    speed_kt = bilinear_wgs84(speed_ms, bbox, lons, lats) * KNOTS_PER_MPS
                    image = render_tile(speed_kt, scale_max_kt, render_alpha)
                    if image is None:
                        continue
                    path = output_root / step_key / "speed" / str(z) / str(x_tile) / f"{y_tile}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(path, compress_level=2)
                    tile_count += 1
        manifest_steps.append(
            {
                "key": step_key,
                "lead_hour": lead_hour,
                "lead_minutes": step.get("lead_minutes"),
                "valid_time_utc": step.get("valid_time_utc"),
            }
        )

    manifest = {
        "format": "corsewind.model.raster_tiles.v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "modelLabel": payload.get("model_label"),
        "runTimeUtc": payload.get("run_time_utc"),
        "bounds_wgs84": list(bbox),
        "tileSize": TILE_SIZE,
        "renderScale": scale,
        "tilePixels": TILE_SIZE * scale,
        "zooms": list(zooms),
        "modes": ["speed"],
        "encoding": "color",
        "steps": manifest_steps,
        "urlTemplate": f"./tiles/{model}/{{step}}/{{mode}}/{{z}}/{{x}}/{{y}}.png",
        "speedScaleMaxKt": scale_max_kt,
        "renderAlpha": render_alpha,
        "opacity": 1.0,
        "tileCount": tile_count,
        "source": {"json": str(json_path.relative_to(ROOT))},
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="arome")
    parser.add_argument("--all", action="store_true", help="Build tiles for every available model JSON.")
    parser.add_argument("--zooms", nargs="+", type=int, default=list(DEFAULT_ZOOMS))
    parser.add_argument("--scale-max-kt", type=float, default=DEFAULT_SCALE_MAX_KT)
    parser.add_argument("--scale", type=int, default=2, choices=[1, 2, 3], help="Physical-pixel supersampling per 256 CSS tile (2 = retina).")
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
        manifest = build(model, tuple(args.zooms), args.scale_max_kt, args.scale)
        print(f"{model}: {manifest['tileCount']} tiles · {len(manifest['steps'])} steps · zooms {manifest['zooms']} · {manifest['tilePixels']}px")
        print(f"  wrote {(WIND2D / 'tiles' / model / 'manifest.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
