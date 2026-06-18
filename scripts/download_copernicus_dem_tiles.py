#!/usr/bin/env python3
"""Download Copernicus GLO-30 DEM tiles needed for a WGS84 bbox."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data/raw/dem/copernicus_glo30"
BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"


def tile_name(lat_floor: int, lon_floor: int) -> str:
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00_DEM.tif"


def required_tiles(bbox: list[float]) -> list[tuple[str, str]]:
    min_lon, min_lat, max_lon, max_lat = bbox
    tiles = []
    for lat_floor in range(math.floor(min_lat), math.ceil(max_lat)):
        for lon_floor in range(math.floor(min_lon), math.ceil(max_lon)):
            name = tile_name(lat_floor, lon_floor)
            tiles.append((name, f"{BASE_URL}/{name.removesuffix('.tif')}/{name}"))
    return tiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", nargs=4, type=float, default=[8.62, 41.82, 8.90, 42.00])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, url in required_tiles(args.bbox):
        output = output_dir / name
        if output.exists() and not args.force:
            print(f"exists {output}")
            continue
        response = requests.get(url, timeout=180)
        if response.status_code >= 400:
            raise SystemExit(f"HTTP {response.status_code} for {url}: {response.text[:200]}")
        output.write_bytes(response.content)
        print(f"downloaded {output} {len(response.content)} bytes")


if __name__ == "__main__":
    main()
