#!/usr/bin/env python3
"""Download ESA WorldCover 10 m tiles needed for a WGS84 bbox."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data/raw/landcover/esa_worldcover_v200_2021"
BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"


def tile_name(north_lat: int, west_lon: int) -> str:
    ns = "N" if north_lat >= 0 else "S"
    ew = "E" if west_lon >= 0 else "W"
    return f"ESA_WorldCover_10m_2021_v200_{ns}{abs(north_lat):02d}{ew}{abs(west_lon):03d}_Map.tif"


def required_tiles(bbox: list[float]) -> list[tuple[str, str]]:
    min_lon, min_lat, max_lon, max_lat = bbox
    eps = 1e-9
    lon_start = math.floor(min_lon / 3.0) * 3
    lon_end = math.floor((max_lon - eps) / 3.0) * 3
    lat_north_min = math.ceil((min_lat + eps) / 3.0) * 3
    lat_north_max = math.ceil(max_lat / 3.0) * 3
    tiles = []
    for north_lat in range(lat_north_min, lat_north_max + 1, 3):
        for west_lon in range(lon_start, lon_end + 1, 3):
            name = tile_name(north_lat, west_lon)
            tiles.append((name, f"{BASE_URL}/{name}"))
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
