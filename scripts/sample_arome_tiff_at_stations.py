#!/usr/bin/env python3
"""Sample an AROME WCS GeoTIFF at station coordinates.

This is intentionally lightweight for Phase 0. The WCS request bbox is passed
explicitly, and the script samples the nearest raster cell for each station.
"""

from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path

import numpy as np


STATIONS = [
    {"station_id": "20092001", "name": "CONCA", "lat": 41.7357, "lon": 9.3402},
    {"station_id": "20247001", "name": "LA CHIAPPA", "lat": 41.5962, "lon": 9.3628},
    {"station_id": "20114002", "name": "FIGARI", "lat": 41.5022, "lon": 9.0978},
    {"station_id": "20272004", "name": "SARTENE", "lat": 41.6210, "lon": 8.9830},
    {"station_id": "20041001", "name": "CAP PERTUSATO", "lat": 41.3672, "lon": 9.1840},
]


TIFF_TYPE_SIZES = {
    1: 1,  # BYTE
    2: 1,  # ASCII
    3: 2,  # SHORT
    4: 4,  # LONG
    5: 8,  # RATIONAL
    12: 8,  # DOUBLE
}


def read_tiff_values(data: bytes, endian: str, field_type: int, count: int, value_or_offset: int) -> tuple[int | float | str, ...]:
    size = TIFF_TYPE_SIZES[field_type] * count
    raw = data[value_or_offset : value_or_offset + size] if size > 4 else struct.pack(endian + "I", value_or_offset)[:size]
    if field_type == 2:
        return (raw.rstrip(b"\x00").decode("utf-8", errors="replace"),)
    if field_type == 3:
        return struct.unpack(endian + f"{count}H", raw)
    if field_type == 4:
        return struct.unpack(endian + f"{count}I", raw)
    if field_type == 12:
        return struct.unpack(endian + f"{count}d", raw)
    raise ValueError(f"Unsupported TIFF field type: {field_type}")


def read_float64_tiff(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if data[:2] == b"II":
        endian = "<"
    elif data[:2] == b"MM":
        endian = ">"
    else:
        raise ValueError("Not a TIFF file")

    magic, ifd_offset = struct.unpack(endian + "HI", data[2:8])
    if magic != 42:
        raise ValueError(f"Unsupported TIFF magic: {magic}")

    entry_count = struct.unpack(endian + "H", data[ifd_offset : ifd_offset + 2])[0]
    tags: dict[int, tuple[int | float | str, ...]] = {}
    for index in range(entry_count):
        offset = ifd_offset + 2 + index * 12
        tag, field_type, count, value_or_offset = struct.unpack(endian + "HHII", data[offset : offset + 12])
        tags[tag] = read_tiff_values(data, endian, field_type, count, value_or_offset)

    width = int(tags[256][0])
    height = int(tags[257][0])
    bits_per_sample = int(tags[258][0])
    compression = int(tags[259][0])
    sample_format = int(tags[339][0])
    offsets = [int(value) for value in tags[273]]
    byte_counts = [int(value) for value in tags[279]]

    if bits_per_sample != 64 or compression != 1 or sample_format != 3:
        raise ValueError(
            "Only uncompressed float64 GeoTIFF rasters are supported by this Phase 0 sampler"
        )

    raw = b"".join(data[offset : offset + byte_count] for offset, byte_count in zip(offsets, byte_counts))
    array = np.frombuffer(raw, dtype=endian + "f8", count=width * height)
    return array.reshape((height, width))


def sample_nearest(array: np.ndarray, lon: float, lat: float, bbox: tuple[float, float, float, float]) -> tuple[int, int, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    height, width = array.shape
    col = round((lon - min_lon) / (max_lon - min_lon) * (width - 1))
    # WCS GeoTIFF rows are north-to-south.
    row = round((max_lat - lat) / (max_lat - min_lat) * (height - 1))
    row = max(0, min(height - 1, row))
    col = max(0, min(width - 1, col))
    return row, col, float(array[row, col])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--valid-time", required=True)
    parser.add_argument("--variable", required=True)
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"), required=True)
    args = parser.parse_args()

    array = read_float64_tiff(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "station_id",
        "name",
        "run_time_utc",
        "valid_time_utc",
        "variable",
        "lat",
        "lon",
        "row",
        "col",
        "value",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for station in STATIONS:
            row, col, value = sample_nearest(array, station["lon"], station["lat"], tuple(args.bbox))
            writer.writerow(
                {
                    **station,
                    "run_time_utc": args.run_time,
                    "valid_time_utc": args.valid_time,
                    "variable": args.variable,
                    "row": row,
                    "col": col,
                    "value": value,
                }
            )

    finite = array[np.isfinite(array)]
    if finite.size:
        print(
            f"Wrote {len(STATIONS)} samples to {args.output}; "
            f"raster shape={array.shape}, min={finite.min():.3f}, max={finite.max():.3f}"
        )
    else:
        print(f"Wrote {len(STATIONS)} samples to {args.output}; raster has no finite values")


if __name__ == "__main__":
    main()
