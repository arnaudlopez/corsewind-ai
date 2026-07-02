#!/usr/bin/env python3
"""Normalize MeteoNet ground-station archives for Corsica pretraining."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_INPUT_ROOT = DEFAULT_ML_ROOT / "research/meteonet/raw/SE/ground_stations"
DEFAULT_OUTPUT_ROOT = DEFAULT_ML_ROOT / "research/meteonet/normalized/ground_stations"
DEFAULT_PROFILE = DEFAULT_ML_ROOT / "research/meteonet/normalized/ground_stations/profile.json"
DEFAULT_BBOX = "41.0,8.4,43.2,9.8"
KELVIN_OFFSET = 273.15


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def finite_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def kelvin_to_c(value: Any) -> float | None:
    number = finite_float(value)
    return round(number - KELVIN_OFFSET, 3) if number is not None else None


def pa_to_hpa(value: Any) -> float | None:
    number = finite_float(value)
    return round(number / 100.0, 3) if number is not None else None


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be min_lat,min_lon,max_lat,max_lon")
    return parts[0], parts[1], parts[2], parts[3]


def in_bbox(row: dict[str, str], bbox: tuple[float, float, float, float]) -> bool:
    lat = finite_float(row.get("lat"))
    lon = finite_float(row.get("lon"))
    if lat is None or lon is None:
        return False
    min_lat, min_lon, max_lat, max_lon = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def parse_meteonet_time(value: str) -> str | None:
    if not value:
        return None
    try:
        parsed = datetime.strptime(value, "%Y%m%d %H:%M").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return parsed.isoformat().replace("+00:00", "Z")


def year_from_archive(path: Path) -> int | None:
    for token in path.name.replace(".", "_").split("_"):
        if token in {"2016", "2017", "2018"}:
            return int(token)
    return None


def normalize_row(row: dict[str, str], zone: str, year: int) -> dict[str, Any] | None:
    timestamp = parse_meteonet_time(row.get("date") or "")
    station_id = row.get("number_sta")
    if not station_id or timestamp is None:
        return None
    return {
        "format": "corsewind.meteonet_ground_observation.v1",
        "source_project": "meteonet",
        "source_dataset": "meteonet_ground_stations_6min",
        "zone": zone,
        "year": year,
        "station_id": str(station_id),
        "timestamp_utc": timestamp,
        "latitude": finite_float(row.get("lat")),
        "longitude": finite_float(row.get("lon")),
        "altitude_m": finite_float(row.get("height_sta")),
        "wind_direction_deg": finite_float(row.get("dd")),
        "wind_mean_ms": finite_float(row.get("ff")),
        "precipitation_mm": finite_float(row.get("precip")),
        "humidity_pct": finite_float(row.get("hu")),
        "dewpoint_c": kelvin_to_c(row.get("td")),
        "temperature_c": kelvin_to_c(row.get("t")),
        "sea_level_pressure_hpa": pa_to_hpa(row.get("psl")),
        "normalized_at_utc": utc_now(),
    }


def archive_member_name(zone: str, year: int) -> str:
    return f"{zone}{year}.csv"


def iter_archive_rows(path: Path, zone: str, year: int):
    member_name = archive_member_name(zone, year)
    with tarfile.open(path, "r:gz") as archive:
        member = archive.extractfile(member_name)
        if member is None:
            raise FileNotFoundError(f"{member_name} not found in {path}")
        with io.TextIOWrapper(member, encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)


def normalize_archive(
    archive_path: Path,
    output_root: Path,
    zone: str,
    bbox: tuple[float, float, float, float],
    max_rows: int | None,
) -> dict[str, Any]:
    year = year_from_archive(archive_path)
    if year is None:
        raise ValueError(f"Cannot infer year from {archive_path}")
    output_path = output_root / f"zone={zone}" / f"year={year}" / "observations.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    station_meta: dict[str, dict[str, Any]] = {}
    field_counts: Counter[str] = Counter()
    row_count = 0
    total_rows_seen = 0
    first_time = None
    last_time = None
    with output_path.open("w", encoding="utf-8") as out:
        for raw in iter_archive_rows(archive_path, zone, year):
            total_rows_seen += 1
            if not in_bbox(raw, bbox):
                continue
            row = normalize_row(raw, zone, year)
            if row is None:
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            row_count += 1
            timestamp = row["timestamp_utc"]
            first_time = timestamp if first_time is None or timestamp < first_time else first_time
            last_time = timestamp if last_time is None or timestamp > last_time else last_time
            station_id = row["station_id"]
            station_meta.setdefault(station_id, {
                "station_id": station_id,
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "altitude_m": row["altitude_m"],
            })
            for key, value in row.items():
                if key.endswith(("_ms", "_deg", "_c", "_hpa", "_pct", "_mm")) and value is not None:
                    field_counts[key] += 1
            if max_rows is not None and row_count >= max_rows:
                break
    return {
        "year": year,
        "archive": str(archive_path),
        "output_path": str(output_path),
        "total_rows_seen": total_rows_seen,
        "row_count": row_count,
        "station_count": len(station_meta),
        "first_timestamp_utc": first_time,
        "last_timestamp_utc": last_time,
        "field_non_null_counts": dict(sorted(field_counts.items())),
        "stations": sorted(station_meta.values(), key=lambda item: item["station_id"]),
    }


def write_station_registry(output_root: Path, zone: str, summaries: list[dict[str, Any]]) -> Path:
    stations: dict[str, dict[str, Any]] = {}
    years_by_station: dict[str, set[int]] = {}
    for summary in summaries:
        year = int(summary["year"])
        for station in summary.get("stations", []):
            station_id = station["station_id"]
            stations.setdefault(station_id, station)
            years_by_station.setdefault(station_id, set()).add(year)
    payload = {
        "format": "corsewind.meteonet_ground_station_registry.v1",
        "generated_at_utc": utc_now(),
        "zone": zone,
        "station_count": len(stations),
        "stations": [
            {**stations[station_id], "years": sorted(years_by_station.get(station_id, set()))}
            for station_id in sorted(stations)
        ],
    }
    path = output_root / f"zone={zone}" / "stations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--zone", default="SE")
    parser.add_argument("--bbox", type=parse_bbox, default=parse_bbox(DEFAULT_BBOX))
    parser.add_argument("--year", action="append", type=int)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archives = sorted(args.input_root.glob(f"{args.zone}_ground_stations_*.tar.gz"))
    if args.year:
        selected_years = set(args.year)
        archives = [path for path in archives if year_from_archive(path) in selected_years]
    summaries = [
        normalize_archive(path, args.output_root, args.zone, args.bbox, args.max_rows)
        for path in archives
    ]
    station_registry = write_station_registry(args.output_root, args.zone, summaries)
    profile = {
        "format": "corsewind.meteonet_ground_observation_profile.v1",
        "generated_at_utc": utc_now(),
        "zone": args.zone,
        "bbox": args.bbox,
        "summary_count": len(summaries),
        "row_count": sum(item["row_count"] for item in summaries),
        "station_count": len({station["station_id"] for item in summaries for station in item.get("stations", [])}),
        "first_timestamp_utc": min((item["first_timestamp_utc"] for item in summaries if item["first_timestamp_utc"]), default=None),
        "last_timestamp_utc": max((item["last_timestamp_utc"] for item in summaries if item["last_timestamp_utc"]), default=None),
        "station_registry": str(station_registry),
        "summaries": summaries,
    }
    args.profile.parent.mkdir(parents=True, exist_ok=True)
    args.profile.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
