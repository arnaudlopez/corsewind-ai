#!/usr/bin/env python3
"""Normalize Beacon Live weather-state observations for the CorseWind ML dataset."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_BEACON_STATE = Path("/Users/arnaud/Documents/beacon-live-app/data/weather-state.json")
DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_ML_ROOT / "observations/beacon_live"
KNOT_TO_MS = 0.514444


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


def finite_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def knots_to_ms(value: Any) -> float | None:
    number = finite_float(value)
    return None if number is None else round(number * KNOT_TO_MS, 3)


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["beacon_source_id"]): item
        for item in payload.get("spots", [])
        if item.get("beacon_source_id")
    }


def normalize_observation(observation: dict[str, Any], registry: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    source_id = observation.get("sourceId")
    source = registry.get(str(source_id))
    if not source:
        return None
    live = observation.get("payload", {}).get("live")
    if not isinstance(live, dict):
        return None
    observed_at = parse_time(observation.get("observedAt"))
    received_at = parse_time(observation.get("receivedAt"))
    if observed_at is None:
        return None
    return {
        "format": "corsewind.ml_observation.v1",
        "source_project": "beacon-live-app",
        "source_id": source_id,
        "spot_id": source["spot_id"],
        "station_id": source.get("station_id"),
        "source_type": source.get("source_type"),
        "timestamp_utc": observed_at.isoformat().replace("+00:00", "Z"),
        "received_at_utc": received_at.isoformat().replace("+00:00", "Z") if received_at else None,
        "latitude": source.get("latitude"),
        "longitude": source.get("longitude"),
        "source_resolution_minutes": source.get("source_resolution_minutes"),
        "wind_mean_ms": knots_to_ms(live.get("windSpeed")),
        "gust_ms": knots_to_ms(live.get("windGust")),
        "wind_direction_deg": finite_float(live.get("windDirection")),
        "temperature_c": finite_float(live.get("temperature")),
        "humidity_pct": finite_float(live.get("humidity")),
        "pressure_hpa": finite_float(live.get("pressure")),
        "raw_units": {
            "windSpeed": "kt",
            "windGust": "kt",
            "windDirection": "deg",
            "temperature": "C",
            "humidity": "%",
            "pressure": "hPa",
        },
    }


def output_path(output_root: Path, timestamp: str) -> Path:
    day = timestamp[:10]
    return output_root / f"date={day}" / "observations.jsonl"


def write_jsonl_by_day(output_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_path[output_path(output_root, row["timestamp_utc"])].append(row)

    written = {}
    for path, path_rows in by_path.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        deduped = {
            (row["source_id"], row["timestamp_utc"]): row
            for row in path_rows
        }
        ordered = sorted(deduped.values(), key=lambda row: (row["timestamp_utc"], row["source_id"]))
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")
        written[str(path)] = len(ordered)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-state", type=Path, default=DEFAULT_BEACON_STATE)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_registry(args.registry)
    state = json.loads(args.weather_state.read_text(encoding="utf-8"))
    rows = [
        row
        for row in (
            normalize_observation(observation, registry)
            for observation in state.get("observations", [])
        )
        if row is not None
    ]
    written = write_jsonl_by_day(args.output_root, rows)
    print(json.dumps({
        "generated_at_utc": utc_now(),
        "input": str(args.weather_state),
        "registry": str(args.registry),
        "normalized_rows": len(rows),
        "written": written,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
