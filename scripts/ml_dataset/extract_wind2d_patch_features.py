#!/usr/bin/env python3
"""Extract local Wind2D grid-patch features around ML spots."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_spots(path: Path, include_context: bool) -> list[dict[str, Any]]:
    payload = read_json(path)
    spots = payload.get("spots", []) if isinstance(payload, dict) else payload
    out = []
    for spot in spots:
        if not isinstance(spot, dict):
            continue
        if not include_context and not spot.get("use_for_ml", False):
            continue
        if finite_float(spot.get("latitude")) is None or finite_float(spot.get("longitude")) is None:
            continue
        out.append(spot)
    return out


def grid_position(lat: float, lon: float, bbox: list[float], rows: int, cols: int) -> tuple[float, float, bool]:
    west, south, east, north = bbox
    x = (lon - west) / (east - west) * (cols - 1)
    y = (north - lat) / (north - south) * (rows - 1)
    return x, y, 0 <= x <= cols - 1 and 0 <= y <= rows - 1


def grid_value(grid: list[list[Any]], row: int, col: int) -> float | None:
    if row < 0 or row >= len(grid):
        return None
    line = grid[row]
    if col < 0 or col >= len(line):
        return None
    return finite_float(line[col])


def patch_values(grid: Any, x: float, y: float, radius: int) -> list[float]:
    if not isinstance(grid, list) or not grid:
        return []
    row0 = int(round(y))
    col0 = int(round(x))
    values = []
    for row in range(row0 - radius, row0 + radius + 1):
        for col in range(col0 - radius, col0 + radius + 1):
            value = grid_value(grid, row, col)
            if value is not None:
                values.append(value)
    return values


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    values = sorted(values)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "min": round(values[0], 6),
        "mean": round(mean, 6),
        "max": round(values[-1], 6),
        "std": round(math.sqrt(variance), 6),
        "range": round(values[-1] - values[0], 6),
    }


def center_value(grid: Any, x: float, y: float) -> float | None:
    if not isinstance(grid, list) or not grid:
        return None
    return grid_value(grid, int(round(y)), int(round(x)))


def infer_lead_minutes(step: dict[str, Any]) -> int | None:
    lead = step.get("lead_minutes")
    if isinstance(lead, int):
        return lead
    lead_hour = finite_float(step.get("lead_hour"))
    return None if lead_hour is None else int(round(lead_hour * 60))


def add_field_features(row: dict[str, Any], prefix: str, grid: Any, x: float, y: float, radii: list[int]) -> None:
    center = center_value(grid, x, y)
    row[f"{prefix}_center_ms"] = None if center is None else round(center, 6)
    for radius in radii:
        stats = summarize(patch_values(grid, x, y, radius))
        for key, value in stats.items():
            row[f"{prefix}_patch{radius}_{key}"] = value
        if center is not None and stats.get("max") is not None:
            row[f"{prefix}_patch{radius}_max_minus_center_ms"] = round(float(stats["max"]) - center, 6)
        if center is not None and stats.get("mean") is not None:
            row[f"{prefix}_patch{radius}_center_minus_mean_ms"] = round(center - float(stats["mean"]), 6)


def extract(layer: dict[str, Any], spots: list[dict[str, Any]], radii: list[int]) -> list[dict[str, Any]]:
    bbox = layer.get("bbox_wgs84")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise SystemExit("Layer has no bbox_wgs84.")
    rows = []
    for step in layer.get("forecast_steps") or []:
        shape = step.get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            continue
        grid_rows, grid_cols = int(shape[0]), int(shape[1])
        for spot in spots:
            lat = float(spot["latitude"])
            lon = float(spot["longitude"])
            x, y, inside = grid_position(lat, lon, bbox, grid_rows, grid_cols)
            row: dict[str, Any] = {
                "format": "corsewind.wind2d_patch_features.v1",
                "source": layer.get("product"),
                "model_label": layer.get("model_label"),
                "run_time_utc": layer.get("run_time_utc"),
                "generated_at_utc": layer.get("generated_at_utc"),
                "valid_time_utc": step.get("valid_time_utc"),
                "lead_minutes": infer_lead_minutes(step),
                "spot_id": spot.get("spot_id"),
                "spot_name": spot.get("name"),
                "latitude": lat,
                "longitude": lon,
                "grid_x": round(x, 4),
                "grid_y": round(y, 4),
                "inside_grid": inside,
                "radii_cells": radii,
                "extracted_at_utc": utc_now(),
            }
            if inside:
                add_field_features(row, "wind_speed", step.get("speed_ms"), x, y, radii)
                add_field_features(row, "wind_u", step.get("u_ms"), x, y, radii)
                add_field_features(row, "wind_v", step.get("v_ms"), x, y, radii)
                if "gust_speed_ms" in step:
                    add_field_features(row, "gust_speed", step.get("gust_speed_ms"), x, y, radii)
            rows.append(row)
    return rows


def output_path(root: Path, source: str, valid_time_utc: str | None) -> Path:
    day = (valid_time_utc or utc_now())[:10]
    return root / f"source={source}" / f"date={day}" / "patch_features.jsonl"


def write_by_day(root: Path, rows: list[dict[str, Any]]) -> dict[str, int]:
    grouped: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[output_path(root, str(row.get("source") or "unknown"), row.get("valid_time_utc"))].append(row)
    written = {}
    for path, new_rows in grouped.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if path.exists():
            existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        deduped = {
            (row.get("source"), row.get("run_time_utc"), row.get("valid_time_utc"), row.get("spot_id")): row
            for row in [*existing, *new_rows]
        }
        ordered = sorted(deduped.values(), key=lambda row: (row.get("run_time_utc") or "", row.get("valid_time_utc") or "", row.get("spot_id") or ""))
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")
        tmp.replace(path)
        written[str(path)] = len(ordered)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ML_ROOT / "model_patch_features")
    parser.add_argument("--patch-radius", type=int, action="append", default=[1, 2])
    parser.add_argument("--include-context-spots", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layer = read_json(args.input)
    spots = load_spots(args.registry, args.include_context_spots)
    rows = extract(layer, spots, sorted(set(args.patch_radius)))
    written = write_by_day(args.output_root, rows)
    print(json.dumps({
        "generated_at_utc": utc_now(),
        "input": str(args.input),
        "output_root": str(args.output_root),
        "row_count": len(rows),
        "spot_count": len(spots),
        "written": written,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
