#!/usr/bin/env python3
"""Sample Wind2D model-layer JSON grids at ML spot coordinates."""

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
DEFAULT_OUTPUT_ROOT = DEFAULT_ML_ROOT / "model_samples"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def finite_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def read_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"Input does not exist: {path}")
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def load_spots(path: Path, include_context: bool, selected_ids: set[str]) -> list[dict[str, Any]]:
    payload = read_json(path)
    spots = payload.get("spots", []) if isinstance(payload, dict) else payload
    if not isinstance(spots, list):
        raise SystemExit(f"Registry has no spots list: {path}")
    selected = []
    for spot in spots:
        if not isinstance(spot, dict):
            continue
        if selected_ids and spot.get("spot_id") not in selected_ids:
            continue
        if not include_context and not spot.get("use_for_ml", False):
            continue
        if finite_float(spot.get("latitude")) is None or finite_float(spot.get("longitude")) is None:
            continue
        selected.append(spot)
    return selected


def grid_position(lat: float, lon: float, bbox: list[float], rows: int, cols: int) -> tuple[float, float, bool]:
    west, south, east, north = bbox
    if rows < 2 or cols < 2 or east == west or north == south:
        raise SystemExit("Invalid grid geometry.")
    x = (lon - west) / (east - west) * (cols - 1)
    y = (north - lat) / (north - south) * (rows - 1)
    inside = 0 <= x <= cols - 1 and 0 <= y <= rows - 1
    return x, y, inside


def grid_value(grid: list[list[Any]], row: int, col: int) -> float | None:
    if row < 0 or col < 0 or row >= len(grid):
        return None
    line = grid[row]
    if col >= len(line):
        return None
    return finite_float(line[col])


def sample_nearest(grid: list[list[Any]], x: float, y: float) -> float | None:
    return grid_value(grid, int(round(y)), int(round(x)))


def sample_bilinear(grid: list[list[Any]], x: float, y: float) -> float | None:
    x0 = math.floor(x)
    y0 = math.floor(y)
    x1 = math.ceil(x)
    y1 = math.ceil(y)
    q11 = grid_value(grid, y0, x0)
    q21 = grid_value(grid, y0, x1)
    q12 = grid_value(grid, y1, x0)
    q22 = grid_value(grid, y1, x1)
    if None in {q11, q21, q12, q22}:
        return sample_nearest(grid, x, y)
    if x0 == x1 and y0 == y1:
        return q11
    if x0 == x1:
        return q11 * (y1 - y) + q12 * (y - y0)
    if y0 == y1:
        return q11 * (x1 - x) + q21 * (x - x0)
    return (
        q11 * (x1 - x) * (y1 - y)
        + q21 * (x - x0) * (y1 - y)
        + q12 * (x1 - x) * (y - y0)
        + q22 * (x - x0) * (y - y0)
    )


def sample_grid(grid: Any, x: float, y: float, method: str) -> float | None:
    if not isinstance(grid, list) or not grid:
        return None
    value = sample_nearest(grid, x, y) if method == "nearest" else sample_bilinear(grid, x, y)
    return round(value, 4) if value is not None else None


def wind_direction_deg(u_ms: float | None, v_ms: float | None) -> float | None:
    if u_ms is None or v_ms is None:
        return None
    # Meteorological direction: degrees from which the wind blows.
    return round((math.degrees(math.atan2(-u_ms, -v_ms)) + 360.0) % 360.0, 2)


def infer_lead_minutes(step: dict[str, Any]) -> int | None:
    lead = step.get("lead_minutes")
    if isinstance(lead, int):
        return lead
    lead_hour = finite_float(step.get("lead_hour"))
    return int(round(lead_hour * 60)) if lead_hour is not None else None


def sample_layer(
    source: str,
    layer: dict[str, Any],
    spots: list[dict[str, Any]],
    sample_method: str,
) -> list[dict[str, Any]]:
    bbox = layer.get("bbox_wgs84")
    steps = layer.get("forecast_steps") or []
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise SystemExit("Layer has no valid bbox_wgs84.")
    if not steps:
        raise SystemExit("Layer has no forecast_steps.")

    rows: list[dict[str, Any]] = []
    for step in steps:
        shape = step.get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            continue
        grid_rows, grid_cols = int(shape[0]), int(shape[1])
        for spot in spots:
            lat = float(spot["latitude"])
            lon = float(spot["longitude"])
            x, y, inside = grid_position(lat, lon, bbox, grid_rows, grid_cols)
            speed = sample_grid(step.get("speed_ms"), x, y, sample_method) if inside else None
            u_ms = sample_grid(step.get("u_ms"), x, y, sample_method) if inside else None
            v_ms = sample_grid(step.get("v_ms"), x, y, sample_method) if inside else None
            gust = sample_grid(step.get("gust_speed_ms"), x, y, sample_method) if inside else None
            rows.append({
                "format": "corsewind.ml_model_spot_sample.v1",
                "source": source,
                "product": layer.get("product"),
                "model_label": layer.get("model_label"),
                "run_time_utc": layer.get("run_time_utc"),
                "generated_at_utc": layer.get("generated_at_utc"),
                "valid_time_utc": step.get("valid_time_utc"),
                "lead_hour": step.get("lead_hour"),
                "lead_minutes": infer_lead_minutes(step),
                "spot_id": spot.get("spot_id"),
                "spot_name": spot.get("name"),
                "spot_kind": spot.get("kind"),
                "spot_source_type": spot.get("source_type"),
                "station_id": spot.get("station_id"),
                "latitude": lat,
                "longitude": lon,
                "use_for_ml": bool(spot.get("use_for_ml", False)),
                "sample_method": sample_method,
                "grid_x": round(x, 4),
                "grid_y": round(y, 4),
                "inside_grid": inside,
                "wind_speed_ms": speed,
                "wind_u_ms": u_ms,
                "wind_v_ms": v_ms,
                "wind_direction_deg": wind_direction_deg(u_ms, v_ms),
                "gust_speed_ms": gust,
                "grid": layer.get("grid"),
                "bbox_wgs84": bbox,
                "sampled_at_utc": utc_now(),
            })
    return rows


def output_path(output_root: Path, source: str, valid_time: str | None) -> Path:
    day = (valid_time or utc_now())[:10]
    return output_root / f"source={source}" / f"date={day}" / "samples.jsonl"


def write_jsonl_by_valid_day(output_root: Path, source: str, rows: list[dict[str, Any]]) -> dict[str, int]:
    by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_path[output_path(output_root, source, row.get("valid_time_utc"))].append(row)
    written: dict[str, int] = {}
    for path, path_rows in by_path.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        deduped = {
            (row.get("source"), row.get("run_time_utc"), row.get("valid_time_utc"), row.get("spot_id")): row
            for row in path_rows
        }
        ordered = sorted(
            deduped.values(),
            key=lambda row: (row.get("valid_time_utc") or "", row.get("spot_id") or ""),
        )
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")
        tmp.replace(path)
        written[str(path)] = len(ordered)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=["arome", "aromepi", "moloch", "icon2i"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-method", choices=["bilinear", "nearest"], default="bilinear")
    parser.add_argument("--include-context-spots", action="store_true", help="Include spots with use_for_ml=false.")
    parser.add_argument("--spot-id", action="append", default=[], help="Sample only specific spot ids. Repeatable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    registry_path = resolve_path(args.registry)
    output_root = resolve_path(args.output_root)
    layer = read_json(input_path)
    spots = load_spots(registry_path, args.include_context_spots, set(args.spot_id))
    rows = sample_layer(args.source, layer, spots, args.sample_method)
    written = write_jsonl_by_valid_day(output_root, args.source, rows)
    print(json.dumps({
        "generated_at_utc": utc_now(),
        "source": args.source,
        "input": str(input_path),
        "registry": str(registry_path),
        "output_root": str(output_root),
        "sample_method": args.sample_method,
        "spot_count": len(spots),
        "row_count": len(rows),
        "written": written,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
