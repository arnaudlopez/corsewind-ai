#!/usr/bin/env python3
"""Backfill Open-Meteo previous model runs at ML spots."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from requests import RequestException


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_ML_ROOT / "open_meteo/previous_runs"
API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
DEFAULT_MODEL = "best_match"
DEFAULT_BASE_VARIABLES = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "precipitation",
    "rain",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def load_spots(path: Path, include_context: bool, selected_ids: set[str]) -> list[dict[str, Any]]:
    payload = read_json(path)
    spots = payload.get("spots", []) if isinstance(payload, dict) else payload
    out = []
    for spot in spots:
        if not isinstance(spot, dict) or not spot.get("spot_id"):
            continue
        if selected_ids and str(spot["spot_id"]) not in selected_ids:
            continue
        if not include_context and not spot.get("use_for_ml", False):
            continue
        if finite_float(spot.get("latitude")) is None or finite_float(spot.get("longitude")) is None:
            continue
        out.append(spot)
    return out


def date_chunks(start: date, end: date, max_days: int) -> list[tuple[date, date]]:
    chunks = []
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=max_days - 1))
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def previous_run_variables(base_variables: list[str], lead_days: list[int]) -> list[str]:
    return [f"{variable}_previous_day{lead_day}" for variable in base_variables for lead_day in lead_days]


def day_output_path(output_root: Path, model: str, valid_time: str) -> Path:
    return output_root / f"model={model}" / f"date={valid_time[:10]}" / "previous_runs.jsonl"


def request_with_retries(url: str, params: dict[str, Any], timeout: int, attempts: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < attempts:
                time.sleep(min(30, attempt * 3))
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"Open-Meteo HTTP {response.status_code}: {response.text[:500]}")
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(f"Open-Meteo error: {payload}")
            return payload
        except (RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(30, attempt * 3))
                continue
            break
    raise RuntimeError(f"Open-Meteo request failed after {attempts} attempts: {last_error}")


def fetch_spot_chunk(
    spot: dict[str, Any],
    model: str,
    variables: list[str],
    start: date,
    end: date,
    timeout: int,
    attempts: int,
) -> dict[str, Any]:
    params = {
        "latitude": spot["latitude"],
        "longitude": spot["longitude"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(variables),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "models": model,
    }
    return request_with_retries(API_URL, params, timeout, attempts)


def response_rows(
    payload: dict[str, Any],
    spot: dict[str, Any],
    model: str,
    base_variables: list[str],
    lead_days: list[int],
) -> list[dict[str, Any]]:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    units = payload.get("hourly_units") or {}
    rows = []
    for idx, timestamp in enumerate(times):
        valid_time = str(timestamp)
        if len(valid_time) == 16:
            valid_time = f"{valid_time}:00Z"
        elif not valid_time.endswith("Z"):
            valid_time = f"{valid_time}Z"
        for lead_day in lead_days:
            values = {}
            for variable in base_variables:
                source_key = f"{variable}_previous_day{lead_day}"
                series = hourly.get(source_key)
                value = series[idx] if isinstance(series, list) and idx < len(series) else None
                values[variable] = finite_float(value)
            rows.append({
                "format": "corsewind.open_meteo_previous_runs.v1",
                "source": "open_meteo",
                "source_dataset": "previous_runs",
                "model": model,
                "valid_time_utc": valid_time,
                "lead_days": lead_day,
                "nominal_forecast_age_hours": lead_day * 24,
                "spot_id": spot.get("spot_id"),
                "spot_name": spot.get("name"),
                "spot_kind": spot.get("kind"),
                "spot_source_type": spot.get("source_type"),
                "station_id": spot.get("station_id"),
                "latitude": finite_float(spot.get("latitude")),
                "longitude": finite_float(spot.get("longitude")),
                "use_for_ml": bool(spot.get("use_for_ml", False)),
                "api_latitude": finite_float(payload.get("latitude")),
                "api_longitude": finite_float(payload.get("longitude")),
                "api_elevation_m": finite_float(payload.get("elevation")),
                "hourly_units": units,
                "features": values,
                "non_null_feature_count": sum(value is not None for value in values.values()),
                "fetched_at_utc": utc_now(),
            })
    return rows


def write_jsonl_by_day(output_root: Path, model: str, rows: list[dict[str, Any]]) -> dict[str, int]:
    by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_path[day_output_path(output_root, model, row.get("valid_time_utc") or utc_now())].append(row)
    written = {}
    for path, path_rows in by_path.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = iter_jsonl(path)
        deduped = {
            (row.get("model"), row.get("spot_id"), row.get("valid_time_utc"), row.get("lead_days")): row
            for row in [*existing, *path_rows]
        }
        ordered = sorted(
            deduped.values(),
            key=lambda row: (row.get("valid_time_utc") or "", row.get("spot_id") or "", row.get("lead_days") or 0),
        )
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")
        tmp.replace(path)
        written[str(path)] = len(ordered)
    return written


def summarize_written(written: dict[str, int]) -> dict[str, Any]:
    return {
        "file_count": len(written),
        "row_refs": sum(written.values()),
        "sample": dict(list(written.items())[:10]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--variables", default=",".join(DEFAULT_BASE_VARIABLES), help="Comma-separated base variable names without _previous_day suffix.")
    parser.add_argument("--lead-days", default="1,2,3,4,5,6,7", help="Comma-separated previous_day offsets.")
    parser.add_argument("--spot-id", action="append", default=[])
    parser.add_argument("--include-context-spots", action="store_true")
    parser.add_argument("--max-days-per-request", type=int, default=31)
    parser.add_argument("--request-sleep-sec", type=float, default=0.5)
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    if end < start:
        raise SystemExit("--end-date must be after --start-date")
    base_variables = [item.strip() for item in args.variables.split(",") if item.strip()]
    lead_days = [int(item.strip()) for item in args.lead_days.split(",") if item.strip()]
    variables = previous_run_variables(base_variables, lead_days)
    spots = load_spots(resolve_path(args.registry), args.include_context_spots, set(args.spot_id))
    chunks = date_chunks(start, end, args.max_days_per_request)
    output_root = resolve_path(args.output_root)
    plan = {
        "generated_at_utc": utc_now(),
        "source": "open_meteo_previous_runs",
        "model": args.model,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "spot_count": len(spots),
        "chunk_count_per_spot": len(chunks),
        "base_variables": base_variables,
        "lead_days": lead_days,
        "output_root": str(output_root),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps({**plan, "spots": [spot.get("spot_id") for spot in spots]}, indent=2, ensure_ascii=False))
        return

    total_rows = 0
    written: dict[str, int] = {}
    errors = []
    for spot in spots:
        for chunk_start, chunk_end in chunks:
            try:
                payload = fetch_spot_chunk(spot, args.model, variables, chunk_start, chunk_end, args.timeout_sec, args.attempts)
                rows = response_rows(payload, spot, args.model, base_variables, lead_days)
                total_rows += len(rows)
                written.update(write_jsonl_by_day(output_root, args.model, rows))
            except Exception as exc:  # noqa: BLE001
                errors.append({
                    "spot_id": spot.get("spot_id"),
                    "start_date": chunk_start.isoformat(),
                    "end_date": chunk_end.isoformat(),
                    "error": str(exc),
                })
            time.sleep(args.request_sleep_sec)
    print(json.dumps({**plan, "row_count": total_rows, "written": summarize_written(written), "errors": errors}, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
