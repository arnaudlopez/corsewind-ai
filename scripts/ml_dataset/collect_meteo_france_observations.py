#!/usr/bin/env python3
"""Collect Météo-France observations for the CorseWind ML dataset."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests import RequestException


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_OUTPUT_ROOT = DEFAULT_ML_ROOT / "observations/meteo_france"
DPOBS_BASE = "https://public-api.meteofrance.fr/public/DPObs/v2"
DPPAQUET_BASE = "https://public-api.meteofrance.fr/public/DPPaquetObs/v2"
DEFAULT_SYNOPTIC_CORSICA_IDS = "07753,07754,07761,07765,07770,07775,07780,07785,07790"
DEFAULT_BUOY_IDS = "6101031"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def api_key() -> str:
    token = os.environ.get("METEOFRANCE_API_KEY") or os.environ.get("METEOFRANCE_TOKEN")
    if not token:
        raise SystemExit("Missing METEOFRANCE_API_KEY. Add it to .env or export it in the shell.")
    return token


def auth_headers() -> dict[str, str]:
    return {"apikey": api_key(), "User-Agent": "CorseWind.ai Météo-France observation collector"}


def request_json(url: str, params: dict[str, str]) -> Any:
    max_attempts = int(os.environ.get("METEOFRANCE_MAX_ATTEMPTS", "5"))
    response: requests.Response | None = None
    last_error: RequestException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, params=params, headers=auth_headers(), timeout=90)
        except RequestException as exc:
            last_error = exc
            if attempt >= max_attempts:
                raise SystemExit(f"Request failed after {attempt} attempt(s): {exc}") from exc
            time.sleep(min(30.0, 2.0 * attempt))
            continue
        if response.status_code not in {429, 500, 502, 503, 504} or attempt >= max_attempts:
            break
        time.sleep(min(60.0, 3.0 * attempt))
    if response is None:
        raise SystemExit(f"Request failed before response: {last_error}")
    if response.status_code >= 400:
        raise SystemExit(
            f"Météo-France API returned HTTP {response.status_code} for {url}\n"
            f"Response preview: {response.text[:600]}"
        )
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Expected JSON from {url}, got {response.headers.get('content-type')}: {exc}") from exc


def finite_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def kelvin_to_c(value: Any) -> float | None:
    number = finite_float(value)
    if number is None:
        return None
    if number > 150:
        number -= 273.15
    return round(number, 3)


def pa_to_hpa(value: Any) -> float | None:
    number = finite_float(value)
    if number is None:
        return None
    if number > 2000:
        number /= 100.0
    return round(number, 3)


def parse_time(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        return str(value)


def first_present(*values: Any) -> Any:
    for value in values:
        if value not in {None, ""}:
            return value
    return None


def load_registry(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_station: dict[str, dict[str, Any]] = {}
    by_spot: dict[str, dict[str, Any]] = {}
    for item in payload.get("spots", []):
        if item.get("station_id"):
            by_station[str(item["station_id"])] = item
        if item.get("spot_id"):
            by_spot[str(item["spot_id"])] = item
    return by_station, by_spot


def normalize_row(row: dict[str, Any], source_dataset: str, station_registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    station_id = str(
        row.get("geo_id_insee")
        or row.get("geo_id_wmo")
        or row.get("geo_id_wigos")
        or row.get("id_station")
        or row.get("id_bouee")
        or ""
    )
    station = station_registry.get(station_id, {})
    valid_time = parse_time(row.get("validity_time") or row.get("reference_time") or row.get("date"))
    return {
        "format": "corsewind.ml_observation.v1",
        "source_project": "meteo_france_public_api",
        "source_dataset": source_dataset,
        "source_id": station_id or None,
        "spot_id": station.get("spot_id"),
        "station_id": station.get("station_id") or station_id or None,
        "source_type": "meteofrance",
        "timestamp_utc": valid_time,
        "reference_time_utc": parse_time(row.get("reference_time")),
        "insert_time_utc": parse_time(row.get("insert_time")),
        "latitude": finite_float(row.get("lat")) or station.get("latitude"),
        "longitude": finite_float(row.get("lon")) or station.get("longitude"),
        "wind_mean_ms": finite_float(row.get("ff")),
        "gust_ms": finite_float(first_present(row.get("raf10"), row.get("raf"), row.get("rafper"), row.get("fxy"))),
        "wind_direction_deg": finite_float(row.get("dd")),
        "gust_direction_deg": finite_float(first_present(row.get("ddraf10"), row.get("ddraf"), row.get("dxy"))),
        "temperature_c": kelvin_to_c(row.get("t")),
        "dewpoint_c": kelvin_to_c(row.get("td")),
        "humidity_pct": finite_float(row.get("u")),
        "pressure_hpa": pa_to_hpa(row.get("pres")),
        "sea_level_pressure_hpa": pa_to_hpa(row.get("pmer")),
        "precipitation_mm": finite_float(first_present(row.get("rr_per"), row.get("rr1"))),
        "precipitation_3h_mm": finite_float(row.get("rr3")),
        "precipitation_6h_mm": finite_float(row.get("rr6")),
        "precipitation_12h_mm": finite_float(row.get("rr12")),
        "precipitation_24h_mm": finite_float(row.get("rr24")),
        "visibility_m": finite_float(row.get("vv")),
        "cloud_cover_code": finite_float(row.get("n")),
        "sunshine_minutes": finite_float(row.get("insolh")),
        "global_radiation_raw": finite_float(row.get("ray_glo01")),
        "soil_temperature_10cm_c": kelvin_to_c(row.get("t_10")),
        "soil_temperature_20cm_c": kelvin_to_c(row.get("t_20")),
        "soil_temperature_50cm_c": kelvin_to_c(row.get("t_50")),
        "soil_temperature_100cm_c": kelvin_to_c(row.get("t_100")),
        "weather_code": row.get("ww"),
        "raw_units": {
            "ff": "m/s",
            "gust": "m/s",
            "dd": "deg",
            "t": "K",
            "td": "K",
            "pres": "Pa",
            "pmer": "Pa",
            "ray_glo01": "raw_api",
        },
        "raw": row,
    }


def rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("features", "data", "observations", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                if key == "features":
                    rows = []
                    for feature in value:
                        props = feature.get("properties") if isinstance(feature, dict) else None
                        if isinstance(props, dict):
                            rows.append(props)
                    return rows
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def station_ids_from_registry(station_registry: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        station_id
        for station_id, station in station_registry.items()
        if station.get("source_type") == "meteofrance" and station_id.isdigit()
    )


def collect_station_6m(station_registry: dict[str, dict[str, Any]], station_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for station_id in station_ids:
        payload = request_json(
            f"{DPOBS_BASE}/station/infrahoraire-6m",
            {"id_station": station_id, "format": "json"},
        )
        rows.extend(normalize_row(row, "dpobs_station_infrahoraire_6m", station_registry) for row in rows_from_payload(payload))
    return rows


def collect_station_hourly(station_registry: dict[str, dict[str, Any]], station_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for station_id in station_ids:
        payload = request_json(
            f"{DPOBS_BASE}/station/horaire",
            {"id_station": station_id, "format": "json"},
        )
        rows.extend(normalize_row(row, "dpobs_station_horaire", station_registry) for row in rows_from_payload(payload))
    return rows


def collect_synop(station_registry: dict[str, dict[str, Any]], synop_ids: str) -> list[dict[str, Any]]:
    payload = request_json(f"{DPOBS_BASE}/synop", {"format": "json", "id_station": synop_ids})
    return [normalize_row(row, "dpobs_synop", station_registry) for row in rows_from_payload(payload)]


def collect_buoys(station_registry: dict[str, dict[str, Any]], buoy_ids: str) -> list[dict[str, Any]]:
    payload = request_json(f"{DPOBS_BASE}/bouees", {"format": "json", "id_bouees": buoy_ids})
    return [normalize_row(row, "dpobs_bouees", station_registry) for row in rows_from_payload(payload)]


def collect_bulk_6m(station_registry: dict[str, dict[str, Any]], date: str) -> list[dict[str, Any]]:
    payload = request_json(
        f"{DPPAQUET_BASE}/paquet/stations/infrahoraire-6m",
        {"date": date, "format": "json"},
    )
    registry_ids = set(station_ids_from_registry(station_registry))
    return [
        normalize_row(row, "dppaquetobs_stations_infrahoraire_6m", station_registry)
        for row in rows_from_payload(payload)
        if not registry_ids or str(row.get("geo_id_insee")) in registry_ids
    ]


def output_path(output_root: Path, row: dict[str, Any]) -> Path:
    day = (row.get("timestamp_utc") or utc_now())[:10]
    dataset = row.get("source_dataset") or "unknown"
    return output_root / f"source_dataset={dataset}" / f"date={day}" / "observations.jsonl"


def write_jsonl_by_day(output_root: Path, rows: list[dict[str, Any]]) -> dict[str, int]:
    by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_path[output_path(output_root, row)].append(row)

    written: dict[str, int] = {}
    for path, path_rows in by_path.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_rows = []
        if path.exists():
            existing_rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        deduped = {
            (row.get("source_dataset"), row.get("source_id"), row.get("timestamp_utc")): row
            for row in [*existing_rows, *path_rows]
        }
        ordered = sorted(deduped.values(), key=lambda row: (row.get("timestamp_utc") or "", row.get("source_id") or ""))
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")
        written[str(path)] = len(ordered)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        action="append",
        choices=["station-6m", "station-hourly", "synop", "bouees", "bulk-6m"],
        default=[],
        help="Collection mode. Repeatable. Default: station-6m.",
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--station-id", action="append", default=[], help="Override registry station ids. Repeatable.")
    parser.add_argument("--synop-ids", default=DEFAULT_SYNOPTIC_CORSICA_IDS)
    parser.add_argument("--buoy-ids", default=DEFAULT_BUOY_IDS)
    parser.add_argument("--date", help="Required for bulk-6m, e.g. 2026-06-23T10:00:00Z.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    station_registry, _ = load_registry(args.registry)
    modes = args.mode or ["station-6m"]
    station_ids = args.station_id or station_ids_from_registry(station_registry)
    rows: list[dict[str, Any]] = []

    if "station-6m" in modes:
        rows.extend(collect_station_6m(station_registry, station_ids))
    if "station-hourly" in modes:
        rows.extend(collect_station_hourly(station_registry, station_ids))
    if "synop" in modes:
        rows.extend(collect_synop(station_registry, args.synop_ids))
    if "bouees" in modes:
        rows.extend(collect_buoys(station_registry, args.buoy_ids))
    if "bulk-6m" in modes:
        if not args.date:
            raise SystemExit("--date is required with --mode bulk-6m")
        rows.extend(collect_bulk_6m(station_registry, args.date))

    rows = [row for row in rows if row.get("timestamp_utc")]
    written = write_jsonl_by_day(args.output_root, rows)
    print(json.dumps({
        "generated_at_utc": utc_now(),
        "modes": modes,
        "registry": str(args.registry),
        "station_ids": station_ids,
        "row_count": len(rows),
        "written": written,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
