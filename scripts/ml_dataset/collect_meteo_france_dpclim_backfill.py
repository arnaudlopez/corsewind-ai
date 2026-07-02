#!/usr/bin/env python3
"""Backfill Météo-France DPClim station climatology CSVs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from requests import RequestException


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_CONTEXT_REGISTRY = ROOT / "configs/ml_context_stations.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_ML_ROOT / "observations/meteo_france_climatology"
BASE_URL = "https://public-api.meteofrance.fr/public/DPClim/v1"
FREQUENCY_ENDPOINTS = {
    "6min": "infrahoraire-6m",
    "hourly": "horaire",
    "daily": "quotidienne",
}
FIELD_MAP = {
    "FF": "wind_mean_ms",
    "DD": "wind_direction_deg",
    "FXI": "gust_instant_ms",
    "DXI": "gust_instant_direction_deg",
    "FXY": "gust_max_ms",
    "DXY": "gust_max_direction_deg",
    "T": "temperature_c",
    "TD": "dewpoint_c",
    "U": "humidity_pct",
    "PSTAT": "pressure_station_hpa",
    "PMER": "sea_level_pressure_hpa",
    "GLO": "global_radiation_j_cm2",
    "DIR": "direct_radiation_j_cm2",
    "DIF": "diffuse_radiation_j_cm2",
    "INS": "sunshine_duration_minutes",
    "N": "cloud_cover_octa",
    "NBAS": "low_cloud_cover_octa",
    "VV": "visibility_m",
    "RR1": "precipitation_1h_mm",
    "TMER": "sea_temperature_c",
    "ETATMER": "sea_state_code",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.strip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def api_key() -> str:
    token = os.environ.get("METEOFRANCE_API_KEY") or os.environ.get("METEOFRANCE_TOKEN")
    if not token:
        raise SystemExit("Missing METEOFRANCE_API_KEY or METEOFRANCE_TOKEN.")
    return token


def headers() -> dict[str, str]:
    return {"apikey": api_key(), "User-Agent": "CorseWind.ai DPClim backfill"}


def request_with_retries(url: str, params: dict[str, Any], timeout: int, attempts: int = 5) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, headers=headers(), params=params, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < attempts:
                time.sleep(min(20, attempt * 2))
                continue
            return response
        except RequestException as exc:
            last_error = exc
            time.sleep(min(20, attempt * 2))
    raise RuntimeError(f"Request failed after {attempts} attempts: {last_error}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    spots = payload.get("spots", []) if isinstance(payload, dict) else payload
    return {
        str(spot.get("station_id")): spot
        for spot in spots
        if isinstance(spot, dict) and spot.get("station_id")
    }


def load_context_registry(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = read_json(path)
    stations = payload.get("stations", []) if isinstance(payload, dict) else payload
    out = {}
    for station in stations:
        if not isinstance(station, dict) or not station.get("station_id"):
            continue
        item = dict(station)
        item.setdefault("use_for_ml", bool(item.get("use_as_target", False)))
        item.setdefault("kind", "official_weather_context")
        item.setdefault("source_type", "meteofrance_dpclim")
        out[str(item["station_id"])] = item
    return out


def station_id_from_station(station: dict[str, Any]) -> str:
    for key in ("id", "station_id", "id_station", "Id_station", "ID_STATION", "POSTE", "poste", "code"):
        value = station.get(key)
        if value not in {None, ""}:
            return str(value).strip()
    return ""


def station_name_from_station(station: dict[str, Any]) -> str | None:
    for key in ("name", "nom", "Nom", "NOM", "libelle", "Libelle", "LIBELLE"):
        value = station.get(key)
        if value not in {None, ""}:
            return str(value).strip()
    return None


def station_coord_from_station(station: dict[str, Any], axis: str) -> float | None:
    keys = (
        ("latitude", "lat", "Latitude", "LAT") if axis == "latitude"
        else ("longitude", "lon", "lng", "Longitude", "LON")
    )
    for key in keys:
        value = parse_number(station.get(key))
        if value is not None:
            return value
    return None


def build_station_metadata(
    stations: list[dict[str, Any]],
    station_registry: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    metadata = {station_id: dict(station) for station_id, station in station_registry.items()}
    for station in stations:
        station_id = station_id_from_station(station)
        if not station_id:
            continue
        merged = {**station, **metadata.get(station_id, {})}
        name = station_name_from_station(merged)
        latitude = station_coord_from_station(merged, "latitude")
        longitude = station_coord_from_station(merged, "longitude")
        if name and not merged.get("name"):
            merged["name"] = name
        if latitude is not None and merged.get("latitude") is None:
            merged["latitude"] = latitude
        if longitude is not None and merged.get("longitude") is None:
            merged["longitude"] = longitude
        metadata[station_id] = merged
    return metadata


def list_stations(department: int, frequency: str, parametre: str | None, timeout: int) -> list[dict[str, Any]]:
    endpoint = FREQUENCY_ENDPOINTS[frequency]
    params: dict[str, Any] = {"id-departement": department}
    if parametre:
        params["parametre"] = parametre
    response = request_with_retries(f"{BASE_URL}/liste-stations/{endpoint}", params, timeout)
    if response.status_code != 200:
        raise RuntimeError(f"Station list HTTP {response.status_code}: {response.text[:500]}")
    payload = response.json()
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def order_csv(station_id: str, frequency: str, start: str, end: str, timeout: int) -> str:
    endpoint = FREQUENCY_ENDPOINTS[frequency]
    params = {
        "id-station": station_id,
        "date-deb-periode": start,
        "date-fin-periode": end,
    }
    response = request_with_retries(f"{BASE_URL}/commande-station/{endpoint}", params, timeout)
    if response.status_code != 202:
        raise RuntimeError(f"Order HTTP {response.status_code} for {station_id}: {response.text[:500]}")
    match = re.search(r"\d+", response.text)
    if not match:
        raise RuntimeError(f"No command id returned for {station_id}: {response.text[:500]}")
    return match.group(0)


def download_order(command_id: str, timeout: int, poll_sleep_sec: float, max_polls: int) -> str:
    for _ in range(max_polls):
        response = request_with_retries(f"{BASE_URL}/commande/fichier", {"id-cmde": command_id}, timeout)
        if response.status_code in {200, 201}:
            return response.content.decode("utf-8", errors="replace")
        if response.status_code == 204:
            time.sleep(poll_sleep_sec)
            continue
        raise RuntimeError(f"Download HTTP {response.status_code} for command {command_id}: {response.text[:500]}")
    raise RuntimeError(f"Command {command_id} not ready after {max_polls} polls")


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def parse_dpclim_time(value: str, frequency: str) -> str | None:
    text = str(value or "").strip()
    formats_by_length = {
        12: "%Y%m%d%H%M",
        10: "%Y%m%d%H",
        8: "%Y%m%d",
    }
    fmt = formats_by_length.get(len(text))
    if fmt is None:
        return None
    try:
        parsed = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        if frequency == "daily" and fmt == "%Y%m%d":
            parsed = parsed.replace(hour=12)
        return parsed.isoformat().replace("+00:00", "Z")
    except ValueError:
        return None
    return None


def normalize_csv(text: str, frequency: str, station_metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text), delimiter=";")
    rows = []
    for raw in reader:
        station_id = str(raw.get("POSTE") or "").strip()
        timestamp = parse_dpclim_time(str(raw.get("DATE") or ""), frequency)
        if not station_id or not timestamp:
            continue
        station = station_metadata.get(station_id, {})
        features = {target: parse_number(raw.get(source)) for source, target in FIELD_MAP.items()}
        quality = {
            source: parse_number(raw.get(f"Q{source}"))
            for source in FIELD_MAP
            if raw.get(f"Q{source}") not in {None, ""}
        }
        rows.append({
            "format": "corsewind.meteo_france_dpclim_observation.v1",
            "source_project": "meteo_france_public_api",
            "source_dataset": f"dpclim_station_{frequency}",
            "station_id": station_id,
            "spot_id": station.get("spot_id"),
            "spot_name": station.get("name"),
            "spot_kind": station.get("kind"),
            "spot_source_type": station.get("source_type"),
            "context_role": station.get("context_role"),
            "use_as_context": bool(station.get("use_as_context", False)),
            "nearest_ml_spot": station.get("nearest_ml_spot"),
            "nearest_any_spot": station.get("nearest_any_spot"),
            "timestamp_utc": timestamp,
            "latitude": parse_number(station.get("latitude")),
            "longitude": parse_number(station.get("longitude")),
            "altitude_m": parse_number(station.get("altitude_m") or station.get("alt")),
            "use_for_ml": bool(station.get("use_for_ml", False)),
            **features,
            "quality_flags": quality,
            "raw": raw,
            "normalized_at_utc": utc_now(),
        })
    return rows


def parse_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def align_to_step(value: datetime, step: timedelta, direction: str) -> datetime:
    midnight = value.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = value - midnight
    step_seconds = int(step.total_seconds())
    elapsed_seconds = int(elapsed.total_seconds())
    aligned_seconds = (elapsed_seconds // step_seconds) * step_seconds
    aligned = midnight + timedelta(seconds=aligned_seconds)
    if direction == "ceil" and aligned < value.replace(microsecond=0):
        aligned += step
    return aligned


def split_datetime_ranges(start: str, end: str, max_days: int, frequency: str) -> list[tuple[str, str]]:
    start_dt = parse_datetime(start)
    end_dt = parse_datetime(end)
    step = timedelta(minutes=6) if frequency == "6min" else timedelta(seconds=1)
    if frequency == "6min":
        start_dt = align_to_step(start_dt, step, "ceil")
        end_dt = align_to_step(end_dt, step, "floor")
    if end_dt < start_dt:
        raise ValueError(f"end datetime must be after start datetime: {start} > {end}")
    if max_days <= 0:
        return [(format_datetime(start_dt), format_datetime(end_dt))]
    ranges = []
    cursor = start_dt
    while cursor <= end_dt:
        chunk_end = min(end_dt, cursor + timedelta(days=max_days) - step)
        ranges.append((format_datetime(cursor), format_datetime(chunk_end)))
        cursor = chunk_end + step
    return ranges


def raw_path(output_root: Path, frequency: str, station_id: str, start: str, end: str) -> Path:
    safe_start = start.replace(":", "").replace("-", "")
    safe_end = end.replace(":", "").replace("-", "")
    return output_root / "raw_csv" / f"frequency={frequency}" / f"station={station_id}" / f"{safe_start}_{safe_end}.csv"


def normalized_path(output_root: Path, frequency: str, timestamp: str) -> Path:
    return output_root / "normalized" / f"frequency={frequency}" / f"date={timestamp[:10]}" / "observations.jsonl"


def write_outputs(output_root: Path, frequency: str, station_id: str, start: str, end: str, csv_text: str, rows: list[dict[str, Any]]) -> dict[str, int]:
    csv_path = raw_path(output_root, frequency, station_id, start, end)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(csv_text, encoding="utf-8")
    by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_path[normalized_path(output_root, frequency, row["timestamp_utc"])].append(row)
    written = {"raw_csv": 1}
    for path, path_rows in by_path.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if path.exists():
            existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        deduped = {
            (row.get("source_dataset"), row.get("station_id"), row.get("timestamp_utc")): row
            for row in [*existing, *path_rows]
        }
        ordered = sorted(deduped.values(), key=lambda row: (row.get("timestamp_utc") or "", row.get("station_id") or ""))
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")
        tmp.replace(path)
        written[str(path)] = len(ordered)
    return written


def summarize_written(written: dict[str, int]) -> dict[str, Any]:
    path_items = [(path, count) for path, count in written.items() if path != "raw_csv"]
    return {
        "raw_csv_batches": written.get("raw_csv", 0),
        "normalized_file_count": len(path_items),
        "normalized_row_refs": sum(path_items_count for _, path_items_count in path_items),
        "sample": dict(path_items[:10]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--context-registry", type=Path, default=DEFAULT_CONTEXT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--frequency", choices=sorted(FREQUENCY_ENDPOINTS), default="hourly")
    parser.add_argument("--department", type=int, default=20)
    parser.add_argument("--parametre", default="vent")
    parser.add_argument("--station-id", action="append", default=[])
    parser.add_argument("--max-stations", type=int)
    parser.add_argument("--start-datetime", required=True, help="ISO UTC start datetime accepted by DPClim.")
    parser.add_argument("--end-datetime", required=True, help="ISO UTC end datetime accepted by DPClim.")
    parser.add_argument("--max-days-per-order", type=int, help="Split long periods into several DPClim orders. Defaults to 7 for 6min, 31 otherwise.")
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--poll-sleep-sec", type=float, default=2)
    parser.add_argument("--max-polls", type=int, default=30)
    parser.add_argument("--request-sleep-sec", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    station_registry = load_registry(resolve_path(args.registry))
    context_registry = load_context_registry(resolve_path(args.context_registry))
    station_registry = {
        station_id: {**context_station, **station_registry.get(station_id, {})}
        for station_id, context_station in context_registry.items()
    } | {
        station_id: station
        for station_id, station in station_registry.items()
        if station_id not in context_registry
    }
    if args.station_id:
        stations = [{"id": station_id, **station_registry.get(station_id, {})} for station_id in args.station_id]
    else:
        stations = list_stations(args.department, args.frequency, args.parametre or None, args.timeout_sec)
    if args.max_stations is not None:
        stations = stations[: args.max_stations]
    station_metadata = build_station_metadata(stations, station_registry)
    max_days_per_order = args.max_days_per_order if args.max_days_per_order is not None else (7 if args.frequency == "6min" else 31)
    datetime_ranges = split_datetime_ranges(args.start_datetime, args.end_datetime, max_days_per_order, args.frequency)
    output_root = resolve_path(args.output_root)
    plan = {
        "generated_at_utc": utc_now(),
        "source": "meteo_france_dpclim",
        "frequency": args.frequency,
        "parametre": args.parametre,
        "start_datetime": args.start_datetime,
        "end_datetime": args.end_datetime,
        "max_days_per_order": max_days_per_order,
        "order_chunk_count_per_station": len(datetime_ranges),
        "station_count": len(stations),
        "output_root": str(output_root),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps({**plan, "stations": stations[:20], "datetime_ranges": datetime_ranges[:20]}, indent=2, ensure_ascii=False))
        return

    total_rows = 0
    written: dict[str, int] = {}
    errors = []
    for station in stations:
        station_id = station_id_from_station(station)
        for chunk_start, chunk_end in datetime_ranges:
            try:
                command_id = order_csv(station_id, args.frequency, chunk_start, chunk_end, args.timeout_sec)
                csv_text = download_order(command_id, args.timeout_sec, args.poll_sleep_sec, args.max_polls)
                rows = normalize_csv(csv_text, args.frequency, station_metadata)
                total_rows += len(rows)
                write_result = write_outputs(output_root, args.frequency, station_id, chunk_start, chunk_end, csv_text, rows)
                written["raw_csv"] = written.get("raw_csv", 0) + write_result.pop("raw_csv", 0)
                written.update(write_result)
            except Exception as exc:  # noqa: BLE001
                errors.append({"station_id": station_id, "start_datetime": chunk_start, "end_datetime": chunk_end, "error": str(exc)})
            time.sleep(args.request_sleep_sec)
    print(json.dumps({**plan, "row_count": total_rows, "written": summarize_written(written), "errors": errors}, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
