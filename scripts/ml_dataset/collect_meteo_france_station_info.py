#!/usr/bin/env python3
"""Backfill Météo-France DPClim station metadata and parameter history."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests import RequestException


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_CONTEXT_REGISTRY = ROOT / "configs/ml_context_stations.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_ML_ROOT / "observations/meteo_france_climatology/station_info"
BASE_URL = "https://public-api.meteofrance.fr/public/DPClim/v1"

PARAMETER_GROUP_KEYWORDS = {
    "wind": ("VENT", "RAFALE"),
    "temperature": ("TEMPERATURE", "TEMPÉRATURE", "TN", "TX"),
    "humidity": ("HUMIDITE", "HUMIDITÉ", "HUM"),
    "pressure": ("PRESSION",),
    "radiation": ("RAYONNEMENT", "INSOLATION", "GLOBAL", "DIRECT", "DIFFUS"),
    "cloud": ("NUAGE", "NEBULOSITE", "NÉBULOSITÉ"),
    "precipitation": ("PRECIPITATION", "PRÉCIPITATION", "PLUIE", "HAUTEUR"),
    "visibility": ("VISIBILITE", "VISIBILITÉ"),
    "sea": ("MER", "HOULE", "VAGUE"),
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
    return {"apikey": api_key(), "User-Agent": "CorseWind.ai DPClim station info backfill"}


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
        if isinstance(spot, dict) and spot.get("station_id") and spot.get("source_type") == "meteofrance"
    }


def load_context_registry(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = read_json(path)
    stations = payload.get("stations", []) if isinstance(payload, dict) else payload
    return {
        str(station.get("station_id")): station
        for station in stations
        if isinstance(station, dict) and station.get("station_id")
    }


def fetch_station_info(station_id: str, timeout: int) -> list[dict[str, Any]]:
    response = request_with_retries(f"{BASE_URL}/information-station", {"id-station": station_id}, timeout)
    if response.status_code != 200:
        raise RuntimeError(f"Station info HTTP {response.status_code} for {station_id}: {response.text[:500]}")
    payload = response.json()
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def parameter_groups(param_names: list[str]) -> dict[str, bool]:
    normalized = [name.upper() for name in param_names]
    groups = {}
    for group, keywords in PARAMETER_GROUP_KEYWORDS.items():
        groups[group] = any(any(keyword in name for keyword in keywords) for name in normalized)
    return groups


def summarize_station_info(station_id: str, payload: list[dict[str, Any]], registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    station = payload[0] if payload else {}
    spot = registry.get(station_id, {})
    parametres = []
    for item in payload:
        parametres.extend(item.get("parametres") or [])
    param_names = sorted({
        str(param.get("nom")).strip()
        for param in parametres
        if isinstance(param, dict) and param.get("nom")
    })
    type_periods = []
    for item in payload:
        type_periods.extend(item.get("typesPoste") or [])
    return {
        "format": "corsewind.meteo_france_dpclim_station_info.v1",
        "source_project": "meteo_france_public_api",
        "source_dataset": "dpclim_station_info",
        "station_id": station_id,
        "meteo_france_name": station.get("nom"),
        "meteo_france_place": station.get("lieuDit"),
        "basin": station.get("bassin"),
        "station_start": station.get("dateDebut"),
        "station_end": station.get("dateFin"),
        "spot_id": spot.get("spot_id"),
        "spot_name": spot.get("name"),
        "spot_kind": spot.get("kind"),
        "latitude": spot.get("latitude"),
        "longitude": spot.get("longitude"),
        "altitude_m": spot.get("altitude_m") or spot.get("alt"),
        "context_role": spot.get("context_role"),
        "use_as_context": bool(spot.get("use_as_context", False)),
        "nearest_ml_spot": spot.get("nearest_ml_spot"),
        "nearest_any_spot": spot.get("nearest_any_spot"),
        "use_for_ml": bool(spot.get("use_for_ml", False)),
        "type_period_count": len(type_periods),
        "type_periods": type_periods,
        "parameter_count": len(param_names),
        "parameter_names": param_names,
        "parameter_groups": parameter_groups(param_names),
        "raw_record_count": len(payload),
        "fetched_at_utc": utc_now(),
    }


def write_outputs(output_root: Path, station_id: str, payload: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, str]:
    raw_path = output_root / "raw" / f"station={station_id}" / "information_station.json"
    summary_path = output_root / "normalized" / f"station={station_id}" / "station_info.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"raw": str(raw_path), "summary": str(summary_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--context-registry", type=Path, default=DEFAULT_CONTEXT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--station-id", action="append", default=[])
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--request-sleep-sec", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    registry = load_registry(resolve_path(args.registry))
    context_registry = load_context_registry(resolve_path(args.context_registry))
    registry = {
        station_id: {**context_station, **registry.get(station_id, {})}
        for station_id, context_station in context_registry.items()
    } | {
        station_id: station
        for station_id, station in registry.items()
        if station_id not in context_registry
    }
    station_ids = args.station_id or sorted(registry)
    output_root = resolve_path(args.output_root)
    plan = {
        "generated_at_utc": utc_now(),
        "source": "meteo_france_dpclim_station_info",
        "station_count": len(station_ids),
        "station_ids": station_ids,
        "output_root": str(output_root),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    written = {}
    summaries = []
    errors = []
    for station_id in station_ids:
        try:
            payload = fetch_station_info(station_id, args.timeout_sec)
            summary = summarize_station_info(station_id, payload, registry)
            summaries.append(summary)
            written[station_id] = write_outputs(output_root, station_id, payload, summary)
        except Exception as exc:  # noqa: BLE001
            errors.append({"station_id": station_id, "error": str(exc)})
        time.sleep(args.request_sleep_sec)

    index_path = output_root / "normalized" / "station_info_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in summaries), encoding="utf-8")
    print(json.dumps({
        **plan,
        "summary_count": len(summaries),
        "index_path": str(index_path),
        "written_station_count": len(written),
        "sample": dict(list(written.items())[:3]),
        "errors": errors,
    }, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
