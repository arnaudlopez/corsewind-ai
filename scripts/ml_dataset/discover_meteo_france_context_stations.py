#!/usr/bin/env python3
"""Discover Météo-France DPClim contextual wind stations around ML spots."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests import RequestException


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPOTS = ROOT / "configs/ml_spots.json"
DEFAULT_OUTPUT = ROOT / "configs/ml_context_stations.json"
BASE_URL = "https://public-api.meteofrance.fr/public/DPClim/v1"
PARAMETER_GROUP_KEYWORDS = {
    "wind": ("VENT", "RAFALE"),
    "temperature": ("TEMPERATURE", "TEMPÉRATURE", "TN", "TX"),
    "humidity": ("HUMIDITE", "HUMIDITÉ", "HUM"),
    "pressure": ("PRESSION", "PRESS", "PSTAT", "PMER", "BAROMET"),
    "radiation": ("RAYONNEMENT", "INSOLATION", "GLOBAL", "DIRECT", "DIFFUS"),
    "cloud": ("NUAGE", "NEBULOSITE", "NÉBULOSITÉ"),
    "precipitation": ("PRECIPITATION", "PRÉCIPITATION", "PLUIE", "HAUTEUR"),
    "visibility": ("VISIBILITE", "VISIBILITÉ"),
    "sea": ("MER", "HOULE", "VAGUE"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
    return {"apikey": api_key(), "User-Agent": "CorseWind.ai DPClim station discovery"}


def request_json(path: str, params: dict[str, Any], timeout: int, attempts: int = 5) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(f"{BASE_URL}{path}", headers=headers(), params=params, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < attempts:
                time.sleep(min(20, attempt * 2))
                continue
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            return response.json()
        except (RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(20, attempt * 2))
                continue
            break
    raise RuntimeError(f"Request failed after {attempts} attempts: {last_error}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def nearest_spot(station: dict[str, Any], spots: list[dict[str, Any]]) -> dict[str, Any] | None:
    values = []
    for spot in spots:
        if spot.get("latitude") is None or spot.get("longitude") is None:
            continue
        values.append((
            haversine_km(float(station["lat"]), float(station["lon"]), float(spot["latitude"]), float(spot["longitude"])),
            spot,
        ))
    if not values:
        return None
    distance, spot = min(values, key=lambda item: item[0])
    return {
        "spot_id": spot.get("spot_id"),
        "spot_name": spot.get("name"),
        "spot_kind": spot.get("kind"),
        "distance_km": round(distance, 3),
    }


def parameter_groups(parameter_names: list[str]) -> dict[str, bool]:
    normalized = [name.upper() for name in parameter_names]
    groups = {}
    for group, keywords in PARAMETER_GROUP_KEYWORDS.items():
        groups[group] = any(any(keyword in name for keyword in keywords) for name in normalized)
    return groups


def infer_context_role(station: dict[str, Any], nearest_ml: dict[str, Any] | None, nearest_any: dict[str, Any] | None) -> str:
    altitude = float(station.get("alt") or 0)
    name = str(station.get("nom") or "").upper()
    distance_any = float((nearest_any or {}).get("distance_km") or 999)
    if "CAP" in name or distance_any <= 10 or altitude < 120:
        return "coastal_official_context"
    if altitude >= 700:
        return "mountain_relief_context"
    if altitude >= 250:
        return "inland_thermal_context"
    if nearest_ml and float(nearest_ml.get("distance_km") or 999) <= 20:
        return "nearby_official_context"
    return "regional_official_context"


def discover(args: argparse.Namespace) -> dict[str, Any]:
    spot_payload = read_json(args.spots)
    spots = spot_payload.get("spots", []) if isinstance(spot_payload, dict) else spot_payload
    all_spots = [spot for spot in spots if isinstance(spot, dict) and spot.get("latitude") is not None and spot.get("longitude") is not None]
    ml_spots = [spot for spot in all_spots if spot.get("use_for_ml")]
    target_station_ids = {
        str(spot.get("station_id"))
        for spot in spots
        if isinstance(spot, dict) and spot.get("source_type") == "meteofrance" and spot.get("station_id")
    }
    station_list = request_json(
        "/liste-stations/horaire",
        {"id-departement": args.department, "parametre": args.parametre},
        args.timeout_sec,
    )
    if not isinstance(station_list, list):
        raise RuntimeError(f"Unexpected station list payload: {type(station_list)}")

    stations = []
    errors = []
    for station in station_list:
        station_id = str(station.get("id") or "")
        if not station_id:
            continue
        try:
            info_payload = request_json("/information-station", {"id-station": station_id}, args.timeout_sec)
            records = info_payload if isinstance(info_payload, list) else [info_payload]
            parameter_names = sorted({
                str(parameter.get("nom")).strip()
                for record in records
                if isinstance(record, dict)
                for parameter in (record.get("parametres") or [])
                if isinstance(parameter, dict) and parameter.get("nom")
            })
            type_periods = [
                type_period
                for record in records
                if isinstance(record, dict)
                for type_period in (record.get("typesPoste") or [])
            ]
            starts = [record.get("dateDebut") for record in records if isinstance(record, dict) and record.get("dateDebut")]
            ends = [record.get("dateFin") for record in records if isinstance(record, dict) and record.get("dateFin")]
            groups = parameter_groups(parameter_names)
            nearest_ml = nearest_spot(station, ml_spots)
            nearest_any = nearest_spot(station, all_spots)
            stations.append({
                "station_id": station_id,
                "name": station.get("nom"),
                "source_type": "meteofrance_dpclim",
                "context_role": infer_context_role(station, nearest_ml, nearest_any),
                "latitude": station.get("lat"),
                "longitude": station.get("lon"),
                "altitude_m": station.get("alt"),
                "poste_ouvert": bool(station.get("posteOuvert")),
                "poste_public": bool(station.get("postePublic")),
                "type_poste": station.get("typePoste"),
                "already_target_station": station_id in target_station_ids,
                "use_as_target": station_id in target_station_ids,
                "use_as_context": True,
                "nearest_ml_spot": nearest_ml,
                "nearest_any_spot": nearest_any,
                "station_start": min(starts) if starts else None,
                "station_end": max(ends) if ends else None,
                "type_period_count": len(type_periods),
                "parameter_count": len(parameter_names),
                "parameter_groups": groups,
                "parameter_names": parameter_names,
            })
        except Exception as exc:  # noqa: BLE001
            errors.append({"station_id": station_id, "error": str(exc)})
        time.sleep(args.request_sleep_sec)

    return {
        "format": "corsewind.ml_context_station_registry.v1",
        "generated_at_utc": utc_now(),
        "provider": "meteo_france_dpclim",
        "department": args.department,
        "parametre": args.parametre,
        "source_endpoint": "/liste-stations/horaire",
        "count": len(stations),
        "target_station_count": sum(station["already_target_station"] for station in stations),
        "context_only_count": sum(not station["already_target_station"] for station in stations),
        "stations": sorted(stations, key=lambda item: (not item["already_target_station"], item["station_id"])),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--spots", type=Path, default=DEFAULT_SPOTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--department", type=int, default=20)
    parser.add_argument("--parametre", default="vent")
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--request-sleep-sec", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    payload = discover(args)
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "count": payload["count"],
        "target_station_count": payload["target_station_count"],
        "context_only_count": payload["context_only_count"],
        "errors": payload["errors"],
    }, indent=2, ensure_ascii=False))
    if payload["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
