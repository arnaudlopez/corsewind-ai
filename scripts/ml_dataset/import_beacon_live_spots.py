#!/usr/bin/env python3
"""Import Beacon Live spots/stations into a CorseWind ML registry."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BEACON_ROOT = Path("/Users/arnaud/Documents/beacon-live-app")
DEFAULT_OUTPUT = ROOT / "configs/ml_spots.json"


SOURCE_ID_OVERRIDES = {
    "la_tonnara": "windsup_tonnara",
    "owm-1202": "pioupiou_1202",
}


SOURCE_RESOLUTION_MINUTES = {
    "meteofrance": 6,
    "windsup": 6,
    "wunderground": 15,
    "esurfmar": 60,
    "owm": 15,
    "candhis": 60,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_beacon_config(beacon_root: Path) -> dict[str, Any]:
    config_path = beacon_root / "src/config/sources.js"
    if not config_path.exists():
        raise SystemExit(f"Beacon Live sources config not found: {config_path}")
    script = """
      import { SOURCES, CANDHIS_STATIONS } from './src/config/sources.js';
      console.log(JSON.stringify({ sources: SOURCES, candhisStations: CANDHIS_STATIONS }));
    """
    proc = subprocess.run(
        ("node", "--input-type=module", "-e", script),
        cwd=beacon_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"Failed to import Beacon Live sources:\n{proc.stderr}")
    return json.loads(proc.stdout)


def source_id_for(source: dict[str, Any]) -> str:
    source_type = source["type"]
    station_id = source.get("stationId")
    source_id = source["id"]
    if source_id in SOURCE_ID_OVERRIDES:
        return SOURCE_ID_OVERRIDES[source_id]
    if source_type == "meteofrance":
        return f"meteofrance_{station_id}"
    if source_type == "windsup":
        return f"windsup_{source_id}"
    if source_type == "wunderground":
        return f"wunderground_{station_id}"
    if source_type == "esurfmar":
        return f"esurfmar_{station_id}"
    if source_type == "owm":
        return f"pioupiou_{source.get('pioupiouId') or station_id}"
    return f"{source_type}_{station_id or source_id}"


def station_record(source: dict[str, Any]) -> dict[str, Any]:
    lat, lon = source["coords"]
    source_type = source["type"]
    return {
        "spot_id": source["id"],
        "name": source["name"],
        "kind": "wind_observation",
        "beacon_source_id": source_id_for(source),
        "beacon_app_id": source["id"],
        "source_type": source_type,
        "station_id": source.get("stationId") or source.get("pioupiouId"),
        "latitude": float(lat),
        "longitude": float(lon),
        "source_resolution_minutes": SOURCE_RESOLUTION_MINUTES.get(source_type),
        "use_for_ml": source_type in {"meteofrance", "windsup", "wunderground", "owm"},
        "notes": None,
    }


def candhis_record(key: str, source: dict[str, Any]) -> dict[str, Any]:
    lat, lon = source["coords"]
    return {
        "spot_id": key,
        "name": source["name"],
        "kind": "marine_observation",
        "beacon_source_id": f"candhis_{key}",
        "beacon_app_id": key,
        "source_type": "candhis",
        "station_id": source.get("code") or source.get("id"),
        "candhis_campaign": source.get("id"),
        "latitude": float(lat),
        "longitude": float(lon),
        "source_resolution_minutes": SOURCE_RESOLUTION_MINUTES["candhis"],
        "use_for_ml": False,
        "notes": "Marine/surf observation; useful as contextual feature, not direct wind target by default.",
    }


def build_registry(beacon_root: Path) -> dict[str, Any]:
    config = read_beacon_config(beacon_root)
    spots = [station_record(source) for source in config["sources"]]
    spots.extend(
        candhis_record(key, value)
        for key, value in sorted(config["candhisStations"].items())
    )
    spots = sorted(spots, key=lambda item: (item["kind"], item["spot_id"]))
    return {
        "format": "corsewind.ml_spot_registry.v1",
        "generated_at_utc": utc_now(),
        "source_project": "beacon-live-app",
        "source_root": str(beacon_root),
        "source_config": str(beacon_root / "src/config/sources.js"),
        "count": len(spots),
        "spots": spots,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beacon-root", type=Path, default=DEFAULT_BEACON_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = build_registry(args.beacon_root)
    write_json(args.output, registry)
    print(json.dumps({"output": str(args.output), "count": registry["count"]}, indent=2))


if __name__ == "__main__":
    main()
