#!/usr/bin/env python3
"""Generate virtual Open-Meteo offset points around ML spots."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_OUTPUT = ROOT / "configs/ml_open_meteo_offset_spots.json"
DEFAULT_OFFSETS = "n10:0:10,e10:90:10,s10:180:10,w10:270:10"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def finite_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_offsets(value: str) -> list[dict[str, float | str]]:
    offsets = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3:
            raise SystemExit(f"Invalid offset '{item}'. Expected name:bearing_deg:distance_km.")
        name, bearing, distance = parts
        bearing_value = finite_float(bearing)
        distance_value = finite_float(distance)
        if not name or bearing_value is None or distance_value is None:
            raise SystemExit(f"Invalid offset '{item}'.")
        offsets.append({
            "name": name,
            "bearing_deg": bearing_value % 360.0,
            "distance_km": distance_value,
        })
    return offsets


def offset_spot_id(base_spot_id: str, name: str) -> str:
    safe_name = "".join(char.lower() if char.isalnum() else "_" for char in str(name)).strip("_")
    return f"{base_spot_id}__nwp_offset_{safe_name}"


def destination_point(latitude: float, longitude: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    radius_km = 6371.0088
    angular = distance_km / radius_km
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(latitude)
    lon1 = math.radians(longitude)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular)
        + math.cos(lat1) * math.sin(angular) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(lat1),
        math.cos(angular) - math.sin(lat1) * math.sin(lat2),
    )
    return round(math.degrees(lat2), 6), round(((math.degrees(lon2) + 540.0) % 360.0) - 180.0, 6)


def load_spots(path: Path, include_context: bool) -> list[dict[str, Any]]:
    payload = read_json(path)
    spots = payload.get("spots", []) if isinstance(payload, dict) else payload
    out = []
    for spot in spots:
        if not isinstance(spot, dict) or not spot.get("spot_id"):
            continue
        if not include_context and not spot.get("use_for_ml", False):
            continue
        if finite_float(spot.get("latitude")) is None or finite_float(spot.get("longitude")) is None:
            continue
        out.append(spot)
    return out


def build_registry(spots: list[dict[str, Any]], offsets: list[dict[str, float | str]]) -> dict[str, Any]:
    generated = []
    for spot in spots:
        base_id = str(spot["spot_id"])
        base_lat = finite_float(spot.get("latitude"))
        base_lon = finite_float(spot.get("longitude"))
        if base_lat is None or base_lon is None:
            continue
        for offset in offsets:
            name = str(offset["name"])
            bearing = float(offset["bearing_deg"])
            distance = float(offset["distance_km"])
            lat, lon = destination_point(base_lat, base_lon, bearing, distance)
            generated.append({
                "spot_id": offset_spot_id(base_id, name),
                "name": f"{spot.get('name') or base_id} NWP offset {name}",
                "kind": "nwp_offset",
                "source_type": "open_meteo_offset",
                "base_spot_id": base_id,
                "offset_name": name,
                "offset_bearing_deg": round(bearing, 6),
                "offset_distance_km": round(distance, 6),
                "latitude": lat,
                "longitude": lon,
                "use_for_ml": False,
            })
    return {
        "format": "corsewind.open_meteo_offset_registry.v1",
        "generated_at_utc": utc_now(),
        "base_spot_count": len(spots),
        "offset_count_per_spot": len(offsets),
        "spot_count": len(generated),
        "offsets": offsets,
        "spots": generated,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offsets", default=DEFAULT_OFFSETS, help="Comma-separated name:bearing_deg:distance_km values.")
    parser.add_argument("--include-context-spots", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spots = load_spots(resolve_path(args.registry), args.include_context_spots)
    offsets = parse_offsets(args.offsets)
    registry = build_registry(spots, offsets)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "base_spot_count": registry["base_spot_count"],
        "spot_count": registry["spot_count"],
        "offsets": offsets,
    }, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
