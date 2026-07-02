#!/usr/bin/env python3
"""Build the first 15-minute spot feature store for ML training."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import glob
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_CONTEXT_REGISTRY = ROOT / "configs/ml_context_stations.json"
DEFAULT_SPOT_STATIC_FEATURES = ROOT / "configs/ml_spot_static_features.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_ML_ROOT / "feature_store"
DEFAULT_SCHEMA_DOC = ROOT / "docs/ml_nowcasting/feature_store_schema.md"
MODEL_SOURCES = ("arome", "aromepi", "moloch", "icon2i")
OPEN_METEO_MODELS = ("meteofrance_arome_france",)
OPEN_METEO_PREVIOUS_RUN_MODELS = ("best_match",)
OPEN_METEO_PREVIOUS_RUN_LEAD_DAYS = (1, 2)
OPEN_METEO_OFFSET_POINTS = (
    "n10:0:10",
    "e10:90:10",
    "s10:180:10",
    "w10:270:10",
)
EUMETSAT_SPOT_PRODUCTS = ("cloud_type", "land_surface_temperature", "global_instability_indices")
MODEL_FIELDS = ("wind_speed_ms", "wind_u_ms", "wind_v_ms", "wind_direction_deg", "gust_speed_ms")
OBS_FIELDS = (
    "wind_mean_ms",
    "gust_ms",
    "wind_direction_deg",
    "temperature_c",
    "dewpoint_c",
    "humidity_pct",
    "pressure_hpa",
    "sea_level_pressure_hpa",
    "precipitation_mm",
    "global_radiation_raw",
)
TARGET_SOURCE_PRIORITIES = {
    "windsup": 100,
    "meteofrance": 90,
    "wunderground": 80,
    "owm": 70,
    "esurfmar": 60,
    "candhis": 50,
}
CONTEXT_ROLE_SLUGS = {
    "coastal_official_context": "coastal",
    "inland_thermal_context": "inland",
    "mountain_relief_context": "relief",
    "regional_official_context": "regional",
}
CONTEXT_DELTA_FIELDS = ("wind_mean_ms", "gust_ms", "temperature_c", "humidity_pct", "pressure_hpa")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_time(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def finite_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def observation_value(row: dict[str, Any], field: str) -> float | None:
    aliases = {
        "gust_ms": ("gust_ms", "gust_max_ms", "gust_instant_ms"),
        "pressure_hpa": ("pressure_hpa", "pressure_station_hpa"),
        "precipitation_mm": ("precipitation_mm", "precipitation_1h_mm"),
        "global_radiation_raw": ("global_radiation_raw", "global_radiation_j_cm2"),
    }
    for key in aliases.get(field, (field,)):
        value = finite_float(row.get(key))
        if value is not None:
            return value
    return None


def minutes_between(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start).total_seconds() / 60.0, 3)


def round_to_step(value: datetime, step_minutes: int) -> datetime:
    step_seconds = step_minutes * 60
    epoch = int(value.timestamp())
    rounded = ((epoch + step_seconds // 2) // step_seconds) * step_seconds
    return datetime.fromtimestamp(rounded, tz=timezone.utc)


def read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def date_strings_between(start: datetime, end: datetime, margin_days_before: int = 0, margin_days_after: int = 0) -> set[str]:
    first = (start - timedelta(days=margin_days_before)).date()
    last = (end + timedelta(days=margin_days_after)).date()
    values = set()
    cursor = first
    while cursor <= last:
        values.add(cursor.isoformat())
        cursor += timedelta(days=1)
    return values


def path_matches_dates(path: Path, include_dates: set[str] | None) -> bool:
    if include_dates is None:
        return True
    text = str(path)
    return any(f"date={date_value}" in text for date_value in include_dates)


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    spots = payload.get("spots", []) if isinstance(payload, dict) else payload
    return {str(spot["spot_id"]): spot for spot in spots if isinstance(spot, dict) and spot.get("spot_id")}


def load_context_stations(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = read_json(path)
    stations = payload.get("stations", []) if isinstance(payload, dict) else payload
    return [station for station in stations if isinstance(station, dict) and station.get("station_id")]


def load_spot_static_features(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = read_json(path)
    rows = payload.get("spots", []) if isinstance(payload, dict) else payload
    out = {}
    for row in rows:
        if isinstance(row, dict) and row.get("spot_id"):
            features = row.get("features") if isinstance(row.get("features"), dict) else row
            out[str(row["spot_id"])] = {
                str(key): value
                for key, value in features.items()
                if key not in {"spot_id", "name"}
            }
    return out


def attach_spot_static_features(features: dict[str, Any], static_features: dict[str, Any], coverage: Counter[str]) -> None:
    for key, value in sorted(static_features.items()):
        if isinstance(value, (dict, list)):
            continue
        feature_key = f"spot_static_{key}"
        features[feature_key] = finite_float(value) if not isinstance(value, str) else value
        if value not in {None, ""}:
            coverage["spot_static"] += 1


def context_role_slug(role: Any) -> str:
    text = str(role or "other")
    return CONTEXT_ROLE_SLUGS.get(text, text.replace("_context", "").replace("_official", "").replace("_thermal", "").replace("_relief", ""))


def station_active_for_window(station: dict[str, Any], start_time: datetime | None, end_time: datetime | None) -> bool:
    station_start = parse_time(station.get("station_start"))
    station_end = parse_time(station.get("station_end"))
    if start_time is not None and station_end is not None and station_end < start_time:
        return False
    if end_time is not None and station_start is not None and station_start > end_time:
        return False
    return True


def distance_km(lat_a: Any, lon_a: Any, lat_b: Any, lon_b: Any) -> float | None:
    lat1 = finite_float(lat_a)
    lon1 = finite_float(lon_a)
    lat2 = finite_float(lat_b)
    lon2 = finite_float(lon_b)
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    hav = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return round(2 * radius_km * math.asin(math.sqrt(hav)), 3)


def bearing_deg(lat_a: Any, lon_a: Any, lat_b: Any, lon_b: Any) -> float | None:
    lat1 = finite_float(lat_a)
    lon1 = finite_float(lon_a)
    lat2 = finite_float(lat_b)
    lon2 = finite_float(lon_b)
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_lambda = math.radians(lon2 - lon1)
    y = math.sin(d_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    return round((math.degrees(math.atan2(y, x)) + 360.0) % 360.0, 3)


def relative_offsets_km(lat_a: Any, lon_a: Any, lat_b: Any, lon_b: Any) -> tuple[float | None, float | None]:
    distance = distance_km(lat_a, lon_a, lat_b, lon_b)
    bearing = bearing_deg(lat_a, lon_a, lat_b, lon_b)
    if distance is None or bearing is None:
        return None, None
    radians = math.radians(bearing)
    return round(distance * math.sin(radians), 3), round(distance * math.cos(radians), 3)


def offset_spot_id(base_spot_id: str, name: str) -> str:
    safe_name = "".join(char.lower() if char.isalnum() else "_" for char in str(name)).strip("_")
    return f"{base_spot_id}__nwp_offset_{safe_name}"


def parse_open_meteo_offset_points(value: str) -> list[dict[str, float | str]]:
    out = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Invalid Open-Meteo offset point '{item}'. Expected name:bearing_deg:distance_km.")
        name, bearing, distance = parts
        bearing_value = finite_float(bearing)
        distance_value = finite_float(distance)
        if not name or bearing_value is None or distance_value is None:
            raise ValueError(f"Invalid Open-Meteo offset point '{item}'.")
        out.append({
            "name": name,
            "bearing_deg": bearing_value % 360.0,
            "distance_km": distance_value,
        })
    return out


def angle_diff_deg(a: Any, b: Any) -> float | None:
    first = finite_float(a)
    second = finite_float(b)
    if first is None or second is None:
        return None
    return round(((first - second + 180.0) % 360.0) - 180.0, 3)


def upwind_alignment_score(wind_from_deg: Any, bearing_from_spot_to_station_deg: Any) -> float | None:
    diff = angle_diff_deg(wind_from_deg, bearing_from_spot_to_station_deg)
    if diff is None:
        return None
    return round(math.cos(math.radians(diff)), 4)


def enrich_context_station_geometry(spot: dict[str, Any], station: dict[str, Any]) -> dict[str, Any]:
    item = dict(station)
    distance = finite_float(item.get("_distance_km"))
    if distance is None:
        distance = distance_km(spot.get("latitude"), spot.get("longitude"), item.get("latitude"), item.get("longitude"))
    bearing_from_spot = bearing_deg(spot.get("latitude"), spot.get("longitude"), item.get("latitude"), item.get("longitude"))
    east_offset, north_offset = relative_offsets_km(spot.get("latitude"), spot.get("longitude"), item.get("latitude"), item.get("longitude"))
    spot_altitude = finite_float(spot.get("altitude_m"))
    station_altitude = finite_float(item.get("altitude_m"))
    item["_distance_km"] = distance
    item["_bearing_from_spot_deg"] = bearing_from_spot
    item["_bearing_to_spot_deg"] = bearing_deg(item.get("latitude"), item.get("longitude"), spot.get("latitude"), spot.get("longitude"))
    item["_east_offset_km"] = east_offset
    item["_north_offset_km"] = north_offset
    item["_altitude_delta_m"] = (
        round(station_altitude - spot_altitude, 3)
        if station_altitude is not None and spot_altitude is not None
        else None
    )
    return item


def load_jsonl_glob(pattern: str, include_dates: set[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(pattern)):
        candidate = Path(path)
        if path_matches_dates(candidate, include_dates):
            rows.extend(iter_jsonl(candidate))
    return rows


def dedupe_by_latest(rows: list[dict[str, Any]], key_fields: tuple[str, ...], time_field: str = "sampled_at_utc") -> list[dict[str, Any]]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        current = selected.get(key)
        if current is None:
            selected[key] = row
            continue
        current_time = parse_time(current.get(time_field)) or datetime.min.replace(tzinfo=timezone.utc)
        row_time = parse_time(row.get(time_field)) or datetime.min.replace(tzinfo=timezone.utc)
        if row_time >= current_time:
            selected[key] = row
    return list(selected.values())


def make_time_index_by_field(rows: list[dict[str, Any]], id_field: str, time_field: str) -> dict[str, tuple[list[datetime], list[dict[str, Any]]]]:
    grouped: dict[str, list[tuple[datetime, dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        item_id = row.get(id_field)
        timestamp = parse_time(row.get(time_field))
        if not item_id or timestamp is None:
            continue
        grouped[str(item_id)].append((timestamp, row))
    index = {}
    for item_id, pairs in grouped.items():
        pairs.sort(key=lambda item: item[0])
        index[item_id] = ([item[0] for item in pairs], [item[1] for item in pairs])
    return index


def make_time_index(rows: list[dict[str, Any]], time_field: str) -> dict[str, tuple[list[datetime], list[dict[str, Any]]]]:
    return make_time_index_by_field(rows, "spot_id", time_field)


def asof_row(
    index: dict[str, tuple[list[datetime], list[dict[str, Any]]]],
    spot_id: str,
    target_time: datetime,
    max_age_minutes: float,
    strictly_before: bool = False,
) -> tuple[dict[str, Any] | None, float | None]:
    item = index.get(spot_id)
    if item is None:
        return None, None
    times, rows = item
    pos = bisect.bisect_left(times, target_time) if strictly_before else bisect.bisect_right(times, target_time)
    if pos <= 0:
        return None, None
    row_time = times[pos - 1]
    age = minutes_between(row_time, target_time)
    if age is None or age < 0 or age > max_age_minutes:
        return None, age
    return rows[pos - 1], age


def row_has_any_observation_value(row: dict[str, Any], fields: tuple[str, ...]) -> bool:
    return any(observation_value(row, field) is not None for field in fields)


def asof_row_with_any_value(
    index: dict[str, tuple[list[datetime], list[dict[str, Any]]]],
    item_id: str,
    target_time: datetime,
    max_age_minutes: float,
    fields: tuple[str, ...],
    strictly_before: bool = False,
) -> tuple[dict[str, Any] | None, float | None]:
    item = index.get(item_id)
    if item is None:
        return None, None
    times, rows = item
    pos = bisect.bisect_left(times, target_time) if strictly_before else bisect.bisect_right(times, target_time)
    last_age = None
    for idx in range(pos - 1, -1, -1):
        age = minutes_between(times[idx], target_time)
        last_age = age
        if age is None or age < 0:
            continue
        if age > max_age_minutes:
            break
        if row_has_any_observation_value(rows[idx], fields):
            return rows[idx], age
    return None, last_age


def nearest_row(
    index: dict[str, tuple[list[datetime], list[dict[str, Any]]]],
    spot_id: str,
    target_time: datetime,
    tolerance_minutes: float,
) -> tuple[dict[str, Any] | None, float | None]:
    item = index.get(spot_id)
    if item is None:
        return None, None
    times, rows = item
    pos = bisect.bisect_left(times, target_time)
    candidates = []
    for idx in (pos - 1, pos):
        if 0 <= idx < len(times):
            delta = abs(minutes_between(target_time, times[idx]) or 0)
            candidates.append((delta, idx))
    if not candidates:
        return None, None
    delta, idx = min(candidates, key=lambda item: item[0])
    if delta > tolerance_minutes:
        return None, delta
    return rows[idx], delta


def load_observations(ml_root: Path, include_dates: set[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ml_root / "observations").glob("**/observations.jsonl")):
        if path_matches_dates(path, include_dates):
            rows.extend(iter_jsonl(path))
    rows = [row for row in rows if (row.get("spot_id") or row.get("station_id")) and parse_time(row.get("timestamp_utc"))]
    return dedupe_by_latest(rows, ("source_project", "source_dataset", "station_id", "spot_id", "timestamp_utc"), "received_at_utc")


def target_source_priority(row: dict[str, Any]) -> int:
    haystack = " ".join(
        str(row.get(key) or "").lower()
        for key in ("spot_source_type", "source_project", "source_dataset")
    )
    for source, priority in TARGET_SOURCE_PRIORITIES.items():
        if source in haystack:
            return priority
    return 0


def target_source_matches_registry(row: dict[str, Any], spot: dict[str, Any]) -> bool:
    expected = str(spot.get("source_type") or "").lower()
    if not expected:
        return False
    values = {
        str(row.get("spot_source_type") or "").lower(),
        str(row.get("source_project") or "").lower(),
    }
    dataset = str(row.get("source_dataset") or "").lower()
    return expected in values or expected in dataset


def target_station_matches_registry(row: dict[str, Any], spot: dict[str, Any]) -> bool:
    expected = spot.get("station_id")
    if expected in {None, ""}:
        return False
    return str(row.get("station_id") or "") == str(expected)


def target_resolution_minutes(row: dict[str, Any], spot: dict[str, Any]) -> float | None:
    return finite_float(row.get("source_resolution_minutes")) or finite_float(spot.get("source_resolution_minutes"))


def target_selection_score(row: dict[str, Any], spot: dict[str, Any], distance_minutes: float) -> tuple[int, int, int, float, float]:
    resolution = target_resolution_minutes(row, spot)
    return (
        int(target_source_matches_registry(row, spot)),
        int(target_station_matches_registry(row, spot)),
        target_source_priority(row),
        -(resolution if resolution is not None else 9999.0),
        -distance_minutes,
    )


def target_candidates(
    observations: list[dict[str, Any]],
    registry: dict[str, dict[str, Any]],
    step_minutes: int,
    tolerance_minutes: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for row in observations:
        if finite_float(row.get("wind_mean_ms")) is None and finite_float(row.get("gust_ms")) is None:
            continue
        spot_id = str(row.get("spot_id") or "")
        if not spot_id:
            continue
        spot = registry.get(spot_id, {})
        timestamp = parse_time(row.get("timestamp_utc"))
        if timestamp is None:
            continue
        target_time = round_to_step(timestamp, step_minutes)
        distance = abs((timestamp - target_time).total_seconds()) / 60.0
        if distance > tolerance_minutes:
            continue
        key = (spot_id, iso_z(target_time))
        score = target_selection_score(row, spot, distance)
        previous = candidates.get(key)
        previous_score = previous.get("_target_selection_score") if previous is not None else None
        if previous is None or score >= tuple(previous_score):
            candidates[key] = {
                **row,
                "_target_time": target_time,
                "_target_distance_minutes": round(distance, 3),
                "_target_source_priority": target_source_priority(row),
                "_target_source_matches_registry": target_source_matches_registry(row, spot),
                "_target_station_matches_registry": target_station_matches_registry(row, spot),
                "_target_resolution_minutes": target_resolution_minutes(row, spot),
                "_target_selection_score": score,
            }
    return candidates


def add_inference_grid_targets(
    candidates: dict[tuple[str, str], dict[str, Any]],
    registry: dict[str, dict[str, Any]],
    start_time: datetime,
    end_time: datetime,
    step_minutes: int,
) -> int:
    added = 0
    cursor = start_time
    while cursor <= end_time:
        target_time_iso = iso_z(cursor)
        for spot_id, spot in registry.items():
            if not spot.get("use_for_ml", False):
                continue
            key = (str(spot_id), target_time_iso)
            if key in candidates:
                continue
            candidates[key] = {
                "spot_id": spot_id,
                "spot_name": spot.get("name"),
                "spot_kind": spot.get("kind"),
                "spot_source_type": spot.get("source_type"),
                "source_project": "corsewind",
                "source_dataset": "inference_grid",
                "source_type": spot.get("source_type"),
                "station_id": spot.get("station_id"),
                "latitude": spot.get("latitude"),
                "longitude": spot.get("longitude"),
                "use_for_ml": True,
                "_target_time": cursor,
                "_target_distance_minutes": None,
                "_target_source_priority": -1,
                "_target_source_matches_registry": False,
                "_target_station_matches_registry": False,
                "_target_resolution_minutes": None,
                "_target_selection_score": (-1, -1, -1, -9999.0, -9999.0),
            }
            added += 1
        cursor += timedelta(minutes=step_minutes)
    return added


def context_station_slots(
    context_stations: list[dict[str, Any]],
    registry: dict[str, dict[str, Any]],
    nearest_count: int,
    per_role_count: int,
    global_nearest_count: int,
    global_role_count: int,
    global_max_distance_km: float,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for station in context_stations:
        if not station_active_for_window(station, start_time, end_time):
            continue
        nearest = station.get("nearest_ml_spot") or {}
        spot_id = nearest.get("spot_id")
        if not spot_id or not station.get("use_as_context", True):
            continue
        spot = registry.get(str(spot_id), {})
        if spot.get("station_id") and str(station.get("station_id")) == str(spot.get("station_id")):
            continue
        item = dict(station)
        item["_distance_km"] = finite_float(nearest.get("distance_km"))
        item = enrich_context_station_geometry(spot, item)
        grouped[str(spot_id)].append(item)

    out: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for spot_id, spot in registry.items():
        stations = grouped.get(spot_id, [])
        ordered = sorted(stations, key=lambda station: (finite_float(station.get("_distance_km")) or 9999.0, str(station.get("station_id"))))
        slots: list[tuple[str, dict[str, Any]]] = []
        used_slot_names: set[str] = set()
        used_station_ids: set[str] = set()

        def add_slot(slot: str, station: dict[str, Any], track_station: bool = True) -> None:
            if slot in used_slot_names:
                return
            slots.append((slot, station))
            used_slot_names.add(slot)
            if track_station:
                used_station_ids.add(str(station.get("station_id")))

        for idx, station in enumerate(ordered[:nearest_count], start=1):
            add_slot(f"nearest_{idx}", station)

        by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for station in ordered:
            by_role[context_role_slug(station.get("context_role"))].append(station)
        for role in ("coastal", "inland", "relief", "regional"):
            for idx, station in enumerate(by_role.get(role, [])[:per_role_count], start=1):
                add_slot(f"{role}_{idx}", station)

        global_candidates = []
        for station in context_stations:
            if not station.get("use_as_context", True):
                continue
            if not station_active_for_window(station, start_time, end_time):
                continue
            station_id = str(station.get("station_id"))
            if station_id in used_station_ids:
                continue
            if spot.get("station_id") and station_id == str(spot.get("station_id")):
                continue
            distance = distance_km(spot.get("latitude"), spot.get("longitude"), station.get("latitude"), station.get("longitude"))
            if distance is None or distance > global_max_distance_km:
                continue
            item = dict(station)
            item["_distance_km"] = distance
            item = enrich_context_station_geometry(spot, item)
            global_candidates.append(item)
        global_candidates.sort(key=lambda station: (finite_float(station.get("_distance_km")) or 9999.0, str(station.get("station_id"))))
        global_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for station in global_candidates:
            global_by_role[context_role_slug(station.get("context_role"))].append(station)
        for role in ("coastal", "relief", "inland", "regional"):
            for idx, station in enumerate(global_by_role.get(role, [])[:global_role_count], start=1):
                add_slot(f"global_{role}_{idx}", station, track_station=False)
        for idx, station in enumerate(global_candidates[:global_nearest_count], start=1):
            add_slot(f"global_nearest_{idx}", station)
        out[spot_id] = slots
    return out


def flatten_feature_map(prefix: str, values: dict[str, Any], out: dict[str, Any], coverage: Counter[str]) -> None:
    for key, value in sorted(values.items()):
        out[f"{prefix}_{key}"] = finite_float(value)
        if value not in {None, ""}:
            coverage[prefix] += 1


def wind_components(speed: float | None, direction_deg: float | None) -> tuple[float | None, float | None]:
    if speed is None or direction_deg is None:
        return None, None
    radians = math.radians(direction_deg)
    return round(-speed * math.sin(radians), 4), round(-speed * math.cos(radians), 4)


def row_wind_components(row: dict[str, Any] | None) -> tuple[float | None, float | None]:
    if not row:
        return None, None
    return wind_components(observation_value(row, "wind_mean_ms"), observation_value(row, "wind_direction_deg"))


def add_aggregate_stats(features: dict[str, Any], prefix: str, records: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        station_id = str(record.get("station_id") or "")
        if station_id and station_id not in deduped:
            deduped[station_id] = record
    values = list(deduped.values())
    features[f"{prefix}_station_count"] = len(values)
    for field in fields:
        numbers = [finite_float(record.get(field)) for record in values]
        numbers = [value for value in numbers if value is not None]
        features[f"{prefix}_{field}_count"] = len(numbers)
        if not numbers:
            continue
        features[f"{prefix}_{field}_mean"] = round(sum(numbers) / len(numbers), 4)
        features[f"{prefix}_{field}_min"] = round(min(numbers), 4)
        features[f"{prefix}_{field}_max"] = round(max(numbers), 4)


def upwind_context_weight(record: dict[str, Any]) -> float | None:
    upwind_score = finite_float(record.get("upwind_score_from_target_wind"))
    if upwind_score is None or upwind_score <= 0:
        return None
    distance = finite_float(record.get("distance_km"))
    age = finite_float(record.get("age_minutes"))
    distance_weight = 1.0 / (1.0 + max(distance or 0.0, 0.0) / 20.0)
    age_weight = 1.0 / (1.0 + max(age or 0.0, 0.0) / 60.0)
    return round(upwind_score * distance_weight * age_weight, 8)


def add_upwind_weighted_stats(features: dict[str, Any], prefix: str, records: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        station_id = str(record.get("station_id") or "")
        if station_id and station_id not in deduped:
            deduped[station_id] = record
    weighted_records = []
    for record in deduped.values():
        weight = upwind_context_weight(record)
        if weight is not None and weight > 0:
            weighted_records.append((record, weight))
    features[f"{prefix}_upwind_weighted_station_count"] = len(weighted_records)
    features[f"{prefix}_upwind_weight_sum"] = round(sum(weight for _, weight in weighted_records), 6)
    if not weighted_records:
        return
    for field in fields:
        numerator = 0.0
        denominator = 0.0
        count = 0
        for record, weight in weighted_records:
            value = finite_float(record.get(field))
            if value is None:
                continue
            numerator += value * weight
            denominator += weight
            count += 1
        features[f"{prefix}_upwind_weighted_{field}_count"] = count
        if denominator > 0:
            features[f"{prefix}_upwind_weighted_{field}_mean"] = round(numerator / denominator, 4)


def context_record_groups(slot: str, station: dict[str, Any]) -> set[str]:
    role = context_role_slug(station.get("context_role"))
    groups = {"all"}
    if slot.startswith("nearest_") or slot.startswith("global_nearest_"):
        groups.add("nearby")
    if role in {"coastal", "inland", "relief", "regional"}:
        groups.add(role)
    if slot.startswith("global_coastal_"):
        groups.add("coastal")
    if slot.startswith("global_relief_"):
        groups.add("relief")
    if slot.startswith("global_inland_"):
        groups.add("inland")
    return groups


def attach_context_station_features(
    features: dict[str, Any],
    station_index: dict[str, tuple[list[datetime], list[dict[str, Any]]]],
    slots: list[tuple[str, dict[str, Any]]],
    target_latest: dict[str, Any] | None,
    target_time: datetime,
    max_age_minutes: float,
    coverage: Counter[str],
) -> bool:
    available_count = 0
    aggregate_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_u, target_v = row_wind_components(target_latest)
    target_wind_direction = observation_value(target_latest, "wind_direction_deg") if target_latest else None
    features["context_station_slot_count"] = len(slots)

    for slot, station in slots:
        prefix = f"context_{slot}"
        station_id = str(station.get("station_id"))
        sample, age = asof_row_with_any_value(station_index, station_id, target_time, max_age_minutes, OBS_FIELDS, strictly_before=True)
        features[f"{prefix}_available"] = sample is not None
        features[f"{prefix}_age_minutes"] = age
        features[f"{prefix}_station_id"] = station_id
        features[f"{prefix}_role"] = context_role_slug(station.get("context_role"))
        features[f"{prefix}_distance_km"] = finite_float(station.get("_distance_km"))
        features[f"{prefix}_bearing_from_spot_deg"] = finite_float(station.get("_bearing_from_spot_deg"))
        features[f"{prefix}_bearing_to_spot_deg"] = finite_float(station.get("_bearing_to_spot_deg"))
        features[f"{prefix}_east_offset_km"] = finite_float(station.get("_east_offset_km"))
        features[f"{prefix}_north_offset_km"] = finite_float(station.get("_north_offset_km"))
        features[f"{prefix}_altitude_m"] = finite_float(station.get("altitude_m"))
        features[f"{prefix}_altitude_delta_m"] = finite_float(station.get("_altitude_delta_m"))
        features[f"{prefix}_latitude"] = finite_float(station.get("latitude"))
        features[f"{prefix}_longitude"] = finite_float(station.get("longitude"))
        features[f"{prefix}_upwind_score_from_target_wind"] = upwind_alignment_score(
            target_wind_direction,
            station.get("_bearing_from_spot_deg"),
        )
        if sample is None:
            continue

        available_count += 1
        coverage[f"context_{slot}"] += 1
        coverage["context_station"] += 1
        for field in OBS_FIELDS:
            features[f"{prefix}_{field}"] = observation_value(sample, field)
        sample_u, sample_v = row_wind_components(sample)
        features[f"{prefix}_wind_u_ms"] = sample_u
        features[f"{prefix}_wind_v_ms"] = sample_v
        record = {
            "station_id": station_id,
            "distance_km": finite_float(station.get("_distance_km")),
            "bearing_from_spot_deg": finite_float(station.get("_bearing_from_spot_deg")),
            "east_offset_km": finite_float(station.get("_east_offset_km")),
            "north_offset_km": finite_float(station.get("_north_offset_km")),
            "altitude_delta_m": finite_float(station.get("_altitude_delta_m")),
            "upwind_score_from_target_wind": features[f"{prefix}_upwind_score_from_target_wind"],
            "age_minutes": age,
            "altitude_m": finite_float(station.get("altitude_m")),
            "wind_mean_ms": observation_value(sample, "wind_mean_ms"),
            "gust_ms": observation_value(sample, "gust_ms"),
            "temperature_c": observation_value(sample, "temperature_c"),
            "humidity_pct": observation_value(sample, "humidity_pct"),
            "pressure_hpa": observation_value(sample, "pressure_hpa"),
            "wind_u_ms": sample_u,
            "wind_v_ms": sample_v,
        }

        if target_latest:
            for field in CONTEXT_DELTA_FIELDS:
                context_value = observation_value(sample, field)
                target_value = observation_value(target_latest, field)
                delta_value = (
                    round(context_value - target_value, 4)
                    if context_value is not None and target_value is not None
                    else None
                )
                features[f"{prefix}_delta_vs_target_{field}"] = delta_value
                record[f"delta_vs_target_{field}"] = delta_value
            features[f"{prefix}_delta_vs_target_wind_u_ms"] = (
                round(sample_u - target_u, 4)
                if sample_u is not None and target_u is not None
                else None
            )
            record["delta_vs_target_wind_u_ms"] = features[f"{prefix}_delta_vs_target_wind_u_ms"]
            features[f"{prefix}_delta_vs_target_wind_v_ms"] = (
                round(sample_v - target_v, 4)
                if sample_v is not None and target_v is not None
                else None
            )
            record["delta_vs_target_wind_v_ms"] = features[f"{prefix}_delta_vs_target_wind_v_ms"]

        for group in context_record_groups(slot, station):
            aggregate_records[group].append(record)

    features["context_station_available_count"] = available_count
    aggregate_fields = (
        "distance_km",
        "age_minutes",
        "altitude_m",
        "altitude_delta_m",
        "bearing_from_spot_deg",
        "east_offset_km",
        "north_offset_km",
        "upwind_score_from_target_wind",
        "wind_mean_ms",
        "gust_ms",
        "temperature_c",
        "humidity_pct",
        "pressure_hpa",
        "wind_u_ms",
        "wind_v_ms",
        "delta_vs_target_wind_mean_ms",
        "delta_vs_target_gust_ms",
        "delta_vs_target_temperature_c",
        "delta_vs_target_humidity_pct",
        "delta_vs_target_pressure_hpa",
        "delta_vs_target_wind_u_ms",
        "delta_vs_target_wind_v_ms",
    )
    for group in ("all", "nearby", "coastal", "inland", "relief", "regional"):
        add_aggregate_stats(features, f"context_agg_{group}", aggregate_records.get(group, []), aggregate_fields)
        add_upwind_weighted_stats(features, f"context_agg_{group}", aggregate_records.get(group, []), (
            "distance_km",
            "age_minutes",
            "altitude_delta_m",
            "wind_mean_ms",
            "gust_ms",
            "temperature_c",
            "humidity_pct",
            "pressure_hpa",
            "wind_u_ms",
            "wind_v_ms",
            "delta_vs_target_wind_mean_ms",
            "delta_vs_target_gust_ms",
            "delta_vs_target_temperature_c",
            "delta_vs_target_pressure_hpa",
            "delta_vs_target_wind_u_ms",
            "delta_vs_target_wind_v_ms",
        ))
    return available_count > 0


def attach_eumetsat_spot_product_features(
    features: dict[str, Any],
    product: str,
    sample: dict[str, Any] | None,
    age_minutes: float | None,
    coverage: Counter[str],
) -> None:
    prefix = f"eumetsat_{product}"
    features[f"{prefix}_available"] = sample is not None
    features[f"{prefix}_age_minutes"] = age_minutes
    if not sample:
        return
    coverage[prefix] += 1
    features[f"{prefix}_sensing_start_utc"] = sample.get("sensing_start_utc")
    features[f"{prefix}_sample_distance_km"] = finite_float(sample.get("sample_distance_km"))
    for key in ("product_quality", "product_completeness", "product_timeliness"):
        features[f"{prefix}_{key}"] = finite_float(sample.get(key))
    for key, value in sorted((sample.get("sampled_values") or {}).items()):
        features[f"{prefix}_{key}"] = finite_float(value)
        if value not in {None, ""}:
            coverage[f"{prefix}_sampled_values"] += 1
    for key, value in sorted((sample.get("sampled_values_c") or {}).items()):
        features[f"{prefix}_{key}"] = finite_float(value)
        if value not in {None, ""}:
            coverage[f"{prefix}_sampled_values_c"] += 1
    for variable, summary in sorted((sample.get("neighborhoods") or {}).items()):
        if not isinstance(summary, dict):
            continue
        for key in ("valid_count", "mean", "min", "max"):
            if key in summary:
                features[f"{prefix}_{variable}_neighborhood_{key}"] = finite_float(summary.get(key))
        if "mode" in summary:
            features[f"{prefix}_{variable}_neighborhood_mode"] = finite_float(summary.get("mode"))
        fractions = summary.get("fractions")
        if isinstance(fractions, dict):
            for value_key, fraction in sorted(fractions.items()):
                features[f"{prefix}_{variable}_neighborhood_fraction_{value_key}"] = finite_float(fraction)


def attach_model_features(
    row: dict[str, Any],
    source: str,
    sample: dict[str, Any] | None,
    target_time: datetime,
    coverage: Counter[str],
) -> None:
    prefix = f"model_{source}"
    if sample is None:
        row[f"{prefix}_available"] = False
        return
    row[f"{prefix}_available"] = bool(sample.get("inside_grid", True))
    run_time = parse_time(sample.get("run_time_utc"))
    row[f"{prefix}_run_time_utc"] = sample.get("run_time_utc")
    row[f"{prefix}_run_age_minutes"] = minutes_between(run_time, target_time)
    row[f"{prefix}_lead_minutes"] = finite_float(sample.get("lead_minutes"))
    for field in MODEL_FIELDS:
        row[f"{prefix}_{field}"] = finite_float(sample.get(field))
    coverage[prefix] += 1


def attach_open_meteo_features(
    row: dict[str, Any],
    model: str,
    sample: dict[str, Any] | None,
    valid_offset_minutes: float | None,
    coverage: Counter[str],
) -> None:
    prefix = f"model_open_meteo_{model}"
    row[f"{prefix}_available"] = sample is not None
    if sample is None:
        return
    row[f"{prefix}_valid_time_utc"] = sample.get("valid_time_utc")
    row[f"{prefix}_valid_offset_minutes"] = valid_offset_minutes
    for key in ("api_latitude", "api_longitude", "api_elevation_m"):
        row[f"{prefix}_{key}"] = finite_float(sample.get(key))

    values = sample.get("features") or {}
    if isinstance(values, dict):
        flatten_feature_map(prefix, values, row, coverage)
        speed = finite_float(values.get("wind_speed_10m"))
        direction = finite_float(values.get("wind_direction_10m"))
        wind_u, wind_v = wind_components(speed, direction)
        row[f"{prefix}_wind_u_10m"] = wind_u
        row[f"{prefix}_wind_v_10m"] = wind_v
    coverage[prefix] += 1


def attach_open_meteo_offset_features(
    row: dict[str, Any],
    *,
    model: str,
    offset_name: str,
    offset_bearing_deg: float,
    offset_distance_km: float,
    sample: dict[str, Any] | None,
    center_sample: dict[str, Any] | None,
    valid_offset_minutes: float | None,
    coverage: Counter[str],
) -> None:
    prefix = f"nwp_offset_{offset_name}"
    row[f"{prefix}_available"] = sample is not None
    row[f"{prefix}_bearing_deg"] = round(offset_bearing_deg, 6)
    row[f"{prefix}_distance_km"] = round(offset_distance_km, 6)
    if sample is None:
        return
    row[f"{prefix}_model"] = model
    row[f"{prefix}_valid_time_utc"] = sample.get("valid_time_utc")
    row[f"{prefix}_valid_offset_minutes"] = valid_offset_minutes
    for key in ("api_latitude", "api_longitude", "api_elevation_m"):
        row[f"{prefix}_{key}"] = finite_float(sample.get(key))

    values = sample.get("features") or {}
    center_values = center_sample.get("features") if isinstance(center_sample, dict) else {}
    if not isinstance(center_values, dict):
        center_values = {}
    if isinstance(values, dict):
        for key in (
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "pressure_msl",
            "surface_pressure",
            "cloud_cover",
            "cloud_cover_low",
            "shortwave_radiation",
            "cape",
            "boundary_layer_height",
        ):
            value = finite_float(values.get(key))
            row[f"{prefix}_{key}"] = value
            center = finite_float(center_values.get(key))
            row[f"{prefix}_delta_vs_center_{key}"] = (
                None if value is None or center is None else round(value - center, 6)
            )
    coverage[prefix] += 1


def add_open_meteo_offset_gradient_features(row: dict[str, Any], offset_points: list[dict[str, float | str]]) -> None:
    by_name = {str(item["name"]): item for item in offset_points}
    pairs = [
        ("e10", "w10", "east_west"),
        ("n10", "s10", "north_south"),
    ]
    fields = ("pressure_msl", "surface_pressure", "temperature_2m", "wind_speed_10m", "cloud_cover", "shortwave_radiation")
    for positive_name, negative_name, axis_name in pairs:
        positive = by_name.get(positive_name)
        negative = by_name.get(negative_name)
        if not positive or not negative:
            continue
        positive_distance = finite_float(positive.get("distance_km"))
        negative_distance = finite_float(negative.get("distance_km"))
        span_km = None if positive_distance is None or negative_distance is None else positive_distance + negative_distance
        for field in fields:
            positive_value = finite_float(row.get(f"nwp_offset_{positive_name}_{field}"))
            negative_value = finite_float(row.get(f"nwp_offset_{negative_name}_{field}"))
            delta = None if positive_value is None or negative_value is None else round(positive_value - negative_value, 6)
            row[f"nwp_offset_gradient_{axis_name}_{field}_delta"] = delta
            row[f"nwp_offset_gradient_{axis_name}_{field}_per_km"] = (
                None if delta is None or not span_km else round(delta / span_km, 8)
            )

    pressure_east = finite_float(row.get("nwp_offset_gradient_east_west_pressure_msl_per_km"))
    pressure_north = finite_float(row.get("nwp_offset_gradient_north_south_pressure_msl_per_km"))
    if pressure_east is not None and pressure_north is not None:
        row["nwp_offset_gradient_pressure_msl_magnitude_hpa_per_km"] = round(
            math.sqrt(pressure_east * pressure_east + pressure_north * pressure_north),
            8,
        )
        wind_from = finite_float(row.get("model_open_meteo_meteofrance_arome_france_wind_direction_10m"))
        if wind_from is not None:
            wind_to = math.radians((wind_from + 180.0) % 360.0)
            wind_east = math.sin(wind_to)
            wind_north = math.cos(wind_to)
            row["nwp_offset_gradient_pressure_msl_aligned_with_wind_hpa_per_km"] = round(
                pressure_east * wind_east + pressure_north * wind_north,
                8,
            )


def attach_open_meteo_previous_run_features(
    row: dict[str, Any],
    model: str,
    lead_day: int,
    sample: dict[str, Any] | None,
    valid_offset_minutes: float | None,
    coverage: Counter[str],
) -> None:
    prefix = f"previous_run_open_meteo_{model}_day{lead_day}"
    row[f"{prefix}_available"] = sample is not None
    if sample is None:
        return
    row[f"{prefix}_valid_time_utc"] = sample.get("valid_time_utc")
    row[f"{prefix}_valid_offset_minutes"] = valid_offset_minutes
    row[f"{prefix}_nominal_forecast_age_hours"] = finite_float(sample.get("nominal_forecast_age_hours"))
    for key in ("api_latitude", "api_longitude", "api_elevation_m"):
        row[f"{prefix}_{key}"] = finite_float(sample.get(key))

    values = sample.get("features") or {}
    if isinstance(values, dict):
        flatten_feature_map(prefix, values, row, coverage)
        speed = finite_float(values.get("wind_speed_10m"))
        direction = finite_float(values.get("wind_direction_10m"))
        wind_u, wind_v = wind_components(speed, direction)
        row[f"{prefix}_wind_u_10m"] = wind_u
        row[f"{prefix}_wind_v_10m"] = wind_v
    coverage[prefix] += 1


def attach_observation_history(
    features: dict[str, Any],
    obs_index: dict[str, tuple[list[datetime], list[dict[str, Any]]]],
    spot_id: str,
    target_time: datetime,
    max_age_minutes: float,
    coverage: Counter[str],
) -> None:
    latest, age = asof_row_with_any_value(obs_index, spot_id, target_time, max_age_minutes, OBS_FIELDS, strictly_before=True)
    features["obs_last_available"] = latest is not None
    features["obs_last_age_minutes"] = age
    if latest:
        coverage["obs_last"] += 1
        for field in OBS_FIELDS:
            features[f"obs_last_{field}"] = observation_value(latest, field)
    for lag_minutes in (15, 60):
        lag_row, lag_age = asof_row_with_any_value(obs_index, spot_id, target_time - timedelta(minutes=lag_minutes), max_age_minutes, OBS_FIELDS, strictly_before=False)
        features[f"obs_lag_{lag_minutes}m_available"] = lag_row is not None
        features[f"obs_lag_{lag_minutes}m_age_minutes"] = lag_age
        if lag_row:
            coverage[f"obs_lag_{lag_minutes}m"] += 1
            for field in ("wind_mean_ms", "gust_ms", "wind_direction_deg", "temperature_c", "pressure_hpa"):
                features[f"obs_lag_{lag_minutes}m_{field}"] = observation_value(lag_row, field)
        if latest and lag_row:
            for field in ("wind_mean_ms", "gust_ms", "temperature_c", "pressure_hpa"):
                current = observation_value(latest, field)
                previous = observation_value(lag_row, field)
                features[f"obs_delta_{lag_minutes}m_{field}"] = round(current - previous, 4) if current is not None and previous is not None else None


def latest_valid_time_index(rows: list[dict[str, Any]], source_field: str | None = None) -> dict[tuple[str, str, str], dict[str, Any]]:
    out = {}
    for item in rows:
        spot_id = item.get("spot_id")
        valid_time = item.get("valid_time_utc")
        source = item.get(source_field) if source_field else item.get("source")
        if not spot_id or not valid_time or not source:
            continue
        valid_dt = parse_time(valid_time)
        item_run = parse_time(item.get("run_time_utc")) or datetime.min.replace(tzinfo=timezone.utc)
        if valid_dt is not None and item_run > valid_dt:
            continue
        key = (str(source), str(spot_id), str(valid_time))
        current = out.get(key)
        if current is None:
            out[key] = item
            continue
        current_run = parse_time(current.get("run_time_utc")) or datetime.min.replace(tzinfo=timezone.utc)
        if item_run >= current_run:
            out[key] = item
    return out


def forecast_time_index(rows: list[dict[str, Any]], source_field: str | None = None) -> dict[tuple[str, str], tuple[list[datetime], list[dict[str, Any]]]]:
    latest = latest_valid_time_index(rows, source_field)
    grouped: dict[tuple[str, str], list[tuple[datetime, dict[str, Any]]]] = defaultdict(list)
    for (source, spot_id, valid_time), row in latest.items():
        valid_dt = parse_time(valid_time)
        if valid_dt is not None:
            grouped[(source, spot_id)].append((valid_dt, row))
    out = {}
    for key, pairs in grouped.items():
        pairs.sort(key=lambda item: item[0])
        out[key] = ([item[0] for item in pairs], [item[1] for item in pairs])
    return out


def previous_run_time_index(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], tuple[list[datetime], list[dict[str, Any]]]]:
    latest: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for item in rows:
        model = item.get("model")
        spot_id = item.get("spot_id")
        valid_time = item.get("valid_time_utc")
        lead_day = item.get("lead_days")
        if not model or not spot_id or not valid_time or lead_day in {None, ""}:
            continue
        try:
            lead_day_int = int(lead_day)
        except (TypeError, ValueError):
            continue
        key = (str(model), lead_day_int, str(spot_id), str(valid_time))
        current = latest.get(key)
        if current is None or str(item.get("fetched_at_utc") or "") >= str(current.get("fetched_at_utc") or ""):
            latest[key] = item

    grouped: dict[tuple[str, int, str], list[tuple[datetime, dict[str, Any]]]] = defaultdict(list)
    for (model, lead_day, spot_id, valid_time), row in latest.items():
        valid_dt = parse_time(valid_time)
        if valid_dt is not None:
            grouped[(model, lead_day, spot_id)].append((valid_dt, row))

    out = {}
    for key, pairs in grouped.items():
        pairs.sort(key=lambda item: item[0])
        out[key] = ([item[0] for item in pairs], [item[1] for item in pairs])
    return out


def nearest_forecast_row(
    index: dict[tuple[str, str], tuple[list[datetime], list[dict[str, Any]]]],
    source: str,
    spot_id: str,
    target_time: datetime,
    tolerance_minutes: float,
) -> tuple[dict[str, Any] | None, float | None]:
    item = index.get((source, spot_id))
    if item is None:
        return None, None
    times, rows = item
    pos = bisect.bisect_left(times, target_time)
    candidates = []
    for idx in (pos - 1, pos):
        if not 0 <= idx < len(times):
            continue
        row = rows[idx]
        run_time = parse_time(row.get("run_time_utc"))
        if run_time is not None and run_time > target_time:
            continue
        offset = minutes_between(target_time, times[idx])
        if offset is None:
            continue
        distance = abs(offset)
        if distance <= tolerance_minutes:
            candidates.append((distance, idx, offset, row))
    if not candidates:
        return None, None
    _, _, offset, row = min(candidates, key=lambda item: (item[0], item[1]))
    return row, offset


def nearest_previous_run_row(
    index: dict[tuple[str, int, str], tuple[list[datetime], list[dict[str, Any]]]],
    model: str,
    lead_day: int,
    spot_id: str,
    target_time: datetime,
    tolerance_minutes: float,
) -> tuple[dict[str, Any] | None, float | None]:
    item = index.get((model, lead_day, spot_id))
    if item is None:
        return None, None
    times, rows = item
    pos = bisect.bisect_left(times, target_time)
    candidates = []
    for idx in (pos - 1, pos):
        if not 0 <= idx < len(times):
            continue
        offset = minutes_between(target_time, times[idx])
        if offset is None:
            continue
        distance = abs(offset)
        if distance <= tolerance_minutes:
            candidates.append((distance, idx, offset, rows[idx]))
    if not candidates:
        return None, None
    _, _, offset, row = min(candidates, key=lambda item: (item[0], item[1]))
    return row, offset


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ml_root = resolve_path(args.ml_root)
    registry = load_registry(resolve_path(args.registry))
    context_stations = load_context_stations(resolve_path(args.context_registry))
    spot_static_features = load_spot_static_features(resolve_path(args.spot_static_features))
    start_time = parse_time(args.start_datetime) if args.start_datetime else None
    end_time = parse_time(args.end_datetime) if args.end_datetime else None
    context_slots = context_station_slots(
        context_stations,
        registry,
        nearest_count=args.context_nearest_count,
        per_role_count=args.context_per_role_count,
        global_nearest_count=args.context_global_nearest_count,
        global_role_count=args.context_global_role_count,
        global_max_distance_km=args.context_global_max_distance_km,
        start_time=start_time,
        end_time=end_time,
    )
    read_dates = (
        date_strings_between(start_time, end_time, margin_days_before=args.read_margin_days_before)
        if start_time is not None and end_time is not None
        else None
    )
    target_dates = (
        date_strings_between(start_time, end_time)
        if start_time is not None and end_time is not None
        else None
    )
    observations = load_observations(ml_root, read_dates)
    target_observations = [row for row in observations if bool(row.get("use_for_ml", False))]
    targets = target_candidates(target_observations, registry, args.step_minutes, args.target_tolerance_minutes)
    inference_grid_target_count = 0
    if args.include_inference_grid:
        if start_time is None or end_time is None:
            raise SystemExit("--include-inference-grid requires --start-datetime and --end-datetime")
        inference_grid_target_count = add_inference_grid_targets(
            targets,
            registry,
            start_time,
            end_time,
            args.step_minutes,
        )
    obs_index = make_time_index(target_observations, "timestamp_utc")
    context_station_index = make_time_index_by_field(observations, "station_id", "timestamp_utc")

    model_rows = dedupe_by_latest(load_jsonl_glob(str(ml_root / "model_samples/source=*/date=*/samples.jsonl"), read_dates), ("source", "run_time_utc", "valid_time_utc", "spot_id"))
    model_forecasts = forecast_time_index(model_rows)
    open_meteo_rows = dedupe_by_latest(
        load_jsonl_glob(str(ml_root / "open_meteo/historical_forecast/model=*/date=*/forecast.jsonl"), read_dates),
        ("model", "valid_time_utc", "spot_id"),
        "fetched_at_utc",
    )
    open_meteo_forecasts = forecast_time_index(open_meteo_rows, "model")
    open_meteo_models = tuple(model.strip() for model in args.open_meteo_models.split(",") if model.strip())
    open_meteo_offset_points = parse_open_meteo_offset_points(args.open_meteo_offset_points)
    previous_run_rows = load_jsonl_glob(str(ml_root / "open_meteo/previous_runs/model=*/date=*/previous_runs.jsonl"), read_dates)
    previous_run_forecasts = previous_run_time_index(previous_run_rows)
    previous_run_models = tuple(model.strip() for model in args.open_meteo_previous_run_models.split(",") if model.strip())
    previous_run_lead_days = tuple(int(item.strip()) for item in args.open_meteo_previous_run_lead_days.split(",") if item.strip())
    extra_rows = dedupe_by_latest(load_jsonl_glob(str(ml_root / "meteo_france_nwp/extra_field_samples/source=*/date=*/extra_fields.jsonl"), read_dates), ("source", "run_time_utc", "valid_time_utc", "spot_id"))
    extra_forecasts = forecast_time_index(extra_rows)
    vertical_rows = dedupe_by_latest(load_jsonl_glob(str(ml_root / "meteo_france_nwp/vertical_profiles/source=*/resolution=*/date=*/vertical_profiles.jsonl"), read_dates), ("source", "resolution", "run_time_utc", "valid_time_utc", "spot_id"))
    vertical_forecasts = forecast_time_index(vertical_rows)
    sst_index = make_time_index(load_jsonl_glob(str(ml_root / "copernicus_marine/sst_samples/date=*/sst_samples.jsonl"), read_dates), "timestamp_utc")
    cloud_index = make_time_index(load_jsonl_glob(str(ml_root / "eumetsat/cloud_mask_samples/date=*/cloud_mask_samples.jsonl"), read_dates), "sensing_start_utc")
    eumetsat_product_indices = {
        product: make_time_index(
            load_jsonl_glob(str(ml_root / f"eumetsat/{product}_samples/date=*/{product}_samples.jsonl"), read_dates),
            "sensing_start_utc",
        )
        for product in EUMETSAT_SPOT_PRODUCTS
    }

    rows = []
    coverage: Counter[str] = Counter()
    for (spot_id, target_time_iso), target_obs in sorted(targets.items(), key=lambda item: (item[0][1], item[0][0])):
        spot = registry.get(spot_id, {})
        target_time = parse_time(target_time_iso)
        if target_time is None:
            continue
        if start_time is not None and target_time < start_time:
            continue
        if end_time is not None and target_time > end_time:
            continue
        target_gust = finite_float(target_obs.get("gust_ms"))
        if target_gust is None:
            target_gust = finite_float(target_obs.get("gust_max_ms"))
        if target_gust is None:
            target_gust = finite_float(target_obs.get("gust_instant_ms"))
        target_pressure = finite_float(target_obs.get("pressure_hpa"))
        if target_pressure is None:
            target_pressure = finite_float(target_obs.get("pressure_station_hpa"))
        targets_payload = {
            "wind_mean_ms": finite_float(target_obs.get("wind_mean_ms")),
            "gust_ms": target_gust,
            "wind_direction_deg": finite_float(target_obs.get("wind_direction_deg")),
            "temperature_c": finite_float(target_obs.get("temperature_c")),
            "pressure_hpa": target_pressure,
            "observation_timestamp_utc": target_obs.get("timestamp_utc"),
            "observation_distance_minutes": target_obs.get("_target_distance_minutes"),
            "observation_source_project": target_obs.get("source_project"),
            "observation_source_dataset": target_obs.get("source_dataset"),
            "observation_source_type": target_obs.get("spot_source_type"),
            "observation_station_id": target_obs.get("station_id"),
            "observation_source_resolution_minutes": target_obs.get("_target_resolution_minutes"),
            "observation_source_priority": target_obs.get("_target_source_priority"),
            "observation_source_matches_registry": target_obs.get("_target_source_matches_registry"),
            "observation_station_matches_registry": target_obs.get("_target_station_matches_registry"),
        }
        features: dict[str, Any] = {}
        source_flags: dict[str, bool] = {}
        attach_spot_static_features(features, spot_static_features.get(spot_id, {}), coverage)
        source_flags["spot_static"] = spot_id in spot_static_features
        target_latest, _ = asof_row_with_any_value(obs_index, spot_id, target_time, args.observation_history_max_age_minutes, OBS_FIELDS, strictly_before=True)
        attach_observation_history(features, obs_index, spot_id, target_time, args.observation_history_max_age_minutes, coverage)
        has_context = attach_context_station_features(
            features,
            context_station_index,
            context_slots.get(spot_id, []),
            target_latest,
            target_time,
            args.context_station_max_age_minutes,
            coverage,
        )
        source_flags["context_stations"] = has_context

        for source in MODEL_SOURCES:
            sample, sample_offset = nearest_forecast_row(model_forecasts, source, spot_id, target_time, args.forecast_valid_tolerance_minutes)
            attach_model_features(features, source, sample, target_time, coverage)
            if sample is not None:
                features[f"model_{source}_valid_time_utc"] = sample.get("valid_time_utc")
                features[f"model_{source}_valid_offset_minutes"] = sample_offset
            source_flags[f"model_{source}"] = sample is not None
            extra, extra_offset = nearest_forecast_row(extra_forecasts, source, spot_id, target_time, args.forecast_valid_tolerance_minutes)
            if extra:
                flatten_feature_map(f"nwp_{source}", extra.get("features") or {}, features, coverage)
                features[f"nwp_{source}_valid_time_utc"] = extra.get("valid_time_utc")
                features[f"nwp_{source}_valid_offset_minutes"] = extra_offset
            source_flags[f"nwp_{source}_extra"] = extra is not None

        for model in open_meteo_models:
            sample, sample_offset = nearest_forecast_row(open_meteo_forecasts, model, spot_id, target_time, args.forecast_valid_tolerance_minutes)
            attach_open_meteo_features(features, model, sample, sample_offset, coverage)
            source_flags[f"model_open_meteo_{model}"] = sample is not None
            for offset in open_meteo_offset_points:
                offset_name = str(offset["name"])
                offset_id = offset_spot_id(spot_id, offset_name)
                offset_sample, offset_sample_offset = nearest_forecast_row(
                    open_meteo_forecasts,
                    model,
                    offset_id,
                    target_time,
                    args.forecast_valid_tolerance_minutes,
                )
                attach_open_meteo_offset_features(
                    features,
                    model=model,
                    offset_name=offset_name,
                    offset_bearing_deg=float(offset["bearing_deg"]),
                    offset_distance_km=float(offset["distance_km"]),
                    sample=offset_sample,
                    center_sample=sample,
                    valid_offset_minutes=offset_sample_offset,
                    coverage=coverage,
                )
                source_flags[f"nwp_offset_{offset_name}_{model}"] = offset_sample is not None
            if open_meteo_offset_points:
                add_open_meteo_offset_gradient_features(features, open_meteo_offset_points)

        for model in previous_run_models:
            for lead_day in previous_run_lead_days:
                sample, sample_offset = nearest_previous_run_row(previous_run_forecasts, model, lead_day, spot_id, target_time, args.forecast_valid_tolerance_minutes)
                attach_open_meteo_previous_run_features(features, model, lead_day, sample, sample_offset, coverage)
                source_flags[f"previous_run_open_meteo_{model}_day{lead_day}"] = sample is not None

        vertical, vertical_offset = nearest_forecast_row(vertical_forecasts, "arome", spot_id, target_time, args.forecast_valid_tolerance_minutes)
        if vertical:
            features["vertical_arome_valid_time_utc"] = vertical.get("valid_time_utc")
            features["vertical_arome_valid_offset_minutes"] = vertical_offset
            flatten_feature_map("vertical_arome", vertical.get("derived_features") or {}, features, coverage)
            profile = vertical.get("profile") or {}
            for profile_name, by_level in sorted(profile.items()):
                if isinstance(by_level, dict):
                    for level, value in sorted(by_level.items(), key=lambda item: int(item[0]), reverse=True):
                        features[f"vertical_arome_{profile_name}_{level}hpa"] = finite_float(value)
                        if value not in {None, ""}:
                            coverage["vertical_arome_profile"] += 1
        source_flags["vertical_arome"] = vertical is not None

        sst, sst_age = asof_row(sst_index, spot_id, target_time, args.sst_max_age_minutes)
        features["sst_available"] = sst is not None
        features["sst_age_minutes"] = sst_age
        if sst:
            coverage["sst"] += 1
            for key in ("sst_c", "sst_k", "sst_sample_distance_km"):
                features[key] = finite_float(sst.get(key))
        source_flags["sst"] = sst is not None

        cloud, cloud_age = asof_row(cloud_index, spot_id, target_time, args.cloud_mask_max_age_minutes)
        features["eumetsat_cloud_mask_available"] = cloud is not None
        features["eumetsat_cloud_mask_age_minutes"] = cloud_age
        if cloud:
            coverage["eumetsat_cloud_mask"] += 1
            for key in ("cloud_state", "cloud_state_mode", "cloud_state_valid_count", "sample_distance_km"):
                features[f"eumetsat_{key}"] = finite_float(cloud.get(key)) if key != "cloud_state_mode" else cloud.get(key)
            fractions = cloud.get("cloud_state_fractions")
            if isinstance(fractions, dict):
                for key, value in sorted(fractions.items()):
                    features[f"eumetsat_cloud_state_fraction_{key}"] = finite_float(value)
        source_flags["eumetsat_cloud_mask"] = cloud is not None

        for product, index in eumetsat_product_indices.items():
            product_sample, product_age = asof_row(index, spot_id, target_time, args.eumetsat_spot_product_max_age_minutes)
            attach_eumetsat_spot_product_features(features, product, product_sample, product_age, coverage)
            source_flags[f"eumetsat_{product}"] = product_sample is not None

        rows.append({
            "format": "corsewind.ml_spot_feature_store_15min.v1",
            "target_time_utc": target_time_iso,
            "spot_id": spot_id,
            "spot_name": spot.get("name") or target_obs.get("spot_name"),
            "spot_kind": spot.get("kind") or target_obs.get("spot_kind"),
            "spot_source_type": spot.get("source_type") or target_obs.get("source_type"),
            "station_id": spot.get("station_id") or target_obs.get("station_id"),
            "latitude": finite_float(spot.get("latitude")) or finite_float(target_obs.get("latitude")),
            "longitude": finite_float(spot.get("longitude")) or finite_float(target_obs.get("longitude")),
            "use_for_ml": bool(spot.get("use_for_ml", target_obs.get("use_for_ml", False))),
            "targets": targets_payload,
            "features": features,
            "feature_sources": source_flags,
            "built_at_utc": utc_now(),
        })
    report = {
        "generated_at_utc": utc_now(),
        "row_count": len(rows),
        "target_candidate_count": len(targets),
        "inference_grid_target_count": inference_grid_target_count,
        "target_candidates_by_source_dataset": dict(sorted(Counter(str(row.get("source_dataset")) for row in targets.values()).items())),
        "target_candidates_by_spot": dict(sorted(Counter(str(row.get("spot_id")) for row in targets.values()).items())),
        "output_rows_by_target_source_dataset": dict(sorted(Counter(str((row.get("targets") or {}).get("observation_source_dataset")) for row in rows).items())),
        "output_rows_by_target_source_type": dict(sorted(Counter(str((row.get("targets") or {}).get("observation_source_type")) for row in rows).items())),
        "observation_row_count": len(observations),
        "target_observation_row_count": len(target_observations),
        "context_station_count": len(context_stations),
        "spot_static_feature_spot_count": len(spot_static_features),
        "context_station_slot_count": sum(len(slots) for slots in context_slots.values()),
        "context_spot_with_slot_count": sum(1 for slots in context_slots.values() if slots),
        "model_sample_row_count": len(model_rows),
        "open_meteo_historical_forecast_row_count": len(open_meteo_rows),
        "open_meteo_previous_run_row_count": len(previous_run_rows),
        "extra_field_row_count": len(extra_rows),
        "vertical_profile_row_count": len(vertical_rows),
        "spots": len({row["spot_id"] for row in rows}),
        "first_target_time_utc": rows[0]["target_time_utc"] if rows else None,
        "last_target_time_utc": rows[-1]["target_time_utc"] if rows else None,
        "feature_source_hits": dict(sorted(coverage.items())),
        "source_flag_counts": source_flag_counts(rows),
        "settings": {
            "step_minutes": args.step_minutes,
            "start_datetime": args.start_datetime,
            "end_datetime": args.end_datetime,
            "read_dates": sorted(read_dates) if read_dates else None,
            "target_dates": sorted(target_dates) if target_dates else None,
            "read_margin_days_before": args.read_margin_days_before,
            "target_tolerance_minutes": args.target_tolerance_minutes,
            "include_inference_grid": args.include_inference_grid,
            "observation_history_max_age_minutes": args.observation_history_max_age_minutes,
            "context_station_max_age_minutes": args.context_station_max_age_minutes,
            "context_nearest_count": args.context_nearest_count,
            "context_per_role_count": args.context_per_role_count,
            "context_global_nearest_count": args.context_global_nearest_count,
            "context_global_role_count": args.context_global_role_count,
            "context_global_max_distance_km": args.context_global_max_distance_km,
            "forecast_valid_tolerance_minutes": args.forecast_valid_tolerance_minutes,
            "open_meteo_models": list(open_meteo_models),
            "open_meteo_offset_points": open_meteo_offset_points,
            "open_meteo_previous_run_models": list(previous_run_models),
            "open_meteo_previous_run_lead_days": list(previous_run_lead_days),
            "sst_max_age_minutes": args.sst_max_age_minutes,
            "cloud_mask_max_age_minutes": args.cloud_mask_max_age_minutes,
            "eumetsat_spot_product_max_age_minutes": args.eumetsat_spot_product_max_age_minutes,
        },
    }
    return rows, report


def source_flag_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for key, value in (row.get("feature_sources") or {}).items():
            if value:
                counts[key] += 1
    return dict(sorted(counts.items()))


def write_outputs(rows: list[dict[str, Any]], report: dict[str, Any], output_root: Path, schema_doc: Path) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    jsonl = output_root / "spot_forecast_15min.jsonl"
    report_path = output_root / "spot_forecast_15min_profile.json"
    fields_path = output_root / "spot_forecast_15min_feature_columns.csv"
    jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    feature_names = sorted({key for row in rows for key in (row.get("features") or {})})
    target_names = sorted({key for row in rows for key in (row.get("targets") or {})})
    with fields_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ["kind", "name"])
        writer.writeheader()
        for name in target_names:
            writer.writerow({"kind": "target", "name": name})
        for name in feature_names:
            writer.writerow({"kind": "feature", "name": name})
    write_schema_doc(schema_doc, report, feature_names, target_names)
    return {"jsonl": str(jsonl), "profile": str(report_path), "columns": str(fields_path), "schema_doc": str(schema_doc)}


def write_schema_doc(path: Path, report: dict[str, Any], feature_names: list[str], target_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Spot Feature Store 15 min",
        "",
        f"Generated at: `{report['generated_at_utc']}`",
        "",
        "## Grain",
        "",
        "One row per `spot_id + target_time_utc`, rounded to a 15-minute grid from available target observations.",
        "",
        "## Outputs",
        "",
        "- `data/processed/ml_dataset/feature_store/spot_forecast_15min.jsonl`",
        "- `data/processed/ml_dataset/feature_store/spot_forecast_15min_profile.json`",
        "- `data/processed/ml_dataset/feature_store/spot_forecast_15min_feature_columns.csv`",
        "",
        "## Current Coverage",
        "",
        f"- rows: `{report['row_count']}`",
        f"- spots: `{report['spots']}`",
        f"- time range: `{report['first_target_time_utc']}` -> `{report['last_target_time_utc']}`",
        "",
        "### Source Flag Counts",
        "",
        "| Source | Rows with source |",
        "| --- | ---: |",
    ]
    for key, value in report.get("source_flag_counts", {}).items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend([
        "",
        "## Targets",
        "",
        "| Name |",
        "| --- |",
    ])
    for name in target_names:
        lines.append(f"| `{name}` |")
    lines.extend(["", "## Features", "", "| Name |", "| --- |"])
    for name in feature_names:
        lines.append(f"| `{name}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-root", type=Path, default=DEFAULT_ML_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--context-registry", type=Path, default=DEFAULT_CONTEXT_REGISTRY)
    parser.add_argument("--spot-static-features", type=Path, default=DEFAULT_SPOT_STATIC_FEATURES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--schema-doc", type=Path, default=DEFAULT_SCHEMA_DOC)
    parser.add_argument("--start-datetime", help="Optional inclusive target start time.")
    parser.add_argument("--end-datetime", help="Optional inclusive target end time.")
    parser.add_argument("--read-margin-days-before", type=int, default=3)
    parser.add_argument("--step-minutes", type=int, default=15)
    parser.add_argument("--target-tolerance-minutes", type=float, default=8)
    parser.add_argument("--include-inference-grid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--observation-history-max-age-minutes", type=float, default=180)
    parser.add_argument("--context-station-max-age-minutes", type=float, default=180)
    parser.add_argument("--context-nearest-count", type=int, default=3)
    parser.add_argument("--context-per-role-count", type=int, default=2)
    parser.add_argument("--context-global-nearest-count", type=int, default=4)
    parser.add_argument("--context-global-role-count", type=int, default=1)
    parser.add_argument("--context-global-max-distance-km", type=float, default=80)
    parser.add_argument("--forecast-valid-tolerance-minutes", type=float, default=31)
    parser.add_argument("--open-meteo-models", default=",".join(OPEN_METEO_MODELS))
    parser.add_argument("--open-meteo-offset-points", default=",".join(OPEN_METEO_OFFSET_POINTS), help="Comma-separated name:bearing_deg:distance_km offset points.")
    parser.add_argument("--open-meteo-previous-run-models", default=",".join(OPEN_METEO_PREVIOUS_RUN_MODELS))
    parser.add_argument("--open-meteo-previous-run-lead-days", default=",".join(str(value) for value in OPEN_METEO_PREVIOUS_RUN_LEAD_DAYS))
    parser.add_argument("--sst-max-age-minutes", type=float, default=48 * 60)
    parser.add_argument("--cloud-mask-max-age-minutes", type=float, default=180)
    parser.add_argument("--eumetsat-spot-product-max-age-minutes", type=float, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, report = build_rows(args)
    outputs = write_outputs(rows, report, resolve_path(args.output_root), resolve_path(args.schema_doc))
    print(json.dumps({**report, "outputs": outputs}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
