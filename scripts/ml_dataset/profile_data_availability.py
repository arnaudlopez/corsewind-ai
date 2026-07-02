#!/usr/bin/env python3
"""Profile currently available ML dataset sources and fields."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_BEACON_STATE = Path("/Users/arnaud/Documents/beacon-live-app/data/weather-state.json")
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_OUTPUT_JSON = DEFAULT_ML_ROOT / "source_inventories/data_availability_profile.json"
DEFAULT_OUTPUT_MD = ROOT / "docs/ml_nowcasting/data_availability_profile.md"
EUMETSAT_SPOT_PRODUCTS = {
    "cloud_type": "Cloud Type",
    "land_surface_temperature": "Land Surface Temperature",
    "global_instability_indices": "Global Instability Indices",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def flatten_keys(value: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            keys.add(name)
            keys.update(flatten_keys(child, name))
    return keys


def is_present(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and value == "")


def non_null_counter(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, int]:
    return {field: sum(1 for row in rows if is_present(row.get(field))) for field in fields}


def minmax_time(rows: list[dict[str, Any]], key: str) -> tuple[str | None, str | None]:
    values = sorted(str(row[key]) for row in rows if row.get(key))
    return (values[0], values[-1]) if values else (None, None)


def profile_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    payload = read_json(path)
    spots = payload.get("spots", [])
    return {
        "exists": True,
        "spot_count": len(spots),
        "use_for_ml": dict(Counter(str(item.get("use_for_ml")) for item in spots)),
        "source_type": dict(Counter(item.get("source_type") for item in spots)),
        "kind": dict(Counter(item.get("kind") for item in spots)),
        "source_resolution_minutes": dict(Counter(str(item.get("source_resolution_minutes")) for item in spots)),
    }


def profile_beacon_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    payload = read_json(path)
    observations = [item for item in payload.get("observations", []) if isinstance(item, dict)]
    live_rows = [
        item.get("payload", {}).get("live", {})
        for item in observations
        if isinstance(item.get("payload", {}).get("live", {}), dict)
    ]
    field_counts = Counter()
    for live in live_rows:
        for key, value in live.items():
            if value not in {None, ""}:
                field_counts[key] += 1
    first_observed, last_observed = minmax_time(observations, "observedAt")
    return {
        "exists": True,
        "updated_at": payload.get("updatedAt"),
        "observation_count": len(observations),
        "first_observed_at": first_observed,
        "last_observed_at": last_observed,
        "source_count": len({item.get("sourceId") for item in observations}),
        "live_field_non_null_counts": dict(sorted(field_counts.items())),
        "source_health_count": len(payload.get("sourceHealth", {}) or {}),
    }


def profile_model_samples(root: Path) -> dict[str, Any]:
    samples_root = root / "model_samples"
    files = sorted(samples_root.glob("source=*/date=*/samples.jsonl"))
    by_source: dict[str, dict[str, Any]] = {}
    fields = [
        "wind_speed_ms",
        "gust_speed_ms",
        "wind_u_ms",
        "wind_v_ms",
        "wind_direction_deg",
        "lead_minutes",
    ]
    for file in files:
        source = file.parts[-3].split("=", 1)[-1]
        rows = iter_jsonl(file)
        item = by_source.setdefault(source, {
            "file_count": 0,
            "row_count": 0,
            "first_valid_time_utc": None,
            "last_valid_time_utc": None,
            "spot_count": 0,
            "fields_non_null": {field: 0 for field in fields},
        })
        item["file_count"] += 1
        item["row_count"] += len(rows)
        times = [item["first_valid_time_utc"], item["last_valid_time_utc"], *[row.get("valid_time_utc") for row in rows]]
        times = sorted(str(value) for value in times if value)
        item["first_valid_time_utc"] = times[0] if times else None
        item["last_valid_time_utc"] = times[-1] if times else None
        item.setdefault("_spots", set()).update(row.get("spot_id") for row in rows if row.get("spot_id"))
        counts = non_null_counter(rows, fields)
        for field, count in counts.items():
            item["fields_non_null"][field] += count
    for item in by_source.values():
        item["spot_count"] = len(item.pop("_spots", set()))
    return {
        "exists": samples_root.exists(),
        "file_count": len(files),
        "row_count": sum(item["row_count"] for item in by_source.values()),
        "sources": by_source,
    }


def profile_meteo_france_observations(root: Path) -> dict[str, Any]:
    obs_roots = [
        root / "observations/meteo_france",
        root / "observations/meteo_france_climatology/normalized",
        root / "observations/windsup/normalized",
    ]
    files = sorted(
        file
        for obs_root in obs_roots
        for file in [
            *obs_root.glob("source_dataset=*/date=*/observations.jsonl"),
            *obs_root.glob("frequency=*/date=*/observations.jsonl"),
            *obs_root.glob("date=*/observations.jsonl"),
        ]
    )
    fields = [
        "wind_mean_ms",
        "gust_ms",
        "wind_mean_kt_raw",
        "gust_kt_raw",
        "gust_instant_ms",
        "gust_max_ms",
        "wind_direction_deg",
        "temperature_c",
        "dewpoint_c",
        "humidity_pct",
        "pressure_hpa",
        "pressure_station_hpa",
        "sea_level_pressure_hpa",
        "precipitation_mm",
        "precipitation_1h_mm",
        "visibility_m",
        "cloud_cover_code",
        "cloud_cover_octa",
        "low_cloud_cover_octa",
        "sunshine_minutes",
        "sunshine_duration_minutes",
        "global_radiation_raw",
        "global_radiation_j_cm2",
        "direct_radiation_j_cm2",
        "diffuse_radiation_j_cm2",
        "soil_temperature_10cm_c",
        "sea_temperature_c",
        "weather_code",
    ]
    by_dataset: dict[str, dict[str, Any]] = {}
    for file in files:
        rows = iter_jsonl(file)
        dataset = rows[0].get("source_dataset") if rows else None
        if not dataset:
            dataset = file.parts[-3].split("=", 1)[-1]
        item = by_dataset.setdefault(dataset, {
            "file_count": 0,
            "row_count": 0,
            "first_timestamp_utc": None,
            "last_timestamp_utc": None,
            "source_count": 0,
            "spot_count": 0,
            "fields_non_null": {field: 0 for field in fields},
        })
        item["file_count"] += 1
        item["row_count"] += len(rows)
        times = [item["first_timestamp_utc"], item["last_timestamp_utc"], *[row.get("timestamp_utc") for row in rows]]
        times = sorted(str(value) for value in times if value)
        item["first_timestamp_utc"] = times[0] if times else None
        item["last_timestamp_utc"] = times[-1] if times else None
        item.setdefault("_sources", set()).update(
            row.get("source_id") or row.get("station_id")
            for row in rows
            if row.get("source_id") or row.get("station_id")
        )
        item.setdefault("_spots", set()).update(row.get("spot_id") for row in rows if row.get("spot_id"))
        spot_stats = item.setdefault("_spot_stats", {})
        for row in rows:
            spot_id = row.get("spot_id")
            if not spot_id:
                continue
            stats = spot_stats.setdefault(spot_id, {
                "row_count": 0,
                "first_timestamp_utc": None,
                "last_timestamp_utc": None,
                "source_ids": set(),
            })
            stats["row_count"] += 1
            timestamp = row.get("timestamp_utc")
            if timestamp:
                times = sorted(
                    str(value)
                    for value in [stats["first_timestamp_utc"], stats["last_timestamp_utc"], timestamp]
                    if value
                )
                stats["first_timestamp_utc"] = times[0]
                stats["last_timestamp_utc"] = times[-1]
            source_id = row.get("source_id") or row.get("station_id")
            if source_id:
                stats["source_ids"].add(str(source_id))
        counts = non_null_counter(rows, fields)
        for field, count in counts.items():
            item["fields_non_null"][field] += count
    for item in by_dataset.values():
        item["source_count"] = len(item.pop("_sources", set()))
        item["spot_count"] = len(item.pop("_spots", set()))
        spot_stats = item.pop("_spot_stats", {})
        item["spot_summaries"] = {
            spot_id: {
                "row_count": stats["row_count"],
                "first_timestamp_utc": stats["first_timestamp_utc"],
                "last_timestamp_utc": stats["last_timestamp_utc"],
                "source_ids": sorted(stats["source_ids"]),
            }
            for spot_id, stats in sorted(spot_stats.items())
        }
    return {
        "exists": any(obs_root.exists() for obs_root in obs_roots),
        "roots": [str(obs_root) for obs_root in obs_roots if obs_root.exists()],
        "file_count": len(files),
        "row_count": sum(item["row_count"] for item in by_dataset.values()),
        "datasets": by_dataset,
    }


def profile_copernicus_marine_sst(root: Path) -> dict[str, Any]:
    sst_root = root / "copernicus_marine/sst_samples"
    files = sorted(sst_root.glob("date=*/sst_samples.jsonl"))
    fields = [
        "sst_c",
        "sst_k",
        "sst_pixel_latitude",
        "sst_pixel_longitude",
        "sst_sample_distance_km",
    ]
    item: dict[str, Any] = {
        "exists": sst_root.exists(),
        "file_count": len(files),
        "row_count": 0,
        "first_timestamp_utc": None,
        "last_timestamp_utc": None,
        "spot_count": 0,
        "fields_non_null": {field: 0 for field in fields},
    }
    spots = set()
    for file in files:
        rows = iter_jsonl(file)
        item["row_count"] += len(rows)
        times = [item["first_timestamp_utc"], item["last_timestamp_utc"], *[row.get("timestamp_utc") for row in rows]]
        times = sorted(str(value) for value in times if value)
        item["first_timestamp_utc"] = times[0] if times else None
        item["last_timestamp_utc"] = times[-1] if times else None
        spots.update(row.get("spot_id") for row in rows if row.get("spot_id"))
        counts = non_null_counter(rows, fields)
        for field, count in counts.items():
            item["fields_non_null"][field] += count
    item["spot_count"] = len(spots)
    return item


def profile_eumetsat_cloud_mask(root: Path) -> dict[str, Any]:
    cloud_root = root / "eumetsat/cloud_mask_samples"
    files = sorted(cloud_root.glob("date=*/cloud_mask_samples.jsonl"))
    fields = [
        "cloud_state",
        "cloud_state_mode",
        "cloud_state_valid_count",
        "sample_distance_km",
        "product_quality",
        "product_completeness",
        "product_timeliness",
    ]
    item: dict[str, Any] = {
        "exists": cloud_root.exists(),
        "file_count": len(files),
        "row_count": 0,
        "first_sensing_start_utc": None,
        "last_sensing_start_utc": None,
        "spot_count": 0,
        "product_count": 0,
        "cloud_state_counts": {},
        "fields_non_null": {field: 0 for field in fields},
    }
    spots = set()
    products = set()
    cloud_states: Counter[Any] = Counter()
    for file in files:
        rows = iter_jsonl(file)
        item["row_count"] += len(rows)
        times = [item["first_sensing_start_utc"], item["last_sensing_start_utc"], *[row.get("sensing_start_utc") for row in rows]]
        times = sorted(str(value) for value in times if value)
        item["first_sensing_start_utc"] = times[0] if times else None
        item["last_sensing_start_utc"] = times[-1] if times else None
        spots.update(row.get("spot_id") for row in rows if row.get("spot_id"))
        products.update(row.get("product_id") for row in rows if row.get("product_id"))
        cloud_states.update(str(row.get("cloud_state")) for row in rows if row.get("cloud_state") is not None)
        counts = non_null_counter(rows, fields)
        for field, count in counts.items():
            item["fields_non_null"][field] += count
    item["spot_count"] = len(spots)
    item["product_count"] = len(products)
    item["cloud_state_counts"] = dict(sorted(cloud_states.items()))
    return item


def profile_eumetsat_spot_product(root: Path, product: str) -> dict[str, Any]:
    output_name = product
    product_root = root / f"eumetsat/{output_name}_samples"
    files = sorted(product_root.glob(f"date=*/{output_name}_samples.jsonl"))
    item: dict[str, Any] = {
        "exists": product_root.exists(),
        "file_count": len(files),
        "row_count": 0,
        "first_sensing_start_utc": None,
        "last_sensing_start_utc": None,
        "spot_count": 0,
        "product_count": 0,
        "sampled_variables": {},
        "fields_non_null": {
            "sample_distance_km": 0,
            "sampled_values": 0,
            "sampled_values_c": 0,
            "neighborhoods": 0,
            "product_quality": 0,
            "product_completeness": 0,
            "product_timeliness": 0,
        },
    }
    spots = set()
    products = set()
    sampled_variables: Counter[str] = Counter()
    fields = list(item["fields_non_null"])
    for file in files:
        rows = iter_jsonl(file)
        item["row_count"] += len(rows)
        times = [item["first_sensing_start_utc"], item["last_sensing_start_utc"], *[row.get("sensing_start_utc") for row in rows]]
        times = sorted(str(value) for value in times if value)
        item["first_sensing_start_utc"] = times[0] if times else None
        item["last_sensing_start_utc"] = times[-1] if times else None
        spots.update(row.get("spot_id") for row in rows if row.get("spot_id"))
        products.update(row.get("product_id") for row in rows if row.get("product_id"))
        for row in rows:
            sampled = row.get("sampled_values")
            if isinstance(sampled, dict):
                sampled_variables.update(key for key, value in sampled.items() if value not in {None, ""})
        counts = non_null_counter(rows, fields)
        for field, count in counts.items():
            item["fields_non_null"][field] += count
    item["spot_count"] = len(spots)
    item["product_count"] = len(products)
    item["sampled_variables"] = dict(sampled_variables.most_common())
    return item


def profile_nwp_extra_fields(root: Path) -> dict[str, Any]:
    sample_root = root / "meteo_france_nwp/extra_field_samples"
    files = sorted(sample_root.glob("source=*/date=*/extra_fields.jsonl"))
    item: dict[str, Any] = {
        "exists": sample_root.exists(),
        "file_count": len(files),
        "row_count": 0,
        "sources": {},
    }
    by_source: dict[str, dict[str, Any]] = {}
    for file in files:
        rows = iter_jsonl(file)
        source = file.parts[-3].split("=", 1)[-1]
        source_item = by_source.setdefault(source, {
            "row_count": 0,
            "spot_count": 0,
            "first_valid_time_utc": None,
            "last_valid_time_utc": None,
            "features_non_null": defaultdict(int),
            "features_seen": set(),
        })
        source_item["row_count"] += len(rows)
        item["row_count"] += len(rows)
        times = [source_item["first_valid_time_utc"], source_item["last_valid_time_utc"], *[row.get("valid_time_utc") for row in rows]]
        times = sorted(str(value) for value in times if value)
        source_item["first_valid_time_utc"] = times[0] if times else None
        source_item["last_valid_time_utc"] = times[-1] if times else None
        spots = set(row.get("spot_id") for row in rows if row.get("spot_id"))
        source_item.setdefault("_spots", set()).update(spots)
        for row in rows:
            features = row.get("features")
            if not isinstance(features, dict):
                continue
            source_item["features_seen"].update(features.keys())
            for key, value in features.items():
                if value not in {None, ""}:
                    source_item["features_non_null"][key] += 1
    for source, source_item in by_source.items():
        source_item["spot_count"] = len(source_item.pop("_spots", set()))
        source_item["features_seen"] = sorted(source_item["features_seen"])
        source_item["features_non_null"] = dict(sorted(source_item["features_non_null"].items()))
    item["sources"] = by_source
    return item


def profile_nwp_vertical_profiles(root: Path) -> dict[str, Any]:
    sample_root = root / "meteo_france_nwp/vertical_profiles"
    files = sorted(sample_root.glob("source=*/resolution=*/date=*/vertical_profiles.jsonl"))
    item: dict[str, Any] = {
        "exists": sample_root.exists(),
        "file_count": len(files),
        "row_count": 0,
        "sources": {},
    }
    by_source: dict[str, dict[str, Any]] = {}
    for file in files:
        rows = iter_jsonl(file)
        source = file.parts[-4].split("=", 1)[-1]
        resolution = file.parts[-3].split("=", 1)[-1]
        key = f"{source}_{resolution}"
        source_item = by_source.setdefault(key, {
            "row_count": 0,
            "spot_count": 0,
            "first_valid_time_utc": None,
            "last_valid_time_utc": None,
            "pressure_levels_hpa": set(),
            "profile_features_seen": set(),
            "derived_features_seen": set(),
        })
        source_item["row_count"] += len(rows)
        item["row_count"] += len(rows)
        times = [source_item["first_valid_time_utc"], source_item["last_valid_time_utc"], *[row.get("valid_time_utc") for row in rows]]
        times = sorted(str(value) for value in times if value)
        source_item["first_valid_time_utc"] = times[0] if times else None
        source_item["last_valid_time_utc"] = times[-1] if times else None
        source_item.setdefault("_spots", set()).update(row.get("spot_id") for row in rows if row.get("spot_id"))
        for row in rows:
            for level in row.get("pressure_levels_hpa") or []:
                source_item["pressure_levels_hpa"].add(int(level))
            profile = row.get("profile")
            if isinstance(profile, dict):
                source_item["profile_features_seen"].update(profile.keys())
            derived = row.get("derived_features")
            if isinstance(derived, dict):
                source_item["derived_features_seen"].update(key for key, value in derived.items() if value not in {None, ""})
    for source_item in by_source.values():
        source_item["spot_count"] = len(source_item.pop("_spots", set()))
        source_item["pressure_levels_hpa"] = sorted(source_item["pressure_levels_hpa"], reverse=True)
        source_item["profile_features_seen"] = sorted(source_item["profile_features_seen"])
        source_item["derived_features_seen"] = sorted(source_item["derived_features_seen"])
    item["sources"] = by_source
    return item


def profile_feature_store(root: Path) -> dict[str, Any]:
    feature_root = root / "feature_store"
    candidates = []
    top_level = feature_root / "spot_forecast_15min.jsonl"
    if top_level.exists():
        candidates.append(top_level)
    candidates.extend(feature_root.glob("*/spot_forecast_15min.jsonl"))
    candidates.extend(feature_root.glob("*/chunk=*/spot_forecast_15min.jsonl"))
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    jsonl = candidates[0] if candidates else top_level
    selected_root = jsonl.parent
    profile_json = selected_root / "spot_forecast_15min_profile.json"
    rows = iter_jsonl(jsonl)
    spot_ids = {row.get("spot_id") for row in rows if row.get("spot_id")}
    first_time, last_time = minmax_time(rows, "target_time_utc")
    source_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    feature_counts: Counter[str] = Counter()
    for row in rows:
        for key, value in (row.get("feature_sources") or {}).items():
            if value:
                source_counts[key] += 1
        for key, value in (row.get("targets") or {}).items():
            if value not in {None, ""}:
                target_counts[key] += 1
        for key, value in (row.get("features") or {}).items():
            if value not in {None, ""}:
                feature_counts[key] += 1
    return {
        "exists": jsonl.exists(),
        "root": str(selected_root),
        "jsonl": str(jsonl),
        "discovered_store_count": len(candidates),
        "profile_exists": profile_json.exists(),
        "row_count": len(rows),
        "spot_count": len(spot_ids),
        "first_target_time_utc": first_time,
        "last_target_time_utc": last_time,
        "source_flag_counts": dict(sorted(source_counts.items())),
        "target_non_null": dict(target_counts.most_common()),
        "top_feature_non_null": dict(feature_counts.most_common(30)),
        "profile": read_json(profile_json) if profile_json.exists() else {},
    }


def profile_training_tables(root: Path) -> dict[str, Any]:
    training_root = root / "training_tables"
    profile_files = sorted(
        training_root.glob("*/training_profile.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    tables: dict[str, dict[str, Any]] = {}
    for profile_file in profile_files:
        payload = read_json(profile_file)
        table_name = profile_file.parent.name
        rows_path = profile_file.parent / "training_rows.jsonl"
        columns_path = profile_file.parent / "training_columns.csv"
        evaluation_path = profile_file.parent / "evaluation.json"
        evaluation = read_json(evaluation_path) if evaluation_path.exists() else {}
        tables[table_name] = {
            "root": str(profile_file.parent),
            "profile": str(profile_file),
            "rows_path": str(rows_path),
            "columns_path": str(columns_path),
            "evaluation_path": str(evaluation_path),
            "rows_exists": rows_path.exists(),
            "columns_exists": columns_path.exists(),
            "evaluation_exists": evaluation_path.exists(),
            "training_row_count": payload.get("training_row_count", 0),
            "source_feature_row_count": payload.get("source_feature_row_count", payload.get("chunk_training_row_refs", 0)),
            "training_rows_by_lead": payload.get("training_rows_by_lead", {}),
            "training_rows_by_spot": payload.get("training_rows_by_spot", {}),
            "missing_baseline_wind_by_lead": payload.get("missing_baseline_wind_by_lead", {}),
            "missing_target_wind_by_lead": payload.get("missing_target_wind_by_lead", {}),
            "evaluation_comparisons": evaluation.get("comparisons", {}),
            "categorical_feature_count": len(payload.get("categorical_features", [])),
            "settings": payload.get("settings", {}),
        }
    latest = next(iter(tables), None)
    return {
        "exists": training_root.exists() and bool(tables),
        "root": str(training_root),
        "table_count": len(tables),
        "latest_table": latest,
        "total_training_rows": sum(item.get("training_row_count", 0) for item in tables.values()),
        "tables": tables,
    }


def profile_trained_models(root: Path) -> dict[str, Any]:
    model_root = root / "models"
    result_files = sorted(
        model_root.glob("*/training_results.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    runs: dict[str, dict[str, Any]] = {}
    for result_file in result_files:
        payload = read_json(result_file)
        run_id = result_file.parent.name
        model_files = sorted(result_file.parent.glob("*.joblib"))
        model_summaries = {}
        for target, item in (payload.get("models") or {}).items():
            model_summaries[target] = {
                "type": item.get("type"),
                "rmse_gain_pct_vs_raw": item.get("rmse_gain_pct_vs_raw"),
                "corrected_nwp_test": item.get("corrected_nwp_test"),
                "raw_nwp_test": item.get("raw_nwp_test"),
                "classification_test": item.get("test"),
            }
        runs[run_id] = {
            "root": str(result_file.parent),
            "training_results": str(result_file),
            "model_file_count": len(model_files),
            "row_count": payload.get("row_count", 0),
            "train_row_count": payload.get("train_row_count", 0),
            "test_row_count": payload.get("test_row_count", 0),
            "temporal_split_issue_time_utc": payload.get("temporal_split_issue_time_utc"),
            "models": model_summaries,
            "skipped_targets": payload.get("skipped_targets", {}),
        }
    latest = next(iter(runs), None)
    return {
        "exists": model_root.exists() and bool(runs),
        "root": str(model_root),
        "run_count": len(runs),
        "latest_run": latest,
        "runs": runs,
    }


def profile_meteonet_ground_stations(root: Path) -> dict[str, Any]:
    profile_path = root / "research/meteonet/normalized/ground_stations/profile.json"
    raw_root = root / "research/meteonet/raw/SE/ground_stations"
    raw_files = sorted(raw_root.glob("*.tar.gz"))
    if not profile_path.exists():
        return {
            "exists": False,
            "raw_file_count": len(raw_files),
            "raw_size_bytes": sum(path.stat().st_size for path in raw_files),
        }
    payload = read_json(profile_path)
    year_summaries = {
        str(item.get("year")): {
            "row_count": item.get("row_count", 0),
            "station_count": item.get("station_count", 0),
            "first_timestamp_utc": item.get("first_timestamp_utc"),
            "last_timestamp_utc": item.get("last_timestamp_utc"),
            "wind_mean_non_null": (item.get("field_non_null_counts") or {}).get("wind_mean_ms", 0),
            "wind_direction_non_null": (item.get("field_non_null_counts") or {}).get("wind_direction_deg", 0),
            "temperature_non_null": (item.get("field_non_null_counts") or {}).get("temperature_c", 0),
            "pressure_non_null": (item.get("field_non_null_counts") or {}).get("sea_level_pressure_hpa", 0),
        }
        for item in payload.get("summaries", [])
    }
    return {
        "exists": True,
        "profile_path": str(profile_path),
        "raw_file_count": len(raw_files),
        "raw_size_bytes": sum(path.stat().st_size for path in raw_files),
        "row_count": payload.get("row_count", 0),
        "station_count": payload.get("station_count", 0),
        "first_timestamp_utc": payload.get("first_timestamp_utc"),
        "last_timestamp_utc": payload.get("last_timestamp_utc"),
        "station_registry": payload.get("station_registry"),
        "year_summaries": year_summaries,
    }


def profile_copernicus_marine_inventory(root: Path) -> dict[str, Any]:
    path = root / "source_inventories/copernicus_marine_products.json"
    if not path.exists():
        return {"exists": False}
    payload = read_json(path)
    candidates = payload.get("candidates", [])
    return {
        "exists": True,
        "generated_at_utc": payload.get("generated_at_utc"),
        "candidate_count": len(candidates),
        "decisions": dict(Counter(item.get("decision") for item in candidates)),
        "feature_families": [item.get("feature_family") for item in candidates],
    }


def profile_eumetsat_inventory(root: Path) -> dict[str, Any]:
    path = root / "source_inventories/eumetsat_products.json"
    if not path.exists():
        return {"exists": False}
    payload = read_json(path)
    candidates = payload.get("candidates", [])
    return {
        "exists": True,
        "generated_at_utc": payload.get("generated_at_utc"),
        "candidate_count": len(candidates),
        "error_count": payload.get("error_count", 0),
        "decisions": dict(Counter(item.get("decision") for item in candidates)),
        "feature_families": [item.get("feature_family") for item in candidates],
    }


def profile_eumetsat_catalog_keyword_inventory(root: Path) -> dict[str, Any]:
    path = root / "source_inventories/eumetsat_catalog_keyword_inventory.json"
    if not path.exists():
        return {"exists": False}
    payload = read_json(path)
    groups = payload.get("groups", {})
    return {
        "exists": True,
        "generated_at_utc": payload.get("generated_at_utc"),
        "collection_count": payload.get("collection_count", 0),
        "matched_count": payload.get("matched_count", 0),
        "group_counts": {group: len(items) for group, items in groups.items()},
        "top_collections": [
            item.get("collection_id")
            for item in payload.get("top_matches", [])[:10]
        ],
    }


def profile_wcs_inventory(root: Path) -> dict[str, Any]:
    path = root / "source_inventories/meteo_france_wcs_variables.json"
    if not path.exists():
        return {"exists": False}
    payload = read_json(path)
    services = {}
    for service in payload.get("services", []):
        key = f"{service.get('product')}_{service.get('resolution')}"
        families = service.get("families", {})
        services[key] = {
            "coverage_count": service.get("coverage_count"),
            "variable_count": service.get("variable_count"),
            "families": {
                name: {
                    "coverage_count": item.get("coverage_count"),
                    "variable_count": item.get("variable_count"),
                    "variables": item.get("variables", [])[:20],
                }
                for name, item in families.items()
            },
        }
    return {
        "exists": True,
        "generated_at_utc": payload.get("generated_at_utc"),
        "service_count": len(services),
        "services": services,
    }


def external_access_status() -> dict[str, Any]:
    return {
        "copernicus_marine": {
            "configured": bool(
                os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
                and os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
            ),
            "required_env": ["COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD"],
            "target_data": ["sst_nearest_c", "land_minus_sea_temp_c"],
        },
        "eumetsat": {
            "configured": bool(os.environ.get("EUMETSAT_CONSUMER_KEY") and os.environ.get("EUMETSAT_CONSUMER_SECRET")),
            "required_env": ["EUMETSAT_CONSUMER_KEY", "EUMETSAT_CONSUMER_SECRET"],
            "target_data": ["cloud_fraction_satellite", "cloud_type", "cloud_top_height"],
        },
        "cds_era5": {
            "configured": bool(os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY")),
            "required_env": ["CDSAPI_URL", "CDSAPI_KEY"],
            "target_data": ["historical_reanalysis_context"],
        },
    }


def pct(count: int, total: int) -> str:
    if total <= 0:
        return "0%"
    return f"{count / total * 100:.0f}%"


def write_markdown(path: Path, profile: dict[str, Any]) -> None:
    lines = [
        "# Data Availability Profile",
        "",
        f"Generated at: `{profile['generated_at_utc']}`",
        "",
        "## Summary",
        "",
        "| Source | Status | Rows / coverage | Notes |",
        "| --- | --- | ---: | --- |",
    ]
    registry = profile["registry"]
    beacon = profile["beacon_live"]
    samples = profile["model_samples"]
    nwp_extra = profile["nwp_extra_fields"]
    nwp_vertical = profile["nwp_vertical_profiles"]
    feature_store = profile["feature_store"]
    training_tables = profile["training_tables"]
    trained_models = profile["trained_models"]
    mfobs = profile["meteo_france_observations"]
    meteonet = profile["meteonet_ground_stations"]
    cop_sst = profile["copernicus_marine_sst"]
    eum_cloud = profile["eumetsat_cloud_mask"]
    eum_spot_products = profile.get("eumetsat_spot_products", {})
    cop_inv = profile["copernicus_marine_inventory"]
    eum_inv = profile["eumetsat_inventory"]
    eum_catalog = profile["eumetsat_catalog_keyword_inventory"]
    wcs = profile["meteo_france_wcs"]
    lines.extend([
        f"| Spot registry | {'OK' if registry.get('exists') else 'missing'} | {registry.get('spot_count', 0)} spots | {registry.get('use_for_ml', {})} |",
        f"| Beacon Live snapshot | {'OK' if beacon.get('exists') else 'missing'} | {beacon.get('observation_count', 0)} obs | live fields: {', '.join(beacon.get('live_field_non_null_counts', {}).keys())} |",
        f"| Model samples | {'OK' if samples.get('exists') else 'missing'} | {samples.get('row_count', 0)} rows | sampled model forecasts at spots |",
        f"| NWP extra fields | {'OK' if nwp_extra.get('exists') else 'missing'} | {nwp_extra.get('row_count', 0)} rows | AROME/AROME-PI thermal/context fields at spots |",
        f"| NWP vertical profiles | {'OK' if nwp_vertical.get('exists') else 'missing'} | {nwp_vertical.get('row_count', 0)} rows | AROME 0.025 isobaric profiles at spots |",
        f"| Feature store 15 min | {'OK' if feature_store.get('exists') else 'missing'} | {feature_store.get('row_count', 0)} rows | canonical training rows by spot/time |",
        f"| Residual training tables | {'OK' if training_tables.get('exists') else 'missing'} | {training_tables.get('total_training_rows', 0)} rows | NWP baseline + issue-time features + residual labels |",
        f"| Trained residual models | {'OK' if trained_models.get('exists') else 'missing'} | {trained_models.get('run_count', 0)} runs | latest: {trained_models.get('latest_run')} |",
        f"| In-situ observations | {'OK' if mfobs.get('exists') else 'missing'} | {mfobs.get('row_count', 0)} rows | normalized Meteo-France + WindsUp spot obs |",
        f"| MeteoNet ground stations | {'OK' if meteonet.get('exists') else 'missing'} | {meteonet.get('row_count', 0)} rows | 6-minute Corsica station observations for pretraining |",
        f"| Copernicus Marine SST | {'OK' if cop_sst.get('exists') else 'missing'} | {cop_sst.get('row_count', 0)} rows | sampled sea-surface temperature at spots |",
        f"| EUMETSAT Cloud Mask | {'OK' if eum_cloud.get('exists') else 'missing'} | {eum_cloud.get('row_count', 0)} rows | sampled MTG cloud mask at spots |",
        *[
            f"| EUMETSAT {EUMETSAT_SPOT_PRODUCTS[product]} | {'OK' if item.get('exists') else 'missing'} | {item.get('row_count', 0)} rows | sampled variables: {', '.join(list(item.get('sampled_variables', {}).keys())[:6]) or 'none yet'} |"
            for product, item in eum_spot_products.items()
        ],
        f"| Copernicus Marine inventory | {'OK' if cop_inv.get('exists') else 'missing'} | {cop_inv.get('candidate_count', 0)} candidates | {cop_inv.get('decisions', {})} |",
        f"| EUMETSAT inventory | {'OK' if eum_inv.get('exists') else 'missing'} | {eum_inv.get('candidate_count', 0)} candidates | {eum_inv.get('decisions', {})} |",
        f"| EUMETSAT catalogue keyword scan | {'OK' if eum_catalog.get('exists') else 'missing'} | {eum_catalog.get('matched_count', 0)} matches | {eum_catalog.get('group_counts', {})} |",
        f"| Meteo-France WCS inventory | {'OK' if wcs.get('exists') else 'missing'} | {wcs.get('service_count', 0)} services | model variables available beyond wind |",
        "",
        "## In-Situ Observation Fields",
        "",
    ])
    for dataset, item in mfobs.get("datasets", {}).items():
        lines.extend([
            f"### `{dataset}`",
            "",
            f"- rows: `{item['row_count']}`",
            f"- time range: `{item['first_timestamp_utc']}` -> `{item['last_timestamp_utc']}`",
            f"- sources: `{item['source_count']}`, mapped spots: `{item['spot_count']}`",
            "",
            "| Field | Non-null | Coverage |",
            "| --- | ---: | ---: |",
        ])
        for field, count in sorted(item["fields_non_null"].items()):
            if count:
                lines.append(f"| `{field}` | {count} | {pct(count, item['row_count'])} |")
        lines.append("")
        spot_summaries = item.get("spot_summaries", {})
        if spot_summaries and len(spot_summaries) <= 50:
            lines.extend([
                "| Spot | Rows | Time range | Source ids |",
                "| --- | ---: | --- | --- |",
            ])
            for spot_id, stats in spot_summaries.items():
                source_ids = ", ".join(stats.get("source_ids", []))
                lines.append(
                    f"| `{spot_id}` | {stats.get('row_count', 0)} | "
                    f"`{stats.get('first_timestamp_utc')}` -> `{stats.get('last_timestamp_utc')}` | "
                    f"`{source_ids}` |"
                )
            lines.append("")
    lines.extend([
        "## MeteoNet Ground Stations",
        "",
        f"- rows: `{meteonet.get('row_count', 0)}`",
        f"- stations: `{meteonet.get('station_count', 0)}`",
        f"- time range: `{meteonet.get('first_timestamp_utc')}` -> `{meteonet.get('last_timestamp_utc')}`",
        f"- raw archives: `{meteonet.get('raw_file_count', 0)}`",
        f"- station registry: `{meteonet.get('station_registry')}`",
        "",
        "| Year | Rows | Stations | Wind speed | Wind direction | Temperature | Pressure |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for year, item in sorted(meteonet.get("year_summaries", {}).items()):
        rows = item.get("row_count", 0)
        lines.append(
            f"| `{year}` | {rows} | {item.get('station_count', 0)} | "
            f"{pct(item.get('wind_mean_non_null', 0), rows)} | "
            f"{pct(item.get('wind_direction_non_null', 0), rows)} | "
            f"{pct(item.get('temperature_non_null', 0), rows)} | "
            f"{pct(item.get('pressure_non_null', 0), rows)} |"
        )
    lines.append("")
    lines.extend([
        "## Model Samples",
        "",
        "| Source | Rows | Spots | Time range | Wind | Gust | Direction |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ])
    for source, item in sorted(samples.get("sources", {}).items()):
        fields = item["fields_non_null"]
        lines.append(
            f"| `{source}` | {item['row_count']} | {item['spot_count']} | "
            f"`{item['first_valid_time_utc']}` -> `{item['last_valid_time_utc']}` | "
            f"{pct(fields.get('wind_speed_ms', 0), item['row_count'])} | "
            f"{pct(fields.get('gust_speed_ms', 0), item['row_count'])} | "
            f"{pct(fields.get('wind_direction_deg', 0), item['row_count'])} |"
        )
    lines.extend(["", "## Meteo-France WCS Families", ""])
    lines.extend(["", "## NWP Extra Fields", "", "| Source | Rows | Spots | Time range | Features |", "| --- | ---: | ---: | --- | --- |"])
    for source, item in sorted(nwp_extra.get("sources", {}).items()):
        features = ", ".join(f"`{feature}`" for feature in item.get("features_seen", []))
        lines.append(
            f"| `{source}` | {item.get('row_count', 0)} | {item.get('spot_count', 0)} | "
            f"`{item.get('first_valid_time_utc')}` -> `{item.get('last_valid_time_utc')}` | {features} |"
        )
    lines.append("")
    lines.extend(["", "## NWP Vertical Profiles", "", "| Source | Rows | Spots | Time range | Pressure levels | Profile features | Derived features |", "| --- | ---: | ---: | --- | --- | --- | --- |"])
    for source, item in sorted(nwp_vertical.get("sources", {}).items()):
        levels = ", ".join(f"`{level}`" for level in item.get("pressure_levels_hpa", []))
        profile_features = ", ".join(f"`{feature}`" for feature in item.get("profile_features_seen", []))
        derived_features = ", ".join(f"`{feature}`" for feature in item.get("derived_features_seen", []))
        lines.append(
            f"| `{source}` | {item.get('row_count', 0)} | {item.get('spot_count', 0)} | "
            f"`{item.get('first_valid_time_utc')}` -> `{item.get('last_valid_time_utc')}` | {levels} | {profile_features} | {derived_features} |"
        )
    lines.append("")
    lines.extend([
        "",
        "## Feature Store 15 min",
        "",
        f"- rows: `{feature_store.get('row_count', 0)}`",
        f"- spots: `{feature_store.get('spot_count', 0)}`",
        f"- time range: `{feature_store.get('first_target_time_utc')}` -> `{feature_store.get('last_target_time_utc')}`",
        "",
        "| Source flag | Rows with source |",
        "| --- | ---: |",
    ])
    for key, value in feature_store.get("source_flag_counts", {}).items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "| Target | Non-null rows |", "| --- | ---: |"])
    for key, value in feature_store.get("target_non_null", {}).items():
        lines.append(f"| `{key}` | {value} |")
    lines.append("")
    lines.extend([
        "## Residual Training Tables",
        "",
        f"- tables: `{training_tables.get('table_count', 0)}`",
        f"- latest: `{training_tables.get('latest_table')}`",
        f"- total rows: `{training_tables.get('total_training_rows', 0)}`",
        "",
        "| Table | Rows | Source rows | Leads | Spots | Wind RMSE gain | Gust RMSE gain | Missing baseline wind | Missing target wind |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ])
    for table, item in training_tables.get("tables", {}).items():
        leads = ", ".join(f"`+{lead}m:{count}`" for lead, count in item.get("training_rows_by_lead", {}).items())
        comparisons = item.get("evaluation_comparisons", {})
        lines.append(
            f"| `{table}` | {item.get('training_row_count', 0)} | {item.get('source_feature_row_count', 0)} | "
            f"{leads} | {len(item.get('training_rows_by_spot', {}))} | "
            f"{comparisons.get('wind_rmse_gain_pct_error_persistence_vs_raw')} | "
            f"{comparisons.get('gust_rmse_gain_pct_error_persistence_vs_raw')} | "
            f"`{item.get('missing_baseline_wind_by_lead', {})}` | "
            f"`{item.get('missing_target_wind_by_lead', {})}` |"
        )
    lines.append("")
    lines.extend([
        "## Trained Residual Models",
        "",
        f"- runs: `{trained_models.get('run_count', 0)}`",
        f"- latest: `{trained_models.get('latest_run')}`",
        "",
        "| Run | Rows | Train | Test | Target | Type | RMSE gain vs raw | Test metric |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ])
    for run_id, run in trained_models.get("runs", {}).items():
        for target, item in run.get("models", {}).items():
            if item.get("type") == "regression":
                metric = item.get("corrected_nwp_test", {})
                metric_text = f"RMSE `{metric.get('rmse')}`, MAE `{metric.get('mae')}`"
            else:
                metric = item.get("classification_test", {})
                metric_text = f"Brier `{metric.get('brier')}`, positives `{metric.get('positive_count')}`"
            lines.append(
                f"| `{run_id}` | {run.get('row_count', 0)} | {run.get('train_row_count', 0)} | "
                f"{run.get('test_row_count', 0)} | `{target}` | `{item.get('type')}` | "
                f"{item.get('rmse_gain_pct_vs_raw')} | {metric_text} |"
            )
    lines.append("")
    for service, item in sorted(wcs.get("services", {}).items()):
        lines.extend([f"### `{service}`", "", "| Family | Variables | Coverages | Examples |", "| --- | ---: | ---: | --- |"])
        for family, family_item in sorted(item.get("families", {}).items()):
            examples = ", ".join(f"`{value}`" for value in family_item.get("variables", [])[:5])
            lines.append(f"| `{family}` | {family_item.get('variable_count')} | {family_item.get('coverage_count')} | {examples} |")
        lines.append("")
    lines.extend([
        "## Copernicus Marine SST",
        "",
        f"- rows: `{cop_sst.get('row_count', 0)}`",
        f"- time range: `{cop_sst.get('first_timestamp_utc')}` -> `{cop_sst.get('last_timestamp_utc')}`",
        f"- spots: `{cop_sst.get('spot_count', 0)}`",
        "",
        "| Field | Non-null | Coverage |",
        "| --- | ---: | ---: |",
    ])
    for field, count in sorted(cop_sst.get("fields_non_null", {}).items()):
        if count:
            lines.append(f"| `{field}` | {count} | {pct(count, cop_sst.get('row_count', 0))} |")
    lines.append("")
    lines.extend([
        "## EUMETSAT Cloud Mask",
        "",
        f"- rows: `{eum_cloud.get('row_count', 0)}`",
        f"- products: `{eum_cloud.get('product_count', 0)}`",
        f"- time range: `{eum_cloud.get('first_sensing_start_utc')}` -> `{eum_cloud.get('last_sensing_start_utc')}`",
        f"- spots: `{eum_cloud.get('spot_count', 0)}`",
        f"- cloud state counts: `{eum_cloud.get('cloud_state_counts', {})}`",
        "",
        "| Field | Non-null | Coverage |",
        "| --- | ---: | ---: |",
    ])
    for field, count in sorted(eum_cloud.get("fields_non_null", {}).items()):
        if count:
            lines.append(f"| `{field}` | {count} | {pct(count, eum_cloud.get('row_count', 0))} |")
    lines.append("")
    for product, item in eum_spot_products.items():
        lines.extend([
            f"## EUMETSAT {EUMETSAT_SPOT_PRODUCTS[product]}",
            "",
            f"- rows: `{item.get('row_count', 0)}`",
            f"- products: `{item.get('product_count', 0)}`",
            f"- time range: `{item.get('first_sensing_start_utc')}` -> `{item.get('last_sensing_start_utc')}`",
            f"- spots: `{item.get('spot_count', 0)}`",
            f"- sampled variables: `{item.get('sampled_variables', {})}`",
            "",
            "| Field | Non-null | Coverage |",
            "| --- | ---: | ---: |",
        ])
        for field, count in sorted(item.get("fields_non_null", {}).items()):
            if count:
                lines.append(f"| `{field}` | {count} | {pct(count, item.get('row_count', 0))} |")
        lines.append("")
    lines.extend([
        "## External Access Configuration",
        "",
        "| Source | Configured | Required env | Target data |",
        "| --- | --- | --- | --- |",
    ])
    for source, item in profile["external_access"].items():
        lines.append(
            f"| `{source}` | `{item['configured']}` | "
            f"{', '.join(f'`{env}`' for env in item['required_env'])} | "
            f"{', '.join(f'`{data}`' for data in item['target_data'])} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--beacon-state", type=Path, default=DEFAULT_BEACON_STATE)
    parser.add_argument("--ml-root", type=Path, default=DEFAULT_ML_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = {
        "format": "corsewind.data_availability_profile.v1",
        "generated_at_utc": utc_now(),
        "registry": profile_registry(args.registry),
        "beacon_live": profile_beacon_state(args.beacon_state),
        "model_samples": profile_model_samples(args.ml_root),
        "nwp_extra_fields": profile_nwp_extra_fields(args.ml_root),
        "nwp_vertical_profiles": profile_nwp_vertical_profiles(args.ml_root),
        "feature_store": profile_feature_store(args.ml_root),
        "training_tables": profile_training_tables(args.ml_root),
        "trained_models": profile_trained_models(args.ml_root),
        "meteo_france_observations": profile_meteo_france_observations(args.ml_root),
        "meteonet_ground_stations": profile_meteonet_ground_stations(args.ml_root),
        "copernicus_marine_sst": profile_copernicus_marine_sst(args.ml_root),
        "eumetsat_cloud_mask": profile_eumetsat_cloud_mask(args.ml_root),
        "eumetsat_spot_products": {
            product: profile_eumetsat_spot_product(args.ml_root, product)
            for product in EUMETSAT_SPOT_PRODUCTS
        },
        "copernicus_marine_inventory": profile_copernicus_marine_inventory(args.ml_root),
        "eumetsat_inventory": profile_eumetsat_inventory(args.ml_root),
        "eumetsat_catalog_keyword_inventory": profile_eumetsat_catalog_keyword_inventory(args.ml_root),
        "meteo_france_wcs": profile_wcs_inventory(args.ml_root),
        "external_access": external_access_status(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.output_md, profile)
    print(json.dumps({
        "generated_at_utc": profile["generated_at_utc"],
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
        "model_sample_rows": profile["model_samples"].get("row_count", 0),
        "nwp_extra_field_rows": profile["nwp_extra_fields"].get("row_count", 0),
        "nwp_vertical_profile_rows": profile["nwp_vertical_profiles"].get("row_count", 0),
        "feature_store_rows": profile["feature_store"].get("row_count", 0),
        "training_table_rows": profile["training_tables"].get("total_training_rows", 0),
        "trained_model_runs": profile["trained_models"].get("run_count", 0),
        "meteo_france_observation_rows": profile["meteo_france_observations"].get("row_count", 0),
        "meteonet_ground_station_rows": profile["meteonet_ground_stations"].get("row_count", 0),
        "copernicus_marine_sst_rows": profile["copernicus_marine_sst"].get("row_count", 0),
        "eumetsat_cloud_mask_rows": profile["eumetsat_cloud_mask"].get("row_count", 0),
        "eumetsat_spot_product_rows": {
            product: item.get("row_count", 0)
            for product, item in profile["eumetsat_spot_products"].items()
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
