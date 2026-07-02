#!/usr/bin/env python3
"""Build residual-correction training rows from the spot feature store."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_FEATURE_STORE = DEFAULT_ML_ROOT / "feature_store/spot_forecast_15min.jsonl"
DEFAULT_OUTPUT_ROOT = DEFAULT_ML_ROOT / "training_tables/residual_correction"
DEFAULT_MODEL_PREFIX = "model_open_meteo_meteofrance_arome_france"
DEFAULT_LEAD_MINUTES = "15,30,45,60,120,180,360"
DEFAULT_WIND_THRESHOLDS_MS = "7.716,10.289"
DEFAULT_GUST_THRESHOLDS_MS = "10.289,12.861"
DEFAULT_FEATURE_PREFIXES = (
    "obs_",
    "context_nearest_",
    "context_coastal_",
    "context_inland_",
    "context_relief_",
    "context_global_",
    "context_agg_",
    "previous_run_open_meteo_",
    "nwp_",
    "nwp_offset_",
    "sst_",
    "eumetsat_",
    "spot_static_",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def first_finite(values: list[Any]) -> float | None:
    for value in values:
        number = finite_float(value)
        if number is not None:
            return number
    return None


def first_feature(features: dict[str, Any], keys: list[str]) -> float | None:
    return first_finite([features.get(key) for key in keys])


def wind_direction_from_uv(u_value: Any, v_value: Any) -> float | None:
    u = finite_float(u_value)
    v = finite_float(v_value)
    if u is None or v is None:
        return None
    return round((math.degrees(math.atan2(-u, -v)) + 360.0) % 360.0, 6)


def circular_mean_degrees(values: list[Any], digits: int = 6) -> float | None:
    degrees = [finite_float(value) for value in values]
    finite_degrees = [value for value in degrees if value is not None]
    if not finite_degrees:
        return None
    sin_sum = sum(math.sin(math.radians(value)) for value in finite_degrees)
    cos_sum = sum(math.cos(math.radians(value)) for value in finite_degrees)
    if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
        return None
    return round((math.degrees(math.atan2(sin_sum, cos_sum)) + 360.0) % 360.0, digits)


def nwp_feature_aliases(model_prefix: str, field: str) -> list[str]:
    aliases = {
        "wind_speed_10m": [
            f"{model_prefix}_wind_speed_10m",
            "nwp_aromepi_wind_speed_10m_ms",
            "nwp_arome_wind_speed_10m_ms",
            "model_aromepi_wind_speed_ms",
            "model_arome_wind_speed_ms",
        ],
        "wind_gusts_10m": [
            f"{model_prefix}_wind_gusts_10m",
            "model_aromepi_gust_speed_ms",
            "model_arome_gust_speed_ms",
        ],
        "wind_direction_10m": [
            f"{model_prefix}_wind_direction_10m",
            "model_aromepi_wind_direction_deg",
            "model_arome_wind_direction_deg",
        ],
        "temperature_2m": [
            f"{model_prefix}_temperature_2m",
            "nwp_aromepi_temperature_2m_c",
            "nwp_arome_temperature_2m_c",
        ],
        "pressure_msl": [
            f"{model_prefix}_pressure_msl",
            "nwp_aromepi_pressure_msl_hpa",
            "nwp_arome_pressure_msl_hpa",
        ],
        "surface_pressure": [
            f"{model_prefix}_surface_pressure",
            "nwp_arome_pressure_surface_hpa",
        ],
        "shortwave_radiation": [
            f"{model_prefix}_shortwave_radiation",
            "nwp_aromepi_downward_shortwave_flux_w_m2",
            "nwp_arome_downward_shortwave_flux_w_m2",
        ],
        "cloud_cover": [
            f"{model_prefix}_cloud_cover",
            "nwp_aromepi_cloud_cover_pct",
            "nwp_arome_total_cloud_cover_pct",
        ],
        "cape": [
            f"{model_prefix}_cape",
            "nwp_arome_cape_j_kg",
        ],
    }
    return aliases.get(field, [f"{model_prefix}_{field}"])


def nwp_feature_value(features: dict[str, Any], model_prefix: str, field: str) -> float | None:
    value = first_feature(features, nwp_feature_aliases(model_prefix, field))
    if value is not None:
        return value
    if field == "wind_speed_10m":
        return mean_finite([features.get(f"nwp_offset_{name}_wind_speed_10m") for name in ("e10", "n10", "s10", "w10")])
    if field == "wind_gusts_10m":
        return mean_finite([features.get(f"nwp_offset_{name}_wind_gusts_10m") for name in ("e10", "n10", "s10", "w10")])
    if field == "wind_direction_10m":
        uv_direction = wind_direction_from_uv(
            first_feature(features, ["nwp_aromepi_wind_u_10m_ms", "nwp_arome_wind_u_10m_ms", "model_aromepi_wind_u_ms"]),
            first_feature(features, ["nwp_aromepi_wind_v_10m_ms", "nwp_arome_wind_v_10m_ms", "model_aromepi_wind_v_ms"]),
        )
        if uv_direction is not None:
            return uv_direction
        return circular_mean_degrees([features.get(f"nwp_offset_{name}_wind_direction_10m") for name in ("e10", "n10", "s10", "w10")])
    return None


def finite_delta(left: Any, right: Any, digits: int = 6) -> float | None:
    left_value = finite_float(left)
    right_value = finite_float(right)
    if left_value is None or right_value is None:
        return None
    return round(left_value - right_value, digits)


def finite_product(*values: Any, digits: int = 6) -> float | None:
    product = 1.0
    for value in values:
        number = finite_float(value)
        if number is None:
            return None
        product *= number
    return round(product, digits)


def bounded_fraction(value: Any, denominator: float = 100.0) -> float | None:
    number = finite_float(value)
    if number is None:
        return None
    return max(0.0, min(1.0, number / denominator))


def mean_finite(values: list[Any], digits: int = 6) -> float | None:
    numbers = [finite_float(value) for value in values]
    finite_numbers = [value for value in numbers if value is not None]
    if not finite_numbers:
        return None
    return round(sum(finite_numbers) / len(finite_numbers), digits)


def parse_int_list(value: str) -> list[int]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return sorted(set(values))


def parse_float_list(value: str) -> list[float]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return sorted(set(values))


def threshold_slug(value_ms: float) -> str:
    knots = value_ms * 1.9438444924406
    rounded = round(knots)
    if abs(knots - rounded) < 0.08:
        return f"{int(rounded)}kt"
    return f"{value_ms:.2f}ms".replace(".", "p")


def cyclical_time_features(timestamp: datetime) -> dict[str, float]:
    minute_of_day = timestamp.hour * 60 + timestamp.minute
    day_of_year = int(timestamp.strftime("%j"))
    return {
        "issue_hour_sin": round(math.sin(2.0 * math.pi * minute_of_day / 1440.0), 10),
        "issue_hour_cos": round(math.cos(2.0 * math.pi * minute_of_day / 1440.0), 10),
        "issue_dayofyear_sin": round(math.sin(2.0 * math.pi * day_of_year / 366.0), 10),
        "issue_dayofyear_cos": round(math.cos(2.0 * math.pi * day_of_year / 366.0), 10),
    }


def read_feature_store(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def should_keep_feature(key: str, prefixes: tuple[str, ...], model_prefix: str) -> bool:
    if key.startswith(f"{model_prefix}_"):
        return True
    if not key.startswith(prefixes):
        return False
    if key.endswith("_utc") or "_sensing_start_utc" in key:
        return False
    return True


def add_thermal_regime_features(features: dict[str, Any], source_features: dict[str, Any], model_prefix: str) -> None:
    sst_c = finite_float(source_features.get("sst_c"))
    land_c = first_finite([
        source_features.get("eumetsat_land_surface_temperature_LST_c"),
        source_features.get("eumetsat_land_surface_temperature_land_surface_temperature_c"),
        source_features.get("eumetsat_land_surface_temperature_lst_c"),
        source_features.get("eumetsat_land_surface_temperature_LST_neighborhood_mean"),
    ])
    air_c = first_finite([
        source_features.get(f"{model_prefix}_temperature_2m"),
        source_features.get("obs_last_temperature_c"),
    ])
    shortwave = finite_float(source_features.get(f"{model_prefix}_shortwave_radiation"))
    cloud_cover = finite_float(source_features.get(f"{model_prefix}_cloud_cover"))
    low_cloud_cover = finite_float(source_features.get(f"{model_prefix}_cloud_cover_low"))
    cape = finite_float(source_features.get(f"{model_prefix}_cape"))
    obs_temp_delta_60m = finite_float(source_features.get("obs_delta_60m_temperature_c"))
    obs_pressure_delta_60m = finite_float(source_features.get("obs_delta_60m_pressure_hpa"))
    clear_sky_fraction = None if cloud_cover is None else round(1.0 - bounded_fraction(cloud_cover), 6)
    low_cloud_fraction = bounded_fraction(low_cloud_cover)
    insolation_proxy = finite_product(shortwave, clear_sky_fraction)

    features["thermal_land_minus_sst_c"] = finite_delta(land_c, sst_c)
    features["thermal_air_minus_sst_c"] = finite_delta(air_c, sst_c)
    features["thermal_land_minus_air_c"] = finite_delta(land_c, air_c)
    features["thermal_clear_sky_fraction"] = clear_sky_fraction
    features["thermal_low_cloud_fraction"] = low_cloud_fraction
    features["thermal_insolation_proxy"] = insolation_proxy
    features["thermal_land_sea_insolation_index"] = finite_product(features["thermal_land_minus_sst_c"], insolation_proxy)
    features["thermal_air_sea_insolation_index"] = finite_product(features["thermal_air_minus_sst_c"], insolation_proxy)
    features["thermal_recent_heating_rate_c_per_h"] = obs_temp_delta_60m
    features["thermal_recent_pressure_tendency_hpa_per_h"] = obs_pressure_delta_60m
    features["thermal_cape_x_land_sea"] = finite_product(cape, features["thermal_land_minus_sst_c"])
    features["thermal_low_cloud_suppression_index"] = finite_product(features["thermal_land_minus_sst_c"], low_cloud_fraction)

    coastal_temp = finite_float(source_features.get("context_agg_coastal_temperature_c_mean"))
    inland_temp = finite_float(source_features.get("context_agg_inland_temperature_c_mean"))
    relief_temp = finite_float(source_features.get("context_agg_relief_temperature_c_mean"))
    coastal_pressure = finite_float(source_features.get("context_agg_coastal_pressure_hpa_mean"))
    inland_pressure = finite_float(source_features.get("context_agg_inland_pressure_hpa_mean"))
    relief_pressure = finite_float(source_features.get("context_agg_relief_pressure_hpa_mean"))
    coastal_wind = finite_float(source_features.get("context_agg_coastal_wind_mean_ms_mean"))
    inland_wind = finite_float(source_features.get("context_agg_inland_wind_mean_ms_mean"))
    relief_wind = finite_float(source_features.get("context_agg_relief_wind_mean_ms_mean"))

    features["thermal_inland_minus_coastal_temperature_c"] = finite_delta(inland_temp, coastal_temp)
    features["thermal_relief_minus_coastal_temperature_c"] = finite_delta(relief_temp, coastal_temp)
    features["thermal_inland_minus_coastal_pressure_hpa"] = finite_delta(inland_pressure, coastal_pressure)
    features["thermal_relief_minus_coastal_pressure_hpa"] = finite_delta(relief_pressure, coastal_pressure)
    features["thermal_coastal_minus_inland_wind_ms"] = finite_delta(coastal_wind, inland_wind)
    features["thermal_coastal_minus_relief_wind_ms"] = finite_delta(coastal_wind, relief_wind)


def add_open_meteo_vertical_features(features: dict[str, Any], model_prefix: str) -> None:
    levels = [1000, 950, 925, 900, 850]

    def value(field: str, level: int) -> float | None:
        return finite_float(features.get(f"{model_prefix}_{field}_{level}hPa"))

    t1000 = value("temperature", 1000)
    t950 = value("temperature", 950)
    t925 = value("temperature", 925)
    t900 = value("temperature", 900)
    t850 = value("temperature", 850)
    z1000 = value("geopotential_height", 1000)
    z850 = value("geopotential_height", 850)
    wind1000 = value("wind_speed", 1000)
    wind850 = value("wind_speed", 850)
    dir1000 = value("wind_direction", 1000)
    dir850 = value("wind_direction", 850)

    thickness = None if z1000 is None or z850 is None else round(z850 - z1000, 6)
    features["open_meteo_vertical_geopotential_thickness_1000_850_m"] = thickness
    features["open_meteo_vertical_temperature_lapse_rate_1000_850_c_per_km"] = (
        None
        if thickness in {None, 0} or t1000 is None or t850 is None
        else round((t1000 - t850) / (thickness / 1000.0), 6)
    )
    features["open_meteo_vertical_temperature_delta_1000_850_c"] = finite_delta(t1000, t850)
    features["open_meteo_vertical_temperature_delta_1000_950_c"] = finite_delta(t1000, t950)
    features["open_meteo_vertical_temperature_delta_950_850_c"] = finite_delta(t950, t850)
    features["open_meteo_vertical_relative_humidity_mean_1000_850_pct"] = mean_finite(
        [value("relative_humidity", level) for level in levels]
    )
    features["open_meteo_vertical_relative_humidity_delta_1000_850_pct"] = finite_delta(
        value("relative_humidity", 1000),
        value("relative_humidity", 850),
    )
    features["open_meteo_vertical_wind_shear_speed_1000_850_ms"] = finite_delta(wind850, wind1000)
    direction_delta = None
    if dir1000 is not None and dir850 is not None:
        direction_delta = ((dir850 - dir1000 + 180.0) % 360.0) - 180.0
    features["open_meteo_vertical_wind_shear_direction_1000_850_deg"] = (
        None if direction_delta is None else round(direction_delta, 6)
    )

    inversions = []
    adjacent_pairs = [(1000, t1000, 950, t950), (950, t950, 925, t925), (925, t925, 900, t900), (900, t900, 850, t850)]
    for _lower_level, lower_temp, _upper_level, upper_temp in adjacent_pairs:
        if lower_temp is not None and upper_temp is not None:
            inversions.append(upper_temp - lower_temp)
    features["open_meteo_vertical_low_level_inversion_strength_c"] = (
        None if not inversions else round(max(inversions), 6)
    )


def add_horizon_nwp_features(features: dict[str, Any], baselines: dict[str, Any], model_prefix: str) -> None:
    issue_wind = nwp_feature_value(features, model_prefix, "wind_speed_10m")
    issue_gust = nwp_feature_value(features, model_prefix, "wind_gusts_10m")
    issue_direction = nwp_feature_value(features, model_prefix, "wind_direction_10m")
    issue_temp = nwp_feature_value(features, model_prefix, "temperature_2m")
    issue_pressure_msl = nwp_feature_value(features, model_prefix, "pressure_msl")
    issue_surface_pressure = nwp_feature_value(features, model_prefix, "surface_pressure")
    issue_shortwave = nwp_feature_value(features, model_prefix, "shortwave_radiation")
    issue_cloud = nwp_feature_value(features, model_prefix, "cloud_cover")
    issue_cape = nwp_feature_value(features, model_prefix, "cape")

    features["nwp_horizon_wind_ramp_ms"] = finite_delta(baselines.get("baseline_wind_mean_ms"), issue_wind)
    features["nwp_horizon_gust_ramp_ms"] = finite_delta(baselines.get("baseline_gust_ms"), issue_gust)
    features["nwp_horizon_temperature_ramp_c"] = finite_delta(baselines.get("baseline_temperature_2m_c"), issue_temp)
    features["nwp_horizon_pressure_msl_ramp_hpa"] = finite_delta(baselines.get("baseline_pressure_msl_hpa"), issue_pressure_msl)
    features["nwp_horizon_surface_pressure_ramp_hpa"] = finite_delta(baselines.get("baseline_surface_pressure_hpa"), issue_surface_pressure)
    features["nwp_horizon_shortwave_ramp"] = finite_delta(baselines.get("baseline_shortwave_radiation"), issue_shortwave)
    features["nwp_horizon_cloud_cover_ramp_pct"] = finite_delta(baselines.get("baseline_cloud_cover_pct"), issue_cloud)
    features["nwp_horizon_cape_ramp"] = finite_delta(baselines.get("baseline_cape"), issue_cape)

    direction_delta = None
    if baselines.get("baseline_wind_direction_deg") is not None and issue_direction is not None:
        direction_delta = ((float(baselines["baseline_wind_direction_deg"]) - issue_direction + 180.0) % 360.0) - 180.0
    features["nwp_horizon_wind_direction_delta_deg"] = None if direction_delta is None else round(direction_delta, 6)

    current_wind_error = finite_float(features.get("model_error_now_wind_mean_ms"))
    current_gust_error = finite_float(features.get("model_error_now_gust_ms"))
    features["nwp_error_persistence_plus_wind_ramp_ms"] = (
        None if current_wind_error is None or features["nwp_horizon_wind_ramp_ms"] is None
        else round(current_wind_error + features["nwp_horizon_wind_ramp_ms"], 6)
    )
    features["nwp_error_persistence_plus_gust_ramp_ms"] = (
        None if current_gust_error is None or features["nwp_horizon_gust_ramp_ms"] is None
        else round(current_gust_error + features["nwp_horizon_gust_ramp_ms"], 6)
    )


def build_issue_features(
    row: dict[str, Any],
    *,
    model_prefix: str,
    prefixes: tuple[str, ...],
    lead_minutes: int,
) -> tuple[dict[str, Any], list[str]]:
    source_features = row.get("features", {}) if isinstance(row.get("features"), dict) else {}
    features: dict[str, Any] = {}
    categorical: list[str] = []

    for key, value in source_features.items():
        if not should_keep_feature(str(key), prefixes, model_prefix):
            continue
        if isinstance(value, str):
            categorical.append(str(key))
        features[str(key)] = value

    issue_time = parse_time(row.get("target_time_utc"))
    if issue_time is not None:
        features.update(cyclical_time_features(issue_time))
    features["lead_time_minutes"] = lead_minutes

    targets = row.get("targets", {}) if isinstance(row.get("targets"), dict) else {}
    model_wind = finite_float(source_features.get(f"{model_prefix}_wind_speed_10m"))
    model_gust = finite_float(source_features.get(f"{model_prefix}_wind_gusts_10m"))
    observed_wind = finite_float(targets.get("wind_mean_ms"))
    observed_gust = finite_float(targets.get("gust_ms"))
    features["model_error_now_wind_mean_ms"] = None if model_wind is None or observed_wind is None else round(observed_wind - model_wind, 6)
    features["model_error_now_gust_ms"] = None if model_gust is None or observed_gust is None else round(observed_gust - model_gust, 6)
    add_thermal_regime_features(features, source_features, model_prefix)
    add_open_meteo_vertical_features(features, model_prefix)

    return features, categorical


def build_baselines(row: dict[str, Any], model_prefix: str) -> dict[str, Any]:
    features = row.get("features", {}) if isinstance(row.get("features"), dict) else {}
    return {
        "baseline_model": model_prefix,
        "baseline_wind_mean_ms": nwp_feature_value(features, model_prefix, "wind_speed_10m"),
        "baseline_gust_ms": nwp_feature_value(features, model_prefix, "wind_gusts_10m"),
        "baseline_wind_direction_deg": nwp_feature_value(features, model_prefix, "wind_direction_10m"),
        "baseline_temperature_2m_c": nwp_feature_value(features, model_prefix, "temperature_2m"),
        "baseline_pressure_msl_hpa": nwp_feature_value(features, model_prefix, "pressure_msl"),
        "baseline_surface_pressure_hpa": nwp_feature_value(features, model_prefix, "surface_pressure"),
        "baseline_shortwave_radiation": nwp_feature_value(features, model_prefix, "shortwave_radiation"),
        "baseline_cloud_cover_pct": nwp_feature_value(features, model_prefix, "cloud_cover"),
        "baseline_cape": nwp_feature_value(features, model_prefix, "cape"),
    }


def build_labels(
    future_row: dict[str, Any],
    baselines: dict[str, Any],
    wind_thresholds_ms: list[float],
    gust_thresholds_ms: list[float],
) -> dict[str, Any]:
    targets = future_row.get("targets", {}) if isinstance(future_row.get("targets"), dict) else {}
    target_wind = finite_float(targets.get("wind_mean_ms"))
    target_gust = finite_float(targets.get("gust_ms"))
    target_direction = finite_float(targets.get("wind_direction_deg"))
    baseline_wind = finite_float(baselines.get("baseline_wind_mean_ms"))
    baseline_gust = finite_float(baselines.get("baseline_gust_ms"))

    labels: dict[str, Any] = {
        "target_wind_mean_ms": target_wind,
        "target_gust_ms": target_gust,
        "target_wind_direction_deg": target_direction,
        "residual_wind_mean_ms": None if target_wind is None or baseline_wind is None else round(target_wind - baseline_wind, 6),
        "residual_gust_ms": None if target_gust is None or baseline_gust is None else round(target_gust - baseline_gust, 6),
        "target_observation_timestamp_utc": targets.get("observation_timestamp_utc"),
        "target_observation_distance_minutes": targets.get("observation_distance_minutes"),
        "target_observation_source_project": targets.get("observation_source_project"),
        "target_observation_source_dataset": targets.get("observation_source_dataset"),
        "target_observation_source_type": targets.get("observation_source_type"),
        "target_observation_station_id": targets.get("observation_station_id"),
        "target_observation_source_resolution_minutes": targets.get("observation_source_resolution_minutes"),
    }
    for threshold in wind_thresholds_ms:
        slug = threshold_slug(threshold)
        labels[f"target_wind_gt_{slug}"] = None if target_wind is None else int(target_wind > threshold)
    for threshold in gust_thresholds_ms:
        slug = threshold_slug(threshold)
        labels[f"target_gust_gt_{slug}"] = None if target_gust is None else int(target_gust > threshold)
    return labels


def build_training_rows(
    feature_rows: list[dict[str, Any]],
    *,
    lead_minutes: list[int],
    model_prefix: str,
    prefixes: tuple[str, ...],
    wind_thresholds_ms: list[float],
    gust_thresholds_ms: list[float],
    issue_start_time: datetime | None = None,
    issue_end_time: datetime | None = None,
    issue_start_hour_utc: int | None = None,
    issue_end_hour_utc: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in feature_rows:
        spot_id = row.get("spot_id")
        timestamp = row.get("target_time_utc")
        if spot_id and timestamp:
            by_key[(str(spot_id), str(timestamp))] = row

    training_rows: list[dict[str, Any]] = []
    missing_future = Counter()
    missing_baseline = Counter()
    missing_label = Counter()
    categorical_features: set[str] = set()

    for issue_row in feature_rows:
        spot_id = issue_row.get("spot_id")
        issue_time = parse_time(issue_row.get("target_time_utc"))
        if not spot_id or issue_time is None:
            continue
        if issue_start_time is not None and issue_time < issue_start_time:
            continue
        if issue_end_time is not None and issue_time > issue_end_time:
            continue
        if issue_start_hour_utc is not None and issue_time.hour < issue_start_hour_utc:
            continue
        if issue_end_hour_utc is not None and issue_time.hour > issue_end_hour_utc:
            continue
        for lead in lead_minutes:
            target_time = issue_time + timedelta(minutes=lead)
            target_time_utc = iso_z(target_time)
            future_row = by_key.get((str(spot_id), target_time_utc))
            if future_row is None:
                missing_future[lead] += 1
                continue

            issue_features, categorical = build_issue_features(
                issue_row,
                model_prefix=model_prefix,
                prefixes=prefixes,
                lead_minutes=lead,
            )
            categorical_features.update(categorical)
            baselines = build_baselines(future_row, model_prefix)
            add_horizon_nwp_features(issue_features, baselines, model_prefix)
            labels = build_labels(future_row, baselines, wind_thresholds_ms, gust_thresholds_ms)
            if baselines["baseline_wind_mean_ms"] is None:
                missing_baseline[lead] += 1
            if labels["target_wind_mean_ms"] is None:
                missing_label[lead] += 1

            training_rows.append({
                "format": "corsewind.residual_correction_training_row.v1",
                "spot_id": spot_id,
                "spot_name": issue_row.get("spot_name"),
                "spot_kind": issue_row.get("spot_kind"),
                "spot_source_type": issue_row.get("spot_source_type"),
                "station_id": issue_row.get("station_id"),
                "latitude": issue_row.get("latitude"),
                "longitude": issue_row.get("longitude"),
                "issue_time_utc": iso_z(issue_time),
                "target_time_utc": target_time_utc,
                "lead_time_minutes": lead,
                "features": issue_features,
                "baselines": baselines,
                "labels": labels,
                "issue_feature_sources": issue_row.get("feature_sources", {}),
                "target_feature_sources": future_row.get("feature_sources", {}),
                "built_at_utc": utc_now(),
            })

    profile = {
        "generated_at_utc": utc_now(),
        "source_feature_row_count": len(feature_rows),
        "training_row_count": len(training_rows),
        "lead_minutes_requested": lead_minutes,
        "issue_start_time_utc": iso_z(issue_start_time) if issue_start_time else None,
        "issue_end_time_utc": iso_z(issue_end_time) if issue_end_time else None,
        "training_rows_by_lead": dict(sorted(Counter(row["lead_time_minutes"] for row in training_rows).items())),
        "training_rows_by_spot": dict(sorted(Counter(str(row["spot_id"]) for row in training_rows).items())),
        "missing_future_rows_by_lead": dict(sorted(missing_future.items())),
        "missing_baseline_wind_by_lead": dict(sorted(missing_baseline.items())),
        "missing_target_wind_by_lead": dict(sorted(missing_label.items())),
        "categorical_features": sorted(categorical_features),
        "settings": {
            "model_prefix": model_prefix,
            "feature_prefixes": list(prefixes),
            "wind_thresholds_ms": wind_thresholds_ms,
            "gust_thresholds_ms": gust_thresholds_ms,
            "issue_start_hour_utc": issue_start_hour_utc,
            "issue_end_hour_utc": issue_end_hour_utc,
        },
    }
    return training_rows, profile


def collect_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns: dict[str, dict[str, Any]] = {}
    for row in rows:
        for group in ("features", "baselines", "labels"):
            values = row.get(group, {}) if isinstance(row.get(group), dict) else {}
            for key, value in values.items():
                name = f"{group}.{key}"
                item = columns.setdefault(name, {
                    "column": name,
                    "group": group,
                    "non_null_count": 0,
                    "types": set(),
                })
                if value is not None:
                    item["non_null_count"] += 1
                    item["types"].add(type(value).__name__)
    output = []
    for name, item in sorted(columns.items()):
        output.append({
            "column": name,
            "group": item["group"],
            "non_null_count": item["non_null_count"],
            "types": "|".join(sorted(item["types"])) if item["types"] else "",
        })
    return output


def write_outputs(rows: list[dict[str, Any]], profile: dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "training_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    profile_path = output_root / "training_profile.json"
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    columns = collect_columns(rows)
    columns_path = output_root / "training_columns.csv"
    with columns_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["column", "group", "non_null_count", "types"])
        writer.writeheader()
        writer.writerows(columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--lead-minutes", default=DEFAULT_LEAD_MINUTES)
    parser.add_argument("--model-prefix", default=DEFAULT_MODEL_PREFIX)
    parser.add_argument("--feature-prefix", action="append", default=None)
    parser.add_argument("--wind-threshold-ms", default=DEFAULT_WIND_THRESHOLDS_MS)
    parser.add_argument("--gust-threshold-ms", default=DEFAULT_GUST_THRESHOLDS_MS)
    parser.add_argument("--issue-start-datetime", help="Optional inclusive issue-time lower bound.")
    parser.add_argument("--issue-end-datetime", help="Optional inclusive issue-time upper bound.")
    parser.add_argument("--issue-start-hour-utc", type=int, help="Optional inclusive per-day issue start hour.")
    parser.add_argument("--issue-end-hour-utc", type=int, help="Optional inclusive per-day issue end hour.")
    args = parser.parse_args()

    feature_store = args.feature_store if args.feature_store.is_absolute() else ROOT / args.feature_store
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    prefixes = tuple(args.feature_prefix or DEFAULT_FEATURE_PREFIXES)
    if args.issue_start_hour_utc is not None and not 0 <= args.issue_start_hour_utc <= 23:
        raise SystemExit("--issue-start-hour-utc must be between 0 and 23")
    if args.issue_end_hour_utc is not None and not 0 <= args.issue_end_hour_utc <= 23:
        raise SystemExit("--issue-end-hour-utc must be between 0 and 23")
    if (
        args.issue_start_hour_utc is not None
        and args.issue_end_hour_utc is not None
        and args.issue_end_hour_utc < args.issue_start_hour_utc
    ):
        raise SystemExit("--issue-end-hour-utc must be greater than or equal to --issue-start-hour-utc")

    feature_rows = read_feature_store(feature_store)
    rows, profile = build_training_rows(
        feature_rows,
        lead_minutes=parse_int_list(args.lead_minutes),
        model_prefix=args.model_prefix,
        prefixes=prefixes,
        wind_thresholds_ms=parse_float_list(args.wind_threshold_ms),
        gust_thresholds_ms=parse_float_list(args.gust_threshold_ms),
        issue_start_time=parse_time(args.issue_start_datetime),
        issue_end_time=parse_time(args.issue_end_datetime),
        issue_start_hour_utc=args.issue_start_hour_utc,
        issue_end_hour_utc=args.issue_end_hour_utc,
    )
    profile["source_feature_store"] = str(feature_store)
    profile["output_root"] = str(output_root)
    write_outputs(rows, profile, output_root)
    print(json.dumps({
        "training_row_count": profile["training_row_count"],
        "training_rows_by_lead": profile["training_rows_by_lead"],
        "missing_future_rows_by_lead": profile["missing_future_rows_by_lead"],
        "output_root": str(output_root),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
