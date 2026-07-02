#!/usr/bin/env python3
"""Summarize ML backfill datasets stored under ML_DATASET_ROOT."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item


def summarize_jsonl_files(files: list[Path], time_key: str, spot_key: str = "spot_id") -> dict[str, Any]:
    row_count = 0
    spots: Counter[str] = Counter()
    dates: Counter[str] = Counter()
    feature_non_null: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    models: Counter[str] = Counter()
    lead_days: Counter[str] = Counter()
    for path in files:
        for row in iter_jsonl(path):
            row_count += 1
            spot = row.get(spot_key) or row.get("station_id") or "unknown"
            spots[str(spot)] += 1
            timestamp = str(row.get(time_key) or row.get("timestamp_utc") or "")
            if len(timestamp) >= 10:
                dates[timestamp[:10]] += 1
            dataset = row.get("source_dataset")
            if dataset:
                datasets[str(dataset)] += 1
            model = row.get("model")
            if model:
                models[str(model)] += 1
            if row.get("lead_days") is not None:
                lead_days[str(row["lead_days"])] += 1
            features = row.get("features")
            if isinstance(features, dict):
                for key, value in features.items():
                    if value is not None:
                        feature_non_null[key] += 1
            else:
                for key, value in row.items():
                    if key.startswith(("wind_", "gust_", "temperature_", "pressure_", "humidity_", "sea_level_", "global_", "direct_", "diffuse_")) and value is not None:
                        feature_non_null[key] += 1
    date_values = sorted(dates)
    return {
        "file_count": len(files),
        "row_count": row_count,
        "date_min": date_values[0] if date_values else None,
        "date_max": date_values[-1] if date_values else None,
        "spot_or_station_count": len(spots),
        "top_spots_or_stations": spots.most_common(20),
        "datasets": dict(datasets),
        "models": dict(models),
        "lead_days": dict(lead_days),
        "non_null_features": dict(feature_non_null.most_common(40)),
    }


def source_summary(root: Path) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    open_meteo_historical = sorted((root / "open_meteo/historical_forecast").glob("model=*/date=*/forecast.jsonl"))
    if open_meteo_historical:
        summaries["open_meteo_historical_forecast"] = summarize_jsonl_files(open_meteo_historical, "valid_time_utc")
    open_meteo_previous = sorted((root / "open_meteo/previous_runs").glob("model=*/date=*/previous_runs.jsonl"))
    if open_meteo_previous:
        summaries["open_meteo_previous_runs"] = summarize_jsonl_files(open_meteo_previous, "valid_time_utc")
    dpclim = sorted((root / "observations/meteo_france_climatology/normalized").glob("frequency=*/date=*/observations.jsonl"))
    if dpclim:
        summaries["meteo_france_dpclim_observations"] = summarize_jsonl_files(dpclim, "timestamp_utc")
    station_info = sorted((root / "observations/meteo_france_climatology/station_info/normalized").glob("station=*/station_info.json"))
    if station_info:
        station_summaries = []
        parameter_groups: Counter[str] = Counter()
        for path in station_info:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            station_summaries.append({
                "station_id": payload.get("station_id"),
                "spot_id": payload.get("spot_id"),
                "station_start": payload.get("station_start"),
                "station_end": payload.get("station_end"),
                "parameter_count": payload.get("parameter_count"),
                "type_period_count": payload.get("type_period_count"),
            })
            groups = payload.get("parameter_groups")
            if isinstance(groups, dict):
                for group, enabled in groups.items():
                    if enabled:
                        parameter_groups[group] += 1
        summaries["meteo_france_dpclim_station_info"] = {
            "station_count": len(station_summaries),
            "parameter_groups_station_counts": dict(parameter_groups),
            "stations": sorted(station_summaries, key=lambda item: str(item.get("station_id") or "")),
        }
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-root", type=Path, default=DEFAULT_ML_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ml_root = args.ml_root if args.ml_root.is_absolute() else ROOT / args.ml_root
    print(json.dumps({
        "format": "corsewind.ml_backfill_summary.v1",
        "ml_root": str(ml_root),
        "sources": source_summary(ml_root),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
