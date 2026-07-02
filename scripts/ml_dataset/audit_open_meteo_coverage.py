#!/usr/bin/env python3
"""Audit Open-Meteo historical forecast coverage by spot and day."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_INPUT_ROOT = DEFAULT_ML_ROOT / "open_meteo/historical_forecast"


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_dates(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def iter_jsonl(path: Path):
    if not path.exists():
        return
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


def load_spots(path: Path, include_context: bool, selected_ids: set[str]) -> list[dict[str, Any]]:
    payload = read_json(path)
    spots = payload.get("spots", []) if isinstance(payload, dict) else payload
    out = []
    for spot in spots:
        if not isinstance(spot, dict) or not spot.get("spot_id"):
            continue
        if selected_ids and str(spot["spot_id"]) not in selected_ids:
            continue
        if not include_context and not spot.get("use_for_ml", False):
            continue
        out.append(spot)
    return out


def compress_dates(days: list[str]) -> list[dict[str, str]]:
    if not days:
        return []
    parsed = [parse_date(day) for day in sorted(set(days))]
    ranges = []
    start = previous = parsed[0]
    for day in parsed[1:]:
        if day == previous + timedelta(days=1):
            previous = day
            continue
        ranges.append({"start": start.isoformat(), "end": previous.isoformat()})
        start = previous = day
    ranges.append({"start": start.isoformat(), "end": previous.isoformat()})
    return ranges


def audit(
    input_root: Path,
    registry: Path,
    model: str,
    start: date,
    end: date,
    include_context: bool,
    selected_ids: set[str],
    required_features: list[str],
) -> dict[str, Any]:
    spots = load_spots(registry, include_context, selected_ids)
    expected_days = [day.isoformat() for day in iter_dates(start, end)]
    expected_rows_per_spot = len(expected_days) * 24
    counts_by_spot_day: dict[tuple[str, str], int] = defaultdict(int)
    feature_counts_by_spot_day: dict[tuple[str, str], int] = defaultdict(int)
    null_feature_rows = Counter()
    missing_required_feature_rows = Counter()
    for day in expected_days:
        path = input_root / f"model={model}" / f"date={day}" / "forecast.jsonl"
        for row in iter_jsonl(path) or []:
            if row.get("model") != model:
                continue
            spot_id = str(row.get("spot_id") or "")
            if selected_ids and spot_id not in selected_ids:
                continue
            counts_by_spot_day[(spot_id, day)] += 1
            features = row.get("features")
            if isinstance(features, dict) and not any(value is not None for value in features.values()):
                null_feature_rows[spot_id] += 1
            if required_features:
                if isinstance(features, dict) and all(features.get(feature) is not None for feature in required_features):
                    feature_counts_by_spot_day[(spot_id, day)] += 1
                else:
                    missing_required_feature_rows[spot_id] += 1

    by_spot = []
    total_expected = 0
    total_observed = 0
    total_feature_complete = 0
    for spot in spots:
        spot_id = str(spot["spot_id"])
        missing_days = []
        partial_days = []
        feature_missing_days = []
        feature_partial_days = []
        observed = 0
        feature_complete = 0
        for day in expected_days:
            count = counts_by_spot_day.get((spot_id, day), 0)
            feature_count = feature_counts_by_spot_day.get((spot_id, day), 0)
            observed += count
            feature_complete += feature_count
            if count == 0:
                missing_days.append(day)
            elif count < 24:
                partial_days.append({"date": day, "rows": count, "missing_rows": 24 - count})
            if required_features:
                if feature_count == 0:
                    feature_missing_days.append(day)
                elif feature_count < 24:
                    feature_partial_days.append({"date": day, "rows": feature_count, "missing_rows": 24 - feature_count})
        total_expected += expected_rows_per_spot
        total_observed += observed
        total_feature_complete += feature_complete
        by_spot.append({
            "spot_id": spot_id,
            "spot_name": spot.get("name"),
            "expected_rows": expected_rows_per_spot,
            "observed_rows": observed,
            "missing_rows": expected_rows_per_spot - observed,
            "required_feature_complete_rows": feature_complete,
            "required_feature_missing_rows": expected_rows_per_spot - feature_complete if required_features else None,
            "required_feature_missing_day_count": len(feature_missing_days) if required_features else None,
            "required_feature_partial_day_count": len(feature_partial_days) if required_features else None,
            "missing_day_count": len(missing_days),
            "partial_day_count": len(partial_days),
            "missing_day_ranges": compress_dates(missing_days),
            "partial_days_sample": partial_days[:20],
            "required_feature_missing_day_ranges": compress_dates(feature_missing_days) if required_features else [],
            "required_feature_partial_days": feature_partial_days if required_features else [],
            "required_feature_partial_days_sample": feature_partial_days[:20],
            "all_null_feature_rows": null_feature_rows.get(spot_id, 0),
            "rows_missing_required_features": missing_required_feature_rows.get(spot_id, 0),
        })
    return {
        "format": "corsewind.open_meteo_coverage_audit.v1",
        "input_root": str(input_root),
        "model": model,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "spot_count": len(spots),
        "expected_rows": total_expected,
        "observed_rows": total_observed,
        "missing_rows": total_expected - total_observed,
        "required_features": required_features,
        "required_feature_complete_rows": total_feature_complete,
        "required_feature_missing_rows": total_expected - total_feature_complete if required_features else None,
        "complete_spot_count": sum(item["missing_rows"] == 0 for item in by_spot),
        "required_feature_complete_spot_count": (
            sum(item["required_feature_missing_rows"] == 0 for item in by_spot)
            if required_features
            else None
        ),
        "by_spot": by_spot,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--model", default="meteofrance_arome_france")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--include-context-spots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--spot-id", action="append", default=[])
    parser.add_argument("--required-feature", action="append", default=[])
    parser.add_argument("--required-features", help="Comma-separated feature names that must be non-null in each row.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(
        resolve_path(args.input_root),
        resolve_path(args.registry),
        args.model,
        parse_date(args.start_date),
        parse_date(args.end_date),
        args.include_context_spots,
        set(args.spot_id),
        [
            item.strip()
            for value in [*(args.required_feature or []), args.required_features or ""]
            for item in value.split(",")
            if item.strip()
        ],
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
