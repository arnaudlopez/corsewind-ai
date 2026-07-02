#!/usr/bin/env python3
"""Probe EUMETSAT product availability by collection and date window."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_OUTPUT = DEFAULT_ML_ROOT / "source_inventories/eumetsat_availability_probe.jsonl"
DEFAULT_BBOX = "7.5,41.0,10.2,43.3"
DEFAULT_TMP_PATHS = [
    ROOT / "tmp/eumdac_test_pkgs",
    ROOT / "tmp/copernicusmarine_test_pkgs",
]

PRODUCTS: dict[str, str] = {
    "cloud_type": "EO:EUM:DAT:0680",
    "land_surface_temperature": "EO:EUM:DAT:1088",
    "global_instability_indices": "EO:EUM:DAT:0683",
    "cloud_mask": "EO:EUM:DAT:0678",
    "msg_cloud_mask": "EO:EUM:DAT:MSG:CLM",
    "mtg_sst": "EO:EUM:DAT:0694",
    "sarah_radiation": "EO:EUM:DAT:0863",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def add_tmp_paths() -> None:
    for path in reversed(DEFAULT_TMP_PATHS):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def import_eumdac() -> Any:
    try:
        import eumdac

        return eumdac
    except ModuleNotFoundError:
        add_tmp_paths()
        import eumdac

        return eumdac


def parse_utc_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def month_starts(start: date, end: date, step_months: int) -> list[date]:
    if step_months <= 0:
        raise SystemExit("--month-step must be greater than zero")
    cursor = date(start.year, start.month, 1)
    values = []
    while cursor <= end:
        values.append(cursor)
        month = cursor.month + step_months
        year = cursor.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        cursor = date(year, month, 1)
    return values


def probe_starts(args: argparse.Namespace) -> list[datetime]:
    starts: list[datetime] = []
    for item in args.start_datetime:
        starts.append(parse_utc_datetime(item))
    for item in args.date:
        starts.append(datetime.combine(parse_date(item), datetime.min.time(), tzinfo=timezone.utc))
    if args.start_date and args.end_date:
        starts.extend(
            datetime.combine(item, datetime.min.time(), tzinfo=timezone.utc)
            for item in month_starts(parse_date(args.start_date), parse_date(args.end_date), args.month_step)
        )
    if not starts:
        raise SystemExit("Pass --date, --start-datetime, or --start-date/--end-date.")
    return sorted(set(starts))


def connect_datastore() -> Any:
    key = os.environ.get("EUMETSAT_CONSUMER_KEY")
    secret = os.environ.get("EUMETSAT_CONSUMER_SECRET")
    if not key or not secret:
        raise SystemExit("Set EUMETSAT_CONSUMER_KEY and EUMETSAT_CONSUMER_SECRET.")
    eumdac = import_eumdac()
    token = eumdac.AccessToken((key, secret), cache=False)
    return eumdac.DataStore(token)


def product_keys(args: argparse.Namespace) -> list[str]:
    keys = args.product or ["cloud_type", "land_surface_temperature", "global_instability_indices"]
    unknown = sorted(set(keys) - set(PRODUCTS))
    if unknown:
        raise SystemExit(f"Unknown product key(s): {', '.join(unknown)}")
    return keys


def product_id(product: Any) -> str:
    return str(product)


def iso_attr(product: Any, attr: str) -> str | None:
    value = getattr(product, attr, None)
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return iso_z(value)
    return str(value)


def sample_products(products: Any, max_products: int) -> tuple[int, list[dict[str, Any]]]:
    count = 0
    samples: list[dict[str, Any]] = []
    for product in products:
        count += 1
        if len(samples) < max_products:
            samples.append(
                {
                    "product_id": product_id(product),
                    "sensing_start_utc": iso_attr(product, "sensing_start"),
                    "sensing_end_utc": iso_attr(product, "sensing_end"),
                    "ingested_utc": iso_attr(product, "ingested"),
                    "timeliness": getattr(product, "timeliness", None),
                    "quality_status": getattr(product, "qualityStatus", None),
                    "size_bytes": getattr(product, "size", None),
                }
            )
    return count, samples


def write_jsonl(path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", action="append", choices=sorted(PRODUCTS), default=[])
    parser.add_argument("--date", action="append", default=[], help="UTC date to probe, YYYY-MM-DD. Repeatable.")
    parser.add_argument("--start-datetime", action="append", default=[], help="UTC datetime to probe. Repeatable.")
    parser.add_argument("--start-date", help="First monthly probe date, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Last monthly probe date, YYYY-MM-DD.")
    parser.add_argument("--month-step", type=int, default=1)
    parser.add_argument("--window-hours", type=float, default=2)
    parser.add_argument("--bbox", default=DEFAULT_BBOX)
    parser.add_argument("--sample-products", type=int, default=2)
    parser.add_argument("--request-sleep-sec", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datastore = connect_datastore()
    starts = probe_starts(args)
    keys = product_keys(args)
    rows: list[dict[str, Any]] = []
    generated_at = utc_now()
    for key in keys:
        collection_id = PRODUCTS[key]
        try:
            collection = datastore.get_collection(collection_id)
        except Exception as exc:  # eumdac exposes provider-specific errors.
            rows.append(
                {
                    "format": "corsewind.eumetsat_availability_probe.v1",
                    "generated_at_utc": generated_at,
                    "product_key": key,
                    "collection_id": collection_id,
                    "status": "collection_error",
                    "error": str(exc),
                }
            )
            continue
        for start in starts:
            end = start + timedelta(hours=args.window_hours)
            row = {
                "format": "corsewind.eumetsat_availability_probe.v1",
                "generated_at_utc": generated_at,
                "product_key": key,
                "collection_id": collection_id,
                "bbox": args.bbox,
                "start_datetime_utc": iso_z(start),
                "end_datetime_utc": iso_z(end),
                "window_hours": args.window_hours,
                "status": "ok",
                "product_count": 0,
                "sample_products": [],
            }
            try:
                products = collection.search(dtstart=start, dtend=end, bbox=args.bbox)
                count, samples = sample_products(products, args.sample_products)
                row["product_count"] = count
                row["sample_products"] = samples
            except Exception as exc:  # eumdac exposes provider-specific errors.
                row["status"] = "search_error"
                row["error"] = str(exc)
            rows.append(row)
            if args.request_sleep_sec:
                time.sleep(args.request_sleep_sec)
    output = resolve_path(args.output)
    write_jsonl(output, rows, args.append)
    summary = {
        "generated_at_utc": generated_at,
        "output": str(output),
        "row_count": len(rows),
        "products": keys,
        "probe_count": len(starts),
        "available_rows": sum(1 for row in rows if int(row.get("product_count") or 0) > 0),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
