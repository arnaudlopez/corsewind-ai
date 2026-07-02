#!/usr/bin/env python3
"""Backfill public WindsUp spot observation pages into normalized JSONL."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_ML_ROOT / "observations/windsup/normalized"
DEFAULT_RAW_ROOT = DEFAULT_ML_ROOT / "observations/windsup/raw"
WINDSUP_BASE_URL = "https://www.winds-up.com"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
PARIS = ZoneInfo("Europe/Paris")
KT_TO_MS = 0.5144444444444445
CARDINAL_TO_DEG = {
    "N": 0,
    "NNE": 22,
    "NE": 45,
    "ENE": 67,
    "E": 90,
    "ESE": 112,
    "SE": 135,
    "SSE": 157,
    "S": 180,
    "SSO": 202,
    "SO": 225,
    "OSO": 247,
    "O": 270,
    "ONO": 292,
    "NO": 315,
    "NNO": 337,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def knots_to_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * KT_TO_MS, 6)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise SystemExit("--end-date must be after or equal to --start-date")
    values = []
    cursor = start
    while cursor <= end:
        values.append(cursor)
        cursor += timedelta(days=1)
    return values


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def load_windsup_spots(path: Path, selected_ids: set[str]) -> list[dict[str, Any]]:
    payload = read_json(path)
    spots = payload.get("spots", []) if isinstance(payload, dict) else payload
    output = []
    for spot in spots:
        if not isinstance(spot, dict) or spot.get("source_type") != "windsup":
            continue
        if selected_ids and str(spot.get("spot_id")) not in selected_ids and str(spot.get("station_id")) not in selected_ids:
            continue
        if not spot.get("station_id"):
            continue
        output.append(spot)
    return output


def fetch_html(spot_id: str, day: date, timeout: int) -> str:
    url = f"{WINDSUP_BASE_URL}/spot/{spot_id}/?date={day.isoformat()}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def normalize_hour_minute(value: str) -> str:
    hour, minute = value.split(":")
    return f"{hour.zfill(2)}:{minute}"


def hour_minute_paris(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, PARIS).strftime("%H:%M")


def parse_degree(value: Any) -> int | None:
    number = finite_float(value)
    if number is None:
        return None
    degree = int(number)
    return degree if 0 <= degree <= 360 else None


def parse_windsup_html(html: str, requested_day: date) -> list[dict[str, Any]]:
    avg_regex = re.compile(r'\{x:(\d{13}),y:(\d+(?:\.\d+)?),o:"([^"]*)",color:"[^"]*",img:"[^"]*",?\}')
    gust_regex = re.compile(r"\{x:(\d{13}),low:(\d+(?:\.\d+)?),high:(\d+(?:\.\d+)?),?\}")
    row_regex = re.compile(
        r'<div\b[^>]*class=["\'][^"\']*\bspotObsLine\b[^"\']*["\'][^>]*>([\s\S]*?)(?=<div\b[^>]*class=["\'][^"\']*\bspotObsLine\b|<script|$)',
        re.IGNORECASE,
    )
    degree_by_minute: dict[str, int | None] = {}
    for match in row_regex.finditer(html):
        block = match.group(1)
        text = re.sub(r"<[^>]+>", " ", block)
        time_match = re.search(r"\b(\d{1,2}:\d{2})\b", text)
        degree_match = re.search(r'class=["\'][^"\']*\bdeg\b[^"\']*["\'][^>]*>\s*(\d{1,3})\s*<', block, re.IGNORECASE)
        if time_match and degree_match:
            degree_by_minute[normalize_hour_minute(time_match.group(1))] = parse_degree(degree_match.group(1))

    avg_by_time: dict[int, dict[str, Any]] = {}
    gust_by_time: dict[int, float] = {}
    for match in avg_regex.finditer(html):
        timestamp_ms = int(match.group(1))
        avg_kt = finite_float(match.group(2))
        cardinal = match.group(3)
        minute = hour_minute_paris(timestamp_ms)
        direction = degree_by_minute.get(minute)
        if direction is None:
            direction = CARDINAL_TO_DEG.get(cardinal)
        avg_by_time[timestamp_ms] = {
            "wind_mean_kt": avg_kt,
            "wind_direction_deg": direction,
            "wind_direction_cardinal": cardinal,
        }
    for match in gust_regex.finditer(html):
        gust_by_time[int(match.group(1))] = finite_float(match.group(3)) or 0.0

    rows = []
    for timestamp_ms, values in sorted(avg_by_time.items()):
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc)
        local_day = timestamp.astimezone(PARIS).date()
        if local_day != requested_day:
            continue
        wind_kt = values["wind_mean_kt"]
        gust_kt = gust_by_time.get(timestamp_ms, wind_kt)
        rows.append({
            "timestamp_utc": iso_z(timestamp),
            "timestamp_local_date": local_day.isoformat(),
            "timestamp_epoch_ms": timestamp_ms,
            "wind_mean_ms": knots_to_ms(wind_kt),
            "gust_ms": knots_to_ms(gust_kt),
            "wind_direction_deg": values["wind_direction_deg"],
            "wind_mean_kt_raw": wind_kt,
            "gust_kt_raw": gust_kt,
            "wind_direction_cardinal_raw": values["wind_direction_cardinal"],
        })
    return rows


def output_path(output_root: Path, timestamp_utc: str) -> Path:
    return output_root / f"date={timestamp_utc[:10]}" / "observations.jsonl"


def write_rows(output_root: Path, rows: list[dict[str, Any]]) -> dict[str, int]:
    grouped: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[output_path(output_root, row["timestamp_utc"])].append(row)
    written = {}
    for path, new_rows in grouped.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = iter_jsonl(path)
        deduped = {
            (
                row.get("source_project"),
                row.get("source_dataset"),
                row.get("station_id"),
                row.get("spot_id"),
                row.get("timestamp_utc"),
            ): row
            for row in [*existing, *new_rows]
        }
        ordered = sorted(deduped.values(), key=lambda row: (row.get("timestamp_utc") or "", row.get("spot_id") or ""))
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered), encoding="utf-8")
        tmp.replace(path)
        written[str(path)] = len(ordered)
    return written


def save_raw(raw_root: Path, spot: dict[str, Any], day: date, html: str) -> Path:
    path = raw_root / f"station={spot['station_id']}" / f"date={day.isoformat()}" / "page.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def collect_spot_day(spot: dict[str, Any], day: date, args: argparse.Namespace) -> dict[str, Any]:
    html = fetch_html(str(spot["station_id"]), day, args.timeout_sec)
    parsed = parse_windsup_html(html, day)
    fetched_at = utc_now()
    rows = []
    for item in parsed:
        rows.append({
            "format": "corsewind.windsup_observation.v1",
            "source_project": "windsup",
            "source_dataset": "windsup_public_spot_history",
            "source_url": f"{WINDSUP_BASE_URL}/spot/{spot['station_id']}/?date={day.isoformat()}",
            "spot_id": spot.get("spot_id"),
            "spot_name": spot.get("name"),
            "spot_kind": spot.get("kind"),
            "spot_source_type": spot.get("source_type"),
            "station_id": str(spot.get("station_id")),
            "latitude": finite_float(spot.get("latitude")),
            "longitude": finite_float(spot.get("longitude")),
            "source_resolution_minutes": finite_float(spot.get("source_resolution_minutes")),
            "use_for_ml": bool(spot.get("use_for_ml", False)),
            "received_at_utc": fetched_at,
            **item,
        })
    written = {} if args.dry_run else write_rows(resolve_path(args.output_root), rows)
    raw_path = None
    if args.save_raw_html and not args.dry_run:
        raw_path = save_raw(resolve_path(args.raw_root), spot, day, html)
    return {
        "spot_id": spot.get("spot_id"),
        "station_id": spot.get("station_id"),
        "date": day.isoformat(),
        "parsed_rows": len(rows),
        "first_timestamp_utc": rows[0]["timestamp_utc"] if rows else None,
        "last_timestamp_utc": rows[-1]["timestamp_utc"] if rows else None,
        "written": written,
        "raw_path": str(raw_path) if raw_path else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--spot-id", action="append", default=[], help="Filter by CorseWind spot_id or WindsUp station_id.")
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--request-sleep-sec", type=float, default=0.5)
    parser.add_argument("--save-raw-html", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Only print the plan and final summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spots = load_windsup_spots(resolve_path(args.registry), set(args.spot_id))
    days = date_range(parse_date(args.start_date), parse_date(args.end_date))
    plan = {
        "format": "corsewind.windsup_backfill_plan.v1",
        "generated_at_utc": utc_now(),
        "spot_count": len(spots),
        "day_count": len(days),
        "request_count": len(spots) * len(days),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "output_root": str(resolve_path(args.output_root)),
        "dry_run": args.dry_run,
    }
    results = []
    errors = []
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    for day in days:
        for spot in spots:
            try:
                result = collect_spot_day(spot, day, args)
                results.append(result)
                if not args.quiet:
                    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
                error = {"spot_id": spot.get("spot_id"), "station_id": spot.get("station_id"), "date": day.isoformat(), "error": str(exc)}
                errors.append(error)
                if not args.quiet:
                    print(json.dumps(error, ensure_ascii=False, sort_keys=True))
            time.sleep(args.request_sleep_sec)
    summary = {
        **plan,
        "parsed_rows": sum(item.get("parsed_rows", 0) for item in results),
        "successful_requests": len(results),
        "error_count": len(errors),
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
