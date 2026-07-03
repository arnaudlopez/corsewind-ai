#!/usr/bin/env python3
"""Audit observation label semantics and short-window noise proxies."""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def expand_patterns(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            candidate = Path(pattern)
            if candidate.exists():
                paths.append(candidate)
    return sorted(dict.fromkeys(paths))


def finite_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    spots = payload.get("spots") if isinstance(payload, dict) else payload
    if not isinstance(spots, list):
        return {}
    return {str(spot.get("spot_id")): spot for spot in spots if spot.get("spot_id")}


def read_observations(paths: list[Path], registry: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                spot_id = str(row.get("spot_id") or "")
                timestamp = parse_utc(row.get("timestamp_utc"))
                if not spot_id or timestamp is None:
                    continue
                spot = registry.get(spot_id, {})
                rows.append(
                    {
                        "spot_id": spot_id,
                        "timestamp": timestamp,
                        "source_dataset": str(row.get("source_dataset") or ""),
                        "source_project": str(row.get("source_project") or ""),
                        "station_id": str(row.get("station_id") or spot.get("station_id") or ""),
                        "registry_source_type": str(spot.get("source_type") or ""),
                        "score_track": str(spot.get("score_track") or ""),
                        "registry_resolution_minutes": finite_float(spot.get("source_resolution_minutes")),
                        "wind_mean_ms": finite_float(row.get("wind_mean_ms")),
                        "gust_ms": finite_float(row.get("gust_ms")),
                        "wind_direction_deg": finite_float(row.get("wind_direction_deg")),
                    }
                )
    return rows


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    return ordered[lower] * (upper - pos) + ordered[upper] * (pos - lower)


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None, "p95": None}
    return {
        "p50": round(float(quantile(values, 0.50)), 6),
        "p90": round(float(quantile(values, 0.90)), 6),
        "p95": round(float(quantile(values, 0.95)), 6),
    }


def pct(count: int, total: int) -> float:
    return round(count / total * 100.0, 3) if total else 0.0


def offset_to_15m_grid_minutes(timestamp: datetime) -> float:
    minute_float = timestamp.minute + timestamp.second / 60.0 + timestamp.microsecond / 60_000_000.0
    remainder = minute_float % 15.0
    return min(remainder, 15.0 - remainder)


def group_summary(rows_in: list[dict[str, Any]], group_cols: list[str]) -> list[dict[str, Any]]:
    rows = []
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows_in:
        groups[tuple(str(row.get(column) or "") for column in group_cols)].append(row)
    for key_tuple, group_rows in groups.items():
        item = {column: str(value) for column, value in zip(group_cols, key_tuple)}
        group_rows = sorted(group_rows, key=lambda row: (row["spot_id"], row["timestamp"]))
        by_spot: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            by_spot[row["spot_id"]].append(row)
        cadence_values: list[float] = []
        wind_step_delta: list[float] = []
        gust_step_delta: list[float] = []
        for spot_rows in by_spot.values():
            previous = None
            for row in spot_rows:
                if previous is not None:
                    cadence = (row["timestamp"] - previous["timestamp"]).total_seconds() / 60.0
                    if cadence > 0:
                        cadence_values.append(cadence)
                    if row.get("wind_mean_ms") is not None and previous.get("wind_mean_ms") is not None:
                        wind_step_delta.append(abs(float(row["wind_mean_ms"]) - float(previous["wind_mean_ms"])))
                    if row.get("gust_ms") is not None and previous.get("gust_ms") is not None:
                        gust_step_delta.append(abs(float(row["gust_ms"]) - float(previous["gust_ms"])))
                previous = row
        offsets = [offset_to_15m_grid_minutes(row["timestamp"]) for row in group_rows]
        timestamps = [row["timestamp"] for row in group_rows]
        row_count = len(group_rows)
        item.update(
            {
                "rows": row_count,
                "spot_count": len({row["spot_id"] for row in group_rows}),
                "first_timestamp_utc": min(timestamps).isoformat().replace("+00:00", "Z"),
                "last_timestamp_utc": max(timestamps).isoformat().replace("+00:00", "Z"),
                "wind_present_pct": pct(sum(row.get("wind_mean_ms") is not None for row in group_rows), row_count),
                "gust_present_pct": pct(sum(row.get("gust_ms") is not None for row in group_rows), row_count),
                "direction_present_pct": pct(sum(row.get("wind_direction_deg") is not None for row in group_rows), row_count),
                "median_cadence_minutes": None if not cadence_values else round(float(statistics.median(cadence_values)), 6),
                "p90_cadence_minutes": None if not cadence_values else round(float(quantile(cadence_values, 0.90)), 6),
                "median_offset_to_15m_grid_minutes": round(float(statistics.median(offsets)), 6),
                "p90_offset_to_15m_grid_minutes": round(float(quantile(offsets, 0.90)), 6),
                "wind_abs_step_delta_ms": quantiles(wind_step_delta),
                "gust_abs_step_delta_ms": quantiles(gust_step_delta),
            }
        )
        rows.append(item)
    return sorted(rows, key=lambda row: (-row["rows"], tuple(str(row.get(column, "")) for column in group_cols)))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    paths = expand_patterns(args.observations_jsonl)
    registry = load_registry(args.registry)
    rows = read_observations(paths, registry)
    if args.start_utc:
        start = parse_utc(args.start_utc)
        if start is not None:
            rows = [row for row in rows if row["timestamp"] >= start]
    if args.end_utc:
        end = parse_utc(args.end_utc)
        if end is not None:
            rows = [row for row in rows if row["timestamp"] <= end]
    if not rows:
        return {
            "format": "corsewind.label_semantics_audit.v1",
            "generated_at_utc": utc_now(),
            "observation_file_count": len(paths),
            "row_count": 0,
            "error": "no observations loaded",
        }
    timestamps = [row["timestamp"] for row in rows]
    return {
        "format": "corsewind.label_semantics_audit.v1",
        "generated_at_utc": utc_now(),
        "registry": str(args.registry),
        "observation_file_count": len(paths),
        "row_count": int(len(rows)),
        "first_timestamp_utc": min(timestamps).isoformat().replace("+00:00", "Z"),
        "last_timestamp_utc": max(timestamps).isoformat().replace("+00:00", "Z"),
        "by_score_track": group_summary(rows, ["score_track"]),
        "by_source_type": group_summary(rows, ["registry_source_type"]),
        "by_source_dataset": group_summary(rows, ["source_project", "source_dataset"]),
        "by_spot": group_summary(rows, ["score_track", "spot_id"]),
    }


def render_table(rows: list[dict[str, Any]], columns: list[str], limit: int) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows[:limit]:
        values = []
        for column in columns:
            value = row.get(column)
            values.append(f"`{value}`" if isinstance(value, str) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def render_markdown(report: dict[str, Any], limit: int) -> str:
    lines = [
        "# Label Semantics Audit V1",
        "",
        f"- generated: `{report.get('generated_at_utc')}`",
        f"- rows: `{report.get('row_count')}`",
        f"- files: `{report.get('observation_file_count')}`",
        f"- first timestamp: `{report.get('first_timestamp_utc')}`",
        f"- last timestamp: `{report.get('last_timestamp_utc')}`",
        "",
    ]
    if report.get("error"):
        lines.append(f"- error: `{report['error']}`")
        return "\n".join(lines) + "\n"
    sections = [
        ("By Score Track", "by_score_track", ["score_track", "rows", "spot_count", "wind_present_pct", "gust_present_pct", "median_cadence_minutes", "median_offset_to_15m_grid_minutes"]),
        ("By Source Type", "by_source_type", ["registry_source_type", "rows", "spot_count", "wind_present_pct", "gust_present_pct", "median_cadence_minutes", "median_offset_to_15m_grid_minutes"]),
        ("By Source Dataset", "by_source_dataset", ["source_project", "source_dataset", "rows", "spot_count", "wind_present_pct", "gust_present_pct", "median_cadence_minutes"]),
        ("By Spot", "by_spot", ["score_track", "spot_id", "rows", "wind_present_pct", "gust_present_pct", "median_cadence_minutes", "p90_offset_to_15m_grid_minutes"]),
    ]
    for title, key, columns in sections:
        lines.extend([f"## {title}", ""])
        lines.extend(render_table(report.get(key) or [], columns, limit))
        lines.append("")
    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- `median_cadence_minutes` is computed from consecutive observations per spot.",
            "- `median_offset_to_15m_grid_minutes` estimates how far native observations sit from the 15-minute forecast grid.",
            "- `wind_abs_step_delta_ms` and `gust_abs_step_delta_ms` in JSON are short-step volatility proxies, not a final noise floor.",
            "- A true canonical label pass should build centered windows and compare source-native versus canonical labels.",
            "",
        ]
    )
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("configs/ml_spots.json"))
    parser.add_argument("--observations-jsonl", action="append", required=True)
    parser.add_argument("--start-utc")
    parser.add_argument("--end-utc")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--markdown-limit", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.output_json:
        write_json(args.output_json, report)
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(report, args.markdown_limit), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
