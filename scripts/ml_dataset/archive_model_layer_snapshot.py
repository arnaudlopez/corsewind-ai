#!/usr/bin/env python3
"""Archive Wind2D model-layer JSON snapshots for ML dataset construction."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_OUTPUT_ROOT = DEFAULT_ML_ROOT / "model_runs"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_time_slug(value: str) -> str:
    return value.replace(":", "").replace("-", "").replace("+00:00", "Z")


def read_layer(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Input layer does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Input layer is not valid JSON: {path}: {exc}") from exc
    if not payload.get("run_time_utc"):
        raise SystemExit(f"Input layer has no run_time_utc: {path}")
    if not payload.get("forecast_steps"):
        raise SystemExit(f"Input layer has no forecast_steps: {path}")
    return payload


def layer_summary(payload: dict[str, Any], source: str, archive_path: Path, input_path: Path) -> dict[str, Any]:
    steps = list(payload.get("forecast_steps") or [])
    valid_times = [step.get("valid_time_utc") for step in steps if step.get("valid_time_utc")]
    lead_minutes = [
        int(step["lead_minutes"])
        for step in steps
        if isinstance(step.get("lead_minutes"), int)
    ]
    lead_hours = [
        float(step["lead_hour"])
        for step in steps
        if isinstance(step.get("lead_hour"), int | float)
    ]
    return {
        "source": source,
        "format": payload.get("format"),
        "product": payload.get("product"),
        "run_time_utc": payload.get("run_time_utc"),
        "generated_at_utc": payload.get("generated_at_utc"),
        "archived_at_utc": utc_now(),
        "input_path": str(input_path),
        "archive_path": str(archive_path),
        "step_count": len(steps),
        "first_valid_time_utc": min(valid_times) if valid_times else None,
        "last_valid_time_utc": max(valid_times) if valid_times else None,
        "lead_minutes": lead_minutes or None,
        "lead_hours": lead_hours or None,
        "bbox_wgs84": payload.get("bbox_wgs84"),
        "grid": payload.get("grid"),
        "timeline": payload.get("timeline"),
        "source_file": payload.get("source_file"),
        "dataset_id": payload.get("dataset_id"),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def archive_layer(source: str, input_path: Path, output_root: Path) -> dict[str, Any]:
    input_path = resolve_path(input_path)
    output_root = resolve_path(output_root)
    payload = read_layer(input_path)
    run_time_utc = str(payload["run_time_utc"])
    run_slug = safe_time_slug(run_time_utc)
    source_root = output_root / source
    run_root = source_root / f"run_{run_slug}"
    archive_path = run_root / f"{source}_{run_slug}.json.gz"
    summary_path = run_root / "summary.json"
    latest_path = source_root / "latest.json"

    run_root.mkdir(parents=True, exist_ok=True)
    raw_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if not archive_path.exists():
        tmp = archive_path.with_suffix(archive_path.suffix + f".{os.getpid()}.tmp")
        with gzip.open(tmp, "wb", compresslevel=6) as handle:
            handle.write(raw_json)
        tmp.replace(archive_path)

    summary = layer_summary(payload, source, archive_path, input_path)
    write_json(summary_path, summary)
    write_json(latest_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=["arome", "aromepi", "moloch", "icon2i"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = archive_layer(args.source, args.input, args.output_root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
