#!/usr/bin/env python3
"""Create a minimal Meteo-France forecast layer manifest for spot sampling."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def parse_utc_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_layer(args: argparse.Namespace) -> dict[str, Any]:
    run_time = parse_utc_datetime(args.run_time_utc)
    steps = []
    for lead in range(args.start_lead_minutes, args.end_lead_minutes + 1, args.step_minutes):
        valid_time = run_time + timedelta(minutes=lead)
        steps.append(
            {
                "lead_minutes": lead,
                "lead_hour": round(lead / 60.0, 6),
                "valid_time_utc": iso_z(valid_time),
                "shape": [1, 1],
            }
        )
    return {
        "format": "corsewind.meteo_france_forecast_grid_layer_manifest.v1",
        "source": args.source,
        "model_label": args.source.upper(),
        "run_time_utc": iso_z(run_time),
        "bbox_wgs84": args.bbox,
        "forecast_steps": steps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["arome", "aromepi"], required=True)
    parser.add_argument("--run-time-utc", required=True)
    parser.add_argument("--start-lead-minutes", type=int, default=15)
    parser.add_argument("--end-lead-minutes", type=int, default=360)
    parser.add_argument("--step-minutes", type=int, default=15)
    parser.add_argument("--bbox", type=float, nargs=4, default=[8.45, 41.25, 9.75, 43.1])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.step_minutes <= 0:
        raise SystemExit("--step-minutes must be positive")
    if args.end_lead_minutes < args.start_lead_minutes:
        raise SystemExit("--end-lead-minutes must be greater than or equal to --start-lead-minutes")
    layer = build_layer(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(layer, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "step_count": len(layer["forecast_steps"])}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
