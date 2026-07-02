#!/usr/bin/env python3
"""Apply a shadow-only wind floor guard confirmed by corrected gust signal."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KT_PER_MS = 1.9438444924406
MS_PER_KT = 1.0 / KT_PER_MS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def import_dependencies() -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Missing pandas. Run inside the CorseWind ML venv.") from exc
    return {"pd": pd}


def require_columns(frame: Any, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")


def first_existing_column(frame: Any, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def apply_wind_gust_floor_guard(
    frame: Any,
    *,
    base_col: str,
    gust_col: str,
    base_min_kt: float,
    gust_min_kt: float,
    wind_floor_kt: float,
) -> dict[str, Any]:
    require_columns(frame, [base_col, gust_col])
    base = frame[base_col].astype(float)
    gust = frame[gust_col].astype(float)
    output = base.copy()
    source = frame.get("wind_high_event_guard_v1_wind_mean_source", "wind_high_event_guard")
    source = source.astype(str).copy() if hasattr(source, "astype") else "wind_high_event_guard"

    floor_mask = (base >= base_min_kt) & (base < wind_floor_kt) & (gust >= gust_min_kt)
    output.loc[floor_mask] = wind_floor_kt
    if hasattr(source, "loc"):
        source.loc[floor_mask] = f"gust_confirmed_wind_floor_{int(wind_floor_kt)}kt"

    frame["wind_gust_floor_guard_v1_wind_mean_kt"] = output
    frame["wind_gust_floor_guard_v1_wind_mean_ms"] = output * MS_PER_KT
    frame["wind_gust_floor_guard_v1_wind_mean_source"] = source
    delta = output - base
    return {
        "target": "wind",
        "rows": int(len(frame)),
        "base_column": base_col,
        "gust_column": gust_col,
        "base_min_kt": base_min_kt,
        "gust_min_kt": gust_min_kt,
        "wind_floor_kt": wind_floor_kt,
        "floor_rows": int(floor_mask.sum()),
        "changed_rows": int((delta != 0).sum()),
        "mean_delta_kt": float(delta.mean()),
        "max_delta_kt": float(delta.max()),
        "source_share": frame["wind_gust_floor_guard_v1_wind_mean_source"].value_counts(normalize=True).round(6).to_dict(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]
    frame = pd.read_parquet(args.input_parquet)
    base_col = args.base_column or first_existing_column(
        frame,
        [
            "wind_high_event_guard_v1_wind_mean_kt",
            "threshold_guard_v1_wind_mean_kt",
            "strong_gated_wind_mean_kt",
            "champion_wind_mean_kt",
        ],
    )
    gust_col = args.gust_column or first_existing_column(
        frame,
        [
            "gust_recall_floor_guard_v1_gust_kt",
            "probability_event_guard_v1_gust_kt",
            "local_fallback_guard_v1_gust_kt",
            "threshold_guard_v1_gust_kt",
            "gust_high_kt",
        ],
    )
    if base_col is None:
        raise SystemExit("No usable wind base column found.")
    if gust_col is None:
        raise SystemExit("No usable gust confirmation column found.")
    summary = apply_wind_gust_floor_guard(
        frame,
        base_col=base_col,
        gust_col=gust_col,
        base_min_kt=args.base_min_kt,
        gust_min_kt=args.gust_min_kt,
        wind_floor_kt=args.wind_floor_kt,
    )
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_parquet, index=False, compression=args.compression)
    result = {
        "format": "corsewind.wind_gust_floor_guard_v1_application",
        "generated_at_utc": utc_now(),
        "input_parquet": str(args.input_parquet),
        "output_parquet": str(args.output_parquet),
        "policy": {
            "base_min_kt": args.base_min_kt,
            "gust_min_kt": args.gust_min_kt,
            "wind_floor_kt": args.wind_floor_kt,
            "shadow_only": True,
        },
        "targets": {"wind": summary},
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--base-column")
    parser.add_argument("--gust-column")
    parser.add_argument("--base-min-kt", type=float, default=8.0)
    parser.add_argument("--gust-min-kt", type=float, default=14.0)
    parser.add_argument("--wind-floor-kt", type=float, default=12.0)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
