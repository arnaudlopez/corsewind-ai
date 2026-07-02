#!/usr/bin/env python3
"""Apply a shadow-only low-threshold gust recall floor guard."""

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


def apply_recall_floor_guard(
    frame: Any,
    *,
    base_col: str,
    raw_col: str,
    near_kt: float,
    floor12_kt: float,
    floor15_kt: float,
) -> dict[str, Any]:
    require_columns(frame, [base_col, raw_col])
    base = frame[base_col].astype(float)
    raw = frame[raw_col].astype(float)
    output = base.copy()
    source = frame.get("probability_event_guard_v1_gust_source", frame.get("local_fallback_guard_v1_gust_source", "base"))
    source = source.astype(str).copy() if hasattr(source, "astype") else "base"

    floor12 = (raw >= floor12_kt) & (output < floor12_kt) & (output >= floor12_kt - near_kt)
    output.loc[floor12] = floor12_kt
    if hasattr(source, "loc"):
        source.loc[floor12] = f"recall_floor_raw_ge_{int(floor12_kt)}kt_near_{near_kt:g}kt"

    floor15 = (raw >= floor15_kt) & (output < floor15_kt) & (output >= floor15_kt - near_kt)
    output.loc[floor15] = floor15_kt
    if hasattr(source, "loc"):
        source.loc[floor15] = f"recall_floor_raw_ge_{int(floor15_kt)}kt_near_{near_kt:g}kt"

    frame["gust_recall_floor_guard_v1_gust_kt"] = output
    frame["gust_recall_floor_guard_v1_gust_ms"] = output * MS_PER_KT
    frame["gust_recall_floor_guard_v1_gust_source"] = source
    delta = output - base
    return {
        "target": "gust",
        "rows": int(len(frame)),
        "base_column": base_col,
        "raw_column": raw_col,
        "near_kt": near_kt,
        "floor12_kt": floor12_kt,
        "floor15_kt": floor15_kt,
        "floor12_rows": int(floor12.sum()),
        "floor15_rows": int(floor15.sum()),
        "changed_rows": int((delta != 0).sum()),
        "mean_delta_kt": float(delta.mean()),
        "max_delta_kt": float(delta.max()),
        "source_share": frame["gust_recall_floor_guard_v1_gust_source"].value_counts(normalize=True).round(6).to_dict(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]
    frame = pd.read_parquet(args.input_parquet)
    base_col = args.base_column or first_existing_column(
        frame,
        [
            "probability_event_guard_v1_gust_kt",
            "local_fallback_guard_v1_gust_kt",
            "threshold_guard_v1_gust_kt",
            "champion_gust_kt",
        ],
    )
    if base_col is None:
        raise SystemExit("No usable gust base column found.")
    summary = apply_recall_floor_guard(
        frame,
        base_col=base_col,
        raw_col=args.raw_column,
        near_kt=args.near_kt,
        floor12_kt=args.floor12_kt,
        floor15_kt=args.floor15_kt,
    )
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_parquet, index=False, compression=args.compression)
    result = {
        "format": "corsewind.gust_recall_floor_guard_v1_application",
        "generated_at_utc": utc_now(),
        "input_parquet": str(args.input_parquet),
        "output_parquet": str(args.output_parquet),
        "policy": {
            "near_kt": args.near_kt,
            "floor12_kt": args.floor12_kt,
            "floor15_kt": args.floor15_kt,
            "shadow_only": True,
        },
        "targets": {"gust": summary},
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
    parser.add_argument("--raw-column", default="raw_gust_kt")
    parser.add_argument("--near-kt", type=float, default=3.0)
    parser.add_argument("--floor12-kt", type=float, default=12.0)
    parser.add_argument("--floor15-kt", type=float, default=15.0)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
