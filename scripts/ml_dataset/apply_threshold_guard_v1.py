#!/usr/bin/env python3
"""Add threshold-guard shadow prediction columns to an existing prediction parquet."""

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


def add_wind_guard(frame: Any) -> dict[str, Any]:
    require_columns(frame, ["raw_wind_mean_kt", "shadow_router_v1_wind_mean_kt"])
    output = frame["shadow_router_v1_wind_mean_kt"].astype(float).copy()
    source = frame.get("shadow_router_v1_wind_mean_choice", "router")
    source = source.astype(str).copy() if hasattr(source, "astype") else "router"
    raw15_repair = (frame["raw_wind_mean_kt"].astype(float) >= 15.0) & (output < 15.0)
    output.loc[raw15_repair] = frame.loc[raw15_repair, "raw_wind_mean_kt"].astype(float)
    if hasattr(source, "loc"):
        source.loc[raw15_repair] = "raw_15kt_repair"
    frame["threshold_guard_v1_wind_mean_kt"] = output
    frame["threshold_guard_v1_wind_mean_ms"] = output * MS_PER_KT
    frame["threshold_guard_v1_wind_mean_source"] = source
    return {
        "target": "wind",
        "rows": int(len(frame)),
        "raw15_repair_rows": int(raw15_repair.sum()),
        "source_share": frame["threshold_guard_v1_wind_mean_source"].value_counts(normalize=True).round(6).to_dict(),
    }


def add_gust_guard(frame: Any) -> dict[str, Any]:
    require_columns(
        frame,
        [
            "raw_gust_kt",
            "champion_gust_kt",
            "shadow_guarded_stacker_v1_gust_kt",
        ],
    )
    output = frame["shadow_guarded_stacker_v1_gust_kt"].astype(float).copy()
    source = frame.get("shadow_guarded_stacker_v1_gust_source", "guarded_stacker")
    source = source.astype(str).copy() if hasattr(source, "astype") else "guarded_stacker"
    raw = frame["raw_gust_kt"].astype(float)
    champion = frame["champion_gust_kt"].astype(float)

    # Prefer raw/champion consensus around alert thresholds over guarded false positives.
    false_25 = (output >= 25.0) & (raw < 25.0) & (champion < 25.0)
    output.loc[false_25] = frame.loc[false_25, ["raw_gust_kt", "champion_gust_kt"]].max(axis=1).astype(float)
    false_20 = (output >= 20.0) & (raw < 20.0) & (champion < 20.0)
    output.loc[false_20] = frame.loc[false_20, ["raw_gust_kt", "champion_gust_kt"]].max(axis=1).astype(float)

    miss_25 = (output < 25.0) & ((raw >= 25.0) | (champion >= 25.0))
    output.loc[miss_25] = frame.loc[miss_25, ["raw_gust_kt", "champion_gust_kt", "shadow_guarded_stacker_v1_gust_kt"]].max(axis=1).astype(float)
    miss_20 = (output < 20.0) & ((raw >= 20.0) | (champion >= 20.0))
    output.loc[miss_20] = frame.loc[miss_20, ["raw_gust_kt", "champion_gust_kt", "shadow_guarded_stacker_v1_gust_kt"]].max(axis=1).astype(float)

    if hasattr(source, "loc"):
        source.loc[false_25 | false_20] = "raw_champion_false_positive_guard"
        source.loc[miss_25 | miss_20] = "raw_champion_miss_repair"

    frame["threshold_guard_v1_gust_kt"] = output
    frame["threshold_guard_v1_gust_ms"] = output * MS_PER_KT
    frame["threshold_guard_v1_gust_source"] = source
    return {
        "target": "gust",
        "rows": int(len(frame)),
        "false_25_guard_rows": int(false_25.sum()),
        "false_20_guard_rows": int(false_20.sum()),
        "miss_25_repair_rows": int(miss_25.sum()),
        "miss_20_repair_rows": int(miss_20.sum()),
        "source_share": frame["threshold_guard_v1_gust_source"].value_counts(normalize=True).round(6).to_dict(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]
    frame = pd.read_parquet(args.input_parquet)
    summaries = {}
    if not args.target or "wind" in args.target:
        summaries["wind"] = add_wind_guard(frame)
    if not args.target or "gust" in args.target:
        summaries["gust"] = add_gust_guard(frame)

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_parquet, index=False, compression=args.compression)
    summary = {
        "format": "corsewind.threshold_guard_v1_application",
        "generated_at_utc": utc_now(),
        "input_parquet": str(args.input_parquet),
        "output_parquet": str(args.output_parquet),
        "rows": int(len(frame)),
        "targets": summaries,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--target", choices=("wind", "gust"), action="append", default=[])
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
