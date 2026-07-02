#!/usr/bin/env python3
"""Apply a shadow-only gust event guard from threshold probabilities."""

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


def apply_gust_probability_event_guard(
    frame: Any,
    *,
    base_col: str,
    prob20_col: str,
    prob25_col: str,
    prob20_threshold: float,
    prob25_threshold: float,
    floor20_kt: float,
    floor25_kt: float,
) -> dict[str, Any]:
    require_columns(frame, [base_col, prob20_col, prob25_col])
    base = frame[base_col].astype(float)
    prob20 = frame[prob20_col].astype(float).clip(lower=0.0, upper=1.0)
    prob25 = frame[prob25_col].astype(float).clip(lower=0.0, upper=1.0)
    output = base.copy()
    source = frame.get("local_fallback_guard_v1_gust_source", "local_fallback_guard")
    source = source.astype(str).copy() if hasattr(source, "astype") else "local_fallback_guard"

    event25 = (prob25 >= prob25_threshold) & (output < floor25_kt)
    output.loc[event25] = floor25_kt
    if hasattr(source, "loc"):
        source.loc[event25] = f"probability_event_guard_ge25_{prob25_col}"

    event20 = (prob20 >= prob20_threshold) & (output < floor20_kt)
    output.loc[event20] = floor20_kt
    if hasattr(source, "loc"):
        source.loc[event20] = f"probability_event_guard_ge20_{prob20_col}"

    frame["probability_event_guard_v1_gust_kt"] = output
    frame["probability_event_guard_v1_gust_ms"] = output * MS_PER_KT
    frame["probability_event_guard_v1_gust_source"] = source

    changed = output != base
    return {
        "target": "gust",
        "rows": int(len(frame)),
        "base_column": base_col,
        "prob20_column": prob20_col,
        "prob25_column": prob25_col,
        "prob20_threshold": prob20_threshold,
        "prob25_threshold": prob25_threshold,
        "floor20_kt": floor20_kt,
        "floor25_kt": floor25_kt,
        "event20_rows": int(event20.sum()),
        "event25_rows": int(event25.sum()),
        "changed_rows": int(changed.sum()),
        "mean_delta_kt": float((output - base).mean()),
        "max_delta_kt": float((output - base).max()),
        "source_share": frame["probability_event_guard_v1_gust_source"].value_counts(normalize=True).round(6).to_dict(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]
    frame = pd.read_parquet(args.input_parquet)
    base_col = args.base_column or first_existing_column(
        frame,
        [
            "local_fallback_guard_v1_gust_kt",
            "threshold_guard_v1_gust_kt",
            "gust_high_kt",
            "champion_gust_kt",
        ],
    )
    if base_col is None:
        raise SystemExit("No usable gust base column found.")
    prob20_col = args.prob20_column or first_existing_column(
        frame,
        ["prob_gust_ge_20kt_heuristic", "prob_gust_ge_20kt", "prob_gust_ge_20kt_model"],
    )
    prob25_col = args.prob25_column or first_existing_column(
        frame,
        ["prob_gust_ge_25kt_heuristic", "prob_gust_ge_25kt", "prob_gust_ge_25kt_model"],
    )
    if prob20_col is None or prob25_col is None:
        raise SystemExit("Missing gust probability columns.")

    summary = apply_gust_probability_event_guard(
        frame,
        base_col=base_col,
        prob20_col=prob20_col,
        prob25_col=prob25_col,
        prob20_threshold=args.prob20_threshold,
        prob25_threshold=args.prob25_threshold,
        floor20_kt=args.floor20_kt,
        floor25_kt=args.floor25_kt,
    )
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_parquet, index=False, compression=args.compression)
    result = {
        "format": "corsewind.probability_event_guard_v1_application",
        "generated_at_utc": utc_now(),
        "input_parquet": str(args.input_parquet),
        "output_parquet": str(args.output_parquet),
        "policy": {
            "prob20_threshold": args.prob20_threshold,
            "prob25_threshold": args.prob25_threshold,
            "floor20_kt": args.floor20_kt,
            "floor25_kt": args.floor25_kt,
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
    parser.add_argument("--prob20-column")
    parser.add_argument("--prob25-column")
    parser.add_argument("--prob20-threshold", type=float, default=0.80)
    parser.add_argument("--prob25-threshold", type=float, default=0.40)
    parser.add_argument("--floor20-kt", type=float, default=20.0)
    parser.add_argument("--floor25-kt", type=float, default=25.0)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
