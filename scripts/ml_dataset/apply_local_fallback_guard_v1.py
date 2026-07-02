#!/usr/bin/env python3
"""Apply an inference-safe local fallback guard from risk-audit cells."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KT_PER_MS = 1.9438444924406
MS_PER_KT = 1.0 / KT_PER_MS

INFERENCE_SAFE_GROUPS = {"spot_id", "target_hour_utc", "lead_bucket"}
GUST_BASELINE_COLUMNS = {
    "raw": "raw_gust_kt",
    "champion": "champion_gust_kt",
    "high": "gust_high_kt",
    "guarded_stacker": "shadow_guarded_stacker_v1_gust_kt",
    "threshold_guard": "threshold_guard_v1_gust_kt",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def import_dependencies() -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Missing pandas. Run inside the CorseWind ML venv.") from exc
    return {"pd": pd}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_columns(frame: Any, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")


def lead_bucket_from_minutes(value: Any) -> str:
    try:
        minutes = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if minutes <= 60:
        return "0-1h"
    if minutes <= 180:
        return "1-3h"
    if minutes <= 360:
        return "3-6h"
    return "6h+"


def ensure_inference_groups(frame: Any, pd: Any) -> None:
    if "target_hour_utc" not in frame.columns:
        require_columns(frame, ["target_time_utc"])
        frame["target_hour_utc"] = pd.to_datetime(frame["target_time_utc"], utc=True, errors="coerce").dt.hour
    if "lead_bucket" not in frame.columns:
        require_columns(frame, ["lead_time_minutes"])
        frame["lead_bucket"] = frame["lead_time_minutes"].map(lead_bucket_from_minutes)


def policy_from_audit(
    audit: dict[str, Any],
    *,
    target: str,
    min_rows: int,
    min_regression_kt: float,
) -> list[dict[str, Any]]:
    flags = (((audit.get("risk_flags") or {}).get("by_target") or {}).get(target) or {}).get("flags") or []
    rules: list[dict[str, Any]] = []
    for flag in flags:
        group = str(flag.get("group") or "")
        if group not in INFERENCE_SAFE_GROUPS:
            continue
        rows = int(flag.get("rows") or 0)
        regression = float(flag.get("regression") or flag.get("rmse_regression") or 0.0)
        baseline = str(flag.get("best_baseline") or "")
        if rows < min_rows or regression < min_regression_kt or baseline not in GUST_BASELINE_COLUMNS:
            continue
        rules.append(
            {
                "target": target,
                "group": group,
                "value": str(flag.get("value")),
                "fallback": baseline,
                "fallback_column": GUST_BASELINE_COLUMNS[baseline],
                "audit_rows": rows,
                "audit_regression_kt": regression,
                "candidate_rmse_kt": flag.get("candidate_rmse"),
                "fallback_rmse_kt": flag.get("best_baseline_rmse"),
            }
        )
    rules.sort(key=lambda item: (float(item["audit_regression_kt"]), int(item["audit_rows"])), reverse=True)
    return rules


def apply_gust_rules(frame: Any, rules: list[dict[str, Any]], *, preserve_threshold_kt: float | None) -> dict[str, Any]:
    require_columns(frame, ["threshold_guard_v1_gust_kt"])
    original = frame["threshold_guard_v1_gust_kt"].astype(float).copy()
    output = original.copy()
    source = frame.get("threshold_guard_v1_gust_source", "threshold_guard")
    source = source.astype(str).copy() if hasattr(source, "astype") else "threshold_guard"

    applied: list[dict[str, Any]] = []
    matched_any = None
    for rule in rules:
        group = str(rule["group"])
        value = str(rule["value"])
        fallback_column = str(rule["fallback_column"])
        if group not in frame.columns or fallback_column not in frame.columns:
            applied.append({**rule, "applied_rows": 0, "skipped_reason": "missing_column"})
            continue
        mask = frame[group].astype(str) == value
        applied_rows = int(mask.sum())
        preserved_rows = 0
        if applied_rows:
            fallback = frame.loc[mask, fallback_column].astype(float)
            replacement = fallback.copy()
            if preserve_threshold_kt is not None:
                original_subset = original.loc[mask]
                preserve_mask = (original_subset >= preserve_threshold_kt) & (fallback < preserve_threshold_kt)
                preserved_rows = int(preserve_mask.sum())
                replacement.loc[preserve_mask] = original_subset.loc[preserve_mask]
            output.loc[mask] = replacement
            if hasattr(source, "loc"):
                source.loc[mask] = f"local_fallback_{group}_{value}_to_{rule['fallback']}"
                if preserve_threshold_kt is not None and preserved_rows:
                    source.loc[mask & (original >= preserve_threshold_kt) & (frame[fallback_column].astype(float) < preserve_threshold_kt)] = (
                        f"local_fallback_{group}_{value}_preserve_{int(preserve_threshold_kt)}kt"
                    )
            matched_any = mask if matched_any is None else (matched_any | mask)
        applied.append({**rule, "applied_rows": applied_rows, "preserved_threshold_rows": preserved_rows})

    frame["local_fallback_guard_v1_gust_kt"] = output
    frame["local_fallback_guard_v1_gust_ms"] = output * MS_PER_KT
    frame["local_fallback_guard_v1_gust_source"] = source
    return {
        "target": "gust",
        "rows": int(len(frame)),
        "rules": applied,
        "fallback_rows": 0 if matched_any is None else int(matched_any.sum()),
        "source_share": frame["local_fallback_guard_v1_gust_source"].value_counts(normalize=True).round(6).to_dict(),
    }


def apply_wind_high_event_guard(
    frame: Any,
    *,
    wind_floor_kt: float,
    wind_event_kt: float,
    gust_confirm_kt: float,
) -> dict[str, Any]:
    require_columns(frame, ["threshold_guard_v1_wind_mean_kt", "local_fallback_guard_v1_gust_kt"])
    base = frame["threshold_guard_v1_wind_mean_kt"].astype(float)
    gust = frame["local_fallback_guard_v1_gust_kt"].astype(float)
    output = base.copy()
    source = frame.get("threshold_guard_v1_wind_mean_source", "threshold_guard")
    source = source.astype(str).copy() if hasattr(source, "astype") else "threshold_guard"

    high_event_repair = (base >= wind_floor_kt) & (base < wind_event_kt) & (gust >= gust_confirm_kt)
    output.loc[high_event_repair] = wind_event_kt
    if hasattr(source, "loc"):
        source.loc[high_event_repair] = "gust_confirmed_high_wind_repair"

    frame["wind_high_event_guard_v1_wind_mean_kt"] = output
    frame["wind_high_event_guard_v1_wind_mean_ms"] = output * MS_PER_KT
    frame["wind_high_event_guard_v1_wind_mean_source"] = source
    return {
        "target": "wind",
        "rows": int(len(frame)),
        "wind_floor_kt": wind_floor_kt,
        "wind_event_kt": wind_event_kt,
        "gust_confirm_kt": gust_confirm_kt,
        "repair_rows": int(high_event_repair.sum()),
        "source_share": frame["wind_high_event_guard_v1_wind_mean_source"].value_counts(normalize=True).round(6).to_dict(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]
    frame = pd.read_parquet(args.input_parquet)
    ensure_inference_groups(frame, pd)
    audit = read_json(args.risk_audit_json)
    rules = policy_from_audit(
        audit,
        target="gust",
        min_rows=args.min_rows,
        min_regression_kt=args.min_regression_kt,
    )
    gust_summary = apply_gust_rules(frame, rules, preserve_threshold_kt=args.preserve_threshold_kt)
    wind_summary = apply_wind_high_event_guard(
        frame,
        wind_floor_kt=args.wind_floor_kt,
        wind_event_kt=args.wind_event_kt,
        gust_confirm_kt=args.wind_gust_confirm_kt,
    )
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_parquet, index=False, compression=args.compression)
    result = {
        "format": "corsewind.local_fallback_guard_v1_application",
        "generated_at_utc": utc_now(),
        "input_parquet": str(args.input_parquet),
        "risk_audit_json": str(args.risk_audit_json),
        "output_parquet": str(args.output_parquet),
        "policy": {
            "inference_safe_groups": sorted(INFERENCE_SAFE_GROUPS),
            "min_rows": args.min_rows,
            "min_regression_kt": args.min_regression_kt,
            "preserve_threshold_kt": args.preserve_threshold_kt,
            "wind_floor_kt": args.wind_floor_kt,
            "wind_event_kt": args.wind_event_kt,
            "wind_gust_confirm_kt": args.wind_gust_confirm_kt,
        },
        "targets": {"gust": gust_summary, "wind": wind_summary},
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--risk-audit-json", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--min-rows", type=int, default=20)
    parser.add_argument("--min-regression-kt", type=float, default=0.10)
    parser.add_argument("--preserve-threshold-kt", type=float, default=25.0)
    parser.add_argument("--wind-floor-kt", type=float, default=17.0)
    parser.add_argument("--wind-event-kt", type=float, default=20.0)
    parser.add_argument("--wind-gust-confirm-kt", type=float, default=28.0)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
