#!/usr/bin/env python3
"""Apply a saved hindcast router/stacker artifact to live prediction rows.

This script is intentionally non-destructive: it reads a prediction parquet and
adds shadow columns for the router classifier and stacker regressor. The current
champion columns remain untouched so the shadow rail can be scored side by side.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KT_PER_MS = 1.9438444924406
MS_PER_KT = 1.0 / KT_PER_MS

UNIT_PAIRS = (
    ("raw_wind_mean_ms", "raw_wind_mean_kt"),
    ("champion_wind_mean_ms", "champion_wind_mean_kt"),
    ("strong_gated_wind_mean_ms", "strong_gated_wind_mean_kt"),
    ("raw_gust_ms", "raw_gust_kt"),
    ("champion_gust_ms", "champion_gust_kt"),
    ("gust_high_ms", "gust_high_kt"),
    ("strong_gated_gust_ms", "strong_gated_gust_kt"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def import_dependencies() -> dict[str, Any]:
    try:
        import joblib
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Missing dependencies. Run inside the CorseWind ML venv.") from exc
    return {"joblib": joblib, "pd": pd}


def add_unit_columns(frame: Any) -> list[str]:
    added = []
    for ms_col, kt_col in UNIT_PAIRS:
        if kt_col not in frame.columns and ms_col in frame.columns:
            frame[kt_col] = frame[ms_col].astype(float) * KT_PER_MS
            added.append(kt_col)
    return added


def add_time_features(frame: Any, pd: Any) -> list[str]:
    added = []
    if "target_time_utc" in frame.columns:
        target_dt = pd.to_datetime(frame["target_time_utc"], utc=True, errors="coerce")
        if "target_hour_utc" not in frame.columns:
            frame["target_hour_utc"] = target_dt.dt.hour.astype(float)
            added.append("target_hour_utc")
    if "issue_time_utc" in frame.columns:
        issue_dt = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
        if "issue_hour_utc" not in frame.columns:
            frame["issue_hour_utc"] = issue_dt.dt.hour.astype(float)
            added.append("issue_hour_utc")
        if "issue_month_number" not in frame.columns:
            frame["issue_month_number"] = issue_dt.dt.month.astype(float)
            added.append("issue_month_number")
        dayofyear = issue_dt.dt.dayofyear.fillna(1).astype(float)
        if "issue_dayofyear_sin" not in frame.columns:
            frame["issue_dayofyear_sin"] = (2.0 * math.pi * dayofyear / 366.0).map(math.sin)
            added.append("issue_dayofyear_sin")
        if "issue_dayofyear_cos" not in frame.columns:
            frame["issue_dayofyear_cos"] = (2.0 * math.pi * dayofyear / 366.0).map(math.cos)
            added.append("issue_dayofyear_cos")
    return added


def candidate_prediction(frame: Any, choices: Any, candidate_columns: dict[str, str], pd: Any) -> Any:
    prediction = pd.Series(float("nan"), index=frame.index, dtype="float64")
    for name, column in candidate_columns.items():
        mask = choices.astype(str) == str(name)
        if column in frame.columns:
            prediction.loc[mask] = frame.loc[mask, column].astype(float)
    return prediction


def target_output_prefix(target: str) -> str:
    if target == "wind":
        return "wind_mean"
    return target


def apply_target(frame: Any, target: str, artifact: dict[str, Any], pd: Any, allow_missing_features: bool) -> tuple[Any, dict[str, Any]]:
    metadata = artifact.get("metadata") or {}
    features = list(metadata.get("features") or [])
    candidate_columns = dict(metadata.get("candidate_columns") or {})
    missing_features = [column for column in features if column not in frame.columns]
    missing_candidates = [column for column in candidate_columns.values() if column not in frame.columns]
    if missing_candidates:
        raise SystemExit(f"{target}: missing candidate columns: {missing_candidates}")
    if missing_features and not allow_missing_features:
        sample = ", ".join(missing_features[:20])
        suffix = "" if len(missing_features) <= 20 else f" ... (+{len(missing_features) - 20})"
        raise SystemExit(f"{target}: missing feature columns: {sample}{suffix}")
    if missing_features:
        missing_frame = pd.DataFrame(float("nan"), index=frame.index, columns=missing_features)
        frame = pd.concat([frame, missing_frame], axis=1).copy()

    classifier = artifact.get("classifier")
    constant_choice = metadata.get("constant_classifier_choice")
    if constant_choice is not None:
        choices = pd.Series(str(constant_choice), index=frame.index)
    elif classifier is not None:
        choices = pd.Series(classifier.predict(frame[features]), index=frame.index)
    else:
        raise SystemExit(f"{target}: artifact has neither classifier nor constant choice.")

    regressor = artifact.get("regressor")
    if regressor is None:
        raise SystemExit(f"{target}: artifact has no regressor.")

    output_prefix = target_output_prefix(target)
    router_choice_col = f"shadow_router_v1_{output_prefix}_choice"
    router_kt_col = f"shadow_router_v1_{output_prefix}_kt"
    router_ms_col = f"shadow_router_v1_{output_prefix}_ms"
    stacker_kt_col = f"shadow_stacker_v1_{output_prefix}_kt"
    stacker_ms_col = f"shadow_stacker_v1_{output_prefix}_ms"
    guarded_kt_col = f"shadow_guarded_stacker_v1_{output_prefix}_kt"
    guarded_ms_col = f"shadow_guarded_stacker_v1_{output_prefix}_ms"
    guarded_source_col = f"shadow_guarded_stacker_v1_{output_prefix}_source"

    frame[router_choice_col] = choices.astype(str)
    frame[router_kt_col] = candidate_prediction(frame, choices, candidate_columns, pd)
    frame[router_ms_col] = frame[router_kt_col].astype(float) * MS_PER_KT
    frame[stacker_kt_col] = regressor.predict(frame[features])
    frame[stacker_ms_col] = frame[stacker_kt_col].astype(float) * MS_PER_KT
    strong_threshold = 20.0 if target == "gust" else 15.0
    strong_mask = frame[router_kt_col].astype(float) >= strong_threshold
    frame[guarded_kt_col] = frame[stacker_kt_col].astype(float)
    frame.loc[strong_mask, guarded_kt_col] = frame.loc[strong_mask, [stacker_kt_col, router_kt_col]].max(axis=1)
    frame[guarded_ms_col] = frame[guarded_kt_col].astype(float) * MS_PER_KT
    frame[guarded_source_col] = "stacker"
    frame.loc[strong_mask & (frame[guarded_kt_col] == frame[router_kt_col]), guarded_source_col] = "router_guard"

    summary = {
        "target": target,
        "rows": int(len(frame)),
        "feature_count": len(features),
        "missing_features_filled": missing_features,
        "candidate_columns": candidate_columns,
        "choice_share": choices.value_counts(normalize=True).round(6).to_dict(),
        "guard_threshold_kt": strong_threshold,
        "guarded_source_share": frame[guarded_source_col].value_counts(normalize=True).round(6).to_dict(),
        "columns_added": [
            router_choice_col,
            router_kt_col,
            router_ms_col,
            stacker_kt_col,
            stacker_ms_col,
            guarded_kt_col,
            guarded_ms_col,
            guarded_source_col,
        ],
    }
    return frame, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    joblib = deps["joblib"]
    pd = deps["pd"]

    frame = pd.read_parquet(args.input_parquet)
    payload = joblib.load(args.artifact)
    if payload.get("format") != "corsewind.hindcast_router_v1_final_models":
        raise SystemExit(f"Unexpected artifact format: {payload.get('format')}")

    added_unit_columns = add_unit_columns(frame)
    added_time_features = add_time_features(frame, pd)
    targets = args.target or sorted((payload.get("targets") or {}).keys())
    target_summaries = {}
    for target in targets:
        target_artifact = (payload.get("targets") or {}).get(target)
        if not target_artifact:
            raise SystemExit(f"Artifact does not contain target: {target}")
        frame, target_summaries[target] = apply_target(frame, target, target_artifact, pd, args.allow_missing_features)

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_parquet, index=False, compression=args.compression)

    summary = {
        "format": "corsewind.hindcast_router_v1_shadow_application",
        "generated_at_utc": utc_now(),
        "input_parquet": str(args.input_parquet),
        "artifact": str(args.artifact),
        "output_parquet": str(args.output_parquet),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "added_unit_columns": added_unit_columns,
        "added_time_features": added_time_features,
        "targets": target_summaries,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--target", choices=("wind", "gust"), action="append", default=[])
    parser.add_argument("--allow-missing-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
