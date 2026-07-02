#!/usr/bin/env python3
"""Score pseudo-live CorseWind predictions against observations."""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KT_PER_MS = 1.9438444924406
DEFAULT_SPOTS = (
    "cap_corse",
    "la_parata",
    "lfkf",
    "lfkj",
    "lfks",
    "lfvf",
    "lfvh",
)


def import_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as exc:
        raise SystemExit("Missing scoring dependencies. Run inside the CorseWind ML venv.") from exc
    return {"average_precision_score": average_precision_score, "np": np, "pd": pd, "roc_auc_score": roc_auc_score}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            candidate = Path(pattern)
            if candidate.exists():
                paths.append(candidate)
    return sorted(dict.fromkeys(paths))


def parse_csv(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def metrics(frame: Any, pred_col: str, actual_col: str) -> dict[str, Any]:
    values = frame[[pred_col, actual_col]].dropna()
    if values.empty:
        return {"n": 0}
    err = values[pred_col] - values[actual_col]
    return {
        "n": int(len(values)),
        "mae": float(err.abs().mean()),
        "rmse": float((err.pow(2).mean()) ** 0.5),
        "bias": float(err.mean()),
        "p50_abs_error": float(err.abs().quantile(0.50)),
        "p90_abs_error": float(err.abs().quantile(0.90)),
    }


def threshold_metrics(frame: Any, pred_col: str, actual_col: str, threshold: float) -> dict[str, Any]:
    values = frame[[pred_col, actual_col]].dropna()
    if values.empty:
        return {"n": 0}
    pred = values[pred_col] >= threshold
    actual = values[actual_col] >= threshold
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    return {
        "n": int(len(values)),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
        "csi": None if tp + fp + fn == 0 else tp / (tp + fp + fn),
    }


def threshold_metrics_or_empty(frame: Any, pred_col: str, actual_col: str, threshold: float) -> dict[str, Any]:
    if pred_col not in frame.columns or actual_col not in frame.columns:
        return {"n": 0}
    return threshold_metrics(frame, pred_col, actual_col, threshold)


def build_threshold_summary(frame: Any) -> dict[str, Any]:
    thresholds: dict[str, Any] = {}
    wind_rails = {
        "ml": "champion_wind_mean_kt",
        "raw": "raw_wind_mean_kt",
        "strong_gated": "strong_gated_wind_mean_kt",
        "shadow_router_v1": "shadow_router_v1_wind_mean_kt",
        "shadow_stacker_v1": "shadow_stacker_v1_wind_mean_kt",
        "shadow_guarded_stacker_v1": "shadow_guarded_stacker_v1_wind_mean_kt",
        "threshold_guard_v1": "threshold_guard_v1_wind_mean_kt",
        "wind_high_event_guard_v1": "wind_high_event_guard_v1_wind_mean_kt",
    }
    gust_rails = {
        "ml": "champion_gust_kt",
        "raw": "raw_gust_kt",
        "high": "gust_high_kt",
        "strong_gated": "strong_gated_gust_kt",
        "shadow_router_v1": "shadow_router_v1_gust_kt",
        "shadow_stacker_v1": "shadow_stacker_v1_gust_kt",
        "shadow_guarded_stacker_v1": "shadow_guarded_stacker_v1_gust_kt",
        "threshold_guard_v1": "threshold_guard_v1_gust_kt",
        "local_fallback_guard_v1": "local_fallback_guard_v1_gust_kt",
    }
    for level in (12, 15, 20, 25):
        for rail, column in wind_rails.items():
            thresholds[f"wind_{level}kt_{rail}"] = threshold_metrics_or_empty(
                frame, column, "actual_wind_mean_kt", float(level)
            )
        for rail, column in gust_rails.items():
            thresholds[f"gust_{level}kt_{rail}"] = threshold_metrics_or_empty(frame, column, "actual_gust_kt", float(level))
    return thresholds


def binary_alert_metrics(frame: Any, alert_col: str, actual_col: str, actual_threshold: float) -> dict[str, Any]:
    values = frame[[alert_col, actual_col]].dropna()
    if values.empty:
        return {"n": 0}
    pred = values[alert_col].astype(bool)
    actual = values[actual_col] >= actual_threshold
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    return {
        "n": int(len(values)),
        "actual_threshold": actual_threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
        "csi": None if tp + fp + fn == 0 else tp / (tp + fp + fn),
    }


def probability_threshold_metrics(y_true: Any, probability: Any, threshold: float) -> dict[str, Any]:
    pred = probability >= threshold
    actual = y_true.astype(bool)
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
        "csi": None if tp + fp + fn == 0 else tp / (tp + fp + fn),
    }


def best_probability_csi_threshold(y_true: Any, probability: Any) -> dict[str, Any]:
    best = None
    for threshold in [i / 100.0 for i in range(5, 96, 5)]:
        item = probability_threshold_metrics(y_true, probability, threshold)
        score = -1.0 if item["csi"] is None else float(item["csi"])
        if best is None or score > (-1.0 if best["csi"] is None else float(best["csi"])):
            best = item
    return best or probability_threshold_metrics(y_true, probability, 0.5)


def probability_metrics(frame: Any, prob_col: str, actual_col: str, actual_threshold: float, deps: dict[str, Any]) -> dict[str, Any]:
    pd = deps["pd"]
    np = deps["np"]
    values = frame[[prob_col, actual_col]].dropna()
    if values.empty:
        return {"n": 0}
    probability = pd.to_numeric(values[prob_col], errors="coerce").clip(lower=0.0, upper=1.0)
    actual_values = pd.to_numeric(values[actual_col], errors="coerce")
    mask = probability.notna() & actual_values.notna()
    probability = probability[mask]
    y_true = (actual_values[mask] >= actual_threshold).astype(int)
    if len(y_true) == 0:
        return {"n": 0}
    clipped = probability.clip(lower=1e-15, upper=1.0 - 1e-15)
    out = {
        "n": int(len(y_true)),
        "actual_threshold": actual_threshold,
        "positive_count": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "mean_probability": float(probability.mean()),
        "brier": float(((probability - y_true) ** 2).mean()),
        "log_loss": float((-(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped))).mean()),
        "threshold_0p20": probability_threshold_metrics(y_true, probability, 0.20),
        "threshold_0p50": probability_threshold_metrics(y_true, probability, 0.50),
        "best_csi_threshold": best_probability_csi_threshold(y_true, probability),
    }
    if y_true.nunique() > 1:
        out["roc_auc"] = float(deps["roc_auc_score"](y_true, probability))
        out["average_precision"] = float(deps["average_precision_score"](y_true, probability))
    return out


def probability_summary(frame: Any, deps: dict[str, Any]) -> dict[str, Any]:
    out = {}
    specs = [
        ("gust_ge_20kt_final", "prob_gust_ge_20kt", 20.0),
        ("gust_ge_20kt_heuristic", "prob_gust_ge_20kt_heuristic", 20.0),
        ("gust_ge_20kt_model", "prob_gust_ge_20kt_model", 20.0),
        ("gust_ge_25kt_final", "prob_gust_ge_25kt", 25.0),
        ("gust_ge_25kt_heuristic", "prob_gust_ge_25kt_heuristic", 25.0),
        ("gust_ge_25kt_model", "prob_gust_ge_25kt_model", 25.0),
    ]
    for key, column, threshold in specs:
        if column in frame.columns:
            out[key] = probability_metrics(frame, column, "actual_gust_kt", threshold, deps)
    return out


def alert_summary(frame: Any) -> dict[str, Any]:
    out = {}
    if "gust_alert_ge_20kt" in frame.columns:
        out["gust_alert_ge_20kt"] = binary_alert_metrics(frame, "gust_alert_ge_20kt", "actual_gust_kt", 20.0)
    if "gust_alert_ge_25kt" in frame.columns:
        out["gust_alert_ge_25kt"] = binary_alert_metrics(frame, "gust_alert_ge_25kt", "actual_gust_kt", 25.0)
    return out


def add_unit_columns(frame: Any, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("float64")
            frame[column.replace("_ms", "_kt")] = frame[column] * KT_PER_MS


def load_observations(paths: list[Path], spots: set[str], pd: Any) -> Any:
    rows = []
    source_priority = {
        "dpobs_station_infrahoraire_6m": 30,
        "beacon-live-app": 25,
        "weather_state_snapshot": 25,
        "dpobs_station_horaire": 20,
        "dpobs_synop": 10,
        "dpobs_bouees": 5,
    }
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                spot_id = row.get("spot_id")
                if not spot_id or str(spot_id) not in spots:
                    continue
                timestamp = pd.to_datetime(row.get("timestamp_utc"), utc=True, errors="coerce")
                if pd.isna(timestamp):
                    continue
                dataset = str(row.get("source_dataset") or "")
                project = str(row.get("source_project") or "")
                rows.append(
                    {
                        "spot_id": str(spot_id),
                        "obs_time": timestamp,
                        "actual_wind_mean_ms": row.get("wind_mean_ms"),
                        "actual_gust_ms": row.get("gust_ms"),
                        "actual_wind_direction_deg": row.get("wind_direction_deg"),
                        "observation_source_project": project,
                        "observation_source_dataset": dataset,
                        "observation_station_id": row.get("station_id"),
                        "_priority": max(source_priority.get(dataset, 0), source_priority.get(project, 0)),
                    }
                )
    if not rows:
        return pd.DataFrame()
    observations = pd.DataFrame(rows)
    for column in ("actual_wind_mean_ms", "actual_gust_ms", "actual_wind_direction_deg"):
        observations[column] = pd.to_numeric(observations[column], errors="coerce")
    observations = observations.dropna(subset=["actual_wind_mean_ms", "actual_gust_ms"], how="all")
    observations = (
        observations.sort_values(["spot_id", "obs_time", "_priority"])
        .drop_duplicates(["spot_id", "obs_time"], keep="last")
        .reset_index(drop=True)
    )
    return observations


def join_nearest_observations(predictions: Any, observations: Any, tolerance_minutes: float, pd: Any) -> Any:
    joined = []
    if predictions.empty or observations.empty:
        return pd.DataFrame()
    for spot_id, group in predictions.groupby("spot_id"):
        obs_group = observations[observations["spot_id"] == spot_id].sort_values("obs_time")
        if obs_group.empty:
            continue
        for _, prediction in group.iterrows():
            diffs = (obs_group["obs_time"] - prediction["target_dt"]).abs()
            obs_idx = diffs.idxmin()
            distance_minutes = float(diffs.loc[obs_idx].total_seconds() / 60.0)
            if distance_minutes > tolerance_minutes:
                continue
            item = prediction.to_dict()
            item.update(obs_group.loc[obs_idx].to_dict())
            item["obs_distance_minutes"] = distance_minutes
            joined.append(item)
    return pd.DataFrame(joined)


def peak_summary(frame: Any, pred_col: str, raw_col: str, actual_col: str) -> dict[str, Any]:
    if frame.empty:
        return {}
    out = {}
    for spot_id, group in frame.groupby("spot_id"):
        group = group.dropna(subset=[actual_col])
        if group.empty:
            continue
        actual_idx = group[actual_col].idxmax()
        pred_idx = group[pred_col].idxmax() if pred_col in group else None
        raw_idx = group[raw_col].idxmax() if raw_col in group else None
        out[str(spot_id)] = {
            "actual_peak": float(group.loc[actual_idx, actual_col]),
            "actual_peak_time_utc": str(group.loc[actual_idx, "target_time_utc"]),
            "ml_peak": None if pred_idx is None else float(group.loc[pred_idx, pred_col]),
            "ml_peak_time_utc": None if pred_idx is None else str(group.loc[pred_idx, "target_time_utc"]),
            "raw_peak": None if raw_idx is None else float(group.loc[raw_idx, raw_col]),
            "raw_peak_time_utc": None if raw_idx is None else str(group.loc[raw_idx, "target_time_utc"]),
        }
        if out[str(spot_id)]["ml_peak"] is not None:
            out[str(spot_id)]["ml_peak_error"] = out[str(spot_id)]["ml_peak"] - out[str(spot_id)]["actual_peak"]
        if out[str(spot_id)]["raw_peak"] is not None:
            out[str(spot_id)]["raw_peak_error"] = out[str(spot_id)]["raw_peak"] - out[str(spot_id)]["actual_peak"]
    return out


def grouped_metrics(frame: Any, group_column: str) -> dict[str, Any]:
    out = {}
    if frame.empty or group_column not in frame.columns:
        return out
    for key, group in frame.groupby(group_column, dropna=False):
        item = {
            "n": int(len(group)),
            "wind_ml_kt": metrics(group, "champion_wind_mean_kt", "actual_wind_mean_kt"),
            "wind_raw_kt": metrics(group, "raw_wind_mean_kt", "actual_wind_mean_kt"),
            "gust_ml_kt": metrics(group, "champion_gust_kt", "actual_gust_kt"),
            "gust_raw_kt": metrics(group, "raw_gust_kt", "actual_gust_kt"),
        }
        if "gust_high_kt" in group.columns:
            item["gust_high_kt"] = metrics(group, "gust_high_kt", "actual_gust_kt")
        if "strong_gated_wind_mean_kt" in group.columns:
            item["wind_strong_gated_kt"] = metrics(group, "strong_gated_wind_mean_kt", "actual_wind_mean_kt")
        if "strong_gated_gust_kt" in group.columns:
            item["gust_strong_gated_kt"] = metrics(group, "strong_gated_gust_kt", "actual_gust_kt")
        if "shadow_router_v1_wind_mean_kt" in group.columns:
            item["wind_shadow_router_v1_kt"] = metrics(group, "shadow_router_v1_wind_mean_kt", "actual_wind_mean_kt")
        if "shadow_stacker_v1_wind_mean_kt" in group.columns:
            item["wind_shadow_stacker_v1_kt"] = metrics(group, "shadow_stacker_v1_wind_mean_kt", "actual_wind_mean_kt")
        if "shadow_guarded_stacker_v1_wind_mean_kt" in group.columns:
            item["wind_shadow_guarded_stacker_v1_kt"] = metrics(group, "shadow_guarded_stacker_v1_wind_mean_kt", "actual_wind_mean_kt")
        if "threshold_guard_v1_wind_mean_kt" in group.columns:
            item["wind_threshold_guard_v1_kt"] = metrics(group, "threshold_guard_v1_wind_mean_kt", "actual_wind_mean_kt")
        if "wind_high_event_guard_v1_wind_mean_kt" in group.columns:
            item["wind_high_event_guard_v1_kt"] = metrics(group, "wind_high_event_guard_v1_wind_mean_kt", "actual_wind_mean_kt")
        if "shadow_router_v1_gust_kt" in group.columns:
            item["gust_shadow_router_v1_kt"] = metrics(group, "shadow_router_v1_gust_kt", "actual_gust_kt")
        if "shadow_stacker_v1_gust_kt" in group.columns:
            item["gust_shadow_stacker_v1_kt"] = metrics(group, "shadow_stacker_v1_gust_kt", "actual_gust_kt")
        if "shadow_guarded_stacker_v1_gust_kt" in group.columns:
            item["gust_shadow_guarded_stacker_v1_kt"] = metrics(group, "shadow_guarded_stacker_v1_gust_kt", "actual_gust_kt")
        if "threshold_guard_v1_gust_kt" in group.columns:
            item["gust_threshold_guard_v1_kt"] = metrics(group, "threshold_guard_v1_gust_kt", "actual_gust_kt")
        if "local_fallback_guard_v1_gust_kt" in group.columns:
            item["gust_local_fallback_guard_v1_kt"] = metrics(group, "local_fallback_guard_v1_gust_kt", "actual_gust_kt")
        out[str(key)] = item
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]

    predictions = pd.read_parquet(args.predictions_parquet)
    spots = set(parse_csv(args.spots) or DEFAULT_SPOTS)
    predictions = predictions[predictions["spot_id"].astype(str).isin(spots)].copy()
    if predictions.empty:
        raise SystemExit("No prediction rows left after spot filtering.")
    predictions["spot_id"] = predictions["spot_id"].astype(str)
    predictions["target_dt"] = pd.to_datetime(predictions["target_time_utc"], utc=True, errors="coerce")
    predictions["lead_time_minutes"] = pd.to_numeric(predictions["lead_time_minutes"], errors="coerce")
    predictions = predictions.dropna(subset=["target_dt", "lead_time_minutes"])
    if args.target_start_utc:
        predictions = predictions[predictions["target_dt"] >= pd.Timestamp(args.target_start_utc)]
    if args.target_end_utc:
        predictions = predictions[predictions["target_dt"] <= pd.Timestamp(args.target_end_utc)]

    observation_paths = expand_paths(args.observations_jsonl)
    observations = load_observations(observation_paths, spots, pd)
    scored = join_nearest_observations(predictions, observations, args.tolerance_minutes, pd)
    if scored.empty:
        raise SystemExit("No prediction rows matched observations.")

    add_unit_columns(
        scored,
        [
            "champion_wind_mean_ms",
            "raw_wind_mean_ms",
            "guarded_foundation_wind_mean_ms",
            "champion_gust_ms",
            "raw_gust_ms",
            "guarded_foundation_gust_ms",
            "gust_high_ms",
            "strong_soft_wind_mean_ms",
            "strong_soft_gust_ms",
            "strong_aggressive_wind_mean_ms",
            "strong_aggressive_gust_ms",
            "strong_gated_wind_mean_ms",
            "strong_gated_gust_ms",
            "shadow_router_v1_wind_mean_ms",
            "shadow_stacker_v1_wind_mean_ms",
            "shadow_guarded_stacker_v1_wind_mean_ms",
            "threshold_guard_v1_wind_mean_ms",
            "wind_high_event_guard_v1_wind_mean_ms",
            "shadow_router_v1_gust_ms",
            "shadow_stacker_v1_gust_ms",
            "shadow_guarded_stacker_v1_gust_ms",
            "threshold_guard_v1_gust_ms",
            "local_fallback_guard_v1_gust_ms",
            "actual_wind_mean_ms",
            "actual_gust_ms",
        ],
    )
    scored["lead_bucket"] = scored["lead_time_minutes"].map(
        lambda value: "0-1h"
        if value <= 60
        else "1-3h"
        if value <= 180
        else "3-6h"
        if value <= 360
        else "6h+"
    )
    scored["target_hour_utc"] = scored["target_dt"].dt.hour
    scored["actual_gust_regime_kt"] = scored["actual_gust_kt"].map(
        lambda value: ">=25kt" if value >= 25 else "20-25kt" if value >= 20 else "15-20kt" if value >= 15 else "<15kt"
    )
    scored["actual_wind_regime_kt"] = scored["actual_wind_mean_kt"].map(
        lambda value: ">=25kt"
        if value >= 25
        else "20-25kt"
        if value >= 20
        else "15-20kt"
        if value >= 15
        else "12-15kt"
        if value >= 12
        else "<12kt"
    )
    scored["raw_gust_regime_kt"] = scored["raw_gust_kt"].map(
        lambda value: ">=25kt" if value >= 25 else "20-25kt" if value >= 20 else "15-20kt" if value >= 15 else "<15kt"
    )
    scored["raw_wind_regime_kt"] = scored["raw_wind_mean_kt"].map(
        lambda value: ">=25kt"
        if value >= 25
        else "20-25kt"
        if value >= 20
        else "15-20kt"
        if value >= 15
        else "12-15kt"
        if value >= 12
        else "<12kt"
    )

    overall = {
        "wind_ml_kt": metrics(scored, "champion_wind_mean_kt", "actual_wind_mean_kt"),
        "wind_raw_kt": metrics(scored, "raw_wind_mean_kt", "actual_wind_mean_kt"),
        "gust_ml_kt": metrics(scored, "champion_gust_kt", "actual_gust_kt"),
        "gust_raw_kt": metrics(scored, "raw_gust_kt", "actual_gust_kt"),
    }
    if "guarded_foundation_wind_mean_kt" in scored.columns:
        overall["wind_shadow_kt"] = metrics(scored, "guarded_foundation_wind_mean_kt", "actual_wind_mean_kt")
    if "guarded_foundation_gust_kt" in scored.columns:
        overall["gust_shadow_kt"] = metrics(scored, "guarded_foundation_gust_kt", "actual_gust_kt")
    if "gust_high_kt" in scored.columns:
        overall["gust_high_kt"] = metrics(scored, "gust_high_kt", "actual_gust_kt")
    if "strong_gated_wind_mean_kt" in scored.columns:
        overall["wind_strong_gated_kt"] = metrics(scored, "strong_gated_wind_mean_kt", "actual_wind_mean_kt")
    if "strong_gated_gust_kt" in scored.columns:
        overall["gust_strong_gated_kt"] = metrics(scored, "strong_gated_gust_kt", "actual_gust_kt")
    if "shadow_router_v1_wind_mean_kt" in scored.columns:
        overall["wind_shadow_router_v1_kt"] = metrics(scored, "shadow_router_v1_wind_mean_kt", "actual_wind_mean_kt")
    if "shadow_stacker_v1_wind_mean_kt" in scored.columns:
        overall["wind_shadow_stacker_v1_kt"] = metrics(scored, "shadow_stacker_v1_wind_mean_kt", "actual_wind_mean_kt")
    if "shadow_guarded_stacker_v1_wind_mean_kt" in scored.columns:
        overall["wind_shadow_guarded_stacker_v1_kt"] = metrics(scored, "shadow_guarded_stacker_v1_wind_mean_kt", "actual_wind_mean_kt")
    if "threshold_guard_v1_wind_mean_kt" in scored.columns:
        overall["wind_threshold_guard_v1_kt"] = metrics(scored, "threshold_guard_v1_wind_mean_kt", "actual_wind_mean_kt")
    if "wind_high_event_guard_v1_wind_mean_kt" in scored.columns:
        overall["wind_high_event_guard_v1_kt"] = metrics(scored, "wind_high_event_guard_v1_wind_mean_kt", "actual_wind_mean_kt")
    if "shadow_router_v1_gust_kt" in scored.columns:
        overall["gust_shadow_router_v1_kt"] = metrics(scored, "shadow_router_v1_gust_kt", "actual_gust_kt")
    if "shadow_stacker_v1_gust_kt" in scored.columns:
        overall["gust_shadow_stacker_v1_kt"] = metrics(scored, "shadow_stacker_v1_gust_kt", "actual_gust_kt")
    if "shadow_guarded_stacker_v1_gust_kt" in scored.columns:
        overall["gust_shadow_guarded_stacker_v1_kt"] = metrics(scored, "shadow_guarded_stacker_v1_gust_kt", "actual_gust_kt")
    if "threshold_guard_v1_gust_kt" in scored.columns:
        overall["gust_threshold_guard_v1_kt"] = metrics(scored, "threshold_guard_v1_gust_kt", "actual_gust_kt")
    if "local_fallback_guard_v1_gust_kt" in scored.columns:
        overall["gust_local_fallback_guard_v1_kt"] = metrics(scored, "local_fallback_guard_v1_gust_kt", "actual_gust_kt")

    summary = {
        "format": "corsewind.live_hindcast_score.v1",
        "generated_at_utc": utc_now(),
        "predictions_parquet": str(args.predictions_parquet),
        "observations_jsonl": [str(path) for path in observation_paths],
        "joined_rows": int(len(scored)),
        "spot_count": int(scored["spot_id"].nunique()),
        "spots": sorted(scored["spot_id"].unique().tolist()),
        "target_start_utc": str(scored["target_time_utc"].min()),
        "target_end_utc": str(scored["target_time_utc"].max()),
        "tolerance_minutes": args.tolerance_minutes,
        "overall": overall,
        "probability_heads": probability_summary(scored, deps),
        "alert_flags": alert_summary(scored),
        "thresholds": build_threshold_summary(scored),
        "by_spot": grouped_metrics(scored, "spot_id"),
        "by_lead_bucket": grouped_metrics(scored, "lead_bucket"),
        "by_target_hour_utc": grouped_metrics(scored, "target_hour_utc"),
        "by_actual_wind_regime_kt": grouped_metrics(scored, "actual_wind_regime_kt"),
        "by_actual_gust_regime_kt": grouped_metrics(scored, "actual_gust_regime_kt"),
        "by_raw_wind_regime_kt": grouped_metrics(scored, "raw_wind_regime_kt"),
        "by_raw_gust_regime_kt": grouped_metrics(scored, "raw_gust_regime_kt"),
        "peak_gust_by_spot_kt": peak_summary(scored, "champion_gust_kt", "raw_gust_kt", "actual_gust_kt"),
        "peak_gust_high_by_spot_kt": peak_summary(scored, "gust_high_kt", "raw_gust_kt", "actual_gust_kt") if "gust_high_kt" in scored.columns else {},
        "peak_gust_strong_gated_by_spot_kt": peak_summary(scored, "strong_gated_gust_kt", "raw_gust_kt", "actual_gust_kt") if "strong_gated_gust_kt" in scored.columns else {},
        "peak_gust_shadow_router_v1_by_spot_kt": peak_summary(scored, "shadow_router_v1_gust_kt", "raw_gust_kt", "actual_gust_kt") if "shadow_router_v1_gust_kt" in scored.columns else {},
        "peak_gust_shadow_stacker_v1_by_spot_kt": peak_summary(scored, "shadow_stacker_v1_gust_kt", "raw_gust_kt", "actual_gust_kt") if "shadow_stacker_v1_gust_kt" in scored.columns else {},
        "peak_gust_shadow_guarded_stacker_v1_by_spot_kt": peak_summary(scored, "shadow_guarded_stacker_v1_gust_kt", "raw_gust_kt", "actual_gust_kt") if "shadow_guarded_stacker_v1_gust_kt" in scored.columns else {},
        "peak_gust_threshold_guard_v1_by_spot_kt": peak_summary(scored, "threshold_guard_v1_gust_kt", "raw_gust_kt", "actual_gust_kt") if "threshold_guard_v1_gust_kt" in scored.columns else {},
        "peak_gust_local_fallback_guard_v1_by_spot_kt": peak_summary(scored, "local_fallback_guard_v1_gust_kt", "raw_gust_kt", "actual_gust_kt") if "local_fallback_guard_v1_gust_kt" in scored.columns else {},
        "peak_wind_by_spot_kt": peak_summary(scored, "champion_wind_mean_kt", "raw_wind_mean_kt", "actual_wind_mean_kt"),
        "peak_wind_strong_gated_by_spot_kt": peak_summary(scored, "strong_gated_wind_mean_kt", "raw_wind_mean_kt", "actual_wind_mean_kt") if "strong_gated_wind_mean_kt" in scored.columns else {},
        "peak_wind_shadow_router_v1_by_spot_kt": peak_summary(scored, "shadow_router_v1_wind_mean_kt", "raw_wind_mean_kt", "actual_wind_mean_kt") if "shadow_router_v1_wind_mean_kt" in scored.columns else {},
        "peak_wind_shadow_stacker_v1_by_spot_kt": peak_summary(scored, "shadow_stacker_v1_wind_mean_kt", "raw_wind_mean_kt", "actual_wind_mean_kt") if "shadow_stacker_v1_wind_mean_kt" in scored.columns else {},
        "peak_wind_shadow_guarded_stacker_v1_by_spot_kt": peak_summary(scored, "shadow_guarded_stacker_v1_wind_mean_kt", "raw_wind_mean_kt", "actual_wind_mean_kt") if "shadow_guarded_stacker_v1_wind_mean_kt" in scored.columns else {},
        "peak_wind_threshold_guard_v1_by_spot_kt": peak_summary(scored, "threshold_guard_v1_wind_mean_kt", "raw_wind_mean_kt", "actual_wind_mean_kt") if "threshold_guard_v1_wind_mean_kt" in scored.columns else {},
        "peak_wind_high_event_guard_v1_by_spot_kt": peak_summary(scored, "wind_high_event_guard_v1_wind_mean_kt", "raw_wind_mean_kt", "actual_wind_mean_kt") if "wind_high_event_guard_v1_wind_mean_kt" in scored.columns else {},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_scored_parquet:
        args.output_scored_parquet.parent.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(args.output_scored_parquet, index=False, compression=args.compression)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-parquet", type=Path, required=True)
    parser.add_argument("--observations-jsonl", action="append", default=[], required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-scored-parquet", type=Path)
    parser.add_argument("--spots", help="Comma-separated spot ids. Defaults to Meteo-France scored spots.")
    parser.add_argument("--target-start-utc")
    parser.add_argument("--target-end-utc")
    parser.add_argument("--tolerance-minutes", type=float, default=8.0)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
