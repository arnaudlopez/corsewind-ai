#!/usr/bin/env python3
"""Run live wind-mean and gust inference from flat residual-training rows."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ML_ROOT = Path("/srv/data/corsewind/ml_dataset")
DEFAULT_WIND_BASE_RUN = DEFAULT_ML_ROOT / "benchmarks/tabular_lgbm_225k_prev_lowmem_2024_2025_to_2026_v1"
DEFAULT_GUST_BASE_RUN = DEFAULT_ML_ROOT / "benchmarks/tabular_lgbm_225k_prev_lowmem_gust_from_wind_champion_recipe_2024_2025_to_2026_v1"
DEFAULT_WIND_CALIBRATOR_RUN = DEFAULT_ML_ROOT / "benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1"
DEFAULT_GUST_CALIBRATOR_RUN = DEFAULT_ML_ROOT / "benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_gust_from_wind_champion_recipe_v1"
DEFAULT_GUST_PROBABILITY_RUN = DEFAULT_ML_ROOT / "benchmarks/gust_threshold_probability_extratrees_2024_2025_to_2026_v1"
DEFAULT_GUST_ALERT_THRESHOLDS = DEFAULT_ML_ROOT / "benchmarks/gust_probability_alert_thresholds_hindcast_v1/gust_probability_alert_thresholds.json"
DEFAULT_STRONG_SOFT_RUN = DEFAULT_ML_ROOT / "benchmarks/strong_wind_expert_lgbm_weighted_soft_12_15_20_25_v1"
DEFAULT_STRONG_AGGRESSIVE_RUN = DEFAULT_ML_ROOT / "benchmarks/strong_wind_expert_lgbm_weighted_12_15_20_25_v1"
KNOTS_PER_MS = 1.9438444924406
TARGETS = {
    "wind_mean": {
        "label": "wind mean",
        "baseline": "baselines__baseline_wind_mean_ms",
        "model_filename": "labels__residual_wind_mean_ms.joblib",
        "predicted_residual": "predicted_residual_wind_mean_ms",
        "raw": "raw_wind_mean_ms",
        "corrected": "corrected_wind_mean_ms",
        "calibrated": "calibrated_wind_mean_ms",
        "champion": "champion_wind_mean_ms",
        "guarded": "guarded_foundation_wind_mean_ms",
        "guarded_delta": "guarded_foundation_wind_mean_delta_ms",
        "guarded_raw_delta": "guarded_foundation_wind_mean_raw_delta_ms",
        "guarded_used": "guarded_foundation_wind_mean_used_foundation",
        "guarded_capped": "guarded_foundation_wind_mean_delta_was_capped",
        "foundation_expert": "chronos2_univar_wind_mean_ms_mean",
        "foundation_alpha": 0.10,
        "foundation_cap_delta_ms": 0.50,
        "raw_kt": "raw_wind_mean_kt",
        "corrected_kt": "corrected_wind_mean_kt",
        "calibrated_kt": "calibrated_wind_mean_kt",
        "champion_kt": "champion_wind_mean_kt",
        "guarded_kt": "guarded_foundation_wind_mean_kt",
        "second_stage_raw": "predicted_second_stage_residual_wind_mean_ms_raw",
        "second_stage": "predicted_second_stage_residual_wind_mean_ms",
    },
    "gust": {
        "label": "gust",
        "baseline": "baselines__baseline_gust_ms",
        "model_filename": "labels__residual_gust_ms.joblib",
        "predicted_residual": "predicted_residual_gust_ms",
        "raw": "raw_gust_ms",
        "corrected": "corrected_gust_ms",
        "calibrated": "calibrated_gust_ms",
        "champion": "champion_gust_ms",
        "guarded": "guarded_foundation_gust_ms",
        "guarded_delta": "guarded_foundation_gust_delta_ms",
        "guarded_raw_delta": "guarded_foundation_gust_raw_delta_ms",
        "guarded_used": "guarded_foundation_gust_used_foundation",
        "guarded_capped": "guarded_foundation_gust_delta_was_capped",
        "foundation_expert": "timesfm_gust_ms_mean",
        "foundation_alpha": 0.10,
        "foundation_cap_delta_ms": 0.25,
        "raw_kt": "raw_gust_kt",
        "corrected_kt": "corrected_gust_kt",
        "calibrated_kt": "calibrated_gust_kt",
        "champion_kt": "champion_gust_kt",
        "guarded_kt": "guarded_foundation_gust_kt",
        "second_stage_raw": "predicted_second_stage_residual_gust_ms_raw",
        "second_stage": "predicted_second_stage_residual_gust_ms",
    },
}


def import_dependencies():
    try:
        import joblib
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Missing ML dependencies. Run inside the CorseWind ML venv.") from exc
    return {"joblib": joblib, "np": np, "pd": pd, "pq": pq}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_columns(path: Path) -> list[str]:
    payload = read_json(path)
    return list(payload.get("numeric") or []) + list(payload.get("categorical") or [])


def add_time_features(frame: Any, pd: Any, np: Any) -> Any:
    issue_time = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    dayofyear = issue_time.dt.dayofyear.fillna(1).astype(float)
    frame["issue_hour_utc"] = issue_time.dt.hour.astype("float64")
    frame["issue_month"] = issue_time.dt.month.astype("float64")
    angle = 2.0 * math.pi * dayofyear / 366.0
    frame["issue_dayofyear_sin"] = np.sin(angle)
    frame["issue_dayofyear_cos"] = np.cos(angle)
    frame["issue_month_number"] = issue_time.dt.month.astype("float64")
    return frame


def required_pipeline_columns(model: Any) -> list[str]:
    preprocess = getattr(model, "named_steps", {}).get("preprocess") if hasattr(model, "named_steps") else None
    transformers = getattr(preprocess, "transformers_", None) if preprocess is not None else None
    columns: list[str] = []
    for _name, _transformer, selected in transformers or []:
        if isinstance(selected, (list, tuple)):
            columns.extend(str(column) for column in selected)
    return sorted(set(columns))


def infer_scale(calibration_results: Path | None, default_scale: float) -> float:
    if calibration_results and calibration_results.exists():
        payload = read_json(calibration_results)
        selection = payload.get("scale_selection")
        if isinstance(selection, dict) and selection.get("selected_scale") is not None:
            return float(selection["selected_scale"])
    match = re.search(r"scale(\d{3})", str(calibration_results or ""))
    if match:
        return float(match.group(1)) / 100.0
    return float(default_scale)


def as_knots(series: Any) -> Any:
    return series.astype(float) * KNOTS_PER_MS


def as_ms_from_knots(series: Any) -> Any:
    return series.astype(float) / KNOTS_PER_MS


def maybe_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def add_gust_peak_guard(
    frame: Any,
    *,
    enabled: bool,
    raw_trigger_kt: float,
    gap_trigger_kt: float,
    alpha: float,
    cap_delta_kt: float,
    threshold_20_width_kt: float,
    threshold_25_width_kt: float,
    deps: dict[str, Any],
) -> dict[str, Any]:
    pd = deps["pd"]
    np = deps["np"]
    if not enabled or "champion_gust_ms" not in frame.columns or "raw_gust_ms" not in frame.columns:
        frame["gust_peak_guard_enabled"] = False
        return {
            "enabled": False,
            "fallback_reason": "disabled_or_missing_columns",
        }

    champion_kt = pd.to_numeric(frame["champion_gust_kt"], errors="coerce")
    raw_kt = pd.to_numeric(frame["raw_gust_kt"], errors="coerce")
    positive_gap_kt = (raw_kt - champion_kt).clip(lower=0.0)
    raw_component = ((raw_kt - raw_trigger_kt) / max(1.0, raw_trigger_kt)).clip(lower=0.0, upper=1.0)
    gap_component = (positive_gap_kt / max(1.0, cap_delta_kt)).clip(lower=0.0, upper=1.0)
    risk_score = (raw_component * gap_component).clip(lower=0.0, upper=1.0)
    active = champion_kt.notna() & raw_kt.notna() & (raw_kt >= raw_trigger_kt) & (positive_gap_kt >= gap_trigger_kt)
    raw_delta_kt = positive_gap_kt * float(alpha)
    delta_kt = raw_delta_kt.clip(upper=float(cap_delta_kt)).where(active, 0.0)
    high_kt = champion_kt + delta_kt
    frame["gust_peak_guard_enabled"] = True
    frame["gust_peak_guard_raw_gap_kt"] = raw_kt - champion_kt
    frame["gust_peak_guard_positive_gap_kt"] = positive_gap_kt
    frame["gust_peak_guard_risk_score"] = risk_score.where(active, 0.0)
    frame["gust_peak_guard_delta_kt"] = delta_kt
    frame["gust_peak_guard_active"] = active
    frame["gust_peak_guard_delta_was_capped"] = active & (raw_delta_kt > float(cap_delta_kt))
    frame["gust_high_kt"] = high_kt
    frame["gust_high_ms"] = as_ms_from_knots(high_kt)

    width20 = max(0.5, float(threshold_20_width_kt))
    width25 = max(0.5, float(threshold_25_width_kt))
    frame["prob_gust_ge_20kt"] = 1.0 / (1.0 + np.exp(-((high_kt - 20.0) / width20)))
    frame["prob_gust_ge_25kt"] = 1.0 / (1.0 + np.exp(-((high_kt - 25.0) / width25)))
    frame["peak_risk_level"] = pd.cut(
        frame["prob_gust_ge_20kt"],
        bins=[-0.01, 0.20, 0.45, 0.70, 1.01],
        labels=["low", "moderate", "high", "very_high"],
    ).astype("string")
    return {
        "enabled": True,
        "raw_trigger_kt": raw_trigger_kt,
        "gap_trigger_kt": gap_trigger_kt,
        "alpha": alpha,
        "cap_delta_kt": cap_delta_kt,
        "active_count": int(active.sum()),
        "capped_count": int(frame["gust_peak_guard_delta_was_capped"].sum()),
        "fallback_reason": None,
        "output_columns": [
            "gust_high_ms",
            "gust_high_kt",
            "prob_gust_ge_20kt",
            "prob_gust_ge_25kt",
            "peak_risk_level",
        ],
    }


def add_gust_probability_heads(
    frame: Any,
    *,
    model_root: Path | None,
    enabled: bool,
    deps: dict[str, Any],
) -> dict[str, Any]:
    if "prob_gust_ge_20kt" in frame.columns:
        frame["prob_gust_ge_20kt_heuristic"] = frame["prob_gust_ge_20kt"]
    if "prob_gust_ge_25kt" in frame.columns:
        frame["prob_gust_ge_25kt_heuristic"] = frame["prob_gust_ge_25kt"]
    status: dict[str, Any] = {
        "enabled": bool(enabled),
        "model_root": str(model_root) if model_root else None,
        "loaded": {},
        "fallback_reason": None,
    }
    if not enabled:
        status["fallback_reason"] = "disabled"
        return status
    if model_root is None or not model_root.exists():
        status["fallback_reason"] = "missing_model_root"
        return status

    joblib = deps["joblib"]
    pd = deps["pd"]
    threshold_specs = {
        "20kt": ("labels__target_gust_gt_20kt.joblib", "prob_gust_ge_20kt"),
        "25kt": ("labels__target_gust_gt_25kt.joblib", "prob_gust_ge_25kt"),
    }
    for threshold, (filename, column) in threshold_specs.items():
        path = model_root / filename
        if not path.exists():
            status["loaded"][threshold] = {"loaded": False, "reason": "missing_model_file", "path": str(path)}
            continue
        model = joblib.load(path)
        model_columns = required_pipeline_columns(model)
        x = frame.reindex(columns=model_columns) if model_columns else frame
        probability = model.predict_proba(x)[:, 1]
        frame[column] = pd.Series(probability, index=frame.index).clip(lower=0.0, upper=1.0)
        frame[f"{column}_model"] = frame[column]
        status["loaded"][threshold] = {
            "loaded": True,
            "path": str(path),
            "feature_column_count": len(model_columns),
            "non_null_count": int(frame[column].notna().sum()),
        }
    if "prob_gust_ge_20kt" in frame.columns:
        frame["peak_risk_level"] = pd.cut(
            frame["prob_gust_ge_20kt"],
            bins=[-0.01, 0.20, 0.45, 0.70, 1.01],
            labels=["low", "moderate", "high", "very_high"],
        ).astype("string")
    if not any(item.get("loaded") for item in status["loaded"].values()):
        status["fallback_reason"] = "no_threshold_model_loaded"
    return status


def add_gust_probability_alerts(
    frame: Any,
    *,
    thresholds_path: Path | None,
    enabled: bool,
    deps: dict[str, Any],
) -> dict[str, Any]:
    pd = deps["pd"]
    status: dict[str, Any] = {
        "enabled": bool(enabled),
        "thresholds_path": str(thresholds_path) if thresholds_path else None,
        "alerts": {},
        "fallback_reason": None,
    }
    if not enabled:
        status["fallback_reason"] = "disabled"
        return status
    if thresholds_path is None or not thresholds_path.exists():
        status["fallback_reason"] = "missing_thresholds_file"
        return status
    payload = read_json(thresholds_path)
    for alert_name, alert in (payload.get("alerts") or {}).items():
        probability_column = alert.get("probability_column")
        threshold = alert.get("threshold")
        output_alert_column = alert.get("output_alert_column") or f"{alert_name}_alert"
        output_probability_column = alert.get("output_probability_column") or f"{alert_name}_probability"
        if probability_column not in frame.columns or threshold is None:
            status["alerts"][alert_name] = {
                "applied": False,
                "reason": "missing_probability_column_or_threshold",
                "probability_column": probability_column,
                "threshold": threshold,
            }
            continue
        probability = pd.to_numeric(frame[probability_column], errors="coerce").clip(lower=0.0, upper=1.0)
        frame[output_probability_column] = probability
        frame[f"{output_alert_column}_threshold"] = float(threshold)
        frame[output_alert_column] = probability >= float(threshold)
        status["alerts"][alert_name] = {
            "applied": True,
            "probability_column": probability_column,
            "threshold": float(threshold),
            "output_alert_column": output_alert_column,
            "output_probability_column": output_probability_column,
            "active_count": int(frame[output_alert_column].sum()),
        }
    if "gust_alert_ge_25kt" in frame.columns:
        frame["gust_operational_risk_level"] = frame["gust_alert_ge_25kt"].map({True: "high", False: "none"}).astype("string")
    else:
        frame["gust_operational_risk_level"] = "none"
    if "gust_alert_ge_20kt" in frame.columns:
        frame.loc[
            frame["gust_alert_ge_20kt"].fillna(False) & (frame["gust_operational_risk_level"] == "none"),
            "gust_operational_risk_level",
        ] = "moderate"
    if not any(item.get("applied") for item in status["alerts"].values()):
        status["fallback_reason"] = "no_alert_applied"
    return status


def risk_weight(series: Any, *, start: float, full: float, maximum: float) -> Any:
    if full <= start:
        return series.astype(float) * 0.0
    return ((series.astype(float) - float(start)) / (float(full) - float(start))).clip(lower=0.0, upper=1.0) * float(maximum)


def predict_strong_expert(
    frame: Any,
    *,
    root: Path,
    label: str,
    selected_targets: list[str],
    deps: dict[str, Any],
) -> dict[str, Any]:
    joblib = deps["joblib"]
    pd = deps["pd"]
    status: dict[str, Any] = {"root": str(root), "label": label, "targets": {}, "loaded": False}
    if not root.exists():
        status["fallback_reason"] = "missing_root"
        return status
    feature_json = root / "feature_columns.json"
    if not feature_json.exists():
        status["fallback_reason"] = "missing_feature_columns"
        return status
    columns = feature_columns(feature_json)
    for target in selected_targets:
        config = TARGETS[target]
        model_path = root / config["model_filename"]
        output_prefix = "wind_mean" if target == "wind_mean" else "gust"
        output_ms = f"strong_{label}_{output_prefix}_ms"
        residual_col = f"strong_{label}_{output_prefix}_residual_ms"
        if not model_path.exists() or config["baseline"] not in frame.columns:
            status["targets"][target] = {
                "loaded": False,
                "model_path": str(model_path),
                "reason": "missing_model_or_baseline",
            }
            continue
        model = joblib.load(model_path)
        model_columns = required_pipeline_columns(model) or columns
        residual = model.predict(frame.reindex(columns=model_columns))
        frame[residual_col] = pd.Series(residual, index=frame.index).astype(float)
        frame[output_ms] = pd.to_numeric(frame[config["baseline"]], errors="coerce") + frame[residual_col]
        frame[f"{output_ms[:-3]}_kt"] = as_knots(frame[output_ms])
        status["targets"][target] = {
            "loaded": True,
            "model_path": str(model_path),
            "feature_column_count": len(model_columns),
            "non_null_count": int(frame[output_ms].notna().sum()),
        }
        status["loaded"] = True
    if not status["loaded"] and "fallback_reason" not in status:
        status["fallback_reason"] = "no_target_loaded"
    return status


def add_strong_wind_gated_blend(
    frame: Any,
    *,
    enabled: bool,
    selected_targets: list[str],
    soft_root: Path,
    aggressive_root: Path,
    p20_start: float,
    p20_full: float,
    p25_start: float,
    p25_full: float,
    soft_max_weight: float,
    aggressive_max_weight: float,
    cap_delta_ms: float | None,
    deps: dict[str, Any],
) -> dict[str, Any]:
    pd = deps["pd"]
    if not enabled:
        return {"enabled": False, "fallback_reason": "disabled"}
    status: dict[str, Any] = {
        "enabled": True,
        "soft": predict_strong_expert(frame, root=soft_root, label="soft", selected_targets=selected_targets, deps=deps),
        "aggressive": predict_strong_expert(
            frame,
            root=aggressive_root,
            label="aggressive",
            selected_targets=selected_targets,
            deps=deps,
        ),
        "p20_start": p20_start,
        "p20_full": p20_full,
        "p25_start": p25_start,
        "p25_full": p25_full,
        "soft_max_weight": soft_max_weight,
        "aggressive_max_weight": aggressive_max_weight,
        "cap_delta_ms": cap_delta_ms,
        "targets": {},
    }
    if not status["soft"].get("loaded") and not status["aggressive"].get("loaded"):
        status["fallback_reason"] = "no_expert_loaded"
        return status

    p20_source = (
        frame["gust_alert_ge_20kt_probability"]
        if "gust_alert_ge_20kt_probability" in frame.columns
        else frame["prob_gust_ge_20kt"]
        if "prob_gust_ge_20kt" in frame.columns
        else pd.Series(0.0, index=frame.index)
    )
    p25_source = (
        frame["gust_alert_ge_25kt_probability"]
        if "gust_alert_ge_25kt_probability" in frame.columns
        else frame["prob_gust_ge_25kt"]
        if "prob_gust_ge_25kt" in frame.columns
        else pd.Series(0.0, index=frame.index)
    )
    p20 = pd.to_numeric(p20_source, errors="coerce").fillna(0.0)
    p25 = pd.to_numeric(p25_source, errors="coerce").fillna(0.0)
    soft_weight = risk_weight(p20, start=p20_start, full=p20_full, maximum=soft_max_weight)
    aggressive_weight = risk_weight(p25, start=p25_start, full=p25_full, maximum=aggressive_max_weight)
    total_weight = (soft_weight + aggressive_weight).clip(upper=0.95)
    frame["strong_wind_soft_weight"] = soft_weight
    frame["strong_wind_aggressive_weight"] = aggressive_weight
    frame["strong_wind_total_weight"] = total_weight

    for target in selected_targets:
        config = TARGETS[target]
        output_prefix = "wind_mean" if target == "wind_mean" else "gust"
        champion = pd.to_numeric(frame[config["champion"]], errors="coerce")
        soft_col = f"strong_soft_{output_prefix}_ms"
        aggressive_col = f"strong_aggressive_{output_prefix}_ms"
        if soft_col not in frame.columns and aggressive_col not in frame.columns:
            status["targets"][target] = {"applied": False, "reason": "missing_expert_predictions"}
            continue
        soft_prediction = pd.to_numeric(frame[soft_col], errors="coerce") if soft_col in frame.columns else champion
        aggressive_prediction = (
            pd.to_numeric(frame[aggressive_col], errors="coerce") if aggressive_col in frame.columns else champion
        )
        delta = (
            soft_weight * (soft_prediction - champion).fillna(0.0)
            + aggressive_weight * (aggressive_prediction - champion).fillna(0.0)
        )
        if cap_delta_ms is not None:
            delta = delta.clip(lower=-float(cap_delta_ms), upper=float(cap_delta_ms))
        output_ms = f"strong_gated_{output_prefix}_ms"
        frame[f"strong_gated_{output_prefix}_delta_ms"] = delta
        frame[output_ms] = champion + delta
        frame[f"strong_gated_{output_prefix}_kt"] = as_knots(frame[output_ms])
        status["targets"][target] = {
            "applied": True,
            "non_null_count": int(frame[output_ms].notna().sum()),
            "used_count": int((delta.abs() > 1e-9).sum()),
            "mean_delta_ms": float(delta.mean()) if len(delta) else None,
            "max_abs_delta_ms": float(delta.abs().max()) if len(delta) else None,
        }
    return status


def add_shadow_foundation_blend(
    frame: Any,
    target: str,
    *,
    expert_column: str | None,
    alpha: float | None,
    cap_delta_ms: float | None,
    deps: dict[str, Any],
) -> dict[str, Any]:
    pd = deps["pd"]
    config = TARGETS[target]
    expert = expert_column or config["foundation_expert"]
    blend_alpha = float(config["foundation_alpha"] if alpha is None else alpha)
    cap = config["foundation_cap_delta_ms"] if cap_delta_ms is None else cap_delta_ms

    champion = pd.to_numeric(frame[config["calibrated"]], errors="coerce")
    frame[config["champion"]] = champion
    frame[config["champion_kt"]] = as_knots(frame[config["champion"]])

    status: dict[str, Any] = {
        "enabled": True,
        "expert_column": expert,
        "expert_available": expert in frame.columns,
        "alpha": blend_alpha,
        "cap_delta_ms": cap,
        "used_foundation_count": 0,
        "capped_delta_count": 0,
        "fallback_reason": None,
    }
    if expert not in frame.columns:
        frame[config["guarded_raw_delta"]] = 0.0
        frame[config["guarded_delta"]] = 0.0
        frame[config["guarded"]] = frame[config["champion"]]
        frame[config["guarded_used"]] = False
        frame[config["guarded_capped"]] = False
        frame[config["guarded_kt"]] = frame[config["champion_kt"]]
        status["fallback_reason"] = "missing_foundation_expert_column"
        return status

    expert_values = pd.to_numeric(frame[expert], errors="coerce")
    raw_delta = blend_alpha * (expert_values - champion)
    capped_delta = raw_delta
    if cap is not None:
        capped_delta = raw_delta.clip(lower=-float(cap), upper=float(cap))
    used = champion.notna() & expert_values.notna() & raw_delta.notna()
    output = champion + capped_delta
    frame[config["guarded_raw_delta"]] = raw_delta
    frame[config["guarded_delta"]] = output - champion
    frame[config["guarded"]] = output.where(used, champion)
    frame[config["guarded_used"]] = used & (frame[config["guarded"]] != champion)
    frame[config["guarded_capped"]] = used & (capped_delta != raw_delta)
    frame[config["guarded_kt"]] = as_knots(frame[config["guarded"]])
    status["used_foundation_count"] = int(frame[config["guarded_used"]].sum())
    status["capped_delta_count"] = int(frame[config["guarded_capped"]].sum())
    return status


def run_target(
    frame: Any,
    target: str,
    *,
    base_model_path: Path,
    base_columns: list[str],
    calibrator_path: Path | None,
    calibration_results_json: Path | None,
    default_calibration_scale: float,
    clip_correction_ms: float | None,
    foundation_expert_column: str | None,
    foundation_alpha: float | None,
    foundation_cap_delta_ms: float | None,
    deps: dict[str, Any],
) -> dict[str, Any]:
    joblib = deps["joblib"]
    pd = deps["pd"]
    config = TARGETS[target]
    if not base_model_path.exists():
        raise SystemExit(f"Missing {target} base model: {base_model_path}")
    if config["baseline"] not in frame.columns:
        raise SystemExit(f"Missing {target} baseline column: {config['baseline']}")

    base_model = joblib.load(base_model_path)
    frame[config["predicted_residual"]] = base_model.predict(frame.reindex(columns=base_columns))
    frame[config["raw"]] = pd.to_numeric(frame[config["baseline"]], errors="coerce")
    frame[config["corrected"]] = frame[config["raw"]] + frame[config["predicted_residual"]]

    calibration_status: dict[str, Any] = {
        "enabled": False,
        "scale": None,
        "calibrator_path": str(calibrator_path) if calibrator_path else None,
        "calibration_results_json": str(calibration_results_json) if calibration_results_json else None,
    }
    if calibrator_path and calibrator_path.exists():
        calibrator = joblib.load(calibrator_path)
        calibrator_columns = required_pipeline_columns(calibrator)
        x_calibrator = frame.reindex(columns=calibrator_columns) if calibrator_columns else frame
        frame[config["second_stage_raw"]] = calibrator.predict(x_calibrator)
        scale = infer_scale(calibration_results_json, default_calibration_scale)
        second_stage = frame[config["second_stage_raw"]].astype(float) * scale
        if clip_correction_ms is not None:
            second_stage = second_stage.clip(lower=-float(clip_correction_ms), upper=float(clip_correction_ms))
        frame[config["second_stage"]] = second_stage
        frame[config["calibrated"]] = frame[config["corrected"]] + second_stage
        calibration_status.update({"enabled": True, "scale": scale})
    else:
        frame[config["second_stage_raw"]] = 0.0
        frame[config["second_stage"]] = 0.0
        frame[config["calibrated"]] = frame[config["corrected"]]

    frame[config["raw_kt"]] = as_knots(frame[config["raw"]])
    frame[config["corrected_kt"]] = as_knots(frame[config["corrected"]])
    frame[config["calibrated_kt"]] = as_knots(frame[config["calibrated"]])
    shadow_status = add_shadow_foundation_blend(
        frame,
        target,
        expert_column=foundation_expert_column,
        alpha=foundation_alpha,
        cap_delta_ms=foundation_cap_delta_ms,
        deps=deps,
    )
    return {
        "target": target,
        "label": config["label"],
        "base_model_path": str(base_model_path),
        "baseline_column": config["baseline"],
        "raw_column": config["raw"],
        "corrected_column": config["corrected"],
        "calibrated_column": config["calibrated"],
        "champion_column": config["champion"],
        "guarded_foundation_column": config["guarded"],
        "calibration": calibration_status,
        "shadow_foundation_blend": shadow_status,
    }


def predictions_json(frame: Any, target_summaries: list[dict[str, Any]], limit_rows: int) -> dict[str, Any]:
    output_columns = [
        "spot_id",
        "spot_name",
        "station_id",
        "issue_time_utc",
        "target_time_utc",
        "lead_time_minutes",
    ]
    for config in TARGETS.values():
        output_columns.extend([
            config["raw"],
            config["corrected"],
            config["calibrated"],
            config["champion"],
            config["guarded"],
            config["guarded_delta"],
            config["guarded_used"],
            config["guarded_capped"],
            config["raw_kt"],
            config["corrected_kt"],
            config["calibrated_kt"],
            config["champion_kt"],
            config["guarded_kt"],
        ])
    output_columns.extend([
        "gust_high_ms",
        "gust_high_kt",
        "gust_peak_guard_raw_gap_kt",
        "gust_peak_guard_positive_gap_kt",
        "gust_peak_guard_delta_kt",
        "gust_peak_guard_risk_score",
        "gust_peak_guard_active",
        "gust_peak_guard_delta_was_capped",
        "prob_gust_ge_20kt",
        "prob_gust_ge_25kt",
        "prob_gust_ge_20kt_heuristic",
        "prob_gust_ge_25kt_heuristic",
        "prob_gust_ge_20kt_model",
        "prob_gust_ge_25kt_model",
        "peak_risk_level",
        "gust_alert_ge_20kt",
        "gust_alert_ge_20kt_probability",
        "gust_alert_ge_20kt_threshold",
        "gust_alert_ge_25kt",
        "gust_alert_ge_25kt_probability",
        "gust_alert_ge_25kt_threshold",
        "gust_operational_risk_level",
        "strong_wind_soft_weight",
        "strong_wind_aggressive_weight",
        "strong_wind_total_weight",
        "strong_soft_wind_mean_ms",
        "strong_soft_wind_mean_kt",
        "strong_soft_gust_ms",
        "strong_soft_gust_kt",
        "strong_aggressive_wind_mean_ms",
        "strong_aggressive_wind_mean_kt",
        "strong_aggressive_gust_ms",
        "strong_aggressive_gust_kt",
        "strong_gated_wind_mean_ms",
        "strong_gated_wind_mean_kt",
        "strong_gated_wind_mean_delta_ms",
        "strong_gated_gust_ms",
        "strong_gated_gust_kt",
        "strong_gated_gust_delta_ms",
    ])
    rows = frame[[column for column in output_columns if column in frame.columns]].copy()
    rows = rows.sort_values(["spot_id", "target_time_utc", "lead_time_minutes"])
    by_spot = {}
    for spot_id, group in rows.groupby("spot_id", dropna=False):
        by_spot[str(spot_id)] = group.head(limit_rows).to_dict(orient="records")
    return {
        "format": "corsewind.live_wind_and_gust_predictions.v1",
        "generated_at_utc": utc_now(),
        "row_count": int(len(frame)),
        "spot_count": int(frame["spot_id"].nunique()) if "spot_id" in frame.columns else None,
        "first_target_time_utc": str(rows["target_time_utc"].min()) if "target_time_utc" in rows.columns and not rows.empty else None,
        "last_target_time_utc": str(rows["target_time_utc"].max()) if "target_time_utc" in rows.columns and not rows.empty else None,
        "targets": target_summaries,
        "predictions_by_spot": by_spot,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    np = deps["np"]
    pd = deps["pd"]
    pq = deps["pq"]

    frame = pq.read_table(args.input_parquet).to_pandas()
    if frame.empty:
        raise SystemExit(f"Input parquet is empty: {args.input_parquet}")
    frame = add_time_features(frame, pd, np)
    selected_targets = args.target or ["wind_mean", "gust"]
    target_summaries = []
    for target in selected_targets:
        config = TARGETS[target]
        if target == "wind_mean":
            base_model_root = args.wind_base_model_root
            base_feature_columns_json = args.wind_base_feature_columns_json
            calibrator_path = args.wind_calibrator_path
            calibration_results = args.wind_calibration_results_json
            default_scale = args.wind_default_calibration_scale
        else:
            base_model_root = args.gust_base_model_root
            base_feature_columns_json = args.gust_base_feature_columns_json
            calibrator_path = args.gust_calibrator_path
            calibration_results = args.gust_calibration_results_json
            default_scale = args.gust_default_calibration_scale
        base_model_path = base_model_root / config["model_filename"]
        base_columns = feature_columns(base_feature_columns_json)
        target_summaries.append(
            run_target(
                frame,
                target,
                base_model_path=base_model_path,
                base_columns=base_columns,
                calibrator_path=calibrator_path,
                calibration_results_json=calibration_results,
                default_calibration_scale=default_scale,
                clip_correction_ms=args.clip_correction_ms,
                foundation_expert_column=args.wind_foundation_expert_column if target == "wind_mean" else args.gust_foundation_expert_column,
                foundation_alpha=args.wind_foundation_alpha if target == "wind_mean" else args.gust_foundation_alpha,
                foundation_cap_delta_ms=args.wind_foundation_cap_delta_ms if target == "wind_mean" else args.gust_foundation_cap_delta_ms,
                deps=deps,
            )
        )
    gust_peak_guard_status = add_gust_peak_guard(
        frame,
        enabled=args.gust_peak_guard,
        raw_trigger_kt=args.gust_peak_guard_raw_trigger_kt,
        gap_trigger_kt=args.gust_peak_guard_gap_trigger_kt,
        alpha=args.gust_peak_guard_alpha,
        cap_delta_kt=args.gust_peak_guard_cap_delta_kt,
        threshold_20_width_kt=args.gust_peak_guard_prob20_width_kt,
        threshold_25_width_kt=args.gust_peak_guard_prob25_width_kt,
        deps=deps,
    ) if "gust" in selected_targets else {"enabled": False, "fallback_reason": "gust_target_not_selected"}
    gust_probability_heads_status = add_gust_probability_heads(
        frame,
        model_root=args.gust_probability_model_root,
        enabled=args.gust_probability_heads,
        deps=deps,
    ) if "gust" in selected_targets else {"enabled": False, "fallback_reason": "gust_target_not_selected"}
    gust_probability_alerts_status = add_gust_probability_alerts(
        frame,
        thresholds_path=args.gust_alert_thresholds_json,
        enabled=args.gust_alert_thresholds,
        deps=deps,
    ) if "gust" in selected_targets else {"enabled": False, "fallback_reason": "gust_target_not_selected"}
    strong_wind_gated_blend_status = add_strong_wind_gated_blend(
        frame,
        enabled=args.strong_wind_gated_blend,
        selected_targets=selected_targets,
        soft_root=args.strong_wind_soft_model_root,
        aggressive_root=args.strong_wind_aggressive_model_root,
        p20_start=args.strong_wind_p20_start,
        p20_full=args.strong_wind_p20_full,
        p25_start=args.strong_wind_p25_start,
        p25_full=args.strong_wind_p25_full,
        soft_max_weight=args.strong_wind_soft_max_weight,
        aggressive_max_weight=args.strong_wind_aggressive_max_weight,
        cap_delta_ms=args.strong_wind_cap_delta_ms,
        deps=deps,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_parquet = args.output_parquet or args.output_root / "predictions.parquet"
    output_json = args.output_json or args.output_root / "predictions_by_spot.json"
    frame.to_parquet(output_parquet, compression=args.compression, index=False)
    summary = predictions_json(frame, target_summaries, args.limit_json_rows_per_spot)
    summary.update(
        {
            "input_parquet": str(args.input_parquet),
            "output_parquet": str(output_parquet),
            "wind_base_model_root": str(args.wind_base_model_root),
            "wind_base_feature_columns_json": str(args.wind_base_feature_columns_json),
            "gust_base_model_root": str(args.gust_base_model_root),
            "gust_base_feature_columns_json": str(args.gust_base_feature_columns_json),
            "clip_correction_ms": args.clip_correction_ms,
            "gust_peak_guard": gust_peak_guard_status,
            "gust_probability_heads": gust_probability_heads_status,
            "gust_probability_alerts": gust_probability_alerts_status,
            "strong_wind_gated_blend": strong_wind_gated_blend_status,
        }
    )
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--target", choices=sorted(TARGETS), action="append", default=[])
    parser.add_argument("--wind-base-model-root", type=Path, default=DEFAULT_WIND_BASE_RUN)
    parser.add_argument("--wind-base-feature-columns-json", type=Path, default=DEFAULT_WIND_BASE_RUN / "feature_columns.json")
    parser.add_argument("--gust-base-model-root", type=Path, default=DEFAULT_GUST_BASE_RUN)
    parser.add_argument("--gust-base-feature-columns-json", type=Path, default=DEFAULT_GUST_BASE_RUN / "feature_columns.json")
    parser.add_argument("--wind-calibrator-path", type=Path, default=DEFAULT_WIND_CALIBRATOR_RUN / "calibrator.joblib")
    parser.add_argument("--wind-calibration-results-json", type=Path, default=DEFAULT_WIND_CALIBRATOR_RUN / "calibration_results.json")
    parser.add_argument("--wind-default-calibration-scale", type=float, default=0.70)
    parser.add_argument("--gust-calibrator-path", type=Path, default=DEFAULT_GUST_CALIBRATOR_RUN / "calibrator.joblib")
    parser.add_argument("--gust-calibration-results-json", type=Path, default=DEFAULT_GUST_CALIBRATOR_RUN / "calibration_results.json")
    parser.add_argument("--gust-default-calibration-scale", type=float, default=0.70)
    parser.add_argument("--wind-foundation-expert-column", default=TARGETS["wind_mean"]["foundation_expert"])
    parser.add_argument("--wind-foundation-alpha", type=float, default=TARGETS["wind_mean"]["foundation_alpha"])
    parser.add_argument("--wind-foundation-cap-delta-ms", type=float, default=TARGETS["wind_mean"]["foundation_cap_delta_ms"])
    parser.add_argument("--gust-foundation-expert-column", default=TARGETS["gust"]["foundation_expert"])
    parser.add_argument("--gust-foundation-alpha", type=float, default=TARGETS["gust"]["foundation_alpha"])
    parser.add_argument("--gust-foundation-cap-delta-ms", type=float, default=TARGETS["gust"]["foundation_cap_delta_ms"])
    parser.add_argument("--gust-peak-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gust-peak-guard-raw-trigger-kt", type=float, default=12.0)
    parser.add_argument("--gust-peak-guard-gap-trigger-kt", type=float, default=0.0)
    parser.add_argument("--gust-peak-guard-alpha", type=float, default=0.80)
    parser.add_argument("--gust-peak-guard-cap-delta-kt", type=float, default=5.0)
    parser.add_argument("--gust-peak-guard-prob20-width-kt", type=float, default=3.0)
    parser.add_argument("--gust-peak-guard-prob25-width-kt", type=float, default=3.5)
    parser.add_argument("--gust-probability-heads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gust-probability-model-root", type=Path, default=DEFAULT_GUST_PROBABILITY_RUN)
    parser.add_argument("--gust-alert-thresholds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gust-alert-thresholds-json", type=Path, default=DEFAULT_GUST_ALERT_THRESHOLDS)
    parser.add_argument("--strong-wind-gated-blend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strong-wind-soft-model-root", type=Path, default=DEFAULT_STRONG_SOFT_RUN)
    parser.add_argument("--strong-wind-aggressive-model-root", type=Path, default=DEFAULT_STRONG_AGGRESSIVE_RUN)
    parser.add_argument("--strong-wind-p20-start", type=float, default=0.25)
    parser.add_argument("--strong-wind-p20-full", type=float, default=0.35)
    parser.add_argument("--strong-wind-p25-start", type=float, default=0.15)
    parser.add_argument("--strong-wind-p25-full", type=float, default=0.20)
    parser.add_argument("--strong-wind-soft-max-weight", type=float, default=0.65)
    parser.add_argument("--strong-wind-aggressive-max-weight", type=float, default=0.35)
    parser.add_argument("--strong-wind-cap-delta-ms", type=float, default=2.5)
    parser.add_argument("--clip-correction-ms", type=float, default=2.0)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--limit-json-rows-per-spot", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(
        {
            key: result[key]
            for key in ("row_count", "spot_count", "first_target_time_utc", "last_target_time_utc", "output_parquet")
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
