#!/usr/bin/env python3
"""Run live wind-mean inference from flat residual-training rows."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ML_ROOT = Path("/srv/data/corsewind/ml_dataset")
DEFAULT_BASE_RUN = DEFAULT_ML_ROOT / "benchmarks/tabular_lgbm_225k_prev_lowmem_2024_2025_to_2026_v1"
DEFAULT_CALIBRATOR_RUN = DEFAULT_ML_ROOT / "benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1"
KNOTS_PER_MS = 1.9438444924406


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


def predictions_json(frame: Any, limit_rows: int) -> dict[str, Any]:
    output_columns = [
        "spot_id",
        "spot_name",
        "station_id",
        "issue_time_utc",
        "target_time_utc",
        "lead_time_minutes",
        "raw_wind_mean_ms",
        "corrected_wind_mean_ms",
        "calibrated_wind_mean_ms",
        "raw_wind_mean_kt",
        "corrected_wind_mean_kt",
        "calibrated_wind_mean_kt",
    ]
    rows = frame[[column for column in output_columns if column in frame.columns]].copy()
    rows = rows.sort_values(["spot_id", "target_time_utc", "lead_time_minutes"])
    by_spot = {}
    for spot_id, group in rows.groupby("spot_id", dropna=False):
        by_spot[str(spot_id)] = group.head(limit_rows).to_dict(orient="records")
    return {
        "format": "corsewind.live_wind_mean_predictions.v1",
        "generated_at_utc": utc_now(),
        "row_count": int(len(frame)),
        "spot_count": int(frame["spot_id"].nunique()) if "spot_id" in frame.columns else None,
        "first_target_time_utc": str(rows["target_time_utc"].min()) if "target_time_utc" in rows.columns and not rows.empty else None,
        "last_target_time_utc": str(rows["target_time_utc"].max()) if "target_time_utc" in rows.columns and not rows.empty else None,
        "predictions_by_spot": by_spot,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    joblib = deps["joblib"]
    np = deps["np"]
    pd = deps["pd"]
    pq = deps["pq"]

    frame = pq.read_table(args.input_parquet).to_pandas()
    if frame.empty:
        raise SystemExit(f"Input parquet is empty: {args.input_parquet}")
    frame = add_time_features(frame, pd, np)

    base_columns = feature_columns(args.base_feature_columns_json)
    base_model = joblib.load(args.base_model_path)
    x_base = frame.reindex(columns=base_columns)
    frame["predicted_residual_wind_mean_ms"] = base_model.predict(x_base)
    frame["raw_wind_mean_ms"] = pd.to_numeric(frame["baselines__baseline_wind_mean_ms"], errors="coerce")
    frame["corrected_wind_mean_ms"] = frame["raw_wind_mean_ms"] + frame["predicted_residual_wind_mean_ms"]

    calibrator = joblib.load(args.calibrator_path)
    calibrator_columns = required_pipeline_columns(calibrator)
    if calibrator_columns:
        x_calibrator = frame.reindex(columns=calibrator_columns)
    else:
        x_calibrator = frame
    frame["predicted_second_stage_residual_ms_raw"] = calibrator.predict(x_calibrator)
    scale = infer_scale(args.calibration_results_json, args.default_calibration_scale)
    second_stage = frame["predicted_second_stage_residual_ms_raw"].astype(float) * scale
    if args.clip_correction_ms is not None:
        second_stage = second_stage.clip(lower=-float(args.clip_correction_ms), upper=float(args.clip_correction_ms))
    frame["predicted_second_stage_residual_ms"] = second_stage
    frame["calibrated_wind_mean_ms"] = frame["corrected_wind_mean_ms"] + second_stage
    frame["raw_wind_mean_kt"] = as_knots(frame["raw_wind_mean_ms"])
    frame["corrected_wind_mean_kt"] = as_knots(frame["corrected_wind_mean_ms"])
    frame["calibrated_wind_mean_kt"] = as_knots(frame["calibrated_wind_mean_ms"])

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_parquet = args.output_parquet or args.output_root / "predictions.parquet"
    output_json = args.output_json or args.output_root / "predictions_by_spot.json"
    frame.to_parquet(output_parquet, compression=args.compression, index=False)
    summary = predictions_json(frame, args.limit_json_rows_per_spot)
    summary.update(
        {
            "input_parquet": str(args.input_parquet),
            "output_parquet": str(output_parquet),
            "calibration_scale": scale,
            "clip_correction_ms": args.clip_correction_ms,
            "base_model_path": str(args.base_model_path),
            "calibrator_path": str(args.calibrator_path),
        }
    )
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_RUN / "labels__residual_wind_mean_ms.joblib")
    parser.add_argument("--base-feature-columns-json", type=Path, default=DEFAULT_BASE_RUN / "feature_columns.json")
    parser.add_argument("--calibrator-path", type=Path, default=DEFAULT_CALIBRATOR_RUN / "calibrator.joblib")
    parser.add_argument("--calibration-results-json", type=Path, default=DEFAULT_CALIBRATOR_RUN / "calibration_results.json")
    parser.add_argument("--default-calibration-scale", type=float, default=0.70)
    parser.add_argument("--clip-correction-ms", type=float, default=2.0)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--limit-json-rows-per-spot", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({key: result[key] for key in ("row_count", "spot_count", "first_target_time_utc", "last_target_time_utc", "output_parquet", "calibration_scale")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
