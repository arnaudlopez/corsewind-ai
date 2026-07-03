#!/usr/bin/env python3
"""Apply saved gust quantile calibrators to an existing prediction parquet."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def quantile_suffix_value(suffix: str) -> float:
    match = re.fullmatch(r"q(\d+(?:p\d+)?)", suffix.strip().lower())
    if not match:
        return math.inf
    return float(match.group(1).replace("p", ".")) / 100.0


def import_dependencies() -> dict[str, Any]:
    try:
        import joblib
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Missing ML dependencies. Run inside the CorseWind ML venv.") from exc
    return {"joblib": joblib, "np": np, "pd": pd}


def required_pipeline_columns(model: Any) -> list[str]:
    preprocess = getattr(model, "named_steps", {}).get("preprocess") if hasattr(model, "named_steps") else None
    transformers = getattr(preprocess, "transformers_", None) if preprocess is not None else None
    columns: list[str] = []
    for _name, _transformer, selected in transformers or []:
        if selected is None:
            continue
        if isinstance(selected, str):
            columns.append(selected)
        else:
            columns.extend(str(column) for column in selected)
    return list(dict.fromkeys(columns))


def infer_scale(path: Path | None, default: float) -> float:
    if path is None or not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    scale = (payload.get("scale_selection") or {}).get("selected_scale")
    try:
        return float(default if scale is None else scale)
    except (TypeError, ValueError):
        return default


def apply_quantiles(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    joblib = deps["joblib"]
    np = deps["np"]
    pd = deps["pd"]
    frame = pd.read_parquet(args.predictions)
    if args.base_prediction_column not in frame.columns:
        raise SystemExit(f"Missing base prediction column: {args.base_prediction_column}")
    suffixes = list(dict.fromkeys(suffix.strip().lstrip("_") for suffix in args.suffix if suffix.strip().lstrip("_")))
    status: dict[str, Any] = {
        "format": "corsewind.gust_quantile_application.v1",
        "generated_at_utc": utc_now(),
        "predictions": str(args.predictions),
        "calibrator_root": str(args.calibrator_root),
        "base_prediction_column": args.base_prediction_column,
        "row_count": int(len(frame)),
        "suffixes": suffixes,
        "rails": {},
        "monotone_sort": bool(args.monotone_sort),
        "monotone_crossing_rows_before_sort": 0,
    }
    loaded_suffixes: list[str] = []
    for safe_suffix in suffixes:
        model_path = args.calibrator_root / f"calibrator_{safe_suffix}.joblib"
        results_path = args.calibrator_root / f"results_{safe_suffix}.json"
        output_column = f"{args.output_prefix}_{safe_suffix}"
        raw_column = f"{args.raw_output_prefix}_{safe_suffix}_raw"
        correction_column = f"{args.raw_output_prefix}_{safe_suffix}"
        if not model_path.exists():
            status["rails"][safe_suffix] = {"loaded": False, "model_path": str(model_path), "reason": "missing_model"}
            continue
        model = joblib.load(model_path)
        columns = required_pipeline_columns(model)
        x_values = frame.reindex(columns=columns) if columns else frame
        raw_correction = pd.Series(model.predict(x_values), index=frame.index).astype(float)
        scale = infer_scale(results_path, 1.0)
        correction = raw_correction * scale
        if args.clip_correction_ms is not None:
            correction = correction.clip(lower=-float(args.clip_correction_ms), upper=float(args.clip_correction_ms))
        frame[raw_column] = raw_correction
        frame[correction_column] = correction
        frame[output_column] = pd.to_numeric(frame[args.base_prediction_column], errors="coerce") + correction
        loaded_suffixes.append(safe_suffix)
        status["rails"][safe_suffix] = {
            "loaded": True,
            "model_path": str(model_path),
            "results_path": str(results_path) if results_path.exists() else None,
            "feature_column_count": len(columns),
            "scale": scale,
            "non_null_count": int(frame[output_column].notna().sum()),
            "output_column": output_column,
        }
    ordered_suffixes = sorted(loaded_suffixes, key=lambda value: (quantile_suffix_value(value), value))
    if args.monotone_sort and len(ordered_suffixes) >= 2:
        columns = [f"{args.output_prefix}_{suffix}" for suffix in ordered_suffixes]
        values = frame[columns].apply(pd.to_numeric, errors="coerce")
        valid = values.notna().all(axis=1)
        crossings = (values.diff(axis=1).iloc[:, 1:] < 0).any(axis=1) & valid
        status["monotone_crossing_rows_before_sort"] = int(crossings.sum())
        if valid.any():
            values.loc[valid, columns] = np.sort(values.loc[valid, columns].to_numpy(dtype=float), axis=1)
        for column in columns:
            frame[column] = values[column]
    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_predictions, index=False, compression=args.compression)
    status["output_predictions"] = str(args.output_predictions)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(status, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True, default=str))
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--calibrator-root", type=Path, required=True)
    parser.add_argument("--suffix", action="append", default=["q50", "q60", "q75", "q90"])
    parser.add_argument("--base-prediction-column", default="corrected_gust_ms")
    parser.add_argument("--output-prefix", default="calibrated_gust_ms")
    parser.add_argument("--raw-output-prefix", default="predicted_second_stage_residual_gust_ms")
    parser.add_argument("--clip-correction-ms", type=float, default=5.0)
    parser.add_argument("--monotone-sort", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    apply_quantiles(parse_args())


if __name__ == "__main__":
    main()
