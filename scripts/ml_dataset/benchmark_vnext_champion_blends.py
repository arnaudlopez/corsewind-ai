#!/usr/bin/env python3
"""Benchmark cautious v_next blends around current champion predictions."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KNOTS_PER_MS = 1.9438444924406
DEFAULT_ALPHAS = (0.0, 0.05, 0.10, 0.15, 0.20)
DEFAULT_CLIPS_MS = (0.25, 0.50, 0.75, 1.00, 2.00)
DEFAULT_THRESHOLDS_KT = (12.0, 15.0, 20.0, 25.0)
DEFAULT_HOTSPOTS = ("la_tonnara", "santa_manza", "balistra")
TARGETS = {
    "wind_mean": {
        "label": "wind mean",
        "champion": "calibrated_wind_mean_ms",
        "raw": "raw_wind_mean_ms",
        "actual": "actual_wind_mean_ms",
        "vnext_baseline": "baselines__baseline_wind_mean_ms",
        "vnext_label": "labels__target_wind_mean_ms",
        "vnext_model_file": "labels__residual_wind_mean_ms.joblib",
        "champion_rmse_gate": 1.268019,
    },
    "gust": {
        "label": "gust",
        "champion": "calibrated_gust_ms",
        "raw": "raw_gust_ms",
        "actual": "actual_gust_ms",
        "vnext_baseline": "baselines__baseline_gust_ms",
        "vnext_label": "labels__target_gust_ms",
        "vnext_model_file": "labels__residual_gust_ms.joblib",
        "champion_rmse_gate": 1.484221,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def import_deps() -> dict[str, Any]:
    try:
        import joblib
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Missing pandas/numpy/joblib dependencies.") from exc
    return {"joblib": joblib, "np": np, "pd": pd}


def month_range(start_month: str, end_month: str) -> list[str]:
    start_year, start_m = [int(part) for part in start_month.split("-", 1)]
    end_year, end_m = [int(part) for part in end_month.split("-", 1)]
    out = []
    year, month = start_year, start_m
    while (year, month) <= (end_year, end_m):
        out.append(f"{year:04d}_{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return out


def add_time_features(frame: Any, pd: Any, np: Any) -> Any:
    issue_time = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    dayofyear = issue_time.dt.dayofyear.fillna(1).astype(float)
    frame["issue_hour_utc"] = issue_time.dt.hour.astype("float64")
    frame["issue_month"] = issue_time.dt.month.astype("float64")
    angle = 2.0 * math.pi * dayofyear / 366.0
    frame["issue_dayofyear_sin"] = np.sin(angle)
    frame["issue_dayofyear_cos"] = np.cos(angle)
    return frame


def unique_columns(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def key_columns() -> list[str]:
    return ["spot_id", "issue_time_utc", "target_time_utc", "lead_time_minutes"]


def load_feature_columns(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return unique_columns(list(data.get("numeric") or []) + list(data.get("categorical") or []))


def build_vnext_predictions(args: argparse.Namespace, target: str, deps: dict[str, Any]) -> Any:
    pd = deps["pd"]
    np = deps["np"]
    joblib = deps["joblib"]
    config = TARGETS[target]
    model_root = args.vnext_benchmark_root / f"lgbm_{'wind' if target == 'wind_mean' else 'gust'}"
    model = joblib.load(model_root / config["vnext_model_file"])
    feature_columns = load_feature_columns(model_root / "feature_columns.json")
    rows = []
    for month in month_range(args.start_month, args.end_month):
        path = args.training_table_root / f"{args.vnext_run_id_prefix}_{month}" / "training_rows.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame = add_time_features(frame, pd, np)
        required = unique_columns(
            key_columns()
            + [
                "station_id",
                "spot_kind",
                "latitude",
                "longitude",
                config["vnext_baseline"],
                config["vnext_label"],
            ]
            + feature_columns
        )
        frame = frame.reindex(columns=required)
        residual = model.predict(frame.reindex(columns=feature_columns))
        pred = frame[config["vnext_baseline"]].astype(float).to_numpy() + residual
        out = frame[key_columns() + ["station_id", "spot_kind", "latitude", "longitude"]].copy()
        out["vnext_prediction_ms"] = pred
        out["vnext_actual_ms"] = frame[config["vnext_label"]].astype(float)
        out["vnext_raw_ms"] = frame[config["vnext_baseline"]].astype(float)
        rows.append(out)
    if not rows:
        raise SystemExit("No v_next monthly training shards found.")
    combined = pd.concat(rows, ignore_index=True)
    combined["issue_time_utc"] = pd.to_datetime(combined["issue_time_utc"], utc=True, errors="coerce")
    combined["target_time_utc"] = pd.to_datetime(combined["target_time_utc"], utc=True, errors="coerce")
    return combined.drop_duplicates(key_columns())


def load_champion_predictions(path: Path, target: str, deps: dict[str, Any]) -> Any:
    pd = deps["pd"]
    config = TARGETS[target]
    frame = pd.read_parquet(path)
    frame["issue_time_utc"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    if "target_time_utc" in frame.columns:
        frame["target_time_utc"] = pd.to_datetime(frame["target_time_utc"], utc=True, errors="coerce")
    else:
        frame["target_time_utc"] = frame["issue_time_utc"] + pd.to_timedelta(
            frame["lead_time_minutes"].astype(float),
            unit="m",
        )
    keep = unique_columns(
        key_columns()
        + [
            "station_id",
            "spot_kind",
            "latitude",
            "longitude",
            config["champion"],
            config["raw"],
            config["actual"],
        ]
    )
    out = frame.reindex(columns=keep).copy()
    out = out.rename(
        columns={
            config["champion"]: "champion_prediction_ms",
            config["raw"]: "raw_prediction_ms",
            config["actual"]: "actual_ms",
        }
    )
    return out.drop_duplicates(key_columns())


def build_same_key_table(args: argparse.Namespace, target: str, deps: dict[str, Any]) -> Any:
    champion_path = args.wind_champion_predictions if target == "wind_mean" else args.gust_champion_predictions
    champion = load_champion_predictions(champion_path, target, deps)
    vnext = build_vnext_predictions(args, target, deps)
    merged = champion.merge(
        vnext[key_columns() + ["vnext_prediction_ms", "vnext_actual_ms", "vnext_raw_ms"]],
        on=key_columns(),
        how="inner",
        validate="one_to_one",
    )
    # Actual values can differ only by tiny representation differences. Keep the champion actual as reference.
    merged["target_actual_diff_ms"] = merged["actual_ms"].astype(float) - merged["vnext_actual_ms"].astype(float)
    merged["vnext_delta_vs_champion_ms"] = (
        merged["vnext_prediction_ms"].astype(float) - merged["champion_prediction_ms"].astype(float)
    )
    return merged


def finite_metric(frame: Any, prediction_column: str, actual_column: str, np: Any) -> dict[str, Any]:
    values = frame[[prediction_column, actual_column]].copy()
    pred = values[prediction_column].astype(float).to_numpy()
    obs = values[actual_column].astype(float).to_numpy()
    valid = np.isfinite(pred) & np.isfinite(obs)
    if not valid.any():
        return {"count": 0}
    err = pred[valid] - obs[valid]
    return {
        "count": int(err.size),
        "mae": round(float(np.mean(np.abs(err))), 6),
        "rmse": round(float(math.sqrt(float(np.mean(err * err)))), 6),
        "bias": round(float(np.mean(err)), 6),
        "p90_abs_error": round(float(np.quantile(np.abs(err), 0.90)), 6),
        "p95_abs_error": round(float(np.quantile(np.abs(err), 0.95)), 6),
    }


def threshold_metric(frame: Any, prediction_column: str, actual_column: str, threshold_kt: float) -> dict[str, Any]:
    valid = frame[[prediction_column, actual_column]].dropna()
    if valid.empty:
        return {"count": 0}
    threshold_ms = threshold_kt / KNOTS_PER_MS
    predicted = valid[prediction_column].astype(float) >= threshold_ms
    actual = valid[actual_column].astype(float) >= threshold_ms
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())
    return {
        "count": int(len(valid)),
        "threshold_kt": float(threshold_kt),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": None if tp + fp == 0 else round(float(tp / (tp + fp)), 6),
        "recall": None if tp + fn == 0 else round(float(tp / (tp + fn)), 6),
        "csi": None if tp + fp + fn == 0 else round(float(tp / (tp + fp + fn)), 6),
    }


def grouped_metrics(frame: Any, group_columns: list[str], prediction_column: str, actual_column: str, np: Any, pd: Any) -> list[dict[str, Any]]:
    if any(column not in frame.columns for column in group_columns):
        return []
    rows = []
    valid = frame.dropna(subset=[prediction_column, actual_column])
    for raw_key, group in valid.groupby(group_columns, dropna=False):
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        rows.append({
            "group": dict(zip(group_columns, [json_scalar(pd, value) for value in values], strict=True)),
            **finite_metric(group, prediction_column, actual_column, np),
        })
    rows.sort(key=lambda item: (item.get("rmse") is None, item.get("rmse", -1)), reverse=True)
    return rows


def json_scalar(pd: Any, value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def add_gate_columns(frame: Any, args: argparse.Namespace) -> Any:
    frame = frame.copy()
    frame["gate_all"] = True
    frame["gate_lead_le_30"] = frame["lead_time_minutes"].astype(float) <= 30.0
    frame["gate_lead_15"] = frame["lead_time_minutes"].astype(float) == 15.0
    frame["gate_hotspots"] = frame["spot_id"].astype(str).isin(set(args.hotspot))
    frame["raw_kt"] = frame["raw_prediction_ms"].astype(float) * KNOTS_PER_MS
    frame["champion_kt"] = frame["champion_prediction_ms"].astype(float) * KNOTS_PER_MS
    frame["vnext_kt"] = frame["vnext_prediction_ms"].astype(float) * KNOTS_PER_MS
    frame["actual_kt"] = frame["actual_ms"].astype(float) * KNOTS_PER_MS
    frame["gate_vnext_above_champion"] = frame["vnext_prediction_ms"] > frame["champion_prediction_ms"]
    frame["gate_vnext_below_champion"] = frame["vnext_prediction_ms"] < frame["champion_prediction_ms"]
    frame["gate_raw_above_champion_1kt"] = (frame["raw_kt"] - frame["champion_kt"]) >= 1.0
    for threshold in args.threshold_kt:
        token = f"{threshold:g}".replace(".", "p")
        frame[f"gate_raw_ge_{token}kt"] = frame["raw_kt"] >= threshold
        frame[f"gate_champion_ge_{token}kt"] = frame["champion_kt"] >= threshold
        frame[f"gate_raw_or_champion_ge_{token}kt"] = (frame["raw_kt"] >= threshold) | (frame["champion_kt"] >= threshold)
    return frame


def gate_specs(args: argparse.Namespace) -> list[tuple[str, str]]:
    specs = [
        ("all", "gate_all"),
        ("lead_le_30", "gate_lead_le_30"),
        ("lead_15", "gate_lead_15"),
        ("hotspots", "gate_hotspots"),
        ("vnext_above_champion", "gate_vnext_above_champion"),
        ("vnext_below_champion", "gate_vnext_below_champion"),
        ("raw_above_champion_1kt", "gate_raw_above_champion_1kt"),
    ]
    for threshold in args.threshold_kt:
        token = f"{threshold:g}".replace(".", "p")
        specs.extend([
            (f"raw_ge_{threshold:g}kt", f"gate_raw_ge_{token}kt"),
            (f"champion_ge_{threshold:g}kt", f"gate_champion_ge_{token}kt"),
            (f"raw_or_champion_ge_{threshold:g}kt", f"gate_raw_or_champion_ge_{token}kt"),
        ])
    return specs


def evaluate_variant(frame: Any, prediction_column: str, np: Any, pd: Any, thresholds_kt: list[float]) -> dict[str, Any]:
    out = {
        "metrics": finite_metric(frame, prediction_column, "actual_ms", np),
        "by_lead": grouped_metrics(frame, ["lead_time_minutes"], prediction_column, "actual_ms", np, pd),
        "worst_spots": grouped_metrics(frame, ["spot_id"], prediction_column, "actual_ms", np, pd)[:15],
        "threshold_detection": {
            f">={threshold:g}kt": threshold_metric(frame, prediction_column, "actual_ms", threshold)
            for threshold in thresholds_kt
        },
        "by_observed_threshold": {},
    }
    for threshold in thresholds_kt:
        subset = frame[frame["actual_kt"] >= threshold]
        out["by_observed_threshold"][f">={threshold:g}kt"] = finite_metric(subset, prediction_column, "actual_ms", np)
    return out


def build_blend_predictions(frame: Any, args: argparse.Namespace, np: Any) -> list[dict[str, Any]]:
    variants = []
    delta = frame["vnext_delta_vs_champion_ms"].astype(float)
    for gate_name, gate_column in gate_specs(args):
        gate = frame[gate_column].fillna(False).astype(bool)
        gate_count = int(gate.sum())
        if gate_count == 0:
            continue
        for alpha in args.alpha:
            for clip_ms in args.clip_ms:
                name = f"blend_{gate_name}_a{alpha:g}_clip{clip_ms:g}"
                clipped_delta = delta.clip(lower=-float(clip_ms), upper=float(clip_ms))
                pred = frame["champion_prediction_ms"].astype(float) + (gate.astype(float) * float(alpha) * clipped_delta)
                variants.append({
                    "name": name,
                    "gate": gate_name,
                    "gate_column": gate_column,
                    "gate_count": gate_count,
                    "gate_share": round(gate_count / len(frame), 6) if len(frame) else None,
                    "alpha": float(alpha),
                    "clip_ms": float(clip_ms),
                    "prediction": pred.to_numpy(dtype="float64"),
                })
    oracle_pred = np.where(
        np.abs(frame["champion_prediction_ms"].astype(float) - frame["actual_ms"].astype(float))
        <= np.abs(frame["vnext_prediction_ms"].astype(float) - frame["actual_ms"].astype(float)),
        frame["champion_prediction_ms"].astype(float),
        frame["vnext_prediction_ms"].astype(float),
    )
    variants.append({
        "name": "pair_oracle_champion_or_vnext",
        "gate": "oracle",
        "gate_column": None,
        "gate_count": int(len(frame)),
        "gate_share": 1.0,
        "alpha": None,
        "clip_ms": None,
        "prediction": oracle_pred,
        "oracle": True,
    })
    return variants


def summarize_variant(frame: Any, variant: dict[str, Any], np: Any) -> dict[str, Any]:
    work = frame[["actual_ms"]].copy()
    work["prediction"] = variant["prediction"]
    metrics = finite_metric(work, "prediction", "actual_ms", np)
    return {
        "name": variant["name"],
        "gate": variant["gate"],
        "gate_count": variant["gate_count"],
        "gate_share": variant["gate_share"],
        "alpha": variant["alpha"],
        "clip_ms": variant["clip_ms"],
        "oracle": bool(variant.get("oracle", False)),
        **metrics,
    }


def select_best(calibration_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in calibration_rows if not row.get("oracle") and row.get("rmse") is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (float(row["rmse"]), float(row.get("mae") or 999.0)))


def run_target(args: argparse.Namespace, target: str, deps: dict[str, Any]) -> dict[str, Any]:
    pd = deps["pd"]
    np = deps["np"]
    frame = build_same_key_table(args, target, deps)
    frame = add_gate_columns(frame, args)
    split_time = pd.to_datetime(args.blend_selection_split_utc, utc=True)
    calibration = frame[frame["issue_time_utc"] < split_time].copy()
    holdout = frame[frame["issue_time_utc"] >= split_time].copy()
    for name, source in [
        ("champion_prediction", "champion_prediction_ms"),
        ("raw_prediction", "raw_prediction_ms"),
        ("vnext_prediction", "vnext_prediction_ms"),
    ]:
        frame[name] = frame[source]
        calibration[name] = calibration[source]
        holdout[name] = holdout[source]
    full_rows = []
    calibration_rows = []
    holdout_rows = []
    variants = build_blend_predictions(frame, args, np)
    calibration_variants = build_blend_predictions(calibration, args, np)
    holdout_variants = build_blend_predictions(holdout, args, np)
    for variant in variants:
        full_rows.append(summarize_variant(frame, variant, np))
    for variant in calibration_variants:
        calibration_rows.append(summarize_variant(calibration, variant, np))
    for variant in holdout_variants:
        holdout_rows.append(summarize_variant(holdout, variant, np))
    best_cal = select_best(calibration_rows)
    selected_holdout = None
    selected_full = None
    if best_cal is not None:
        selected_holdout = next((row for row in holdout_rows if row["name"] == best_cal["name"]), None)
        selected_full = next((row for row in full_rows if row["name"] == best_cal["name"]), None)
    full_rows.sort(key=lambda row: (row.get("oracle", False), row.get("rmse") is None, row.get("rmse", 999.0)))
    calibration_rows.sort(key=lambda row: (row.get("oracle", False), row.get("rmse") is None, row.get("rmse", 999.0)))
    holdout_rows.sort(key=lambda row: (row.get("oracle", False), row.get("rmse") is None, row.get("rmse", 999.0)))
    predictions = {
        "champion": evaluate_variant(frame.assign(champion_prediction=frame["champion_prediction_ms"]), "champion_prediction", np, pd, args.threshold_kt),
        "raw": evaluate_variant(frame.assign(raw_prediction=frame["raw_prediction_ms"]), "raw_prediction", np, pd, args.threshold_kt),
        "vnext": evaluate_variant(frame.assign(vnext_prediction=frame["vnext_prediction_ms"]), "vnext_prediction", np, pd, args.threshold_kt),
    }
    if selected_full is not None:
        selected_variant = next(variant for variant in variants if variant["name"] == selected_full["name"])
        selected_frame = frame.copy()
        selected_frame["selected_blend_prediction"] = selected_variant["prediction"]
        predictions["selected_blend_full"] = evaluate_variant(selected_frame, "selected_blend_prediction", np, pd, args.threshold_kt)
    target_table_path = args.output_root / f"{target}_champion_vnext_same_key.parquet"
    if args.write_tables:
        args.output_root.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target_table_path, index=False)
    split_baselines = {
        "calibration": {
            "champion": finite_metric(calibration, "champion_prediction", "actual_ms", np),
            "raw": finite_metric(calibration, "raw_prediction", "actual_ms", np),
            "vnext": finite_metric(calibration, "vnext_prediction", "actual_ms", np),
        },
        "holdout": {
            "champion": finite_metric(holdout, "champion_prediction", "actual_ms", np),
            "raw": finite_metric(holdout, "raw_prediction", "actual_ms", np),
            "vnext": finite_metric(holdout, "vnext_prediction", "actual_ms", np),
        },
    }
    holdout_champion_rmse = split_baselines["holdout"]["champion"].get("rmse")
    selected_rmse = selected_holdout.get("rmse") if selected_holdout else None
    return {
        "target": target,
        "label": TARGETS[target]["label"],
        "row_count": int(len(frame)),
        "calibration_row_count": int(len(calibration)),
        "holdout_row_count": int(len(holdout)),
        "target_actual_max_abs_diff_ms": None if frame.empty else round(float(frame["target_actual_diff_ms"].abs().max()), 9),
        "same_key_table": str(target_table_path) if args.write_tables else None,
        "baseline_metrics": {
            "champion": predictions["champion"]["metrics"],
            "raw": predictions["raw"]["metrics"],
            "vnext": predictions["vnext"]["metrics"],
            "pair_oracle": next((row for row in full_rows if row["name"] == "pair_oracle_champion_or_vnext"), None),
        },
        "split_baseline_metrics": split_baselines,
        "best_full_eval_variants": full_rows[:20],
        "best_calibration_variants": calibration_rows[:20],
        "best_holdout_variants": holdout_rows[:20],
        "selected_by_calibration": best_cal,
        "selected_holdout": selected_holdout,
        "selected_full_eval": selected_full,
        "selected_gain_pct_vs_champion_holdout": (
            None
            if not holdout_champion_rmse or selected_rmse is None
            else round((float(holdout_champion_rmse) - float(selected_rmse)) / float(holdout_champion_rmse) * 100.0, 6)
        ),
        "promotion_gate_rmse": TARGETS[target]["champion_rmse_gate"],
        "promotion_verdict": (
            "beats_promotion_gate"
            if selected_full is not None
            and selected_full.get("rmse") is not None
            and float(selected_full["rmse"]) < float(TARGETS[target]["champion_rmse_gate"])
            else "do_not_promote"
        ),
        "detailed_metrics": predictions,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# v_next Champion Blend Benchmark",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Selection split: `{result['blend_selection_split_utc']}`",
        "",
        "## Summary",
        "",
        "| Target | Rows | Champion RMSE | v_next RMSE | Oracle RMSE | Selected holdout RMSE | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for target in result["targets"]:
        baseline = target["baseline_metrics"]
        selected = target.get("selected_holdout") or {}
        lines.append(
            f"| `{target['target']}` | {target['row_count']} | "
            f"{baseline['champion'].get('rmse')} | {baseline['vnext'].get('rmse')} | "
            f"{(baseline.get('pair_oracle') or {}).get('rmse')} | {selected.get('rmse')} | "
            f"`{target['promotion_verdict']}` |"
        )
    for target in result["targets"]:
        lines.extend([
            "",
            f"## {target['label'].title()}",
            "",
            f"- rows: `{target['row_count']}`",
            f"- calibration rows before split: `{target['calibration_row_count']}`",
            f"- holdout rows after split: `{target['holdout_row_count']}`",
            f"- max absolute actual diff between champion and v_next rows: `{target['target_actual_max_abs_diff_ms']}`",
            "",
            "Baseline metrics:",
            "",
            "| Prediction | RMSE | MAE | Bias | Count |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for name, metric in target["baseline_metrics"].items():
            if name == "pair_oracle":
                label = "pair oracle, not deployable"
            else:
                label = name
            if metric:
                lines.append(
                    f"| `{label}` | {metric.get('rmse')} | {metric.get('mae')} | {metric.get('bias')} | {metric.get('count')} |"
                )
        selected_cal = target.get("selected_by_calibration") or {}
        selected_holdout = target.get("selected_holdout") or {}
        selected_full = target.get("selected_full_eval") or {}
        split_baselines = target.get("split_baseline_metrics") or {}
        holdout_champion = (split_baselines.get("holdout") or {}).get("champion") or {}
        calibration_champion = (split_baselines.get("calibration") or {}).get("champion") or {}
        lines.extend([
            "",
            "Selected by calibration:",
            "",
            f"- variant: `{selected_cal.get('name')}`",
            f"- champion calibration RMSE: `{calibration_champion.get('rmse')}`",
            f"- calibration RMSE: `{selected_cal.get('rmse')}`",
            f"- champion holdout RMSE: `{holdout_champion.get('rmse')}`",
            f"- holdout RMSE: `{selected_holdout.get('rmse')}`",
            f"- holdout gain vs champion: `{target.get('selected_gain_pct_vs_champion_holdout')}%`",
            f"- full eval RMSE: `{selected_full.get('rmse')}`",
            f"- promotion verdict: `{target['promotion_verdict']}`",
            "",
            "Best full-evaluation variants:",
            "",
            "| Variant | RMSE | MAE | Bias | Gate share | Alpha | Clip |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in target["best_full_eval_variants"][:10]:
            lines.append(
                f"| `{row['name']}` | {row.get('rmse')} | {row.get('mae')} | {row.get('bias')} | "
                f"{row.get('gate_share')} | {row.get('alpha')} | {row.get('clip_ms')} |"
            )
        lines.extend([
            "",
            "Best holdout variants:",
            "",
            "| Variant | RMSE | MAE | Bias | Gate share | Alpha | Clip |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in target["best_holdout_variants"][:10]:
            lines.append(
                f"| `{row['name']}` | {row.get('rmse')} | {row.get('mae')} | {row.get('bias')} | "
                f"{row.get('gate_share')} | {row.get('alpha')} | {row.get('clip_ms')} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-table-root", type=Path, default=Path("/srv/data/corsewind/ml_dataset/training_tables"))
    parser.add_argument("--vnext-run-id-prefix", default="residual_windsup_sst_prev_vnext")
    parser.add_argument("--vnext-benchmark-root", type=Path, default=Path("/srv/data/corsewind/ml_dataset/benchmarks/vnext_2025h2_to_2026h1"))
    parser.add_argument(
        "--wind-champion-predictions",
        type=Path,
        default=Path("/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1/calibrated_predictions_2026.parquet"),
    )
    parser.add_argument(
        "--gust-champion-predictions",
        type=Path,
        default=Path("/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_gust_from_wind_champion_recipe_v1/calibrated_predictions_2026.parquet"),
    )
    parser.add_argument("--start-month", default="2026-01")
    parser.add_argument("--end-month", default="2026-06")
    parser.add_argument("--target", choices=sorted(TARGETS), action="append", default=[])
    parser.add_argument("--alpha", type=float, action="append", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--clip-ms", type=float, action="append", default=list(DEFAULT_CLIPS_MS))
    parser.add_argument("--threshold-kt", type=float, action="append", default=list(DEFAULT_THRESHOLDS_KT))
    parser.add_argument("--hotspot", action="append", default=list(DEFAULT_HOTSPOTS))
    parser.add_argument("--blend-selection-split-utc", default="2026-04-01T00:00:00Z")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--write-tables", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.target = args.target or ["wind_mean", "gust"]
    args.alpha = sorted({float(value) for value in args.alpha})
    args.clip_ms = sorted({float(value) for value in args.clip_ms})
    args.threshold_kt = sorted({float(value) for value in args.threshold_kt})
    args.hotspot = sorted({str(value) for value in args.hotspot})
    args.output_root.mkdir(parents=True, exist_ok=True)
    deps = import_deps()
    targets = [run_target(args, target, deps) for target in args.target]
    result = {
        "format": "corsewind.vnext_champion_blend_benchmark.v1",
        "generated_at_utc": utc_now(),
        "blend_selection_split_utc": args.blend_selection_split_utc,
        "settings": {
            "alphas": args.alpha,
            "clip_ms": args.clip_ms,
            "threshold_kt": args.threshold_kt,
            "hotspots": args.hotspot,
            "start_month": args.start_month,
            "end_month": args.end_month,
        },
        "targets": targets,
    }
    output_json = args.output_json or args.output_root / "vnext_champion_blend_results.json"
    output_md = args.output_md or args.output_root / "vnext_champion_blend_results.md"
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_markdown(output_md, result)
    print(json.dumps({
        "output_json": str(output_json),
        "output_md": str(output_md),
        "targets": {
            item["target"]: {
                "rows": item["row_count"],
                "champion_rmse": item["baseline_metrics"]["champion"].get("rmse"),
                "vnext_rmse": item["baseline_metrics"]["vnext"].get("rmse"),
                "oracle_rmse": (item["baseline_metrics"].get("pair_oracle") or {}).get("rmse"),
                "selected_holdout_rmse": (item.get("selected_holdout") or {}).get("rmse"),
                "selected_variant": (item.get("selected_by_calibration") or {}).get("name"),
                "verdict": item["promotion_verdict"],
            }
            for item in targets
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
