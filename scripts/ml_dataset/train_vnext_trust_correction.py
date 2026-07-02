#!/usr/bin/env python3
"""Train constrained trust/correction models using v_next as a champion expert."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KNOTS_PER_MS = 1.9438444924406
KEY_COLUMNS = ["spot_id", "issue_time_utc", "target_time_utc", "lead_time_minutes"]
TARGETS = {
    "wind_mean": {
        "champion_path": "/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1/calibrated_predictions_2026.parquet",
        "champion_col": "calibrated_wind_mean_ms",
        "raw_col": "raw_wind_mean_ms",
        "actual_col": "actual_wind_mean_ms",
        "baseline_col": "baselines__baseline_wind_mean_ms",
        "label_col": "labels__target_wind_mean_ms",
        "model_dir": "lgbm_wind",
        "model_file": "labels__residual_wind_mean_ms.joblib",
        "official_gate_rmse": 1.268019,
    },
    "gust": {
        "champion_path": "/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_gust_from_wind_champion_recipe_v1/calibrated_predictions_2026.parquet",
        "champion_col": "calibrated_gust_ms",
        "raw_col": "raw_gust_ms",
        "actual_col": "actual_gust_ms",
        "baseline_col": "baselines__baseline_gust_ms",
        "label_col": "labels__target_gust_ms",
        "model_dir": "lgbm_gust",
        "model_file": "labels__residual_gust_ms.joblib",
        "official_gate_rmse": 1.484221,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def import_deps() -> dict[str, Any]:
    try:
        import joblib
        import lightgbm as lgb
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OrdinalEncoder
    except ImportError as exc:
        raise SystemExit("Missing ML dependencies.") from exc
    return {
        "joblib": joblib,
        "lgb": lgb,
        "np": np,
        "pd": pd,
        "ColumnTransformer": ColumnTransformer,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "ExtraTreesRegressor": ExtraTreesRegressor,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "SimpleImputer": SimpleImputer,
        "Pipeline": Pipeline,
        "OrdinalEncoder": OrdinalEncoder,
    }


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def month_range(start_month: str, end_month: str) -> list[str]:
    sy, sm = [int(x) for x in start_month.split("-", 1)]
    ey, em = [int(x) for x in end_month.split("-", 1)]
    out = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}_{m:02d}")
        m += 1
        if m == 13:
            y += 1
            m = 1
    return out


def load_feature_columns(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return unique(list(data.get("numeric") or []) + list(data.get("categorical") or []))


def add_time_features(frame: Any, pd: Any, np: Any) -> Any:
    issue = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    dayofyear = issue.dt.dayofyear.fillna(1).astype(float)
    frame["issue_hour_utc"] = issue.dt.hour.astype("float64")
    frame["issue_month"] = issue.dt.month.astype("float64")
    angle = 2.0 * math.pi * dayofyear / 366.0
    frame["issue_dayofyear_sin"] = np.sin(angle)
    frame["issue_dayofyear_cos"] = np.cos(angle)
    return frame


def load_champion(path: Path, config: dict[str, Any], pd: Any) -> Any:
    frame = pd.read_parquet(path)
    frame["issue_time_utc"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    if "target_time_utc" in frame.columns:
        frame["target_time_utc"] = pd.to_datetime(frame["target_time_utc"], utc=True, errors="coerce")
    else:
        frame["target_time_utc"] = frame["issue_time_utc"] + pd.to_timedelta(frame["lead_time_minutes"].astype(float), unit="m")
    keep = unique(KEY_COLUMNS + ["station_id", "spot_kind", "latitude", "longitude", config["champion_col"], config["raw_col"], config["actual_col"]])
    out = frame.reindex(columns=keep).rename(columns={
        config["champion_col"]: "champion_prediction_ms",
        config["raw_col"]: "raw_prediction_ms",
        config["actual_col"]: "actual_ms",
    })
    return out.drop_duplicates(KEY_COLUMNS)


def build_training_frame(args: argparse.Namespace, target: str, deps: dict[str, Any]) -> tuple[Any, list[str]]:
    pd = deps["pd"]
    np = deps["np"]
    joblib = deps["joblib"]
    config = TARGETS[target]
    model_root = args.vnext_benchmark_root / config["model_dir"]
    vnext_model = joblib.load(model_root / config["model_file"])
    vnext_features = load_feature_columns(model_root / "feature_columns.json")
    champion = load_champion(Path(config["champion_path"]), config, pd)
    rows = []
    for month in month_range(args.start_month, args.end_month):
        path = args.training_table_root / f"{args.vnext_run_id_prefix}_{month}" / "training_rows.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame = add_time_features(frame, pd, np)
        required = unique(KEY_COLUMNS + ["station_id", "spot_kind", "latitude", "longitude", config["baseline_col"], config["label_col"]] + vnext_features)
        frame = frame.reindex(columns=required)
        residual = vnext_model.predict(frame.reindex(columns=vnext_features))
        out = frame[KEY_COLUMNS + ["station_id", "spot_kind", "latitude", "longitude"]].copy()
        out["vnext_prediction_ms"] = frame[config["baseline_col"]].astype(float).to_numpy() + residual
        out["vnext_raw_ms"] = frame[config["baseline_col"]].astype(float)
        out["vnext_actual_ms"] = frame[config["label_col"]].astype(float)
        feature_block = frame[vnext_features].copy()
        feature_block.columns = [f"vnextfeat__{col}" for col in feature_block.columns]
        rows.append(pd.concat([out, feature_block], axis=1))
    if not rows:
        raise SystemExit("No v_next rows found.")
    vnext = pd.concat(rows, ignore_index=True)
    vnext["issue_time_utc"] = pd.to_datetime(vnext["issue_time_utc"], utc=True, errors="coerce")
    vnext["target_time_utc"] = pd.to_datetime(vnext["target_time_utc"], utc=True, errors="coerce")
    vnext = vnext.drop_duplicates(KEY_COLUMNS)
    merged = champion.merge(vnext, on=KEY_COLUMNS, how="inner", validate="one_to_one", suffixes=("", "_vnext"))
    merged["actual_diff_ms"] = merged["actual_ms"].astype(float) - merged["vnext_actual_ms"].astype(float)
    merged["champion_error_ms"] = merged["champion_prediction_ms"].astype(float) - merged["actual_ms"].astype(float)
    merged["vnext_error_ms"] = merged["vnext_prediction_ms"].astype(float) - merged["actual_ms"].astype(float)
    merged["champion_abs_error_ms"] = merged["champion_error_ms"].abs()
    merged["vnext_abs_error_ms"] = merged["vnext_error_ms"].abs()
    merged["target_residual_vs_champion_ms"] = merged["actual_ms"].astype(float) - merged["champion_prediction_ms"].astype(float)
    merged["vnext_delta_vs_champion_ms"] = merged["vnext_prediction_ms"].astype(float) - merged["champion_prediction_ms"].astype(float)
    merged["raw_delta_vs_champion_ms"] = merged["raw_prediction_ms"].astype(float) - merged["champion_prediction_ms"].astype(float)
    merged["prediction_spread_ms"] = merged["vnext_delta_vs_champion_ms"].abs()
    merged["champion_kt"] = merged["champion_prediction_ms"].astype(float) * KNOTS_PER_MS
    merged["raw_kt"] = merged["raw_prediction_ms"].astype(float) * KNOTS_PER_MS
    merged["vnext_kt"] = merged["vnext_prediction_ms"].astype(float) * KNOTS_PER_MS
    merged["actual_kt"] = merged["actual_ms"].astype(float) * KNOTS_PER_MS
    merged["vnext_beats_champion"] = merged["vnext_abs_error_ms"] < merged["champion_abs_error_ms"]
    merged["vnext_beats_champion_margin_025"] = (merged["champion_abs_error_ms"] - merged["vnext_abs_error_ms"]) >= 0.25
    feature_columns = unique([
        "spot_id",
        "station_id",
        "spot_kind",
        "latitude",
        "longitude",
        "lead_time_minutes",
        "champion_prediction_ms",
        "raw_prediction_ms",
        "vnext_prediction_ms",
        "vnext_raw_ms",
        "vnext_delta_vs_champion_ms",
        "raw_delta_vs_champion_ms",
        "prediction_spread_ms",
        "champion_kt",
        "raw_kt",
        "vnext_kt",
    ] + [f"vnextfeat__{col}" for col in vnext_features])
    return merged, feature_columns


def split_frame(frame: Any, args: argparse.Namespace, pd: Any) -> tuple[Any, Any, Any]:
    train_end = pd.to_datetime(args.train_end_utc, utc=True)
    validation_end = pd.to_datetime(args.validation_end_utc, utc=True)
    train = frame[frame["issue_time_utc"] < train_end].copy()
    validation = frame[(frame["issue_time_utc"] >= train_end) & (frame["issue_time_utc"] < validation_end)].copy()
    holdout = frame[frame["issue_time_utc"] >= validation_end].copy()
    return train, validation, holdout


def infer_feature_types(frame: Any, feature_columns: list[str], pd: Any, max_cardinality: int) -> tuple[list[str], list[str], dict[str, Any]]:
    numeric: list[str] = []
    categorical: list[str] = []
    dropped: dict[str, Any] = {"sparse": [], "constant": [], "high_cardinality": []}
    for col in feature_columns:
        if col not in frame.columns:
            dropped["sparse"].append({"column": col, "reason": "missing"})
            continue
        non_null = int(frame[col].notna().sum())
        if non_null < 100:
            dropped["sparse"].append({"column": col, "non_null": non_null})
            continue
        unique_count = int(frame[col].dropna().nunique())
        if unique_count <= 1:
            dropped["constant"].append({"column": col})
            continue
        if col in {"spot_id", "station_id", "spot_kind"} or col.endswith("_station_id"):
            if unique_count > max_cardinality:
                dropped["high_cardinality"].append({"column": col, "unique_count": unique_count})
            else:
                categorical.append(col)
            continue
        sample = frame[col].dropna().head(1000)
        converted = pd.to_numeric(sample, errors="coerce")
        if len(sample) and converted.notna().all():
            numeric.append(col)
        elif unique_count <= max_cardinality:
            categorical.append(col)
        else:
            dropped["high_cardinality"].append({"column": col, "unique_count": unique_count})
    return numeric, categorical, dropped


def make_preprocessor(deps: dict[str, Any], numeric: list[str], categorical: list[str]) -> Any:
    transformers = []
    if numeric:
        transformers.append(("numeric", deps["Pipeline"]([("imputer", deps["SimpleImputer"](strategy="median"))]), numeric))
    if categorical:
        transformers.append((
            "categorical",
            deps["Pipeline"]([
                ("imputer", deps["SimpleImputer"](strategy="constant", fill_value="__missing__")),
                ("ordinal", deps["OrdinalEncoder"](handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            categorical,
        ))
    return deps["ColumnTransformer"](transformers=transformers, remainder="drop")


def build_regressor(args: argparse.Namespace, deps: dict[str, Any], numeric: list[str], categorical: list[str], family: str) -> Any:
    if family == "lightgbm":
        model = deps["lgb"].LGBMRegressor(
            n_estimators=args.max_iter,
            learning_rate=args.learning_rate,
            num_leaves=args.max_leaf_nodes,
            min_child_samples=args.min_samples_leaf,
            reg_lambda=args.l2_regularization,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
            verbose=-1,
        )
    elif family == "extra_trees":
        model = deps["ExtraTreesRegressor"](
            n_estimators=args.max_iter,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
        )
    else:
        model = deps["HistGradientBoostingRegressor"](
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf,
            l2_regularization=args.l2_regularization,
            random_state=args.random_seed,
        )
    return deps["Pipeline"]([("preprocess", make_preprocessor(deps, numeric, categorical)), ("model", model)])


def build_classifier(args: argparse.Namespace, deps: dict[str, Any], numeric: list[str], categorical: list[str], family: str) -> Any:
    if family == "lightgbm":
        model = deps["lgb"].LGBMClassifier(
            n_estimators=args.max_iter,
            learning_rate=args.learning_rate,
            num_leaves=args.max_leaf_nodes,
            min_child_samples=args.min_samples_leaf,
            reg_lambda=args.l2_regularization,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
            class_weight="balanced",
            verbose=-1,
        )
    elif family == "extra_trees":
        model = deps["ExtraTreesClassifier"](
            n_estimators=args.max_iter,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
            class_weight="balanced",
        )
    else:
        model = deps["HistGradientBoostingClassifier"](
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf,
            l2_regularization=args.l2_regularization,
            random_state=args.random_seed,
        )
    return deps["Pipeline"]([("preprocess", make_preprocessor(deps, numeric, categorical)), ("model", model)])


def metrics(frame: Any, pred_col: str, np: Any) -> dict[str, Any]:
    pred = frame[pred_col].astype(float).to_numpy()
    obs = frame["actual_ms"].astype(float).to_numpy()
    valid = np.isfinite(pred) & np.isfinite(obs)
    if not valid.any():
        return {"count": 0}
    err = pred[valid] - obs[valid]
    return {
        "count": int(err.size),
        "rmse": round(float(math.sqrt(float(np.mean(err * err)))), 6),
        "mae": round(float(np.mean(np.abs(err))), 6),
        "bias": round(float(np.mean(err)), 6),
        "p90_abs_error": round(float(np.quantile(np.abs(err), 0.90)), 6),
        "p95_abs_error": round(float(np.quantile(np.abs(err), 0.95)), 6),
    }


def threshold_detection(frame: Any, pred_col: str, threshold_kt: float) -> dict[str, Any]:
    valid = frame[[pred_col, "actual_ms"]].dropna()
    if valid.empty:
        return {"count": 0}
    threshold_ms = threshold_kt / KNOTS_PER_MS
    pred = valid[pred_col].astype(float) >= threshold_ms
    actual = valid["actual_ms"].astype(float) >= threshold_ms
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    return {
        "count": int(len(valid)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": None if tp + fp == 0 else round(float(tp / (tp + fp)), 6),
        "recall": None if tp + fn == 0 else round(float(tp / (tp + fn)), 6),
        "csi": None if tp + fp + fn == 0 else round(float(tp / (tp + fp + fn)), 6),
    }


def evaluate_variant(frame: Any, pred_col: str, np: Any, thresholds: list[float]) -> dict[str, Any]:
    return {
        "metrics": metrics(frame, pred_col, np),
        "by_threshold_detection": {f">={t:g}kt": threshold_detection(frame, pred_col, t) for t in thresholds},
        "by_observed_threshold": {
            f">={t:g}kt": metrics(frame[frame["actual_kt"] >= t], pred_col, np)
            for t in thresholds
        },
    }


def select_best(validation_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [row for row in validation_rows if row.get("rmse") is not None]
    return None if not rows else min(rows, key=lambda row: (float(row["rmse"]), float(row.get("mae") or 999.0)))


def build_candidate_table(
    validation_candidates: list[dict[str, Any]],
    holdout_predictions: dict[str, Any],
    full_predictions: dict[str, Any],
    validation: Any,
    holdout: Any,
    frame: Any,
    np: Any,
) -> list[dict[str, Any]]:
    rows = []
    for candidate in validation_candidates:
        name = candidate["name"]
        if name not in holdout_predictions or name not in full_predictions:
            continue
        hold = summarize_prediction(holdout, name, holdout_predictions[name], np)
        full = summarize_prediction(frame, name, full_predictions[name], np)
        rows.append({
            "name": name,
            "candidate_type": candidate.get("candidate_type"),
            "family": candidate.get("family"),
            "scale": candidate.get("scale"),
            "alpha": candidate.get("alpha"),
            "clip_ms": candidate.get("clip_ms"),
            "max_lead_minutes": candidate.get("max_lead_minutes"),
            "probability_threshold": candidate.get("probability_threshold"),
            "validation": {
                key: candidate.get(key)
                for key in ("count", "rmse", "mae", "bias", "p90_abs_error", "p95_abs_error")
            },
            "holdout": {
                key: hold.get(key)
                for key in ("count", "rmse", "mae", "bias", "p90_abs_error", "p95_abs_error")
            },
            "full": {
                key: full.get(key)
                for key in ("count", "rmse", "mae", "bias", "p90_abs_error", "p95_abs_error")
            },
        })
    rows.sort(key=lambda row: (
        row["validation"].get("rmse") is None,
        row["validation"].get("rmse", 999.0),
        row["holdout"].get("rmse", 999.0),
    ))
    return rows


def summarize_prediction(frame: Any, name: str, prediction: Any, np: Any) -> dict[str, Any]:
    work = frame[["actual_ms"]].copy()
    work["pred"] = prediction
    item = metrics(work.rename(columns={"pred": name}), name, np)
    return {"name": name, **item}


def fit_and_score(args: argparse.Namespace, target: str, deps: dict[str, Any]) -> dict[str, Any]:
    pd = deps["pd"]
    np = deps["np"]
    frame, feature_candidates = build_training_frame(args, target, deps)
    if args.feature_mode == "prediction_only":
        feature_candidates = [col for col in feature_candidates if not col.startswith("vnextfeat__")]
    train, validation, holdout = split_frame(frame, args, pd)
    numeric, categorical, dropped = infer_feature_types(train, feature_candidates, pd, args.max_categorical_cardinality)
    feature_columns = numeric + categorical
    if len(train) < 1000 or len(validation) < 1000 or len(holdout) < 1000:
        raise SystemExit(f"Not enough split rows for {target}: train={len(train)} val={len(validation)} holdout={len(holdout)}")
    baseline_cols = {
        "champion": "champion_prediction_ms",
        "raw": "raw_prediction_ms",
        "vnext": "vnext_prediction_ms",
    }
    validation_candidates = []
    holdout_predictions: dict[str, Any] = {}
    full_predictions: dict[str, Any] = {}
    trained_regressors: dict[str, Any] = {}

    for alpha in args.blend_alpha:
        for clip_ms in args.vnext_delta_clip_ms:
            name = f"static_vnext_blend_a{alpha:g}_clip{clip_ms:g}"
            val_delta = validation["vnext_delta_vs_champion_ms"].astype(float).clip(-clip_ms, clip_ms).to_numpy()
            hold_delta = holdout["vnext_delta_vs_champion_ms"].astype(float).clip(-clip_ms, clip_ms).to_numpy()
            full_delta = frame["vnext_delta_vs_champion_ms"].astype(float).clip(-clip_ms, clip_ms).to_numpy()
            val_pred = validation["champion_prediction_ms"].astype(float).to_numpy() + alpha * val_delta
            hold_pred = holdout["champion_prediction_ms"].astype(float).to_numpy() + alpha * hold_delta
            full_pred = frame["champion_prediction_ms"].astype(float).to_numpy() + alpha * full_delta
            validation_candidates.append({
                "candidate_type": "static_vnext_blend",
                "alpha": float(alpha),
                "clip_ms": float(clip_ms),
                **summarize_prediction(validation, name, val_pred, np),
            })
            holdout_predictions[name] = hold_pred
            full_predictions[name] = full_pred

    for family in args.model_family:
        reg = build_regressor(args, deps, numeric, categorical, family)
        reg.fit(train[feature_columns], train["target_residual_vs_champion_ms"].astype(float))
        trained_regressors[family] = reg
        val_corr = reg.predict(validation[feature_columns])
        hold_corr = reg.predict(holdout[feature_columns])
        full_corr = reg.predict(frame[feature_columns])
        for scale in args.regression_scale:
            for clip_ms in args.correction_clip_ms:
                name = f"residual_{family}_scale{scale:g}_clip{clip_ms:g}"
                val_pred = validation["champion_prediction_ms"].astype(float).to_numpy() + np.clip(val_corr, -clip_ms, clip_ms) * scale
                hold_pred = holdout["champion_prediction_ms"].astype(float).to_numpy() + np.clip(hold_corr, -clip_ms, clip_ms) * scale
                full_pred = frame["champion_prediction_ms"].astype(float).to_numpy() + np.clip(full_corr, -clip_ms, clip_ms) * scale
                validation_candidates.append({
                    "candidate_type": "residual_regressor",
                    "family": family,
                    "scale": float(scale),
                    "clip_ms": float(clip_ms),
                    **summarize_prediction(validation, name, val_pred, np),
                })
                holdout_predictions[name] = hold_pred
                full_predictions[name] = full_pred
                for max_lead in args.residual_max_lead_minutes:
                    lead_name = f"residual_{family}_scale{scale:g}_clip{clip_ms:g}_leadlte{max_lead:g}"
                    val_gate = validation["lead_time_minutes"].astype(float).to_numpy() <= max_lead
                    hold_gate = holdout["lead_time_minutes"].astype(float).to_numpy() <= max_lead
                    full_gate = frame["lead_time_minutes"].astype(float).to_numpy() <= max_lead
                    val_lead_pred = validation["champion_prediction_ms"].astype(float).to_numpy() + val_gate.astype(float) * np.clip(val_corr, -clip_ms, clip_ms) * scale
                    hold_lead_pred = holdout["champion_prediction_ms"].astype(float).to_numpy() + hold_gate.astype(float) * np.clip(hold_corr, -clip_ms, clip_ms) * scale
                    full_lead_pred = frame["champion_prediction_ms"].astype(float).to_numpy() + full_gate.astype(float) * np.clip(full_corr, -clip_ms, clip_ms) * scale
                    validation_candidates.append({
                        "candidate_type": "residual_regressor_lead_gate",
                        "family": family,
                        "scale": float(scale),
                        "clip_ms": float(clip_ms),
                        "max_lead_minutes": float(max_lead),
                        **summarize_prediction(validation, lead_name, val_lead_pred, np),
                    })
                    holdout_predictions[lead_name] = hold_lead_pred
                    full_predictions[lead_name] = full_lead_pred

    for family in args.model_family:
        clf = build_classifier(args, deps, numeric, categorical, family)
        y = train["vnext_beats_champion_margin_025"].astype(int)
        if y.nunique() < 2:
            continue
        clf.fit(train[feature_columns], y)
        val_proba = clf.predict_proba(validation[feature_columns])[:, 1]
        hold_proba = clf.predict_proba(holdout[feature_columns])[:, 1]
        full_proba = clf.predict_proba(frame[feature_columns])[:, 1]
        for threshold in args.probability_threshold:
            for alpha in args.blend_alpha:
                for clip_ms in args.vnext_delta_clip_ms:
                    name = f"trust_{family}_p{threshold:g}_a{alpha:g}_clip{clip_ms:g}"
                    val_gate = val_proba >= threshold
                    hold_gate = hold_proba >= threshold
                    full_gate = full_proba >= threshold
                    val_delta = validation["vnext_delta_vs_champion_ms"].astype(float).clip(-clip_ms, clip_ms).to_numpy()
                    hold_delta = holdout["vnext_delta_vs_champion_ms"].astype(float).clip(-clip_ms, clip_ms).to_numpy()
                    full_delta = frame["vnext_delta_vs_champion_ms"].astype(float).clip(-clip_ms, clip_ms).to_numpy()
                    val_pred = validation["champion_prediction_ms"].astype(float).to_numpy() + val_gate.astype(float) * alpha * val_delta
                    hold_pred = holdout["champion_prediction_ms"].astype(float).to_numpy() + hold_gate.astype(float) * alpha * hold_delta
                    full_pred = frame["champion_prediction_ms"].astype(float).to_numpy() + full_gate.astype(float) * alpha * full_delta
                    validation_candidates.append({
                        "candidate_type": "trust_classifier",
                        "family": family,
                        "probability_threshold": float(threshold),
                        "alpha": float(alpha),
                        "clip_ms": float(clip_ms),
                        "gate_rate_validation": round(float(np.mean(val_gate)), 6),
                        "gate_rate_holdout": round(float(np.mean(hold_gate)), 6),
                        **summarize_prediction(validation, name, val_pred, np),
                    })
                    holdout_predictions[name] = hold_pred
                    full_predictions[name] = full_pred

    best = select_best(validation_candidates)
    selected_holdout = None
    selected_full = None
    detailed_selected = None
    if best:
        name = best["name"]
        selected_holdout = summarize_prediction(holdout, name, holdout_predictions[name], np)
        selected_full = summarize_prediction(frame, name, full_predictions[name], np)
        full_eval = frame.copy()
        full_eval["selected_prediction_ms"] = full_predictions[name]
        detailed_selected = evaluate_variant(full_eval, "selected_prediction_ms", np, args.threshold_kt)
    validation_candidates.sort(key=lambda row: (row.get("rmse") is None, row.get("rmse", 999.0)))
    official_gate = TARGETS[target]["official_gate_rmse"]
    champion_validation_rmse = metrics(validation.assign(_pred=validation["champion_prediction_ms"]), "_pred", np).get("rmse")
    champion_holdout_rmse = metrics(holdout.assign(_pred=holdout["champion_prediction_ms"]), "_pred", np).get("rmse")
    champion_full_rmse = metrics(frame.assign(_pred=frame["champion_prediction_ms"]), "_pred", np).get("rmse")
    selected_validation_rmse = best.get("rmse") if best else None
    selected_holdout_rmse = selected_holdout.get("rmse") if selected_holdout else None
    selected_full_rmse = selected_full.get("rmse") if selected_full else None
    reliable = (
        selected_validation_rmse is not None
        and selected_holdout_rmse is not None
        and selected_full_rmse is not None
        and champion_validation_rmse is not None
        and champion_holdout_rmse is not None
        and champion_full_rmse is not None
        and float(selected_validation_rmse) < float(champion_validation_rmse)
        and float(selected_holdout_rmse) < float(champion_holdout_rmse)
        and float(selected_full_rmse) < float(champion_full_rmse)
        and float(selected_full_rmse) < official_gate
    )
    candidate_table = build_candidate_table(validation_candidates, holdout_predictions, full_predictions, validation, holdout, frame, np)
    strict_candidates = [
        row for row in candidate_table
        if row["validation"].get("rmse") is not None
        and row["holdout"].get("rmse") is not None
        and row["full"].get("rmse") is not None
        and float(row["validation"]["rmse"]) < float(champion_validation_rmse)
        and float(row["holdout"]["rmse"]) < float(champion_holdout_rmse)
        and float(row["full"]["rmse"]) < float(champion_full_rmse)
        and float(row["full"]["rmse"]) < official_gate
    ]
    strict_candidates.sort(key=lambda row: (row["full"]["rmse"], row["holdout"]["rmse"]))
    saved_artifact = None
    if args.save_best_strict_artifact and strict_candidates:
        best_strict = strict_candidates[0]
        family = best_strict.get("family")
        if best_strict.get("candidate_type") in {"residual_regressor", "residual_regressor_lead_gate"} and family in trained_regressors:
            artifact_dir = args.output_root / f"{target}_{family}_best_strict"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            model_path = artifact_dir / "residual_model.joblib"
            metadata_path = artifact_dir / "metadata.json"
            deps["joblib"].dump(trained_regressors[family], model_path)
            metadata = {
                "format": "corsewind.vnext_trust_correction_artifact.v1",
                "generated_at_utc": utc_now(),
                "target": target,
                "candidate": best_strict,
                "feature_mode": args.feature_mode,
                "feature_columns": feature_columns,
                "model_path": str(model_path),
                "training_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "holdout_rows": int(len(holdout)),
                "train_end_utc": args.train_end_utc,
                "validation_end_utc": args.validation_end_utc,
                "champion_path": TARGETS[target]["champion_path"],
                "vnext_benchmark_root": str(args.vnext_benchmark_root),
                "vnext_model_dir": TARGETS[target]["model_dir"],
            }
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
            saved_artifact = {
                "artifact_dir": str(artifact_dir),
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
            }
    return {
        "target": target,
        "row_count": int(len(frame)),
        "split_rows": {"train": int(len(train)), "validation": int(len(validation)), "holdout": int(len(holdout))},
        "actual_max_abs_diff_ms": None if frame.empty else round(float(frame["actual_diff_ms"].abs().max()), 9),
        "feature_counts": {"numeric": len(numeric), "categorical": len(categorical), "total": len(feature_columns)},
        "dropped_features": dropped,
        "baselines": {
            name: {
                "full": evaluate_variant(frame.assign(_pred=frame[col]), "_pred", np, args.threshold_kt),
                "validation": evaluate_variant(validation.assign(_pred=validation[col]), "_pred", np, args.threshold_kt),
                "holdout": evaluate_variant(holdout.assign(_pred=holdout[col]), "_pred", np, args.threshold_kt),
            }
            for name, col in baseline_cols.items()
        },
        "oracle": {
            "full": summarize_prediction(
                frame,
                "oracle",
                np.where(
                    frame["champion_abs_error_ms"].to_numpy() <= frame["vnext_abs_error_ms"].to_numpy(),
                    frame["champion_prediction_ms"].to_numpy(),
                    frame["vnext_prediction_ms"].to_numpy(),
                ),
                np,
            )
        },
        "best_validation_candidates": validation_candidates[:25],
        "best_strict_candidates": strict_candidates[:25],
        "candidate_table_top_validation": candidate_table[:50],
        "selected_by_validation": best,
        "selected_holdout": selected_holdout,
        "selected_full": selected_full,
        "selected_gain_vs_champion": {
            "validation_rmse_delta": None if selected_validation_rmse is None or champion_validation_rmse is None else round(float(selected_validation_rmse) - float(champion_validation_rmse), 6),
            "holdout_rmse_delta": None if selected_holdout_rmse is None or champion_holdout_rmse is None else round(float(selected_holdout_rmse) - float(champion_holdout_rmse), 6),
            "full_rmse_delta": None if selected_full_rmse is None or champion_full_rmse is None else round(float(selected_full_rmse) - float(champion_full_rmse), 6),
        },
        "selected_detailed_full": detailed_selected,
        "official_gate_rmse": official_gate,
        "promotion_verdict": "reliable_candidate" if reliable or strict_candidates else "do_not_promote",
        "saved_artifact": saved_artifact,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# v_next Trust Correction Benchmark",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Train end: `{result['settings']['train_end_utc']}`",
        f"Validation end / holdout start: `{result['settings']['validation_end_utc']}`",
        "",
        "| Target | Rows | Champion full RMSE | v_next full RMSE | Oracle RMSE | Selected validation RMSE | Selected holdout RMSE | Selected full RMSE | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in result["targets"]:
        champ = item["baselines"]["champion"]["full"]["metrics"]
        vnext = item["baselines"]["vnext"]["full"]["metrics"]
        oracle = item["oracle"]["full"]
        selected_val = item.get("selected_by_validation") or {}
        selected_hold = item.get("selected_holdout") or {}
        selected_full = item.get("selected_full") or {}
        lines.append(
            f"| `{item['target']}` | {item['row_count']} | {champ.get('rmse')} | {vnext.get('rmse')} | "
            f"{oracle.get('rmse')} | {selected_val.get('rmse')} | {selected_hold.get('rmse')} | "
            f"{selected_full.get('rmse')} | `{item['promotion_verdict']}` |"
        )
    for item in result["targets"]:
        selected = item.get("selected_by_validation") or {}
        hold = item.get("selected_holdout") or {}
        full = item.get("selected_full") or {}
        lines.extend([
            "",
            f"## `{item['target']}`",
            "",
            f"- split rows: `{item['split_rows']}`",
            f"- feature counts: `{item['feature_counts']}`",
            f"- selected candidate: `{selected.get('name')}`",
            f"- selected candidate type: `{selected.get('candidate_type')}`",
            f"- validation RMSE: `{selected.get('rmse')}`",
            f"- holdout RMSE: `{hold.get('rmse')}`",
            f"- full RMSE: `{full.get('rmse')}`",
            f"- deltas vs champion: `{item.get('selected_gain_vs_champion')}`",
            f"- official gate: `{item['official_gate_rmse']}`",
            "",
            "Top validation candidates:",
            "",
            "| Candidate | Type | RMSE | MAE | Bias |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for row in item["best_validation_candidates"][:10]:
            lines.append(
                f"| `{row.get('name')}` | `{row.get('candidate_type')}` | {row.get('rmse')} | {row.get('mae')} | {row.get('bias')} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-table-root", type=Path, default=Path("/srv/data/corsewind/ml_dataset/training_tables"))
    parser.add_argument("--vnext-run-id-prefix", default="residual_windsup_sst_prev_vnext")
    parser.add_argument("--vnext-benchmark-root", type=Path, default=Path("/srv/data/corsewind/ml_dataset/benchmarks/vnext_2025h2_to_2026h1"))
    parser.add_argument("--start-month", default="2026-01")
    parser.add_argument("--end-month", default="2026-06")
    parser.add_argument("--target", choices=sorted(TARGETS), action="append", default=[])
    parser.add_argument("--train-end-utc", default="2026-03-01T00:00:00Z")
    parser.add_argument("--validation-end-utc", default="2026-04-01T00:00:00Z")
    parser.add_argument("--model-family", choices=("hist_gradient_boosting", "extra_trees", "lightgbm"), action="append", default=[])
    parser.add_argument("--max-iter", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    parser.add_argument("--l2-regularization", type=float, default=0.1)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-categorical-cardinality", type=int, default=100)
    parser.add_argument("--feature-mode", choices=("full", "prediction_only"), default="full")
    parser.add_argument("--regression-scale", type=float, action="append", default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    parser.add_argument("--correction-clip-ms", type=float, action="append", default=[0.25, 0.50, 0.75, 1.00, 1.50, 2.00])
    parser.add_argument("--residual-max-lead-minutes", type=float, action="append", default=[])
    parser.add_argument("--blend-alpha", type=float, action="append", default=[0.10, 0.20, 0.30, 0.40])
    parser.add_argument("--vnext-delta-clip-ms", type=float, action="append", default=[0.50, 1.00, 2.00])
    parser.add_argument("--probability-threshold", type=float, action="append", default=[0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
    parser.add_argument("--threshold-kt", type=float, action="append", default=[12.0, 15.0, 20.0, 25.0])
    parser.add_argument("--save-best-strict-artifact", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.target = args.target or ["wind_mean", "gust"]
    args.model_family = unique(args.model_family) or ["hist_gradient_boosting"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    deps = import_deps()
    warnings.filterwarnings("ignore", category=deps["pd"].errors.PerformanceWarning)
    targets = [fit_and_score(args, target, deps) for target in args.target]
    result = {
        "format": "corsewind.vnext_trust_correction.v1",
        "generated_at_utc": utc_now(),
        "settings": {
            "start_month": args.start_month,
            "end_month": args.end_month,
            "train_end_utc": args.train_end_utc,
            "validation_end_utc": args.validation_end_utc,
            "model_family": args.model_family,
            "feature_mode": args.feature_mode,
            "regression_scale": sorted(set(args.regression_scale)),
            "correction_clip_ms": sorted(set(args.correction_clip_ms)),
            "residual_max_lead_minutes": sorted(set(args.residual_max_lead_minutes)),
            "blend_alpha": sorted(set(args.blend_alpha)),
            "vnext_delta_clip_ms": sorted(set(args.vnext_delta_clip_ms)),
            "probability_threshold": sorted(set(args.probability_threshold)),
        },
        "targets": targets,
    }
    out_json = args.output_json or args.output_root / "vnext_trust_correction_results.json"
    out_md = args.output_md or args.output_root / "vnext_trust_correction_results.md"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_markdown(out_md, result)
    print(json.dumps({
        "output_json": str(out_json),
        "output_md": str(out_md),
        "targets": {
            item["target"]: {
                "champion_full_rmse": item["baselines"]["champion"]["full"]["metrics"].get("rmse"),
                "vnext_full_rmse": item["baselines"]["vnext"]["full"]["metrics"].get("rmse"),
                "oracle_rmse": item["oracle"]["full"].get("rmse"),
                "selected": (item.get("selected_by_validation") or {}).get("name"),
                "selected_validation_rmse": (item.get("selected_by_validation") or {}).get("rmse"),
                "selected_holdout_rmse": (item.get("selected_holdout") or {}).get("rmse"),
                "selected_full_rmse": (item.get("selected_full") or {}).get("rmse"),
                "verdict": item["promotion_verdict"],
            }
            for item in targets
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
