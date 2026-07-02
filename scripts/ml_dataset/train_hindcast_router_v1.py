#!/usr/bin/env python3
"""Train and evaluate a hindcast mixture/router on scored pseudo-live rows.

The goal is not to replace the median champion directly. This script measures
whether issue-time information can decide when to trust each available rail:
champion, raw NWP, gust_high, and strong_gated.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KT_PER_MS = 1.9438444924406

LEAKY_PREFIXES = (
    "actual_",
    "labels__",
    "observation_",
)
LEAKY_SUBSTRINGS = (
    "target_observation",
    "obs_distance",
    "obs_time",
)

BASE_NUMERIC_FEATURES = (
    "lead_time_minutes",
    "target_hour_utc",
    "issue_hour_utc",
    "issue_month_number",
    "issue_dayofyear_sin",
    "issue_dayofyear_cos",
    "latitude",
    "longitude",
    "raw_wind_mean_kt",
    "raw_gust_kt",
    "champion_wind_mean_kt",
    "champion_gust_kt",
    "gust_high_kt",
    "strong_gated_wind_mean_kt",
    "strong_gated_gust_kt",
    "strong_wind_total_weight",
    "strong_wind_soft_weight",
    "strong_wind_aggressive_weight",
    "prob_gust_ge_20kt",
    "prob_gust_ge_25kt",
    "prob_gust_ge_20kt_heuristic",
    "prob_gust_ge_25kt_heuristic",
    "prob_gust_ge_20kt_model",
    "prob_gust_ge_25kt_model",
    "gust_alert_ge_20kt_probability",
    "gust_alert_ge_25kt_probability",
)

DEFAULT_FEATURE_PREFIXES = (
    "baselines__",
    "features__thermal_",
    "features__spot_static_",
    "features__open_meteo_vertical_",
    "features__nwp_offset_",
)

TARGET_SPECS = {
    "wind": {
        "actual": "actual_wind_mean_kt",
        "thresholds": (12.0, 15.0, 20.0, 25.0),
        "regimes": ((None, 12.0, "<12kt"), (12.0, 15.0, "12-15kt"), (15.0, 20.0, "15-20kt"), (20.0, 25.0, "20-25kt"), (25.0, None, ">=25kt")),
        "candidates": {
            "champion": "champion_wind_mean_kt",
            "raw": "raw_wind_mean_kt",
            "strong_gated": "strong_gated_wind_mean_kt",
        },
    },
    "gust": {
        "actual": "actual_gust_kt",
        "thresholds": (15.0, 20.0, 25.0),
        "regimes": ((None, 15.0, "<15kt"), (15.0, 20.0, "15-20kt"), (20.0, 25.0, "20-25kt"), (25.0, None, ">=25kt")),
        "candidates": {
            "champion": "champion_gust_kt",
            "raw": "raw_gust_kt",
            "gust_high": "gust_high_kt",
            "strong_gated": "strong_gated_gust_kt",
        },
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def expand_paths(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            out.extend(Path(match) for match in matches)
        else:
            candidate = Path(pattern)
            if candidate.exists():
                out.append(candidate)
    return sorted(dict.fromkeys(out))


def json_scalar(pd: Any, value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def is_leaky_column(column: str) -> bool:
    if column.startswith(LEAKY_PREFIXES):
        return True
    return any(part in column for part in LEAKY_SUBSTRINGS)


def load_rows(paths: list[Path], pd: Any) -> Any:
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["hindcast_id"] = path.parent.name
        frames.append(frame)
    if not frames:
        raise SystemExit("No input parquet files found.")
    data = pd.concat(frames, ignore_index=True)
    if "issue_time_utc" in data.columns:
        data["issue_time_utc"] = pd.to_datetime(data["issue_time_utc"], utc=True, errors="coerce")
    if "target_time_utc" in data.columns:
        data["target_dt"] = pd.to_datetime(data["target_time_utc"], utc=True, errors="coerce")
        data["target_hour_utc"] = data["target_dt"].dt.hour.astype(float)
        data["target_date"] = data["target_dt"].dt.strftime("%Y-%m-%d")
    elif "target_dt" in data.columns:
        data["target_dt"] = pd.to_datetime(data["target_dt"], utc=True, errors="coerce")
        data["target_hour_utc"] = data["target_dt"].dt.hour.astype(float)
        data["target_date"] = data["target_dt"].dt.strftime("%Y-%m-%d")
    if "issue_time_utc" in data.columns:
        data["issue_hour_utc"] = data["issue_time_utc"].dt.hour.astype(float)
        data["issue_month_number"] = data["issue_time_utc"].dt.month.astype(float)
        dayofyear = data["issue_time_utc"].dt.dayofyear.fillna(1).astype(float)
        data["issue_dayofyear_sin"] = (2.0 * math.pi * dayofyear / 366.0).map(math.sin)
        data["issue_dayofyear_cos"] = (2.0 * math.pi * dayofyear / 366.0).map(math.cos)
    return data


def ensure_kt_columns(frame: Any) -> None:
    pairs = (
        ("raw_wind_mean_ms", "raw_wind_mean_kt"),
        ("champion_wind_mean_ms", "champion_wind_mean_kt"),
        ("strong_gated_wind_mean_ms", "strong_gated_wind_mean_kt"),
        ("actual_wind_mean_ms", "actual_wind_mean_kt"),
        ("raw_gust_ms", "raw_gust_kt"),
        ("champion_gust_ms", "champion_gust_kt"),
        ("gust_high_ms", "gust_high_kt"),
        ("strong_gated_gust_ms", "strong_gated_gust_kt"),
        ("actual_gust_ms", "actual_gust_kt"),
    )
    for ms_col, kt_col in pairs:
        if kt_col not in frame.columns and ms_col in frame.columns:
            frame[kt_col] = frame[ms_col].astype(float) * KT_PER_MS


def candidate_columns(frame: Any, target: str) -> dict[str, str]:
    spec = TARGET_SPECS[target]
    return {name: column for name, column in spec["candidates"].items() if column in frame.columns}


def metric(frame: Any, pred_col: str, actual_col: str) -> dict[str, Any]:
    values = frame[[pred_col, actual_col]].dropna()
    if values.empty:
        return {"n": 0}
    err = values[pred_col].astype(float) - values[actual_col].astype(float)
    abs_err = err.abs()
    return {
        "n": int(len(values)),
        "rmse": float((err.pow(2).mean()) ** 0.5),
        "mae": float(abs_err.mean()),
        "bias": float(err.mean()),
        "p50_abs_error": float(abs_err.quantile(0.50)),
        "p90_abs_error": float(abs_err.quantile(0.90)),
    }


def threshold_metric(frame: Any, pred_col: str, actual_col: str, threshold: float) -> dict[str, Any]:
    values = frame[[pred_col, actual_col]].dropna()
    if values.empty:
        return {"n": 0, "threshold": threshold}
    pred = values[pred_col].astype(float) >= threshold
    actual = values[actual_col].astype(float) >= threshold
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


def regime_label(value: float, regimes: tuple[tuple[float | None, float | None, str], ...]) -> str:
    for low, high, label in regimes:
        if low is not None and value < low:
            continue
        if high is not None and value >= high:
            continue
        return label
    return "unknown"


def add_oracle(frame: Any, target: str, candidates: dict[str, str]) -> None:
    actual = TARGET_SPECS[target]["actual"]
    error_cols = []
    for name, column in candidates.items():
        err_col = f"{target}_{name}_abs_error"
        frame[err_col] = (frame[column].astype(float) - frame[actual].astype(float)).abs()
        error_cols.append(err_col)
    labels = frame[error_cols].idxmin(axis=1).str.removeprefix(f"{target}_").str.removesuffix("_abs_error")
    frame[f"{target}_oracle_choice"] = labels
    for name, column in candidates.items():
        frame.loc[labels == name, f"{target}_oracle_prediction_kt"] = frame.loc[labels == name, column]


def make_feature_columns(frame: Any, args: argparse.Namespace, candidates: dict[str, str]) -> tuple[list[str], list[str]]:
    numeric = [column for column in BASE_NUMERIC_FEATURES if column in frame.columns]
    numeric.extend(column for column in candidates.values() if column in frame.columns and column not in numeric)
    for prefix in args.feature_prefix:
        numeric.extend(
            column
            for column in frame.columns
            if column.startswith(prefix) and column not in numeric and not is_leaky_column(column)
        )
    categorical = [column for column in ("spot_id", "spot_kind", "spot_source_type", "station_id") if column in frame.columns]
    numeric = [
        column
        for column in numeric
        if column in frame.columns and not is_leaky_column(column) and frame[column].dtype.kind in "biufc"
    ]
    return numeric, categorical


def build_classifier(args: argparse.Namespace, numeric: list[str], categorical: list[str]) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            categorical,
        ))
    if args.classifier == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            class_weight="balanced",
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf,
            l2_regularization=args.l2_regularization,
            random_state=args.random_state,
        )
    return Pipeline([("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")), ("model", model)])


def build_regressor(args: argparse.Namespace, numeric: list[str], categorical: list[str]) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            categorical,
        ))
    if args.regressor == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf,
            l2_regularization=args.l2_regularization,
            random_state=args.random_state,
        )
    return Pipeline([("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")), ("model", model)])


def apply_choice_prediction(frame: Any, target: str, candidates: dict[str, str], choice_col: str, output_col: str) -> None:
    frame[output_col] = float("nan")
    for name, column in candidates.items():
        mask = frame[choice_col].astype(str) == name
        frame.loc[mask, output_col] = frame.loc[mask, column].astype(float)


def add_rule_predictions(frame: Any, target: str, candidates: dict[str, str]) -> dict[str, str]:
    """Add deterministic no-training router baselines for high-wind regimes."""
    out: dict[str, str] = {}
    if target == "wind" and "strong_gated" in candidates:
        frame["wind_rule_strong_gated_kt"] = frame[candidates["strong_gated"]].astype(float)
        out["rule_strong_gated"] = "wind_rule_strong_gated_kt"
        if "raw" in candidates:
            frame["wind_rule_raw25_else_strong_kt"] = frame[candidates["strong_gated"]].astype(float)
            raw25 = frame[candidates["raw"]].astype(float) >= 25.0
            frame.loc[raw25, "wind_rule_raw25_else_strong_kt"] = frame.loc[raw25, candidates["raw"]].astype(float)
            out["rule_raw25_else_strong"] = "wind_rule_raw25_else_strong_kt"
    if target == "gust" and "raw" in candidates:
        if "gust_high" in candidates:
            frame["gust_rule_raw25_else_high_kt"] = frame[candidates["gust_high"]].astype(float)
            raw25 = frame[candidates["raw"]].astype(float) >= 25.0
            frame.loc[raw25, "gust_rule_raw25_else_high_kt"] = frame.loc[raw25, candidates["raw"]].astype(float)
            out["rule_raw25_else_high"] = "gust_rule_raw25_else_high_kt"
        if "gust_high" in candidates and "champion" in candidates:
            frame["gust_rule_raw25_high20_else_champion_kt"] = frame[candidates["champion"]].astype(float)
            raw20 = frame[candidates["raw"]].astype(float) >= 20.0
            raw25 = frame[candidates["raw"]].astype(float) >= 25.0
            frame.loc[raw20, "gust_rule_raw25_high20_else_champion_kt"] = frame.loc[raw20, candidates["gust_high"]].astype(float)
            frame.loc[raw25, "gust_rule_raw25_high20_else_champion_kt"] = frame.loc[raw25, candidates["raw"]].astype(float)
            out["rule_raw25_high20_else_champion"] = "gust_rule_raw25_high20_else_champion_kt"
        if "gust_high" in candidates and "strong_gated" in candidates:
            frame["gust_rule_raw25_high20_else_strong_kt"] = frame[candidates["strong_gated"]].astype(float)
            raw20 = frame[candidates["raw"]].astype(float) >= 20.0
            raw25 = frame[candidates["raw"]].astype(float) >= 25.0
            frame.loc[raw20, "gust_rule_raw25_high20_else_strong_kt"] = frame.loc[raw20, candidates["gust_high"]].astype(float)
            frame.loc[raw25, "gust_rule_raw25_high20_else_strong_kt"] = frame.loc[raw25, candidates["raw"]].astype(float)
            out["rule_raw25_high20_else_strong"] = "gust_rule_raw25_high20_else_strong_kt"
        if "champion" in candidates:
            frame["gust_rule_raw20_else_champion_kt"] = frame[candidates["champion"]].astype(float)
            raw20 = frame[candidates["raw"]].astype(float) >= 20.0
            frame.loc[raw20, "gust_rule_raw20_else_champion_kt"] = frame.loc[raw20, candidates["raw"]].astype(float)
            out["rule_raw20_else_champion"] = "gust_rule_raw20_else_champion_kt"
    return out


def evaluate_target(data: Any, target: str, args: argparse.Namespace, pd: Any, np: Any) -> tuple[dict[str, Any], Any]:
    spec = TARGET_SPECS[target]
    actual_col = spec["actual"]
    candidates = candidate_columns(data, target)
    if actual_col not in data.columns or not candidates:
        return {"target": target, "error": "missing_actual_or_candidates"}, data.iloc[0:0].copy()

    frame = data.dropna(subset=[actual_col, *candidates.values(), args.fold_column]).copy()
    add_oracle(frame, target, candidates)
    numeric, categorical = make_feature_columns(frame, args, candidates)
    features = [*numeric, *categorical]
    prediction_frames = []
    fold_results = []

    for fold in sorted(frame[args.fold_column].dropna().unique().tolist()):
        train = frame[frame[args.fold_column] != fold].copy()
        test = frame[frame[args.fold_column] == fold].copy()
        if train.empty or test.empty:
            continue
        y_choice = train[f"{target}_oracle_choice"].astype(str)
        if y_choice.nunique() == 1:
            test[f"{target}_router_choice"] = y_choice.iloc[0]
        else:
            classifier = build_classifier(args, numeric, categorical)
            classifier.fit(train[features], y_choice)
            test[f"{target}_router_choice"] = classifier.predict(test[features])
        apply_choice_prediction(test, target, candidates, f"{target}_router_choice", f"{target}_router_prediction_kt")

        regressor = build_regressor(args, numeric, categorical)
        regressor.fit(train[features], train[actual_col].astype(float))
        test[f"{target}_stacker_prediction_kt"] = regressor.predict(test[features])
        prediction_frames.append(test)
        fold_results.append({
            "fold": str(fold),
            "fold_column": args.fold_column,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "oracle_choice_share_train": y_choice.value_counts(normalize=True).round(6).to_dict(),
        })

    if not prediction_frames:
        return {"target": target, "error": "no_valid_folds"}, frame.iloc[0:0].copy()
    predicted = pd.concat(prediction_frames, ignore_index=True)
    predicted[f"{target}_actual_regime"] = predicted[actual_col].astype(float).map(lambda value: regime_label(value, spec["regimes"]))
    rule_candidates = add_rule_predictions(predicted, target, candidates)
    output_columns = {
        **candidates,
        **rule_candidates,
        "oracle": f"{target}_oracle_prediction_kt",
        "router_classifier": f"{target}_router_prediction_kt",
        "stacker_regressor": f"{target}_stacker_prediction_kt",
    }

    metrics = {}
    for name, column in candidates.items():
        metrics[name] = metric(predicted, column, actual_col)
    for name, column in rule_candidates.items():
        metrics[name] = metric(predicted, column, actual_col)
    metrics["oracle"] = metric(predicted, f"{target}_oracle_prediction_kt", actual_col)
    metrics["router_classifier"] = metric(predicted, f"{target}_router_prediction_kt", actual_col)
    metrics["stacker_regressor"] = metric(predicted, f"{target}_stacker_prediction_kt", actual_col)

    threshold_metrics = {}
    for threshold in spec["thresholds"]:
        threshold_key = f">={int(threshold)}kt"
        threshold_metrics[threshold_key] = {}
        for name, column in output_columns.items():
            threshold_metrics[threshold_key][name] = threshold_metric(predicted, column, actual_col, threshold)

    by_regime = {}
    for regime, group in predicted.groupby(f"{target}_actual_regime", dropna=False):
        by_regime[str(regime)] = {
            name: metric(group, column, actual_col)
            for name, column in output_columns.items()
        }

    by_hindcast = {}
    for hindcast_id, group in predicted.groupby("hindcast_id", dropna=False):
        by_hindcast[str(hindcast_id)] = {
            name: metric(group, column, actual_col)
            for name, column in output_columns.items()
        }

    summary = {
        "target": target,
        "rows": int(len(predicted)),
        "fold_column": args.fold_column,
        "folds": sorted(predicted[args.fold_column].astype(str).unique().tolist()),
        "hindcasts": sorted(predicted["hindcast_id"].astype(str).unique().tolist()) if "hindcast_id" in predicted.columns else [],
        "candidate_columns": candidates,
        "rule_columns": rule_candidates,
        "numeric_feature_count": len(numeric),
        "categorical_features": categorical,
        "folds": fold_results,
        "metrics": metrics,
        "oracle_choice_share": predicted[f"{target}_oracle_choice"].value_counts(normalize=True).round(6).to_dict(),
        "router_choice_share": predicted[f"{target}_router_choice"].value_counts(normalize=True).round(6).to_dict(),
        "thresholds": threshold_metrics,
        "by_actual_regime": by_regime,
        "by_hindcast": by_hindcast,
    }
    champion_rmse = metrics.get("champion", {}).get("rmse")
    for name in ("router_classifier", "stacker_regressor", "oracle"):
        rmse = metrics.get(name, {}).get("rmse")
        summary[f"{name}_rmse_gain_pct_vs_champion"] = None if not champion_rmse or rmse is None else (champion_rmse - rmse) / champion_rmse * 100.0
    return summary, predicted


def train_final_target_models(data: Any, target: str, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = TARGET_SPECS[target]
    actual_col = spec["actual"]
    candidates = candidate_columns(data, target)
    if actual_col not in data.columns or not candidates:
        return {"target": target, "error": "missing_actual_or_candidates"}, {}

    frame = data.dropna(subset=[actual_col, *candidates.values()]).copy()
    if frame.empty:
        return {"target": target, "error": "no_training_rows"}, {}

    add_oracle(frame, target, candidates)
    numeric, categorical = make_feature_columns(frame, args, candidates)
    features = [*numeric, *categorical]
    y_choice = frame[f"{target}_oracle_choice"].astype(str)

    classifier: Any | None = None
    constant_choice: str | None = None
    if y_choice.nunique() == 1:
        constant_choice = str(y_choice.iloc[0])
    else:
        classifier = build_classifier(args, numeric, categorical)
        classifier.fit(frame[features], y_choice)

    regressor = build_regressor(args, numeric, categorical)
    regressor.fit(frame[features], frame[actual_col].astype(float))

    metadata = {
        "target": target,
        "rows": int(len(frame)),
        "actual_column": actual_col,
        "candidate_columns": candidates,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "features": features,
        "classifier_type": args.classifier,
        "regressor_type": args.regressor,
        "constant_classifier_choice": constant_choice,
        "oracle_choice_share_train": y_choice.value_counts(normalize=True).round(6).to_dict(),
    }
    artifact = {
        "metadata": metadata,
        "classifier": classifier,
        "regressor": regressor,
    }
    return metadata, artifact


def fmt_metric(payload: dict[str, Any] | None) -> str:
    if not payload or payload.get("n", 0) == 0:
        return "n/a"
    return f"{payload.get('rmse'):.3f} / {payload.get('mae'):.3f} / {payload.get('bias'):.3f}"


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    preferred_order = (
        "champion",
        "raw",
        "gust_high",
        "strong_gated",
        "rule_strong_gated",
        "rule_raw25_else_strong",
        "rule_raw25_else_high",
        "rule_raw25_high20_else_champion",
        "rule_raw25_high20_else_strong",
        "rule_raw20_else_champion",
        "router_classifier",
        "stacker_regressor",
        "oracle",
    )

    def ordered_names(summary: dict[str, Any]) -> list[str]:
        names = list((summary.get("metrics") or {}).keys())
        ordered = [name for name in preferred_order if name in names]
        ordered.extend(sorted(name for name in names if name not in ordered))
        return ordered

    lines = [
        "# Hindcast Router v1 Results",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Input files: `{len(result['input_files'])}`",
        f"Validation: `leave-one-hindcast-out`",
        "",
        "Metric cells are `RMSE / MAE / bias` in knots.",
        "",
    ]
    for target, summary in result["targets"].items():
        names = ordered_names(summary)
        lines.extend([
            f"## {target.title()}",
            "",
            f"Rows: `{summary.get('rows')}`",
            f"Router gain vs champion: `{summary.get('router_classifier_rmse_gain_pct_vs_champion')}`",
            f"Stacker gain vs champion: `{summary.get('stacker_regressor_rmse_gain_pct_vs_champion')}`",
            f"Oracle gain vs champion: `{summary.get('oracle_rmse_gain_pct_vs_champion')}`",
            "",
            "| Rail | RMSE / MAE / Bias |",
            "| --- | ---: |",
        ])
        for name in names:
            payload = (summary.get("metrics") or {}).get(name)
            lines.append(f"| `{name}` | {fmt_metric(payload)} |")
        lines.extend(["", "### Choice Shares", "", "Oracle:", ""])
        for name, share in summary.get("oracle_choice_share", {}).items():
            lines.append(f"- `{name}`: `{share}`")
        lines.extend(["", "Router:", ""])
        for name, share in summary.get("router_choice_share", {}).items():
            lines.append(f"- `{name}`: `{share}`")
        lines.extend(["", "### Threshold CSI", ""])
        lines.append("| Threshold | " + " | ".join(f"`{name}`" for name in names) + " |")
        lines.append("| --- | " + " | ".join("---:" for _ in names) + " |")
        for threshold, values in summary.get("thresholds", {}).items():
            def csi(name: str) -> Any:
                item = values.get(name, {})
                value = item.get("csi")
                return "n/a" if value is None else f"{value:.3f}"
            lines.append("| `" + threshold + "` | " + " | ".join(csi(name) for name in names) + " |")
        lines.extend(["", "### Actual Regimes", ""])
        lines.append("| Regime | " + " | ".join(f"`{name}`" for name in names) + " |")
        lines.append("| --- | " + " | ".join("---:" for _ in names) + " |")
        for regime, values in summary.get("by_actual_regime", {}).items():
            lines.append("| `" + regime + "` | " + " | ".join(fmt_metric(values.get(name)) for name in names) + " |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import joblib
    import numpy as np
    import pandas as pd

    paths = expand_paths(args.input)
    data = load_rows(paths, pd)
    ensure_kt_columns(data)
    targets = args.target or ["wind", "gust"]
    result = {
        "format": "corsewind.hindcast_router_v1",
        "generated_at_utc": utc_now(),
        "input_files": [str(path) for path in paths],
        "classifier": args.classifier,
        "regressor": args.regressor,
        "fold_column": args.fold_column,
        "feature_prefixes": args.feature_prefix,
        "targets": {},
    }
    predictions = []
    final_models: dict[str, Any] = {}
    final_model_metadata: dict[str, Any] = {}
    for target in targets:
        summary, target_predictions = evaluate_target(data, target, args, pd, np)
        result["targets"][target] = summary
        if args.save_final_models:
            metadata, artifact = train_final_target_models(data, target, args)
            final_model_metadata[target] = metadata
            if artifact:
                final_models[target] = artifact
        if not target_predictions.empty:
            keep = [
                column
                for column in (
                    "hindcast_id",
                    "issue_time_utc",
                    "target_time_utc",
                    "spot_id",
                    "lead_time_minutes",
                    TARGET_SPECS[target]["actual"],
                    f"{target}_oracle_choice",
                    f"{target}_oracle_prediction_kt",
                    f"{target}_router_choice",
                    f"{target}_router_prediction_kt",
                    f"{target}_stacker_prediction_kt",
                )
                if column in target_predictions.columns
            ]
            predictions.append(target_predictions[keep].copy())
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "router_v1_results.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_markdown(args.output_root / "router_v1_results.md", result)
    if args.save_final_models:
        model_payload = {
            "format": "corsewind.hindcast_router_v1_final_models",
            "generated_at_utc": result["generated_at_utc"],
            "input_files": result["input_files"],
            "fold_column_used_for_oof_validation": args.fold_column,
            "classifier": args.classifier,
            "regressor": args.regressor,
            "feature_prefixes": args.feature_prefix,
            "targets": final_models,
        }
        joblib.dump(model_payload, args.output_root / "router_v1_final_models.joblib")
        (args.output_root / "router_v1_final_models_metadata.json").write_text(
            json.dumps(
                {
                    "format": "corsewind.hindcast_router_v1_final_models_metadata",
                    "generated_at_utc": result["generated_at_utc"],
                    "input_files": result["input_files"],
                    "targets": final_model_metadata,
                },
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
    if predictions:
        # Keep target outputs separate in columns; row alignment can differ by target.
        for target, target_predictions in zip(targets, predictions, strict=False):
            target_predictions.to_parquet(args.output_root / f"router_v1_{target}_predictions.parquet", index=False)
    print(json.dumps({
        "output_root": str(args.output_root),
        "targets": {
            target: {
                "rows": summary.get("rows"),
                "champion_rmse": summary.get("metrics", {}).get("champion", {}).get("rmse"),
                "router_rmse": summary.get("metrics", {}).get("router_classifier", {}).get("rmse"),
                "stacker_rmse": summary.get("metrics", {}).get("stacker_regressor", {}).get("rmse"),
                "oracle_rmse": summary.get("metrics", {}).get("oracle", {}).get("rmse"),
                "router_gain_pct": summary.get("router_classifier_rmse_gain_pct_vs_champion"),
                "stacker_gain_pct": summary.get("stacker_regressor_rmse_gain_pct_vs_champion"),
            }
            for target, summary in result["targets"].items()
        },
    }, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Scored parquet path or glob.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target", choices=sorted(TARGET_SPECS), action="append", default=[])
    parser.add_argument("--fold-column", default="hindcast_id", help="Leakage boundary for validation, e.g. hindcast_id or target_date.")
    parser.add_argument("--feature-prefix", action="append", default=list(DEFAULT_FEATURE_PREFIXES))
    parser.add_argument("--classifier", choices=("hgb", "extra_trees"), default="extra_trees")
    parser.add_argument("--regressor", choices=("hgb", "extra_trees"), default="hgb")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--save-final-models", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
