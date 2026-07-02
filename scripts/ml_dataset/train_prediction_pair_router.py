#!/usr/bin/env python3
"""Train a leakage-safe router between two prediction files.

The router learns when to use the alternative model instead of the base model.
It uses only issue-time metadata and the two predictions, never the future
observation at evaluation time. A temporal split inside the calibration period
selects the probability threshold.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KEY_COLUMNS = ("issue_time_utc", "spot_id", "lead_time_minutes")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def metric(prediction: Any, actual: Any, np: Any) -> dict[str, Any]:
    prediction = prediction.astype(float)
    actual = actual.astype(float)
    valid = ~(np.isnan(prediction) | np.isnan(actual))
    prediction = prediction[valid]
    actual = actual[valid]
    if len(prediction) == 0:
        return {"count": 0}
    errors = prediction - actual
    abs_errors = np.abs(errors)
    return {
        "count": int(len(errors)),
        "rmse": round(float(math.sqrt(float(np.mean(errors * errors)))), 6),
        "mae": round(float(np.mean(abs_errors)), 6),
        "bias": round(float(np.mean(errors)), 6),
        "p50_abs_error": round(float(np.quantile(abs_errors, 0.50)), 6),
        "p90_abs_error": round(float(np.quantile(abs_errors, 0.90)), 6),
        "p95_abs_error": round(float(np.quantile(abs_errors, 0.95)), 6),
    }


def metric_frame(frame: Any, prediction_column: str, np: Any) -> dict[str, Any]:
    valid = frame[[prediction_column, "actual"]].dropna()
    return metric(valid[prediction_column].to_numpy(), valid["actual"].to_numpy(), np)


def json_scalar(pd: Any, value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def grouped_metrics(frame: Any, group_columns: list[str], prediction_column: str, np: Any, pd: Any) -> list[dict[str, Any]]:
    rows = []
    for raw_key, group in frame.dropna(subset=[prediction_column, "actual"]).groupby(group_columns, dropna=False):
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        rows.append({
            "group": dict(zip(group_columns, [json_scalar(pd, value) for value in values], strict=True)),
            **metric_frame(group, prediction_column, np),
        })
    rows.sort(key=lambda item: (item.get("rmse") is None, item.get("rmse", -1)), reverse=True)
    return rows


def load_prediction_file(path: Path, prediction_column: str, actual_column: str, pd: Any) -> Any:
    frame = pd.read_parquet(path)
    missing = [column for column in (*KEY_COLUMNS, prediction_column, actual_column) if column not in frame.columns]
    if missing:
        raise SystemExit(f"{path} missing required columns: {missing}")
    frame["issue_time_utc"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    frame["spot_id"] = frame["spot_id"].astype(str)
    frame["lead_time_minutes"] = frame["lead_time_minutes"].astype("Int64")
    keep = [*KEY_COLUMNS, prediction_column, actual_column]
    for optional in (
        "raw_wind_mean_ms",
        "corrected_wind_mean_ms",
        "calibrated_wind_mean_ms",
        "predicted_residual_wind_mean_ms",
        "predicted_second_stage_residual_wind_mean_ms",
        "station_id",
        "spot_kind",
        "latitude",
        "longitude",
    ):
        if optional in frame.columns and optional not in keep:
            keep.append(optional)
    return frame[keep].drop_duplicates(subset=list(KEY_COLUMNS), keep="first").copy()


def load_pair(
    base_path: Path,
    alt_path: Path,
    base_prediction_column: str,
    alt_prediction_column: str,
    actual_column: str,
    leads: list[int],
    pd: Any,
) -> Any:
    base = load_prediction_file(base_path, base_prediction_column, actual_column, pd)
    alt = load_prediction_file(alt_path, alt_prediction_column, actual_column, pd)
    if leads:
        keep = [int(lead) for lead in leads]
        base = base[base["lead_time_minutes"].astype("Int64").isin(keep)].copy()
        alt = alt[alt["lead_time_minutes"].astype("Int64").isin(keep)].copy()
    base = base.rename(columns={base_prediction_column: "base_prediction", actual_column: "actual"})
    alt = alt.rename(columns={alt_prediction_column: "alt_prediction", actual_column: "alt_actual"})
    optional_alt = [
        column
        for column in alt.columns
        if column not in (*KEY_COLUMNS, "alt_prediction", "alt_actual")
    ]
    alt = alt.drop(columns=optional_alt, errors="ignore")
    merged = base.merge(alt, on=list(KEY_COLUMNS), how="inner", validate="one_to_one")
    actual_delta = (merged["actual"].astype(float) - merged["alt_actual"].astype(float)).abs()
    if float(actual_delta.max(skipna=True) or 0.0) > 1e-5:
        raise SystemExit(f"Actual labels differ, max abs delta={actual_delta.max()}")
    merged = merged.drop(columns=["alt_actual"])
    return merged


def add_features(frame: Any, pd: Any, np: Any) -> Any:
    out = frame.copy()
    out["issue_hour_utc"] = out["issue_time_utc"].dt.hour.astype(float)
    out["issue_month_number"] = out["issue_time_utc"].dt.month.astype(float)
    dayofyear = out["issue_time_utc"].dt.dayofyear.fillna(1).astype(float)
    out["issue_dayofyear_sin"] = np.sin(2.0 * math.pi * dayofyear / 366.0)
    out["issue_dayofyear_cos"] = np.cos(2.0 * math.pi * dayofyear / 366.0)
    out["prediction_diff_alt_minus_base"] = out["alt_prediction"].astype(float) - out["base_prediction"].astype(float)
    out["prediction_mean"] = out[["base_prediction", "alt_prediction"]].astype(float).mean(axis=1)
    out["prediction_abs_diff"] = out["prediction_diff_alt_minus_base"].abs()
    out["prediction_max"] = out[["base_prediction", "alt_prediction"]].astype(float).max(axis=1)
    out["prediction_min"] = out[["base_prediction", "alt_prediction"]].astype(float).min(axis=1)
    out["base_abs_error"] = (out["base_prediction"].astype(float) - out["actual"].astype(float)).abs()
    out["alt_abs_error"] = (out["alt_prediction"].astype(float) - out["actual"].astype(float)).abs()
    return out


def make_feature_lists(frame: Any) -> tuple[list[str], list[str]]:
    numeric = [
        "lead_time_minutes",
        "issue_hour_utc",
        "issue_month_number",
        "issue_dayofyear_sin",
        "issue_dayofyear_cos",
        "base_prediction",
        "alt_prediction",
        "prediction_diff_alt_minus_base",
        "prediction_mean",
        "prediction_abs_diff",
        "prediction_max",
        "prediction_min",
    ]
    for optional in (
        "raw_wind_mean_ms",
        "corrected_wind_mean_ms",
        "calibrated_wind_mean_ms",
        "predicted_residual_wind_mean_ms",
        "predicted_second_stage_residual_wind_mean_ms",
        "latitude",
        "longitude",
    ):
        if optional in frame.columns:
            numeric.append(optional)
    categorical = [column for column in ("spot_id", "station_id", "spot_kind") if column in frame.columns]
    return numeric, categorical


def build_pipeline(args: argparse.Namespace, numeric: list[str], categorical: list[str]):
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
    if args.model_family == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=args.max_iter,
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
            l2_regularization=args.l2_regularization,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
        )
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
        ("model", model),
    ])


def routed_prediction(frame: Any, use_alt: Any) -> Any:
    return frame["base_prediction"].where(~use_alt, frame["alt_prediction"])


def threshold_candidates() -> list[float]:
    return [round(value / 100.0, 2) for value in range(5, 96, 5)]


def choose_threshold(validation: Any, probabilities: Any, np: Any) -> dict[str, Any]:
    rows = []
    best = None
    for threshold in threshold_candidates():
        use_alt = probabilities >= threshold
        prediction = routed_prediction(validation, use_alt)
        item = {
            "threshold": threshold,
            "use_alt_rate": round(float(np.mean(use_alt)), 6),
            **metric(prediction.to_numpy(), validation["actual"].to_numpy(), np),
        }
        rows.append(item)
        if best is None or float(item["rmse"]) < float(best["rmse"]):
            best = item
    assert best is not None
    return {"best": best, "candidates": rows}


def fit_predict_probability(model: Any, train: Any, target: Any, eval_frame: Any, feature_columns: list[str]) -> Any:
    model.fit(train[feature_columns], target)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(eval_frame[feature_columns])[:, 1]
    return model.predict(eval_frame[feature_columns])


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    calibration = add_features(load_pair(
        args.calibration_base_predictions,
        args.calibration_alt_predictions,
        args.base_prediction_column,
        args.alt_prediction_column,
        args.actual_column,
        args.lead_minute,
        pd,
    ), pd, np)
    evaluation = add_features(load_pair(
        args.evaluation_base_predictions,
        args.evaluation_alt_predictions,
        args.base_prediction_column,
        args.alt_prediction_column,
        args.actual_column,
        args.lead_minute,
        pd,
    ), pd, np)
    calibration = calibration.dropna(subset=["base_prediction", "alt_prediction", "actual"]).sort_values("issue_time_utc")
    evaluation = evaluation.dropna(subset=["base_prediction", "alt_prediction", "actual"]).sort_values("issue_time_utc")

    numeric, categorical = make_feature_lists(calibration)
    numeric = [column for column in numeric if column in evaluation.columns]
    categorical = [column for column in categorical if column in evaluation.columns]
    feature_columns = [*numeric, *categorical]
    target = (calibration["alt_abs_error"] + args.margin < calibration["base_abs_error"]).astype(int)
    split_index = max(1, min(len(calibration) - 1, int(len(calibration) * args.fit_fraction)))
    fit_frame = calibration.iloc[:split_index].copy()
    valid_frame = calibration.iloc[split_index:].copy()
    fit_target = target.iloc[:split_index]

    model = build_pipeline(args, numeric, categorical)
    valid_prob = fit_predict_probability(model, fit_frame, fit_target, valid_frame, feature_columns)
    threshold_selection = choose_threshold(valid_frame, valid_prob, np)

    final_model = build_pipeline(args, numeric, categorical)
    eval_prob = fit_predict_probability(final_model, calibration, target, evaluation, feature_columns)
    threshold = float(threshold_selection["best"]["threshold"])
    evaluation["alt_probability"] = eval_prob
    evaluation["use_alt"] = evaluation["alt_probability"].astype(float) >= threshold
    evaluation["routed_prediction_ms"] = routed_prediction(evaluation, evaluation["use_alt"])

    # Non-deployable reference only.
    evaluation["oracle_prediction_ms"] = evaluation["base_prediction"].where(
        evaluation["base_abs_error"] <= evaluation["alt_abs_error"],
        evaluation["alt_prediction"],
    )

    result = {
        "format": "corsewind.prediction_pair_router.v1",
        "generated_at_utc": utc_now(),
        "model_family": args.model_family,
        "calibration_rows": int(len(calibration)),
        "fit_rows": int(len(fit_frame)),
        "validation_rows": int(len(valid_frame)),
        "evaluation_rows": int(len(evaluation)),
        "feature_columns": feature_columns,
        "target_positive_rate": round(float(target.mean()), 6),
        "threshold_selection": threshold_selection,
        "evaluation": {
            "base_metric": metric_frame(evaluation, "base_prediction", np),
            "alt_metric": metric_frame(evaluation, "alt_prediction", np),
            "router_metric": metric_frame(evaluation, "routed_prediction_ms", np),
            "oracle_metric": metric_frame(evaluation, "oracle_prediction_ms", np),
            "use_alt_rate": round(float(evaluation["use_alt"].mean()), 6),
            "router_by_lead": grouped_metrics(evaluation, ["lead_time_minutes"], "routed_prediction_ms", np, pd),
            "router_by_spot": grouped_metrics(evaluation, ["spot_id"], "routed_prediction_ms", np, pd)[: args.limit],
        },
    }
    base_rmse = result["evaluation"]["base_metric"].get("rmse")
    router_rmse = result["evaluation"]["router_metric"].get("rmse")
    result["evaluation"]["rmse_gain_pct_vs_base"] = (
        None if not base_rmse or router_rmse is None else round((float(base_rmse) - float(router_rmse)) / float(base_rmse) * 100.0, 3)
    )
    result["verdict"] = "improved" if router_rmse is not None and base_rmse is not None and router_rmse < base_rmse else "not_improved"

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "pair_router_results.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(args.output_root / "pair_router_results.md", result)
    if args.output_predictions:
        output = evaluation[[*KEY_COLUMNS, "actual", "base_prediction", "alt_prediction", "alt_probability", "use_alt", "routed_prediction_ms", "oracle_prediction_ms"]].copy()
        output.to_parquet(args.output_root / "pair_router_predictions.parquet", index=False)
    print(json.dumps({
        "output_root": str(args.output_root),
        "verdict": result["verdict"],
        "evaluation_rows": result["evaluation_rows"],
        "base_rmse": base_rmse,
        "alt_rmse": result["evaluation"]["alt_metric"].get("rmse"),
        "router_rmse": router_rmse,
        "oracle_rmse": result["evaluation"]["oracle_metric"].get("rmse"),
        "use_alt_rate": result["evaluation"]["use_alt_rate"],
        "threshold": threshold,
    }, indent=2, sort_keys=True))
    return result


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Prediction Pair Router",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Verdict: `{result['verdict']}`",
        f"Model family: `{result['model_family']}`",
        f"Calibration rows: `{result['calibration_rows']}`",
        f"Evaluation rows: `{result['evaluation_rows']}`",
        f"Target positive rate: `{result['target_positive_rate']}`",
        f"Selected threshold: `{result['threshold_selection']['best']['threshold']}`",
        "",
        "## Evaluation",
        "",
        "| Model | RMSE | MAE | Bias | Rows |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (
        ("base", "base_metric"),
        ("alt", "alt_metric"),
        ("router", "router_metric"),
        ("oracle", "oracle_metric"),
    ):
        metric_payload = result["evaluation"][key]
        lines.append(
            f"| `{label}` | {metric_payload.get('rmse')} | {metric_payload.get('mae')} | "
            f"{metric_payload.get('bias')} | {metric_payload.get('count')} |"
        )
    lines.extend([
        "",
        f"Use alt rate: `{result['evaluation']['use_alt_rate']}`",
        f"Router gain vs base: `{result['evaluation']['rmse_gain_pct_vs_base']}%`",
        "",
        "## Router By Lead",
        "",
        "| Lead | RMSE | MAE | Bias | Rows |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in result["evaluation"]["router_by_lead"]:
        lines.append(
            f"| `{item['group'].get('lead_time_minutes')}` | {item.get('rmse')} | "
            f"{item.get('mae')} | {item.get('bias')} | {item.get('count')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-base-predictions", type=Path, required=True)
    parser.add_argument("--calibration-alt-predictions", type=Path, required=True)
    parser.add_argument("--evaluation-base-predictions", type=Path, required=True)
    parser.add_argument("--evaluation-alt-predictions", type=Path, required=True)
    parser.add_argument("--base-prediction-column", default="calibrated_wind_mean_ms")
    parser.add_argument("--alt-prediction-column", default="corrected_wind_mean_ms")
    parser.add_argument("--actual-column", default="actual_wind_mean_ms")
    parser.add_argument("--lead-minute", type=int, action="append", default=[15, 30, 45, 60])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-predictions", action="store_true")
    parser.add_argument("--model-family", choices=("hgb", "extra_trees"), default="hgb")
    parser.add_argument("--fit-fraction", type=float, default=0.75)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--min-samples-leaf", type=int, default=40)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--limit", type=int, default=15)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
