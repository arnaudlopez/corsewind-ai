#!/usr/bin/env python3
"""Audit whether prediction models contain complementary signal.

The main question is not "which model wins on its own?", but whether weaker
models are wrong on different rows than the champion. If the row-wise oracle is
much better than the champion, a deployable router/blend may be worth building.
If the oracle is barely better, the bottleneck is likely data/noise/target
definition rather than model family.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KEY_COLUMNS = ("issue_time_utc", "spot_id", "lead_time_minutes")


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: Path
    prediction_column: str
    actual_column: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_model_spec(raw: str, default_actual_column: str) -> ModelSpec:
    parts = raw.split("|")
    if len(parts) not in (3, 4):
        raise SystemExit(
            "--model must be label|path|prediction_column or "
            "label|path|prediction_column|actual_column"
        )
    label, path, prediction_column = parts[:3]
    actual_column = parts[3] if len(parts) == 4 else default_actual_column
    if not label:
        raise SystemExit(f"Invalid empty model label in {raw!r}")
    return ModelSpec(
        label=label,
        path=Path(path),
        prediction_column=prediction_column,
        actual_column=actual_column,
    )


def json_scalar(pd: Any, value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def sanitize_label(label: str) -> str:
    safe = []
    for char in label:
        if char.isalnum() or char in ("_", "-"):
            safe.append(char)
        else:
            safe.append("_")
    out = "".join(safe).strip("_")
    return out or "model"


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


def metric_frame(frame: Any, prediction_column: str, actual_column: str, np: Any) -> dict[str, Any]:
    subset = frame[[prediction_column, actual_column]].dropna()
    return metric(
        subset[prediction_column].to_numpy(),
        subset[actual_column].to_numpy(),
        np,
    )


def grouped_metrics(
    frame: Any,
    group_columns: list[str],
    prediction_column: str,
    actual_column: str,
    np: Any,
    pd: Any,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if any(column not in frame.columns for column in group_columns):
        return []
    valid = frame.dropna(subset=[prediction_column, actual_column])
    rows = []
    for raw_key, group in valid.groupby(group_columns, dropna=False):
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        rows.append({
            "group": dict(zip(group_columns, [json_scalar(pd, value) for value in values], strict=True)),
            **metric_frame(group, prediction_column, actual_column, np),
        })
    rows.sort(key=lambda item: (item.get("rmse") is None, item.get("rmse", -1)), reverse=True)
    return rows[:limit] if limit else rows


def normalize_frame(spec: ModelSpec, args: argparse.Namespace, pd: Any) -> Any:
    if not spec.path.exists():
        raise SystemExit(f"Prediction file missing for {spec.label}: {spec.path}")
    frame = pd.read_parquet(spec.path)
    missing = [
        column
        for column in (*KEY_COLUMNS, spec.prediction_column, spec.actual_column)
        if column not in frame.columns
    ]
    if missing:
        raise SystemExit(f"{spec.label} missing required columns: {missing}")
    if args.split and args.split_column in frame.columns:
        frame = frame[frame[args.split_column].astype(str) == args.split].copy()
    if args.lead_minute:
        leads = [int(lead) for lead in args.lead_minute]
        frame = frame[frame["lead_time_minutes"].astype("Int64").isin(leads)].copy()
    frame["issue_time_utc"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    frame["spot_id"] = frame["spot_id"].astype(str)
    frame["lead_time_minutes"] = frame["lead_time_minutes"].astype("Int64")
    keep_columns = [*KEY_COLUMNS, spec.actual_column, spec.prediction_column]
    for optional in ("target_time_utc", "station_id"):
        if optional in frame.columns:
            keep_columns.append(optional)
    out = frame[keep_columns].dropna(subset=list(KEY_COLUMNS)).copy()
    out = out.rename(columns={
        spec.actual_column: "actual",
        spec.prediction_column: f"pred__{sanitize_label(spec.label)}",
    })
    if "target_time_utc" in out.columns:
        out["target_time_utc"] = pd.to_datetime(out["target_time_utc"], utc=True, errors="coerce")
    out = out.drop_duplicates(subset=list(KEY_COLUMNS), keep="first")
    return out


def load_models(specs: list[ModelSpec], args: argparse.Namespace, pd: Any, np: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    frames: dict[str, Any] = {}
    native_metrics = {}
    for spec in specs:
        frame = normalize_frame(spec, args, pd)
        prediction_column = f"pred__{sanitize_label(spec.label)}"
        frames[spec.label] = frame
        native_metrics[spec.label] = {
            "path": str(spec.path),
            "prediction_column": spec.prediction_column,
            "actual_column": spec.actual_column,
            "rows": int(len(frame)),
            "metric": metric_frame(frame, prediction_column, "actual", np),
            "by_lead": grouped_metrics(frame, ["lead_time_minutes"], prediction_column, "actual", np, pd),
            "by_spot": grouped_metrics(frame, ["spot_id"], prediction_column, "actual", np, pd, limit=args.limit),
        }
    return frames, native_metrics


def merge_intersection(frames: dict[str, Any], specs: list[ModelSpec], pd: Any) -> Any:
    labels = [spec.label for spec in specs]
    first_label = labels[0]
    merged = frames[first_label].rename(columns={"actual": "actual__base"})
    for label in labels[1:]:
        right = frames[label].rename(columns={"actual": f"actual__{sanitize_label(label)}"})
        optional = [column for column in ("target_time_utc", "station_id") if column in right.columns]
        right = right.drop(columns=optional, errors="ignore")
        merged = merged.merge(right, on=list(KEY_COLUMNS), how="inner", validate="one_to_one")
    actual_columns = [column for column in merged.columns if column.startswith("actual__")]
    merged["actual"] = merged["actual__base"].astype(float)
    max_delta = 0.0
    for column in actual_columns[1:]:
        delta = (merged["actual"].astype(float) - merged[column].astype(float)).abs()
        if len(delta):
            max_delta = max(max_delta, float(delta.max(skipna=True) or 0.0))
    merged = merged.drop(columns=actual_columns)
    return merged, max_delta


def add_time_bins(frame: Any, pd: Any) -> Any:
    out = frame.copy()
    out["issue_hour_utc"] = out["issue_time_utc"].dt.hour
    out["issue_month"] = out["issue_time_utc"].dt.strftime("%Y-%m")
    out["actual_wind_bin_ms"] = pd.cut(
        out["actual"].astype(float),
        bins=[-0.001, 2.0, 4.0, 6.0, 8.0, 999.0],
        labels=["0-2", "2-4", "4-6", "6-8", "8+"],
    ).astype(str)
    return out


def compute_intersection_metrics(frame: Any, labels: list[str], np: Any, pd: Any, limit: int) -> dict[str, Any]:
    metrics = {}
    for label in labels:
        column = f"pred__{sanitize_label(label)}"
        error_column = f"err__{sanitize_label(label)}"
        abs_error_column = f"abs_err__{sanitize_label(label)}"
        frame[error_column] = frame[column].astype(float) - frame["actual"].astype(float)
        frame[abs_error_column] = frame[error_column].abs()
        metrics[label] = {
            "metric": metric_frame(frame, column, "actual", np),
            "by_lead": grouped_metrics(frame, ["lead_time_minutes"], column, "actual", np, pd),
            "by_spot": grouped_metrics(frame, ["spot_id"], column, "actual", np, pd, limit=limit),
            "by_actual_wind_bin": grouped_metrics(frame, ["actual_wind_bin_ms"], column, "actual", np, pd),
        }
    return metrics


def compute_oracle(frame: Any, labels: list[str], np: Any, pd: Any, limit: int) -> dict[str, Any]:
    abs_error_columns = [f"abs_err__{sanitize_label(label)}" for label in labels]
    pred_columns = [f"pred__{sanitize_label(label)}" for label in labels]
    abs_errors = frame[abs_error_columns].to_numpy(dtype=float)
    best_index = np.nanargmin(abs_errors, axis=1)
    oracle_prediction = np.take_along_axis(frame[pred_columns].to_numpy(dtype=float), best_index[:, None], axis=1)[:, 0]
    out = frame.copy()
    out["oracle_prediction"] = oracle_prediction
    out["oracle_model"] = [labels[int(index)] for index in best_index]
    selection_counts = (
        out["oracle_model"]
        .value_counts(dropna=False)
        .rename_axis("model")
        .reset_index(name="count")
    )
    selection = []
    for _, row in selection_counts.iterrows():
        count = int(row["count"])
        selection.append({
            "model": str(row["model"]),
            "count": count,
            "share": round(count / max(len(out), 1), 6),
        })
    return {
        "metric": metric_frame(out, "oracle_prediction", "actual", np),
        "selection": selection,
        "by_lead": grouped_metrics(out, ["lead_time_minutes"], "oracle_prediction", "actual", np, pd),
        "by_spot": grouped_metrics(out, ["spot_id"], "oracle_prediction", "actual", np, pd, limit=limit),
        "by_actual_wind_bin": grouped_metrics(out, ["actual_wind_bin_ms"], "oracle_prediction", "actual", np, pd),
    }


def compute_pairwise(
    frame: Any,
    base_label: str,
    labels: list[str],
    margin: float,
    np: Any,
) -> list[dict[str, Any]]:
    base_abs = frame[f"abs_err__{sanitize_label(base_label)}"].astype(float)
    base_pred_col = f"pred__{sanitize_label(base_label)}"
    rows = []
    for label in labels:
        if label == base_label:
            continue
        alt_abs = frame[f"abs_err__{sanitize_label(label)}"].astype(float)
        alt_pred_col = f"pred__{sanitize_label(label)}"
        oracle_pred = frame[base_pred_col].where(base_abs <= alt_abs, frame[alt_pred_col])
        alt_better = alt_abs + margin < base_abs
        base_better = base_abs + margin < alt_abs
        rows.append({
            "alt_model": label,
            "base_metric": metric_frame(frame, base_pred_col, "actual", np),
            "alt_metric": metric_frame(frame, alt_pred_col, "actual", np),
            "pair_oracle_metric": metric(oracle_pred.to_numpy(), frame["actual"].to_numpy(), np),
            "alt_better_by_margin_count": int(alt_better.sum()),
            "alt_better_by_margin_share": round(float(alt_better.mean()), 6),
            "base_better_by_margin_count": int(base_better.sum()),
            "base_better_by_margin_share": round(float(base_better.mean()), 6),
            "same_side_error_share": round(float(((frame[f"err__{sanitize_label(base_label)}"] * frame[f"err__{sanitize_label(label)}"]) > 0).mean()), 6),
            "error_correlation": round(float(np.corrcoef(
                frame[f"err__{sanitize_label(base_label)}"].astype(float).to_numpy(),
                frame[f"err__{sanitize_label(label)}"].astype(float).to_numpy(),
            )[0, 1]), 6),
        })
    rows.sort(key=lambda item: item["pair_oracle_metric"].get("rmse", float("inf")))
    return rows


def compute_error_correlation(frame: Any, labels: list[str], np: Any) -> list[dict[str, Any]]:
    rows = []
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            left_errors = frame[f"err__{sanitize_label(left)}"].astype(float).to_numpy()
            right_errors = frame[f"err__{sanitize_label(right)}"].astype(float).to_numpy()
            if len(left_errors) < 2:
                corr = None
            else:
                corr = round(float(np.corrcoef(left_errors, right_errors)[0, 1]), 6)
            rows.append({"left": left, "right": right, "error_correlation": corr})
    rows.sort(key=lambda item: (item["error_correlation"] is None, item["error_correlation"]))
    return rows


def pct_gain(base: float | None, new: float | None) -> float | None:
    if base is None or new is None or base == 0:
        return None
    return round((float(base) - float(new)) / float(base) * 100.0, 3)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    base_label = result["base_model"]
    base_metric = result["intersection_metrics"][base_label]["metric"]
    oracle_metric = result["oracle"]["metric"]
    lines = [
        "# Prediction Complementarity Audit",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Verdict: `{result['verdict']}`",
        f"Base model: `{base_label}`",
        f"Intersection rows: `{result['intersection_rows']}`",
        f"Actual max delta across files: `{result['actual_max_abs_delta']}`",
        "",
        "## Intersection Metrics",
        "",
        "| Model | Rows | RMSE | MAE | Bias |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, payload in sorted(
        result["intersection_metrics"].items(),
        key=lambda item: item[1]["metric"].get("rmse", float("inf")),
    ):
        metric_payload = payload["metric"]
        lines.append(
            f"| `{label}` | {metric_payload.get('count')} | {metric_payload.get('rmse')} | "
            f"{metric_payload.get('mae')} | {metric_payload.get('bias')} |"
        )
    lines.extend([
        "",
        "## Oracle",
        "",
        f"- Base RMSE: `{base_metric.get('rmse')}`",
        f"- Oracle RMSE: `{oracle_metric.get('rmse')}`",
        f"- Oracle gain vs base: `{result['oracle_gain_pct_vs_base']}%`",
        f"- Oracle gap to target RMSE: `{result['oracle_gap_to_threshold']}`",
        "",
        "| Selected model | Count | Share |",
        "| --- | ---: | ---: |",
    ])
    for item in result["oracle"]["selection"]:
        lines.append(f"| `{item['model']}` | {item['count']} | {item['share']} |")
    lines.extend([
        "",
        "## Pairwise Oracle Vs Base",
        "",
        "| Alt model | Alt RMSE | Pair oracle RMSE | Alt better share | Error corr | Same side error share |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in result["pairwise_vs_base"]:
        lines.append(
            f"| `{item['alt_model']}` | {item['alt_metric'].get('rmse')} | "
            f"{item['pair_oracle_metric'].get('rmse')} | "
            f"{item['alt_better_by_margin_share']} | {item['error_correlation']} | "
            f"{item['same_side_error_share']} |"
        )
    lines.extend([
        "",
        "## Oracle By Lead",
        "",
        "| Lead | Rows | RMSE | MAE | Bias |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in result["oracle"]["by_lead"]:
        lines.append(
            f"| `{item['group'].get('lead_time_minutes')}` | {item.get('count')} | "
            f"{item.get('rmse')} | {item.get('mae')} | {item.get('bias')} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        result["interpretation"],
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def interpret(result: dict[str, Any], args: argparse.Namespace) -> str:
    gain = result.get("oracle_gain_pct_vs_base")
    rows = result.get("intersection_rows", 0)
    if rows < args.min_intersection_rows:
        return (
            "The intersection is too small to make a strong decision. Use this as "
            "a smoke signal only, not as production evidence."
        )
    if gain is None:
        return "Oracle gain could not be computed."
    if gain >= args.strong_oracle_gain_pct:
        return (
            "There is strong complementary signal: a row-wise oracle is much "
            "better than the base model. The next experiment should be a "
            "leakage-safe router/blender trained on a calibration period."
        )
    if gain >= args.weak_oracle_gain_pct:
        return (
            "There is some complementary signal, but not enough to assume a "
            "router will beat the champion. Try per-lead/per-spot routing and "
            "validate with a strict calibration/evaluation split."
        )
    return (
        "The oracle gain is weak. The candidate models make mostly overlapping "
        "errors with the base model, so adding model families is unlikely to "
        "move RMSE materially unless new target quality or genuinely new input "
        "signals are introduced."
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    specs = [parse_model_spec(raw, args.actual_column) for raw in args.model]
    labels = [spec.label for spec in specs]
    if len(labels) != len(set(labels)):
        raise SystemExit("Model labels must be unique.")
    base_label = args.base_model or labels[0]
    if base_label not in labels:
        raise SystemExit(f"--base-model {base_label!r} is not in model labels: {labels}")

    frames, native_metrics = load_models(specs, args, pd, np)
    intersection, actual_max_delta = merge_intersection(frames, specs, pd)
    if actual_max_delta > args.max_actual_delta:
        raise SystemExit(
            f"Actual labels differ across prediction files by {actual_max_delta}, "
            f"above --max-actual-delta {args.max_actual_delta}."
        )
    intersection = add_time_bins(intersection, pd)
    intersection_metrics = compute_intersection_metrics(intersection, labels, np, pd, args.limit)
    oracle = compute_oracle(intersection, labels, np, pd, args.limit)
    pairwise = compute_pairwise(intersection, base_label, labels, args.margin, np)
    error_correlation = compute_error_correlation(intersection, labels, np)

    base_rmse = intersection_metrics[base_label]["metric"].get("rmse")
    oracle_rmse = oracle["metric"].get("rmse")
    result = {
        "format": "corsewind.prediction_complementarity_audit.v1",
        "generated_at_utc": utc_now(),
        "base_model": base_label,
        "threshold_rmse": args.threshold_rmse,
        "models": [
            {
                "label": spec.label,
                "path": str(spec.path),
                "prediction_column": spec.prediction_column,
                "actual_column": spec.actual_column,
            }
            for spec in specs
        ],
        "native_metrics": native_metrics,
        "intersection_rows": int(len(intersection)),
        "actual_max_abs_delta": round(float(actual_max_delta), 9),
        "intersection_metrics": intersection_metrics,
        "oracle": oracle,
        "oracle_gain_pct_vs_base": pct_gain(base_rmse, oracle_rmse),
        "oracle_gap_to_threshold": None if oracle_rmse is None else round(float(oracle_rmse) - args.threshold_rmse, 6),
        "pairwise_vs_base": pairwise,
        "error_correlation": error_correlation,
        "verdict": "needs_router" if pct_gain(base_rmse, oracle_rmse) and pct_gain(base_rmse, oracle_rmse) >= args.weak_oracle_gain_pct else "weak_complementarity",
    }
    result["interpretation"] = interpret(result, args)

    args.output_root.mkdir(parents=True, exist_ok=True)
    json_path = args.output_root / "prediction_complementarity_audit.json"
    md_path = args.output_root / "prediction_complementarity_audit.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(md_path, result)
    if args.output_intersection:
        keep = [
            *KEY_COLUMNS,
            "actual",
            "issue_hour_utc",
            "issue_month",
            "actual_wind_bin_ms",
            *[f"pred__{sanitize_label(label)}" for label in labels],
            *[f"err__{sanitize_label(label)}" for label in labels],
            *[f"abs_err__{sanitize_label(label)}" for label in labels],
        ]
        intersection[keep].to_parquet(args.output_root / "prediction_intersection.parquet", index=False)
    print(json.dumps({
        "output_root": str(args.output_root),
        "verdict": result["verdict"],
        "intersection_rows": result["intersection_rows"],
        "base_rmse": base_rmse,
        "oracle_rmse": oracle_rmse,
        "oracle_gain_pct_vs_base": result["oracle_gain_pct_vs_base"],
    }, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec: label|path|prediction_column or label|path|prediction_column|actual_column.",
    )
    parser.add_argument("--base-model", default="", help="Model label used as the champion/base comparison.")
    parser.add_argument("--actual-column", default="actual_wind_mean_ms")
    parser.add_argument("--split-column", default="benchmark_split")
    parser.add_argument("--split", default="eval", help="Filter this split when --split-column exists; empty disables.")
    parser.add_argument("--lead-minute", action="append", type=int, default=[], help="Keep only these lead minutes.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-intersection", action="store_true")
    parser.add_argument("--threshold-rmse", type=float, default=0.9)
    parser.add_argument("--max-actual-delta", type=float, default=1e-5)
    parser.add_argument("--margin", type=float, default=0.25, help="Minimum absolute-error improvement counted as a meaningful row win.")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--min-intersection-rows", type=int, default=1000)
    parser.add_argument("--weak-oracle-gain-pct", type=float, default=3.0)
    parser.add_argument("--strong-oracle-gain-pct", type=float, default=8.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
