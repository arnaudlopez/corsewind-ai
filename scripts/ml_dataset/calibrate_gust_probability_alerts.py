#!/usr/bin/env python3
"""Calibrate operational gust-alert thresholds from scored hindcast rows."""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SPECS = {
    "gust_ge_20kt": {
        "actual_threshold_kt": 20.0,
        "candidate_columns": [
            "prob_gust_ge_20kt_model",
            "prob_gust_ge_20kt",
            "prob_gust_ge_20kt_heuristic",
        ],
        "output_alert_column": "gust_alert_ge_20kt",
        "output_probability_column": "gust_alert_ge_20kt_probability",
    },
    "gust_ge_25kt": {
        "actual_threshold_kt": 25.0,
        "candidate_columns": [
            "prob_gust_ge_25kt_model",
            "prob_gust_ge_25kt",
            "prob_gust_ge_25kt_heuristic",
        ],
        "output_alert_column": "gust_alert_ge_25kt",
        "output_probability_column": "gust_alert_ge_25kt_probability",
    },
}


def import_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as exc:
        raise SystemExit("Missing calibration dependencies. Run inside the CorseWind ML venv.") from exc
    return {
        "average_precision_score": average_precision_score,
        "np": np,
        "pd": pd,
        "roc_auc_score": roc_auc_score,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)
    return sorted(dict.fromkeys(paths))


def threshold_metrics(y_true: Any, probability: Any, threshold: float) -> dict[str, Any]:
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


def score_probability_column(frame: Any, column: str, actual_threshold_kt: float, deps: dict[str, Any]) -> dict[str, Any]:
    np = deps["np"]
    values = frame[[column, "actual_gust_kt"]].dropna().copy()
    if values.empty:
        return {"column": column, "n": 0, "skipped": True}
    probability = values[column].astype(float).clip(lower=0.0, upper=1.0)
    y_true = (values["actual_gust_kt"].astype(float) >= actual_threshold_kt).astype(int)
    eps = 1e-15
    clipped = probability.clip(lower=eps, upper=1.0 - eps)
    thresholds = [i / 100.0 for i in range(1, 100)]
    threshold_scores = [threshold_metrics(y_true, probability, threshold) for threshold in thresholds]
    best = max(threshold_scores, key=lambda item: -1.0 if item["csi"] is None else float(item["csi"]))
    recall_first = {}
    for min_recall in (0.50, 0.70, 0.80, 0.90):
        candidates = [
            item for item in threshold_scores
            if item["recall"] is not None and item["recall"] >= min_recall
        ]
        recall_first[str(min_recall)] = max(
            candidates,
            key=lambda item: -1.0 if item["csi"] is None else float(item["csi"]),
        ) if candidates else None
    out = {
        "column": column,
        "n": int(len(values)),
        "positive_count": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "mean_probability": float(probability.mean()),
        "brier": float(((probability - y_true) ** 2).mean()),
        "log_loss": float((-(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped))).mean()),
        "best_csi_threshold": best,
        "recall_first_thresholds": recall_first,
        "skipped": False,
    }
    if y_true.nunique() > 1:
        out["roc_auc"] = float(deps["roc_auc_score"](y_true, probability))
        out["average_precision"] = float(deps["average_precision_score"](y_true, probability))
    return out


def choose_best_candidate(scores: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [
        score for score in scores
        if not score.get("skipped") and score.get("best_csi_threshold", {}).get("csi") is not None
    ]
    if not valid:
        return None
    return max(valid, key=lambda score: float(score["best_csi_threshold"]["csi"]))


def render_markdown(result: dict[str, Any]) -> str:
    lines = ["# Gust Probability Alert Calibration", ""]
    lines.append(f"Generated at: `{result['generated_at_utc']}`")
    lines.append("")
    lines.append("| Alert | Selected column | Threshold | CSI | Precision | Recall | TP | FP | FN |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for alert_name, alert in sorted(result["alerts"].items()):
        selected = alert.get("selected") or {}
        best = selected.get("best_csi_threshold") or {}
        lines.append(
            f"| `{alert_name}` | `{selected.get('column')}` | {best.get('threshold')} | "
            f"{best.get('csi')} | {best.get('precision')} | {best.get('recall')} | "
            f"{best.get('tp')} | {best.get('fp')} | {best.get('fn')} |"
        )
    lines.append("")
    lines.append("These thresholds are operational alert thresholds from the scored hindcast batch.")
    lines.append("They are not proof of calibrated probabilities.")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]
    paths = expand_paths(args.scored_parquet)
    if not paths:
        raise SystemExit("No scored parquet inputs matched.")
    frame = pd.concat([pd.read_parquet(path).assign(source_parquet=str(path)) for path in paths], ignore_index=True)
    if "actual_gust_kt" not in frame.columns:
        raise SystemExit("Input scored rows must contain actual_gust_kt.")

    result = {
        "format": "corsewind.gust_probability_alert_thresholds.v1",
        "generated_at_utc": utc_now(),
        "source_parquet_count": len(paths),
        "source_parquets": [str(path) for path in paths],
        "row_count": int(len(frame)),
        "alerts": {},
    }
    for alert_name, spec in DEFAULT_SPECS.items():
        candidate_scores = [
            score_probability_column(frame, column, spec["actual_threshold_kt"], deps)
            for column in spec["candidate_columns"]
            if column in frame.columns
        ]
        selected = choose_best_candidate(candidate_scores)
        result["alerts"][alert_name] = {
            "actual_threshold_kt": spec["actual_threshold_kt"],
            "candidate_scores": candidate_scores,
            "selected": selected,
            "probability_column": None if selected is None else selected["column"],
            "threshold": None if selected is None else selected["best_csi_threshold"]["threshold"],
            "output_alert_column": spec["output_alert_column"],
            "output_probability_column": spec["output_probability_column"],
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(result), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-parquet", action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
