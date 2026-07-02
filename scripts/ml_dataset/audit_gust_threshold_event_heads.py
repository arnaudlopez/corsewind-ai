#!/usr/bin/env python3
"""Audit gust threshold event heads on scored shadow rows."""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONTINUOUS_RAILS = {
    "raw": "raw_gust_kt",
    "champion": "champion_gust_kt",
    "high": "gust_high_kt",
    "router": "shadow_router_v1_gust_kt",
    "stacker": "shadow_stacker_v1_gust_kt",
    "guarded_stacker": "shadow_guarded_stacker_v1_gust_kt",
    "threshold_guard": "threshold_guard_v1_gust_kt",
    "local_fallback_guard": "local_fallback_guard_v1_gust_kt",
}

PROBABILITY_HEADS = {
    "prob_final_20": ("prob_gust_ge_20kt", 20.0),
    "prob_heuristic_20": ("prob_gust_ge_20kt_heuristic", 20.0),
    "prob_model_20": ("prob_gust_ge_20kt_model", 20.0),
    "prob_final_25": ("prob_gust_ge_25kt", 25.0),
    "prob_heuristic_25": ("prob_gust_ge_25kt_heuristic", 25.0),
    "prob_model_25": ("prob_gust_ge_25kt_model", 25.0),
}

ALERT_HEADS = {
    "alert_20": ("gust_alert_ge_20kt", 20.0),
    "alert_25": ("gust_alert_ge_25kt", 25.0),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def import_dependencies() -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Missing pandas. Run inside the CorseWind ML venv.") from exc
    return {"pd": pd}


def expand_inputs(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        matches = [Path(path) for path in sorted(glob.glob(item))]
        paths.extend(matches or [Path(item)])
    return paths


def event_counts(pred: Any, actual: Any) -> dict[str, Any]:
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    return {
        "n": int(len(actual)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
        "csi": None if tp + fp + fn == 0 else tp / (tp + fp + fn),
    }


def deterministic_metric(frame: Any, column: str, threshold: float) -> dict[str, Any]:
    if column not in frame.columns or "actual_gust_kt" not in frame.columns:
        return {"n": 0}
    values = frame[[column, "actual_gust_kt"]].dropna()
    if values.empty:
        return {"n": 0}
    pred = values[column].astype(float) >= threshold
    actual = values["actual_gust_kt"].astype(float) >= threshold
    return event_counts(pred, actual)


def probability_metric(frame: Any, column: str, threshold: float) -> dict[str, Any]:
    if column not in frame.columns or "actual_gust_kt" not in frame.columns:
        return {"n": 0}
    values = frame[[column, "actual_gust_kt"]].dropna()
    if values.empty:
        return {"n": 0}
    probability = values[column].astype(float).clip(lower=0.0, upper=1.0)
    actual = values["actual_gust_kt"].astype(float) >= threshold
    y = actual.astype(float)
    brier = float(((probability - y) ** 2).mean())
    best = None
    thresholds = [round(i / 100.0, 2) for i in range(5, 96, 5)]
    by_threshold = {}
    for cutoff in thresholds:
        item = event_counts(probability >= cutoff, actual)
        item["probability_cutoff"] = cutoff
        by_threshold[f"{cutoff:.2f}"] = item
        score = -1.0 if item.get("csi") is None else float(item["csi"])
        if best is None or score > (-1.0 if best.get("csi") is None else float(best["csi"])):
            best = item
    at_default = event_counts(probability >= 0.50, actual)
    at_default["probability_cutoff"] = 0.50
    return {
        "n": int(len(values)),
        "positive_count": int(actual.sum()),
        "positive_rate": float(actual.mean()),
        "mean_probability": float(probability.mean()),
        "brier": brier,
        "threshold_0p50": at_default,
        "best_csi_threshold": best or at_default,
        "by_probability_cutoff": by_threshold,
    }


def alert_metric(frame: Any, column: str, threshold: float) -> dict[str, Any]:
    if column not in frame.columns or "actual_gust_kt" not in frame.columns:
        return {"n": 0}
    values = frame[[column, "actual_gust_kt"]].dropna()
    if values.empty:
        return {"n": 0}
    pred = values[column].astype(bool)
    actual = values["actual_gust_kt"].astype(float) >= threshold
    return event_counts(pred, actual)


def audit(frame: Any, thresholds: list[float]) -> dict[str, Any]:
    by_threshold: dict[str, Any] = {}
    for threshold in thresholds:
        key = f"gust_ge_{int(threshold)}kt"
        deterministic = {
            name: deterministic_metric(frame, column, threshold)
            for name, column in CONTINUOUS_RAILS.items()
            if column in frame.columns
        }
        probability = {
            name: probability_metric(frame, column, head_threshold)
            for name, (column, head_threshold) in PROBABILITY_HEADS.items()
            if column in frame.columns and int(head_threshold) == int(threshold)
        }
        alerts = {
            name: alert_metric(frame, column, head_threshold)
            for name, (column, head_threshold) in ALERT_HEADS.items()
            if column in frame.columns and int(head_threshold) == int(threshold)
        }
        best_deterministic = best_by_csi(deterministic)
        best_probability = best_probability_by_csi(probability)
        by_threshold[key] = {
            "threshold_kt": threshold,
            "deterministic": deterministic,
            "probability": probability,
            "alerts": alerts,
            "best_deterministic": best_deterministic,
            "best_probability": best_probability,
        }
    return by_threshold


def best_by_csi(items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        {"name": name, **item}
        for name, item in items.items()
        if item.get("n") and item.get("csi") is not None
    ]
    if not candidates:
        return {}
    return max(candidates, key=lambda item: (float(item["csi"]), float(item.get("recall") or 0.0), -float(item.get("fp") or 0)))


def best_probability_by_csi(items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for name, item in items.items():
        best = item.get("best_csi_threshold") or {}
        if best.get("csi") is not None:
            candidates.append({"name": name, "brier": item.get("brier"), **best})
    if not candidates:
        return {}
    return max(candidates, key=lambda item: (float(item["csi"]), float(item.get("recall") or 0.0), -float(item.get("fp") or 0)))


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]
    paths = expand_inputs(args.scored_parquet)
    frames = [pd.read_parquet(path) for path in paths if path.exists()]
    if not frames:
        raise SystemExit("No scored parquet inputs found.")
    frame = pd.concat(frames, ignore_index=True)
    result = {
        "format": "corsewind.gust_threshold_event_head_audit.v1",
        "generated_at_utc": utc_now(),
        "scored_parquets": [str(path) for path in paths],
        "rows": int(len(frame)),
        "thresholds": audit(frame, args.threshold_kt),
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return result


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Gust Threshold Event Head Audit",
        "",
        f"- generated: `{result['generated_at_utc']}`",
        f"- rows: `{result['rows']}`",
        "",
    ]
    for threshold_name, item in result["thresholds"].items():
        lines.extend(
            [
                f"## `{threshold_name}`",
                "",
                "| Family | Head | CSI | Precision | Recall | TP | FP | FN | Extra |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for name, metric in (item.get("deterministic") or {}).items():
            lines.append(
                "| deterministic | "
                + " | ".join(
                    [
                        f"`{name}`",
                        fmt(metric.get("csi")),
                        fmt(metric.get("precision")),
                        fmt(metric.get("recall")),
                        fmt(metric.get("tp"), 0),
                        fmt(metric.get("fp"), 0),
                        fmt(metric.get("fn"), 0),
                        "",
                    ]
                )
                + " |"
            )
        for name, metric in (item.get("probability") or {}).items():
            best = metric.get("best_csi_threshold") or {}
            lines.append(
                "| probability | "
                + " | ".join(
                    [
                        f"`{name}`",
                        fmt(best.get("csi")),
                        fmt(best.get("precision")),
                        fmt(best.get("recall")),
                        fmt(best.get("tp"), 0),
                        fmt(best.get("fp"), 0),
                        fmt(best.get("fn"), 0),
                        f"cutoff `{fmt(best.get('probability_cutoff'))}`, brier `{fmt(metric.get('brier'))}`",
                    ]
                )
                + " |"
            )
        for name, metric in (item.get("alerts") or {}).items():
            lines.append(
                "| alert | "
                + " | ".join(
                    [
                        f"`{name}`",
                        fmt(metric.get("csi")),
                        fmt(metric.get("precision")),
                        fmt(metric.get("recall")),
                        fmt(metric.get("tp"), 0),
                        fmt(metric.get("fp"), 0),
                        fmt(metric.get("fn"), 0),
                        "",
                    ]
                )
                + " |"
            )
        best_det = item.get("best_deterministic") or {}
        best_prob = item.get("best_probability") or {}
        lines.extend(
            [
                "",
                f"- best deterministic: `{best_det.get('name')}` CSI `{fmt(best_det.get('csi'))}`",
                f"- best probability: `{best_prob.get('name')}` CSI `{fmt(best_prob.get('csi'))}` cutoff `{fmt(best_prob.get('probability_cutoff'))}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-parquet", action="append", required=True)
    parser.add_argument("--threshold-kt", type=float, action="append", default=[20.0, 25.0])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
