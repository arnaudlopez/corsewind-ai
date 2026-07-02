#!/usr/bin/env python3
"""Audit wind-mean threshold event heads on scored shadow rows."""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONTINUOUS_RAILS = {
    "raw": "raw_wind_mean_kt",
    "champion": "champion_wind_mean_kt",
    "strong_gated": "strong_gated_wind_mean_kt",
    "router": "shadow_router_v1_wind_mean_kt",
    "stacker": "shadow_stacker_v1_wind_mean_kt",
    "guarded_stacker": "shadow_guarded_stacker_v1_wind_mean_kt",
    "threshold_guard": "threshold_guard_v1_wind_mean_kt",
    "high_event_guard": "wind_high_event_guard_v1_wind_mean_kt",
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
    if column not in frame.columns or "actual_wind_mean_kt" not in frame.columns:
        return {"n": 0}
    values = frame[[column, "actual_wind_mean_kt"]].dropna()
    if values.empty:
        return {"n": 0}
    pred = values[column].astype(float) >= threshold
    actual = values["actual_wind_mean_kt"].astype(float) >= threshold
    return event_counts(pred, actual)


def best_by_csi(items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        {"name": name, **item}
        for name, item in items.items()
        if item.get("n") and item.get("csi") is not None
    ]
    if not candidates:
        return {}
    return max(candidates, key=lambda item: (float(item["csi"]), float(item.get("recall") or 0.0), -float(item.get("fp") or 0)))


def audit(frame: Any, thresholds: list[float]) -> dict[str, Any]:
    by_threshold: dict[str, Any] = {}
    for threshold in thresholds:
        key = f"wind_ge_{int(threshold)}kt"
        deterministic = {
            name: deterministic_metric(frame, column, threshold)
            for name, column in CONTINUOUS_RAILS.items()
            if column in frame.columns
        }
        by_threshold[key] = {
            "threshold_kt": threshold,
            "deterministic": deterministic,
            "best_deterministic": best_by_csi(deterministic),
        }
    return by_threshold


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]
    paths = expand_inputs(args.scored_parquet)
    frames = [pd.read_parquet(path) for path in paths if path.exists()]
    if not frames:
        raise SystemExit("No scored parquet inputs found.")
    frame = pd.concat(frames, ignore_index=True)
    result = {
        "format": "corsewind.wind_threshold_event_head_audit.v1",
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
        "# Wind Threshold Event Head Audit",
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
                "| Head | CSI | Precision | Recall | TP | FP | FN |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, metric in (item.get("deterministic") or {}).items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{name}`",
                        fmt(metric.get("csi")),
                        fmt(metric.get("precision")),
                        fmt(metric.get("recall")),
                        fmt(metric.get("tp"), 0),
                        fmt(metric.get("fp"), 0),
                        fmt(metric.get("fn"), 0),
                    ]
                )
                + " |"
            )
        best = item.get("best_deterministic") or {}
        lines.extend(["", f"- best deterministic: `{best.get('name')}` CSI `{fmt(best.get('csi'))}`", ""])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-parquet", action="append", required=True)
    parser.add_argument("--threshold-kt", type=float, action="append", default=[12.0, 15.0, 20.0, 25.0])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
