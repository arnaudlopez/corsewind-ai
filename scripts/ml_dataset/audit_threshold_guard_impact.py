#!/usr/bin/env python3
"""Audit threshold_guard_v1 impact by target, group, and threshold."""

from __future__ import annotations

import argparse
import glob
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGETS = {
    "wind": {
        "actual": "actual_wind_mean_kt",
        "candidate": "threshold_guard_v1_wind_mean_kt",
        "baselines": {
            "raw": "raw_wind_mean_kt",
            "champion": "champion_wind_mean_kt",
            "router": "shadow_router_v1_wind_mean_kt",
        },
        "thresholds": (12.0, 15.0, 20.0, 25.0),
        "source": "threshold_guard_v1_wind_mean_source",
    },
    "gust": {
        "actual": "actual_gust_kt",
        "candidate": "threshold_guard_v1_gust_kt",
        "baselines": {
            "raw": "raw_gust_kt",
            "champion": "champion_gust_kt",
            "high": "gust_high_kt",
            "guarded_stacker": "shadow_guarded_stacker_v1_gust_kt",
        },
        "thresholds": (12.0, 15.0, 20.0, 25.0),
        "source": "threshold_guard_v1_gust_source",
    },
}

GROUP_COLUMNS = (
    "spot_id",
    "lead_bucket",
    "actual_wind_regime_kt",
    "actual_gust_regime_kt",
    "target_hour_utc",
)


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


def metric(frame: Any, pred_col: str, actual_col: str) -> dict[str, Any]:
    if pred_col not in frame.columns or actual_col not in frame.columns:
        return {"n": 0}
    values = frame[[pred_col, actual_col]].dropna()
    n = int(len(values))
    if n == 0:
        return {"n": 0}
    err = values[pred_col].astype(float) - values[actual_col].astype(float)
    abs_err = err.abs()
    return {
        "n": n,
        "rmse": math.sqrt(float((err**2).mean())),
        "mae": float(abs_err.mean()),
        "bias": float(err.mean()),
        "p90_abs_error": float(abs_err.quantile(0.90)),
    }


def threshold_metric(frame: Any, pred_col: str, actual_col: str, threshold: float) -> dict[str, Any]:
    if pred_col not in frame.columns or actual_col not in frame.columns:
        return {"n": 0, "threshold": threshold}
    values = frame[[pred_col, actual_col]].dropna()
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


def target_summary(frame: Any, target: str) -> dict[str, Any]:
    spec = TARGETS[target]
    actual = spec["actual"]
    candidate = spec["candidate"]
    baselines = spec["baselines"]
    candidate_metric = metric(frame, candidate, actual)
    baseline_metrics = {name: metric(frame, column, actual) for name, column in baselines.items()}
    thresholds = {
        f"{int(level)}kt": {
            "candidate": threshold_metric(frame, candidate, actual, level),
            **{name: threshold_metric(frame, column, actual, level) for name, column in baselines.items()},
        }
        for level in spec["thresholds"]
    }
    source_col = spec["source"]
    source_share = {}
    if source_col in frame.columns:
        source_share = frame[source_col].astype(str).value_counts(normalize=True).round(6).to_dict()
    return {
        "candidate": candidate_metric,
        "baselines": baseline_metrics,
        "rmse_gain_vs_baseline": {
            name: None
            if candidate_metric.get("rmse") is None or item.get("rmse") is None
            else float(item["rmse"]) - float(candidate_metric["rmse"])
            for name, item in baseline_metrics.items()
        },
        "thresholds": thresholds,
        "source_share": source_share,
    }


def grouped_summary(frame: Any, group_col: str) -> dict[str, Any]:
    if group_col not in frame.columns:
        return {}
    out: dict[str, Any] = {}
    for value, group in frame.groupby(group_col, dropna=False):
        label = str(value)
        out[label] = {
            "rows": int(len(group)),
            "wind": target_summary(group, "wind"),
            "gust": target_summary(group, "gust"),
        }
    return out


def group_risk_flags(summary: dict[str, Any], *, min_rows: int, max_rmse_regression: float) -> dict[str, Any]:
    flags: list[dict[str, Any]] = []
    by_target = {"wind": [], "gust": []}
    for group_name, groups in (summary.get("groups") or {}).items():
        for value, group in groups.items():
            rows = int(group.get("rows") or 0)
            if rows < min_rows:
                continue
            for target in ("wind", "gust"):
                target_item = group.get(target) or {}
                candidate_rmse = (target_item.get("candidate") or {}).get("rmse")
                baseline_metrics = target_item.get("baselines") or {}
                baseline_rmses = {
                    name: metric_item.get("rmse")
                    for name, metric_item in baseline_metrics.items()
                    if metric_item.get("rmse") is not None
                }
                if candidate_rmse is None or not baseline_rmses:
                    continue
                best_baseline_name, best_baseline_rmse = min(baseline_rmses.items(), key=lambda item: item[1])
                regression = float(candidate_rmse) - float(best_baseline_rmse)
                if regression > max_rmse_regression:
                    flag = {
                        "target": target,
                        "group": group_name,
                        "value": value,
                        "rows": rows,
                        "candidate_rmse": float(candidate_rmse),
                        "best_baseline": best_baseline_name,
                        "best_baseline_rmse": float(best_baseline_rmse),
                        "regression": regression,
                        "max_allowed_regression": max_rmse_regression,
                    }
                    flags.append(flag)
                    by_target[target].append(flag)
    return {
        "min_rows": min_rows,
        "max_rmse_regression": max_rmse_regression,
        "flag_count": len(flags),
        "by_target": {target: {"flag_count": len(items), "flags": items} for target, items in by_target.items()},
        "flags": flags,
    }


def load_frame(paths: list[Path], pd: Any) -> Any:
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        raise SystemExit("No input parquet files found.")
    frame = pd.concat(frames, ignore_index=True)
    if "lead_bucket" not in frame.columns and "lead_time_minutes" in frame.columns:
        lead = pd.to_numeric(frame["lead_time_minutes"], errors="coerce")
        frame["lead_bucket"] = lead.map(
            lambda value: "0-1h"
            if value <= 60
            else "1-3h"
            if value <= 180
            else "3-6h"
            if value <= 360
            else "6h+"
        )
    return frame


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]
    paths = expand_inputs(args.scored_parquet)
    frame = load_frame(paths, pd)
    summary = {
        "format": "corsewind.threshold_guard_impact_audit.v1",
        "generated_at_utc": utc_now(),
        "scored_parquet": [str(path) for path in paths],
        "rows": int(len(frame)),
        "overall": {
            "wind": target_summary(frame, "wind"),
            "gust": target_summary(frame, "gust"),
        },
        "groups": {column: grouped_summary(frame, column) for column in GROUP_COLUMNS},
    }
    summary["risk_flags"] = group_risk_flags(
        summary,
        min_rows=args.min_group_rows,
        max_rmse_regression=args.max_group_rmse_regression_kt,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return summary


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_target_table(summary: dict[str, Any], target: str) -> list[str]:
    item = summary["overall"][target]
    lines = [
        f"## {target.title()} Overall",
        "",
        "| Rail | RMSE kt | MAE kt | Bias kt | Gain vs Candidate kt |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    candidate = item["candidate"]
    lines.append(f"| `threshold_guard` | {fmt(candidate.get('rmse'))} | {fmt(candidate.get('mae'))} | {fmt(candidate.get('bias'))} |  |")
    for name, metric_item in item["baselines"].items():
        gain = item["rmse_gain_vs_baseline"].get(name)
        lines.append(
            f"| `{name}` | {fmt(metric_item.get('rmse'))} | {fmt(metric_item.get('mae'))} | "
            f"{fmt(metric_item.get('bias'))} | {fmt(gain)} |"
        )
    lines.extend(["", "### Threshold CSI", "", "| Threshold | Candidate | " + " | ".join(item["baselines"].keys()) + " |"])
    lines.append("| --- |" + " ---: |" * (len(item["baselines"]) + 1))
    for threshold, metrics_by_rail in item["thresholds"].items():
        values = [fmt((metrics_by_rail.get("candidate") or {}).get("csi"))]
        values.extend(fmt((metrics_by_rail.get(name) or {}).get("csi")) for name in item["baselines"])
        lines.append(f"| `{threshold}` | " + " | ".join(values) + " |")
    lines.append("")
    return lines


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Threshold Guard Impact Audit",
        "",
        f"- generated: `{summary['generated_at_utc']}`",
        f"- rows: `{summary['rows']}`",
        "",
    ]
    lines.extend(render_target_table(summary, "wind"))
    lines.extend(render_target_table(summary, "gust"))
    risk = summary.get("risk_flags") or {}
    lines.extend(
        [
            "## Risk Flags",
            "",
            f"- min rows: `{risk.get('min_rows')}`",
            f"- max RMSE regression kt: `{risk.get('max_rmse_regression')}`",
            f"- flags: `{risk.get('flag_count')}`",
            "",
            "| Target | Group | Value | Rows | Candidate RMSE kt | Best Baseline | Baseline RMSE kt | Regression kt |",
            "| --- | --- | --- | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for flag in risk.get("flags") or []:
        lines.append(
            f"| `{flag.get('target')}` | `{flag.get('group')}` | `{flag.get('value')}` | "
            f"{flag.get('rows')} | {fmt(flag.get('candidate_rmse'))} | `{flag.get('best_baseline')}` | "
            f"{fmt(flag.get('best_baseline_rmse'))} | {fmt(flag.get('regression'))} |"
        )
    lines.append("")
    lines.extend(["## Group Risks", "", "| Group | Value | Target | Rows | Candidate RMSE kt | Best Baseline RMSE kt |", "| --- | --- | --- | ---: | ---: | ---: |"])
    for group_name, groups in summary.get("groups", {}).items():
        for value, group in groups.items():
            rows = group.get("rows")
            for target in ("wind", "gust"):
                target_item = group.get(target) or {}
                candidate_rmse = (target_item.get("candidate") or {}).get("rmse")
                baseline_rmses = [
                    metric_item.get("rmse")
                    for metric_item in (target_item.get("baselines") or {}).values()
                    if metric_item.get("rmse") is not None
                ]
                best_baseline = min(baseline_rmses) if baseline_rmses else None
                lines.append(
                    f"| `{group_name}` | `{value}` | `{target}` | {rows} | {fmt(candidate_rmse)} | {fmt(best_baseline)} |"
                )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-parquet", action="append", required=True)
    parser.add_argument("--min-group-rows", type=int, default=20)
    parser.add_argument("--max-group-rmse-regression-kt", type=float, default=0.10)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
