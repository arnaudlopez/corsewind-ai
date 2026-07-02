#!/usr/bin/env python3
"""Audit local impact for all shadow promotion candidates."""

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
        "candidates": {
            "threshold_guard": "threshold_guard_v1_wind_mean_kt",
            "high_event_guard": "wind_high_event_guard_v1_wind_mean_kt",
        },
        "baselines": {
            "raw": "raw_wind_mean_kt",
            "champion": "champion_wind_mean_kt",
            "strong_gated": "strong_gated_wind_mean_kt",
            "router": "shadow_router_v1_wind_mean_kt",
            "threshold_guard": "threshold_guard_v1_wind_mean_kt",
        },
    },
    "gust": {
        "actual": "actual_gust_kt",
        "candidates": {
            "threshold_guard": "threshold_guard_v1_gust_kt",
            "local_fallback_guard": "local_fallback_guard_v1_gust_kt",
        },
        "baselines": {
            "raw": "raw_gust_kt",
            "champion": "champion_gust_kt",
            "high": "gust_high_kt",
            "guarded_stacker": "shadow_guarded_stacker_v1_gust_kt",
            "threshold_guard": "threshold_guard_v1_gust_kt",
        },
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
    if values.empty:
        return {"n": 0}
    err = values[pred_col].astype(float) - values[actual_col].astype(float)
    return {
        "n": int(len(values)),
        "rmse": math.sqrt(float((err**2).mean())),
        "mae": float(err.abs().mean()),
        "bias": float(err.mean()),
    }


def load_frame(paths: list[Path], pd: Any) -> Any:
    frames = [pd.read_parquet(path) for path in paths if path.exists()]
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


def candidate_group_summary(frame: Any, target: str, candidate: str) -> dict[str, Any]:
    spec = TARGETS[target]
    actual = spec["actual"]
    candidate_col = spec["candidates"][candidate]
    candidate_metric = metric(frame, candidate_col, actual)
    baselines = {
        name: metric(frame, col, actual)
        for name, col in spec["baselines"].items()
        if name != candidate and col in frame.columns
    }
    baseline_rmses = {name: item.get("rmse") for name, item in baselines.items() if item.get("rmse") is not None}
    best_baseline_name = None
    best_baseline_rmse = None
    if baseline_rmses:
        best_baseline_name, best_baseline_rmse = min(baseline_rmses.items(), key=lambda item: item[1])
    return {
        "candidate": candidate_metric,
        "baselines": baselines,
        "best_baseline": best_baseline_name,
        "best_baseline_rmse": best_baseline_rmse,
        "rmse_regression_vs_best": None
        if candidate_metric.get("rmse") is None or best_baseline_rmse is None
        else float(candidate_metric["rmse"]) - float(best_baseline_rmse),
    }


def audit_candidate(frame: Any, target: str, candidate: str, group_columns: tuple[str, ...]) -> dict[str, Any]:
    overall = candidate_group_summary(frame, target, candidate)
    groups: dict[str, Any] = {}
    for group_col in group_columns:
        if group_col not in frame.columns:
            continue
        group_items = {}
        for value, group in frame.groupby(group_col, dropna=False):
            group_items[str(value)] = {
                "rows": int(len(group)),
                **candidate_group_summary(group, target, candidate),
            }
        groups[group_col] = group_items
    return {"overall": overall, "groups": groups}


def risk_flags(audits: dict[str, Any], *, min_rows: int, max_regression: float) -> dict[str, Any]:
    flags: list[dict[str, Any]] = []
    by_target_candidate: dict[str, dict[str, Any]] = {}
    for target, candidates in audits.items():
        by_target_candidate.setdefault(target, {})
        for candidate, audit in candidates.items():
            candidate_flags = []
            for group_name, groups in (audit.get("groups") or {}).items():
                for value, item in groups.items():
                    rows = int(item.get("rows") or 0)
                    regression = item.get("rmse_regression_vs_best")
                    if rows < min_rows or regression is None or float(regression) <= max_regression:
                        continue
                    flag = {
                        "target": target,
                        "candidate": candidate,
                        "group": group_name,
                        "value": value,
                        "rows": rows,
                        "candidate_rmse": (item.get("candidate") or {}).get("rmse"),
                        "best_baseline": item.get("best_baseline"),
                        "best_baseline_rmse": item.get("best_baseline_rmse"),
                        "regression": float(regression),
                        "max_allowed_regression": max_regression,
                    }
                    flags.append(flag)
                    candidate_flags.append(flag)
            by_target_candidate[target][candidate] = {
                "flag_count": len(candidate_flags),
                "flags": sorted(candidate_flags, key=lambda flag: float(flag["regression"]) * int(flag["rows"]), reverse=True),
            }
    return {
        "min_rows": min_rows,
        "max_rmse_regression": max_regression,
        "flag_count": len(flags),
        "by_target_candidate": by_target_candidate,
        "flags": sorted(flags, key=lambda flag: float(flag["regression"]) * int(flag["rows"]), reverse=True),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]
    paths = expand_inputs(args.scored_parquet)
    frame = load_frame(paths, pd)
    audits = {
        target: {
            candidate: audit_candidate(frame, target, candidate, GROUP_COLUMNS)
            for candidate, col in spec["candidates"].items()
            if col in frame.columns
        }
        for target, spec in TARGETS.items()
    }
    result = {
        "format": "corsewind.shadow_candidate_impact_audit.v1",
        "generated_at_utc": utc_now(),
        "scored_parquet": [str(path) for path in paths],
        "rows": int(len(frame)),
        "candidates": audits,
    }
    result["risk_flags"] = risk_flags(
        audits,
        min_rows=args.min_group_rows,
        max_regression=args.max_group_rmse_regression_kt,
    )
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
        "# Shadow Candidate Impact Audit",
        "",
        f"- generated: `{result['generated_at_utc']}`",
        f"- rows: `{result['rows']}`",
        "",
        "## Overall",
        "",
        "| Target | Candidate | RMSE kt | Best Baseline | Baseline RMSE kt | Regression kt |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for target, candidates in (result.get("candidates") or {}).items():
        for candidate, audit in candidates.items():
            overall = audit.get("overall") or {}
            metric_item = overall.get("candidate") or {}
            lines.append(
                f"| `{target}` | `{candidate}` | {fmt(metric_item.get('rmse'))} | "
                f"`{overall.get('best_baseline')}` | {fmt(overall.get('best_baseline_rmse'))} | "
                f"{fmt(overall.get('rmse_regression_vs_best'))} |"
            )
    risk = result.get("risk_flags") or {}
    lines.extend(
        [
            "",
            "## Risk Flags",
            "",
            f"- min rows: `{risk.get('min_rows')}`",
            f"- max RMSE regression kt: `{risk.get('max_rmse_regression')}`",
            f"- flags: `{risk.get('flag_count')}`",
            "",
            "| Target | Candidate | Group | Value | Rows | Candidate RMSE kt | Best Baseline | Baseline RMSE kt | Regression kt |",
            "| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for flag in risk.get("flags") or []:
        lines.append(
            f"| `{flag.get('target')}` | `{flag.get('candidate')}` | `{flag.get('group')}` | `{flag.get('value')}` | "
            f"{flag.get('rows')} | {fmt(flag.get('candidate_rmse'))} | `{flag.get('best_baseline')}` | "
            f"{fmt(flag.get('best_baseline_rmse'))} | {fmt(flag.get('regression'))} |"
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
