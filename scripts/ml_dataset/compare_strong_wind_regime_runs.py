#!/usr/bin/env python3
"""Compare residual training runs on observed wind/gust regimes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_LABELS = {
    "labels__residual_wind_mean_ms": "wind mean",
    "labels__residual_gust_ms": "gust",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(run: dict[str, Any], target: str, path: list[str]) -> Any:
    value: Any = run.get("models", {}).get(target, {})
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def delta_pct(base: float | None, candidate: float | None) -> float | None:
    if base is None or candidate is None or base == 0:
        return None
    return round((base - candidate) / base * 100.0, 3)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare_target(base: dict[str, Any], candidate: dict[str, Any], target: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "target": target,
        "label": TARGET_LABELS.get(target, target),
    }
    for metric_name in ("rmse", "mae", "bias"):
        base_value = as_float(metric_value(base, target, ["corrected_nwp_test", metric_name]))
        candidate_value = as_float(metric_value(candidate, target, ["corrected_nwp_test", metric_name]))
        out[f"base_{metric_name}"] = base_value
        out[f"candidate_{metric_name}"] = candidate_value
        if metric_name in {"rmse", "mae"}:
            out[f"{metric_name}_gain_pct_vs_base"] = delta_pct(base_value, candidate_value)
    out["regimes"] = {}
    base_regimes = metric_value(base, target, ["corrected_nwp_by_observed_regime"]) or {}
    candidate_regimes = metric_value(candidate, target, ["corrected_nwp_by_observed_regime"]) or {}
    raw_regimes = metric_value(candidate, target, ["raw_nwp_by_observed_regime"]) or metric_value(base, target, ["raw_nwp_by_observed_regime"]) or {}
    for regime in sorted(set(base_regimes) | set(candidate_regimes) | set(raw_regimes)):
        base_item = base_regimes.get(regime, {})
        candidate_item = candidate_regimes.get(regime, {})
        raw_item = raw_regimes.get(regime, {})
        base_rmse = as_float(base_item.get("rmse"))
        candidate_rmse = as_float(candidate_item.get("rmse"))
        raw_rmse = as_float(raw_item.get("rmse"))
        out["regimes"][regime] = {
            "count": candidate_item.get("count") or base_item.get("count") or raw_item.get("count"),
            "raw_rmse": raw_rmse,
            "base_rmse": base_rmse,
            "candidate_rmse": candidate_rmse,
            "candidate_gain_pct_vs_base": delta_pct(base_rmse, candidate_rmse),
            "candidate_gain_pct_vs_raw": delta_pct(raw_rmse, candidate_rmse),
            "base_mae": as_float(base_item.get("mae")),
            "candidate_mae": as_float(candidate_item.get("mae")),
            "candidate_bias": as_float(candidate_item.get("bias")),
        }
    out["threshold_detection"] = {}
    base_detection = metric_value(base, target, ["corrected_nwp_threshold_detection"]) or {}
    candidate_detection = metric_value(candidate, target, ["corrected_nwp_threshold_detection"]) or {}
    raw_detection = metric_value(candidate, target, ["raw_nwp_threshold_detection"]) or metric_value(base, target, ["raw_nwp_threshold_detection"]) or {}
    for threshold in sorted(set(base_detection) | set(candidate_detection) | set(raw_detection)):
        out["threshold_detection"][threshold] = {
            "raw": raw_detection.get(threshold, {}),
            "base": base_detection.get(threshold, {}),
            "candidate": candidate_detection.get(threshold, {}),
        }
    return out


def render_markdown(result: dict[str, Any]) -> str:
    lines = ["# Strong Wind Regime Run Comparison", ""]
    lines.append(f"Generated: `{result['generated_at_utc']}`")
    lines.append(f"Base: `{result['base_results']}`")
    lines.append(f"Candidate: `{result['candidate_results']}`")
    lines.append("")
    lines.append("| Target | Base RMSE | Candidate RMSE | RMSE Gain | Base MAE | Candidate MAE | MAE Gain |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for target in result["targets"]:
        lines.append(
            f"| `{target['label']}` | {target.get('base_rmse')} | {target.get('candidate_rmse')} | "
            f"{target.get('rmse_gain_pct_vs_base')}% | {target.get('base_mae')} | "
            f"{target.get('candidate_mae')} | {target.get('mae_gain_pct_vs_base')}% |"
        )
    for target in result["targets"]:
        lines.extend(["", f"## {target['label']} By Observed Regime", ""])
        lines.append("| Regime | Count | Raw RMSE | Base RMSE | Candidate RMSE | Gain vs Base | Gain vs Raw | Candidate Bias |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for regime, item in target["regimes"].items():
            lines.append(
                f"| `{regime}` | {item.get('count')} | {item.get('raw_rmse')} | {item.get('base_rmse')} | "
                f"{item.get('candidate_rmse')} | {item.get('candidate_gain_pct_vs_base')}% | "
                f"{item.get('candidate_gain_pct_vs_raw')}% | {item.get('candidate_bias')} |"
            )
        lines.extend(["", "### Detection", ""])
        lines.append("| Threshold | Raw CSI | Base CSI | Candidate CSI | Raw Recall | Base Recall | Candidate Recall |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for threshold, item in target["threshold_detection"].items():
            lines.append(
                f"| `{threshold}` | {item.get('raw', {}).get('csi')} | {item.get('base', {}).get('csi')} | "
                f"{item.get('candidate', {}).get('csi')} | {item.get('raw', {}).get('recall')} | "
                f"{item.get('base', {}).get('recall')} | {item.get('candidate', {}).get('recall')} |"
            )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    base = load_results(args.base_results)
    candidate = load_results(args.candidate_results)
    targets = [
        target for target in TARGET_LABELS
        if target in base.get("models", {}) and target in candidate.get("models", {})
    ]
    if args.target:
        allowed = set(args.target)
        targets = [target for target in targets if target in allowed]
    result = {
        "format": "corsewind.strong_wind_regime_run_comparison.v1",
        "generated_at_utc": utc_now(),
        "base_results": str(args.base_results),
        "candidate_results": str(args.candidate_results),
        "base_run_id": base.get("run_id"),
        "candidate_run_id": candidate.get("run_id"),
        "targets": [compare_target(base, candidate, target) for target in targets],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(result), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
