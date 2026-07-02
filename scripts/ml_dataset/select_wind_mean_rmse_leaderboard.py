#!/usr/bin/env python3
"""Select the best known leakage-audited wind-mean RMSE result."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def calibration_entry(path: Path) -> dict[str, Any] | None:
    payload = load_json(path)
    metrics = payload.get("calibrated_metrics") or {}
    rmse = as_float(metrics.get("rmse"))
    if rmse is None:
        return None
    return {
        "source_type": "prediction_residual_calibrator",
        "path": str(path),
        "run_id": path.parent.name,
        "verdict": payload.get("verdict") or ("achieved" if rmse < 0.9 else "not_achieved"),
        "rmse": rmse,
        "mae": as_float(metrics.get("mae")),
        "bias": as_float(metrics.get("bias")),
        "metric_count": as_int(metrics.get("count")),
        "threshold_rmse": as_float(payload.get("threshold_rmse")) or 0.9,
        "temporal_split_issue_time_utc": None,
        "evidence_kind": payload.get("format"),
        "warnings": [],
        "reasons": [] if rmse is not None else ["missing calibrated_metrics.rmse"],
    }


def tabular_audit_entry(path: Path) -> dict[str, Any] | None:
    payload = load_json(path)
    rmse = as_float(payload.get("corrected_rmse"))
    if rmse is None:
        return None
    return {
        "source_type": "tabular_rmse09_audit",
        "path": str(path),
        "run_id": payload.get("run_id") or path.parent.name,
        "verdict": payload.get("verdict"),
        "rmse": rmse,
        "mae": as_float(payload.get("corrected_mae")),
        "bias": None,
        "metric_count": as_int(payload.get("metric_count")),
        "threshold_rmse": as_float(payload.get("threshold_rmse")) or 0.9,
        "temporal_split_issue_time_utc": payload.get("temporal_split_issue_time_utc"),
        "evidence_kind": payload.get("format"),
        "warnings": payload.get("warnings") or [],
        "reasons": payload.get("reasons") or [],
    }


def tabular_selection_entries(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    entries = []
    for item in payload.get("runs") or []:
        rmse = as_float(item.get("corrected_rmse"))
        if rmse is None:
            continue
        entries.append({
            "source_type": "tabular_rmse09_selection_run",
            "path": item.get("path") or str(path),
            "selection_path": str(path),
            "run_id": item.get("run_id"),
            "verdict": item.get("verdict"),
            "rmse": rmse,
            "mae": as_float(item.get("corrected_mae")),
            "bias": None,
            "metric_count": as_int(item.get("metric_count")),
            "threshold_rmse": as_float(payload.get("threshold_rmse")) or 0.9,
            "temporal_split_issue_time_utc": item.get("temporal_split_issue_time_utc"),
            "evidence_kind": payload.get("format"),
            "warnings": item.get("warnings") or [],
            "reasons": item.get("reasons") or [],
        })
    return entries


def discover_inputs(args: argparse.Namespace) -> list[Path]:
    paths = list(args.input_json or [])
    for root in args.search_root or []:
        if not root.exists():
            continue
        patterns = (
            "**/calibration_results.json",
            "**/tabular_rmse09_audit.json",
            "**/hpa_tabular_rmse09_selection.json",
            "**/tabular_rmse09_selection.json",
        )
        for pattern in patterns:
            paths.extend(sorted(root.glob(pattern)))
    return sorted({path.resolve() for path in paths if path.exists()})


def entry_from_path(path: Path) -> list[dict[str, Any]]:
    name = path.name
    try:
        if name == "calibration_results.json":
            entry = calibration_entry(path)
            return [entry] if entry else []
        if name == "tabular_rmse09_audit.json":
            entry = tabular_audit_entry(path)
            return [entry] if entry else []
        if name in {"hpa_tabular_rmse09_selection.json", "tabular_rmse09_selection.json"}:
            return tabular_selection_entries(path)
    except (json.JSONDecodeError, OSError) as exc:
        return [{
            "source_type": "invalid_json",
            "path": str(path),
            "run_id": path.parent.name,
            "verdict": "invalid",
            "rmse": None,
            "mae": None,
            "bias": None,
            "metric_count": None,
            "threshold_rmse": 0.9,
            "temporal_split_issue_time_utc": None,
            "evidence_kind": None,
            "warnings": [],
            "reasons": [str(exc)],
        }]
    return []


def valid_for_selection(entry: dict[str, Any]) -> bool:
    if entry.get("rmse") is None:
        return False
    if entry.get("reasons"):
        return False
    if entry.get("verdict") == "invalid":
        return False
    return True


def build_leaderboard(paths: list[Path], threshold: float) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in paths:
        entries.extend(entry_from_path(path))
    valid = [entry for entry in entries if valid_for_selection(entry)]
    valid.sort(key=lambda item: (float(item["rmse"]), str(item.get("run_id"))))
    best = valid[0] if valid else None
    decision = "invalid"
    if best:
        decision = "achieved" if float(best["rmse"]) < threshold else "not_achieved"
    return {
        "format": "corsewind.wind_mean_rmse_leaderboard.v1",
        "generated_at_utc": utc_now(),
        "threshold_rmse": threshold,
        "decision": decision,
        "input_count": len(paths),
        "entry_count": len(entries),
        "valid_entry_count": len(valid),
        "best": best,
        "runs": sorted(
            entries,
            key=lambda item: (
                item.get("rmse") is None,
                float(item.get("rmse") or 999.0),
                str(item.get("run_id")),
            ),
        ),
    }


def write_markdown(path: Path, leaderboard: dict[str, Any]) -> None:
    best = leaderboard.get("best") or {}
    lines = [
        "# Wind Mean RMSE Leaderboard",
        "",
        f"Generated: `{leaderboard['generated_at_utc']}`",
        f"Decision: `{leaderboard['decision']}`",
        f"Threshold RMSE: `{leaderboard['threshold_rmse']}`",
        f"Entries: `{leaderboard['entry_count']}`",
        f"Valid entries: `{leaderboard['valid_entry_count']}`",
        "",
        "## Best Known Result",
        "",
    ]
    if best:
        lines.extend([
            f"- Run: `{best.get('run_id')}`",
            f"- Source: `{best.get('source_type')}`",
            f"- RMSE: `{best.get('rmse')}`",
            f"- MAE: `{best.get('mae')}`",
            f"- Metric rows: `{best.get('metric_count')}`",
            f"- Evidence: `{best.get('path')}`",
            f"- Gap to 0.9: `{round(float(best['rmse']) - float(leaderboard['threshold_rmse']), 6)}`",
        ])
    else:
        lines.append("- None.")
    lines.extend(["", "## Runs", "", "| Run | Source | Verdict | RMSE | MAE | Rows | Evidence |", "| --- | --- | --- | ---: | ---: | ---: | --- |"])
    for item in leaderboard.get("runs") or []:
        lines.append(
            f"| `{item.get('run_id')}` | `{item.get('source_type')}` | `{item.get('verdict')}` | "
            f"`{item.get('rmse')}` | `{item.get('mae')}` | `{item.get('metric_count')}` | `{item.get('path')}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, action="append", default=[])
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument("--threshold-rmse", type=float, default=0.9)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    leaderboard = build_leaderboard(discover_inputs(args), args.threshold_rmse)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(leaderboard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, leaderboard)
    print(json.dumps(leaderboard, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
