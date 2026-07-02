#!/usr/bin/env python3
"""Select a calibrator run from a validation sweep for later locked evaluation."""

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


def coverage_ok(run: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons = []
    train_rows = int(run.get("train_rows", 0) or 0)
    test_rows = int(run.get("test_rows", 0) or 0)
    if train_rows < args.min_train_rows:
        reasons.append(f"train_rows {train_rows} < {args.min_train_rows}")
    if test_rows < args.min_validation_rows:
        reasons.append(f"validation_rows {test_rows} < {args.min_validation_rows}")
    split = run.get("split_coverage", {}) if isinstance(run.get("split_coverage"), dict) else {}
    validation = split.get("test", {}) if isinstance(split.get("test"), dict) else {}
    for field, minimum in (
        ("unique_spots", args.min_validation_spots),
        ("unique_leads", args.min_validation_leads),
        ("unique_issue_days", args.min_validation_days),
    ):
        value = validation.get(field)
        if value is not None and int(value) < minimum:
            reasons.append(f"{field} {value} < {minimum}")
    return not reasons, reasons


def select_run(sweep: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    candidates = []
    rejected = []
    for run in sweep.get("runs", []):
        metric = run.get("metrics", {}).get("calibrator", {})
        rmse = as_float(metric.get("rmse"))
        run_name = str(run.get("run_name") or run.get("model_family"))
        if rmse is None:
            rejected.append({"run_name": run_name, "reasons": ["missing calibrator RMSE"]})
            continue
        ok, reasons = coverage_ok(run, args)
        if not ok:
            rejected.append({"run_name": run_name, "rmse": rmse, "reasons": reasons})
            continue
        candidates.append({
            "run_name": run_name,
            "model_family": run.get("model_family"),
            "fit_group": run.get("fit_group"),
            "rmse": rmse,
            "mae": metric.get("mae"),
            "bias": metric.get("bias"),
            "train_rows": run.get("train_rows"),
            "validation_rows": run.get("test_rows"),
            "split_coverage": run.get("split_coverage"),
        })
    candidates.sort(key=lambda item: (item["rmse"], item["run_name"]))
    selected = candidates[0] if candidates else None
    return {
        "format": "corsewind.sequence_calibrator_run_selection.v1",
        "generated_at_utc": utc_now(),
        "sweep_results": str(args.sweep_results),
        "selection_metric": "validation calibrator RMSE",
        "selected_run_name": selected["run_name"] if selected else None,
        "selected": selected,
        "candidate_count": len(candidates),
        "rejected_count": len(rejected),
        "candidates": candidates,
        "rejected": rejected,
        "train_end": sweep.get("train_end"),
        "eval_start": sweep.get("eval_start"),
        "benchmark_roots": sweep.get("benchmark_roots", []),
        "min_train_rows": args.min_train_rows,
        "min_validation_rows": args.min_validation_rows,
        "min_validation_spots": args.min_validation_spots,
        "min_validation_leads": args.min_validation_leads,
        "min_validation_days": args.min_validation_days,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Sequence Calibrator Run Selection",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Selected run: `{result['selected_run_name']}`",
        f"Candidate count: `{result['candidate_count']}`",
        f"Rejected count: `{result['rejected_count']}`",
        "",
        "## Candidates",
        "",
        "| Run | Family | Fit group | RMSE | MAE | Validation rows |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in result["candidates"]:
        fit_group = ", ".join(item.get("fit_group") or []) or "global"
        lines.append(
            f"| `{item['run_name']}` | `{item.get('model_family')}` | `{fit_group}` | "
            f"{item.get('rmse')} | {item.get('mae')} | {item.get('validation_rows')} |"
        )
    if not result["candidates"]:
        lines.append("| _none_ | | | | | |")
    lines.extend(["", "## Rejected", ""])
    if result["rejected"]:
        for item in result["rejected"]:
            lines.append(f"- `{item['run_name']}`: {', '.join(item.get('reasons') or [])}")
    else:
        lines.append("- None.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_results", type=Path)
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--min-validation-rows", type=int, default=200)
    parser.add_argument("--min-validation-spots", type=int, default=3)
    parser.add_argument("--min-validation-leads", type=int, default=4)
    parser.add_argument("--min-validation-days", type=int, default=20)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--fail-if-empty", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = select_run(load_json(args.sweep_results), args)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if args.fail_if_empty and result["selected_run_name"] is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
