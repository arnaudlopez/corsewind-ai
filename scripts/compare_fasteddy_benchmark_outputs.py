#!/usr/bin/env python3
"""Summarize FastEddy benchmark outputs when NetCDF files are available."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "data/processed/benchmarks/fasteddy/benchmark_plan.json"
DEFAULT_REPORT_JSON = ROOT / "data/processed/benchmarks/fasteddy/comparison_report.json"
DEFAULT_REPORT_MD = ROOT / "reports/fasteddy_benchmark_comparison.md"
KNOTS_PER_MPS = 1.943844492


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def find_fasteddy_outputs(case: dict[str, Any]) -> list[Path]:
    case_dir = ROOT / case["case_dir"]
    return sorted((case_dir / "output").glob("FE_*"))


def summarize(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean_ms": math.nan, "max_ms": math.nan, "p90_ms": math.nan}
    return {
        "mean_ms": round(float(np.mean(finite)), 4),
        "max_ms": round(float(np.max(finite)), 4),
        "p90_ms": round(float(np.percentile(finite, 90)), 4),
        "mean_kt": round(float(np.mean(finite) * KNOTS_PER_MPS), 3),
        "max_kt": round(float(np.max(finite) * KNOTS_PER_MPS), 3),
    }


def read_speed(path: Path, output_height_m: float) -> tuple[np.ndarray | None, str | None]:
    try:
        from netCDF4 import Dataset
    except ImportError:
        return None, "netCDF4 not installed; run pip install -r requirements-benchmark.txt"
    try:
        dataset = Dataset(path)
    except OSError as exc:
        return None, str(exc)
    with dataset:
        variables = dataset.variables
        if "u" not in variables or "v" not in variables:
            return None, f"no u/v variables in {display_path(path)}"
        u = np.array(variables["u"][:], dtype=np.float32)
        v = np.array(variables["v"][:], dtype=np.float32)
        z_values = np.array(variables["z"][:], dtype=np.float32) if "z" in variables else None
    speed = np.hypot(u, v)
    while speed.ndim > 3:
        speed = speed[-1]
    if speed.ndim == 3:
        z_index = int(np.argmin(np.abs(z_values - output_height_m))) if z_values is not None and len(z_values) else 0
        speed = speed[z_index]
    return speed, None


def compare_case(case: dict[str, Any]) -> dict[str, Any]:
    outputs = find_fasteddy_outputs(case)
    result: dict[str, Any] = {
        "zone_id": case["zone"]["id"],
        "label": case["zone"]["label"],
        "outputs": [display_path(path) for path in outputs],
        "speed": None,
        "error": None,
    }
    if not outputs:
        result["error"] = "missing FastEddy output files"
        return result
    speed, error = read_speed(outputs[-1], case["domain"]["output_height_m"])
    result["error"] = error
    result["speed"] = summarize(speed) if speed is not None else None
    return result


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# FastEddy Benchmark Comparison",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Cases: `{len(report['cases'])}`",
        "",
    ]
    for case in report["cases"]:
        lines.extend(
            [
                f"## {case['label']}",
                "",
                f"- Outputs: `{case['outputs']}`",
                f"- Speed: `{case['speed'] or case['error']}`",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_REPORT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    report = {
        "format": "corsewind.fasteddy_benchmark.comparison.v1",
        "generated_at_utc": utc_now(),
        "plan": display_path(plan_path),
        "cases": [compare_case(case) for case in plan["cases"]],
    }
    json_path = args.json_output if args.json_output.is_absolute() else ROOT / args.json_output
    md_path = args.markdown_output if args.markdown_output.is_absolute() else ROOT / args.markdown_output
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"wrote {display_path(json_path)}")
    print(f"wrote {display_path(md_path)}")


if __name__ == "__main__":
    main()
