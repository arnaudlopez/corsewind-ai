#!/usr/bin/env python3
"""Compare WindNinja and QES-Winds benchmark outputs."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "data/processed/benchmarks/qes_winds/benchmark_plan.json"
DEFAULT_STATUS = ROOT / "data/processed/benchmarks/qes_winds/benchmark_status.json"
DEFAULT_REPORT_JSON = ROOT / "data/processed/benchmarks/qes_winds/comparison_report.json"
DEFAULT_REPORT_MD = ROOT / "reports/qes_winds_benchmark_comparison.md"
KNOTS_PER_MPS = 1.943844492


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_ascii_grid(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    header: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for _ in range(6):
            key, value = handle.readline().split()[:2]
            header[key.lower()] = float(value)
        data = np.loadtxt(handle, dtype=np.float32)
    nodata = header.get("nodata_value", -9999.0)
    data[data <= nodata + 1e-3] = np.nan
    return data, header


def find_windninja_speed(case: dict[str, Any]) -> Path | None:
    output_dir = ROOT / case["windninja"]["case_dir"] / case["windninja"]["output_dir"]
    matches = sorted(output_dir.glob("*_vel.asc"))
    return matches[0] if matches else None


def read_qes_speed(case: dict[str, Any], output_height_m: float) -> tuple[np.ndarray | None, str | None]:
    nc_path = ROOT / case["qes_winds"]["expected_winds_out"]
    if not nc_path.exists():
        return None, f"missing {nc_path.relative_to(ROOT)}"
    try:
        from netCDF4 import Dataset
    except ImportError:
        return None, "netCDF4 not installed; run pip install -r requirements-benchmark.txt"

    with Dataset(nc_path) as dataset:
        variables = dataset.variables
        if "mag" in variables:
            mag = variables["mag"][:]
        elif "u" in variables and "v" in variables:
            mag = np.hypot(variables["u"][:], variables["v"][:])
        else:
            return None, f"no mag/u/v variable in {nc_path.relative_to(ROOT)}"
        z_values = np.array(variables["z"][:], dtype=np.float32) if "z" in variables else None
    values = np.array(mag, dtype=np.float32)
    if values.ndim == 4:
        values = values[0]
    if values.ndim != 3:
        return None, f"unexpected QES mag shape {values.shape}"
    if z_values is not None and len(z_values):
        z_index = int(np.argmin(np.abs(z_values - output_height_m)))
    else:
        z_index = min(values.shape[0] - 1, max(0, int(round(output_height_m)) - 1))
    return values[z_index], None


def summarize_speed(values: np.ndarray) -> dict[str, float]:
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


def compare_arrays(windninja: np.ndarray, qes: np.ndarray) -> dict[str, float]:
    rows = min(windninja.shape[0], qes.shape[0])
    cols = min(windninja.shape[1], qes.shape[1])
    wn = windninja[:rows, :cols]
    qe = qes[:rows, :cols]
    mask = np.isfinite(wn) & np.isfinite(qe)
    if not np.any(mask):
        return {"status": "no_overlap"}
    diff = qe[mask] - wn[mask]
    return {
        "status": "compared",
        "sample_count": int(mask.sum()),
        "bias_qes_minus_windninja_ms": round(float(np.mean(diff)), 4),
        "mae_ms": round(float(np.mean(np.abs(diff))), 4),
        "rmse_ms": round(float(np.sqrt(np.mean(diff * diff))), 4),
        "mae_kt": round(float(np.mean(np.abs(diff)) * KNOTS_PER_MPS), 3),
        "rmse_kt": round(float(np.sqrt(np.mean(diff * diff)) * KNOTS_PER_MPS), 3),
    }


def compare_case(case: dict[str, Any]) -> dict[str, Any]:
    wn_path = find_windninja_speed(case)
    windninja_speed = None
    wn_error = None
    if wn_path:
        windninja_speed, _ = read_ascii_grid(wn_path)
    else:
        wn_error = "missing WindNinja *_vel.asc"
    qes_speed, qes_error = read_qes_speed(case, case["domain"]["output_height_m"])
    result = {
        "zone_id": case["zone"]["id"],
        "label": case["zone"]["label"],
        "windninja": {
            "path": str(wn_path.relative_to(ROOT)) if wn_path else None,
            "error": wn_error,
            "speed": summarize_speed(windninja_speed) if windninja_speed is not None else None,
        },
        "qes_winds": {
            "path": case["qes_winds"]["expected_winds_out"],
            "error": qes_error,
            "speed": summarize_speed(qes_speed) if qes_speed is not None else None,
        },
        "comparison": None,
    }
    if windninja_speed is not None and qes_speed is not None:
        result["comparison"] = compare_arrays(windninja_speed, qes_speed)
    return result


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# QES-Winds vs WindNinja Benchmark",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Cases: `{len(report['cases'])}`",
        "",
    ]
    for case in report["cases"]:
        lines.extend([f"## {case['label']}", ""])
        wn = case["windninja"]
        qe = case["qes_winds"]
        cmp = case["comparison"]
        lines.append(f"- WindNinja: `{wn['error'] or wn['speed']}`")
        lines.append(f"- QES-Winds: `{qe['error'] or qe['speed']}`")
        lines.append(f"- Comparison: `{cmp or 'not available'}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_REPORT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    status_path = args.status if args.status.is_absolute() else ROOT / args.status
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else None
    report = {
        "format": "corsewind.qes_winds_benchmark.comparison.v1",
        "generated_at_utc": utc_now(),
        "plan": display_path(plan_path),
        "status": display_path(status_path) if status else None,
        "run_status": status,
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
