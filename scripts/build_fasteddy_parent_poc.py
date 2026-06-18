#!/usr/bin/env python3
"""Build a first FastEddy parent NetCDF manifest from downloaded AROME POC inputs.

This script is intentionally conservative: it does not invent missing 3D physics.
It checks the POC download plan, imports any available GeoTIFF fallback fields, and
writes an explicit readiness manifest plus a minimal parent NetCDF when possible.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sample_arome_tiff_at_stations import read_float64_tiff


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "data/processed/benchmarks/fasteddy/arome_poc/arome_fasteddy_poc_download_plan.json"
DEFAULT_OUTPUT = ROOT / "data/processed/benchmarks/fasteddy/arome_poc/fasteddy_parent_poc.nc"
DEFAULT_MANIFEST = ROOT / "data/processed/benchmarks/fasteddy/arome_poc/fasteddy_parent_poc_manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def collect_inputs(plan: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    arrays: dict[str, np.ndarray] = {}
    missing: list[str] = []
    unsupported: list[str] = []
    for step in plan["steps"]:
        for item in step["downloads"]:
            if item.get("status") != "planned":
                missing.append(item["requirement_id"])
                continue
            path = ROOT / item["output"]
            if not path.exists():
                missing.append(item["requirement_id"])
                continue
            if path.suffix.lower() in {".tiff", ".tif"}:
                arrays[item["requirement_id"]] = read_float64_tiff(path).astype(np.float32)
            else:
                unsupported.append(item["requirement_id"])
    return arrays, sorted(set(missing)), sorted(set(unsupported))


def write_minimal_parent_nc(path: Path, arrays: dict[str, np.ndarray], plan: dict[str, Any]) -> str | None:
    try:
        from netCDF4 import Dataset
    except ImportError:
        return "netCDF4 not installed; run pip install -r requirements-benchmark.txt"
    if not arrays:
        return "no readable GeoTIFF fallback arrays available"
    sample = next(iter(arrays.values()))
    rows, cols = sample.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", 1)
        dataset.createDimension("z", 1)
        dataset.createDimension("y", rows)
        dataset.createDimension("x", cols)
        dataset.setncattr("format", "corsewind.fasteddy_parent_poc.v1")
        dataset.setncattr("warning", "POC fallback package only; not a production FastEddy ICBC file.")
        dataset.setncattr("source_run_time_utc", plan["source"]["run_time_utc"])
        for key, values in arrays.items():
            variable = dataset.createVariable(key, "f4", ("y", "x"), zlib=True)
            variable[:, :] = values
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    manifest_path = args.manifest_output if args.manifest_output.is_absolute() else ROOT / args.manifest_output
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    arrays, missing, unsupported = collect_inputs(plan)
    write_error = write_minimal_parent_nc(output_path, arrays, plan)
    manifest = {
        "format": "corsewind.fasteddy_parent_poc_manifest.v1",
        "generated_at_utc": utc_now(),
        "plan": display_path(plan_path),
        "output": display_path(output_path) if write_error is None else None,
        "production_fasteddy_ready": False,
        "poc_parent_written": write_error is None,
        "readable_arrays": sorted(arrays),
        "missing_inputs": missing,
        "unsupported_grib_inputs": unsupported,
        "error": write_error,
        "next_step": "Decode GRIB 3D fields and generate FastEddy ICBC files; do not use this fallback NetCDF as a production LES parent.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {display_path(manifest_path)}")
    if write_error:
        print(f"parent_poc_not_written={write_error}")
    else:
        print(f"wrote {display_path(output_path)}")


if __name__ == "__main__":
    main()
