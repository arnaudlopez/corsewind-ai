#!/usr/bin/env python3
"""Run or dry-run the prod-like FastEddy real-case preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS = ROOT / "data/processed/benchmarks/fasteddy/prod_like_status.json"
DEFAULT_RUN_STATUS = ROOT / "data/processed/benchmarks/fasteddy/prod_like_run_status.json"
STAGES = ["geospec", "simgrid", "icbc", "fasteddy"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_script(directory: str | None, script_name: str) -> Path | None:
    if not directory:
        return None
    path = Path(directory) / script_name
    return path if path.exists() else None


def resolve_binary(value: str | None) -> str | None:
    if not value:
        return None
    candidate = Path(value)
    if candidate.exists():
        return str(candidate)
    return shutil.which(value)


def run_command(cmd: list[str], cwd: Path, dry_run: bool) -> dict[str, Any]:
    printable = " ".join(cmd)
    if dry_run:
        print(f"dry-run: {printable}", flush=True)
        return {"cmd": printable, "cwd": display_path(cwd), "status": "dry_run", "elapsed_s": 0.0}
    started = time.time()
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return {
        "cmd": printable,
        "cwd": display_path(cwd),
        "status": "pass" if proc.returncode == 0 else "fail",
        "returncode": proc.returncode,
        "elapsed_s": round(time.time() - started, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def stage_command(stage: str, case: dict[str, Any], args: argparse.Namespace) -> tuple[list[str] | None, str | None]:
    case_dir = ROOT / case["case_dir"]
    coupler_dir = args.fasteddy_coupler_dir or os.environ.get("FASTEDDY_COUPLER_DIR")
    if stage == "geospec":
        script = resolve_script(coupler_dir, "GeoSpec.py")
        if not script:
            return None, "Set FASTEDDY_COUPLER_DIR to FastEddy scripts/python_utilities/coupler."
        return [args.python_bin, str(script), "-f", "geospec.json"], None
    if stage == "simgrid":
        script = resolve_script(coupler_dir, "SimGrid.py")
        if not script:
            return None, "Set FASTEDDY_COUPLER_DIR to FastEddy scripts/python_utilities/coupler."
        return [args.python_bin, str(script), "-f", "simgrid.json"], None
    if stage == "icbc":
        adapter = resolve_binary(args.adapter_bin or os.environ.get("CORSEWIND_FASTEDDY_ADAPTER"))
        if not adapter:
            return None, "Set CORSEWIND_FASTEDDY_ADAPTER after implementing the AROME-to-FastEddy IC/BC adapter."
        return [adapter, "-f", "genicbcs_arome_adapter.json"], None
    if stage == "fasteddy":
        fasteddy_bin = resolve_binary(args.fasteddy_bin or os.environ.get("FASTEDDY_BIN"))
        if not fasteddy_bin:
            return None, "Set FASTEDDY_BIN to the compiled FastEddy executable."
        return [fasteddy_bin, "fasteddy_real.in"], None
    return None, f"Unknown stage: {stage}"


def run_case(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    case_dir = ROOT / case["case_dir"]
    results = []
    for stage in args.stages:
        cmd, missing = stage_command(stage, case, args)
        if missing:
            status = "skipped_missing_tool" if args.allow_missing_tools else "blocked_missing_tool"
            results.append({"stage": stage, "status": status, "message": missing})
            if not args.allow_missing_tools:
                break
            continue
        result = run_command(cmd, case_dir, args.dry_run)
        result["stage"] = stage
        results.append(result)
        if result["status"] == "fail":
            break
    return {"zone_id": case["zone"]["id"], "case_dir": case["case_dir"], "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--run-status-output", type=Path, default=DEFAULT_RUN_STATUS)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--fasteddy-coupler-dir")
    parser.add_argument("--adapter-bin")
    parser.add_argument("--fasteddy-bin")
    parser.add_argument("--allow-missing-tools", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status_path = args.status if args.status.is_absolute() else ROOT / args.status
    status = json.loads(status_path.read_text(encoding="utf-8"))
    results = [run_case(case, args) for case in status["cases"]]
    output = {
        "format": "corsewind.fasteddy.prod_like_run_status.v1",
        "generated_at_utc": utc_now(),
        "source_status": display_path(status_path),
        "dry_run": args.dry_run,
        "stages": args.stages,
        "results": results,
    }
    output_path = args.run_status_output if args.run_status_output.is_absolute() else ROOT / args.run_status_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {display_path(output_path)}")


if __name__ == "__main__":
    main()
