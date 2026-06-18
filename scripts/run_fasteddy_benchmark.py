#!/usr/bin/env python3
"""Run prepared FastEddy smoke benchmark cases."""

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
DEFAULT_PLAN = ROOT / "data/processed/benchmarks/fasteddy/benchmark_plan.json"
DEFAULT_STATUS = ROOT / "data/processed/benchmarks/fasteddy/benchmark_status.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_binary(value: str | None) -> str | None:
    if not value:
        return None
    candidate = Path(value)
    if candidate.exists():
        return str(candidate)
    return shutil.which(value)


def has_nvidia_gpu() -> tuple[bool, str]:
    try:
        proc = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    if proc.returncode != 0:
        return False, proc.stderr.strip() or proc.stdout.strip()
    return bool(proc.stdout.strip()), proc.stdout.strip()


def build_command(case: dict[str, Any], fasteddy_bin: str, mpirun_bin: str | None) -> list[str]:
    ranks = int(case["fasteddy"].get("mpi_ranks", 1))
    template = case["fasteddy"]["mpi_command"] if ranks > 1 and mpirun_bin else case["fasteddy"]["command"]
    return [fasteddy_bin if part == "{FASTEDDY_BIN}" else (mpirun_bin if part == "mpirun" and mpirun_bin else part) for part in template]


def run_command(cmd: list[str], cwd: Path, dry_run: bool) -> dict[str, Any]:
    printable = " ".join(cmd)
    started = time.time()
    if dry_run:
        print(f"dry-run: {printable}", flush=True)
        return {"cmd": printable, "cwd": display_path(cwd), "status": "dry_run", "elapsed_s": 0.0}
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    result = {
        "cmd": printable,
        "cwd": display_path(cwd),
        "status": "pass" if proc.returncode == 0 else "fail",
        "returncode": proc.returncode,
        "elapsed_s": round(time.time() - started, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    print(f"{printable} -> {result['status']} {result['elapsed_s']}s", flush=True)
    return result


def run_case(case: dict[str, Any], args: argparse.Namespace, gpu: tuple[bool, str]) -> dict[str, Any]:
    result: dict[str, Any] = {"zone_id": case["zone"]["id"], "case_dir": case["case_dir"], "fasteddy": None}
    fasteddy_bin_value = args.fasteddy_bin or os.environ.get("FASTEDDY_BIN")
    fasteddy_bin = resolve_binary(fasteddy_bin_value)
    mpirun_bin = resolve_binary(args.mpirun_bin or os.environ.get("MPIEXEC") or "mpirun")
    if not fasteddy_bin:
        result["fasteddy"] = {
            "status": "skipped_missing_binary",
            "message": f"Set FASTEDDY_BIN or pass --fasteddy-bin. Received: {fasteddy_bin_value or 'none'}",
        }
    elif not gpu[0] and not args.allow_no_gpu:
        result["fasteddy"] = {"status": "skipped_no_gpu", "message": gpu[1]}
    else:
        case_dir = ROOT / case["case_dir"]
        result["fasteddy"] = run_command(build_command(case, fasteddy_bin, mpirun_bin), case_dir, args.dry_run)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--fasteddy-bin", default=None)
    parser.add_argument("--mpirun-bin", default=None)
    parser.add_argument("--allow-no-gpu", action="store_true", help="Run FastEddy even if nvidia-smi is unavailable.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    gpu = has_nvidia_gpu()
    started = time.time()
    results = [run_case(case, args, gpu) for case in plan["cases"]]
    status = {
        "format": "corsewind.fasteddy_benchmark.status.v1",
        "generated_at_utc": utc_now(),
        "plan": display_path(plan_path),
        "dry_run": args.dry_run,
        "gpu": {"available": gpu[0], "detail": gpu[1]},
        "elapsed_s": round(time.time() - started, 3),
        "results": results,
    }
    status_path = args.status_output if args.status_output.is_absolute() else ROOT / args.status_output
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"wrote {display_path(status_path)}")


if __name__ == "__main__":
    main()
