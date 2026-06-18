#!/usr/bin/env python3
"""Run QES-Winds and/or WindNinja benchmark cases."""

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
DEFAULT_PLAN = ROOT / "data/processed/benchmarks/qes_winds/benchmark_plan.json"
DEFAULT_STATUS = ROOT / "data/processed/benchmarks/qes_winds/benchmark_status.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def has_nvidia_gpu() -> tuple[bool, str]:
    try:
        proc = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    if proc.returncode != 0:
        return False, proc.stderr.strip() or proc.stdout.strip()
    return bool(proc.stdout.strip()), proc.stdout.strip()


def resolve_qes_binary(value: str | None) -> str | None:
    if not value:
        return None
    candidate = Path(value)
    if candidate.exists():
        return str(candidate)
    return shutil.which(value)


def run_command(cmd: list[str], cwd: Path, dry_run: bool) -> dict[str, Any]:
    printable = " ".join(cmd)
    started = time.time()
    if dry_run:
        print(f"dry-run: {printable}", flush=True)
        return {"cmd": printable, "status": "dry_run", "elapsed_s": 0.0}
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    result = {
        "cmd": printable,
        "status": "pass" if proc.returncode == 0 else "fail",
        "returncode": proc.returncode,
        "elapsed_s": round(time.time() - started, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    print(f"{printable} -> {result['status']} {result['elapsed_s']}s", flush=True)
    return result


def qes_command(case: dict[str, Any], qes_bin: str) -> list[str]:
    return [qes_bin if part == "{QES_WINDS_BIN}" else part for part in case["qes_winds"]["command"]]


def run_case(case: dict[str, Any], args: argparse.Namespace, gpu: tuple[bool, str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "zone_id": case["zone"]["id"],
        "case_dir": case["case_dir"],
        "windninja": None,
        "qes_winds": None,
    }
    if args.engine in {"windninja", "both"}:
        result["windninja"] = run_command(case["windninja"]["command"], ROOT, args.dry_run)

    if args.engine in {"qes", "both"}:
        qes_bin_value = args.qes_bin or os.environ.get("QES_WINDS_BIN")
        qes_bin = resolve_qes_binary(qes_bin_value)
        if not qes_bin:
            result["qes_winds"] = {
                "status": "skipped_missing_binary",
                "message": f"Set QES_WINDS_BIN or pass --qes-bin. Received: {qes_bin_value or 'none'}",
            }
        elif not gpu[0] and not args.allow_no_gpu:
            result["qes_winds"] = {"status": "skipped_no_gpu", "message": gpu[1]}
        else:
            result["qes_winds"] = run_command(qes_command(case, qes_bin), ROOT, args.dry_run)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--engine", choices=["windninja", "qes", "both"], default="both")
    parser.add_argument("--qes-bin", default=None)
    parser.add_argument("--allow-no-gpu", action="store_true", help="Run QES even if nvidia-smi is unavailable.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    gpu = has_nvidia_gpu()
    started = time.time()
    case_results = [run_case(case, args, gpu) for case in plan["cases"]]
    status = {
        "format": "corsewind.qes_winds_benchmark.status.v1",
        "generated_at_utc": utc_now(),
        "plan": display_path(plan_path),
        "engine": args.engine,
        "dry_run": args.dry_run,
        "gpu": {"available": gpu[0], "detail": gpu[1]},
        "elapsed_s": round(time.time() - started, 3),
        "results": case_results,
    }
    status_path = args.status_output if args.status_output.is_absolute() else ROOT / args.status_output
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"wrote {display_path(status_path)}")


if __name__ == "__main__":
    main()
