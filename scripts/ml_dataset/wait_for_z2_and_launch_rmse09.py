#!/usr/bin/env python3
"""Wait until z2 is reachable, then launch the strict RMSE09 experiment."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def run(cmd: list[str], *, check: bool = True, dry_run: bool = False) -> subprocess.CompletedProcess[str] | None:
    print("$ " + shlex.join(cmd), flush=True)
    if dry_run:
        return None
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=check)


def probe_ssh(host: str, timeout_seconds: int) -> dict[str, Any]:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={timeout_seconds}",
        host,
        "printf rmse09-z2-ready",
    ]
    started = time.monotonic()
    completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    elapsed = round(time.monotonic() - started, 3)
    return {
        "ok": completed.returncode == 0 and "rmse09-z2-ready" in completed.stdout,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "stdout": completed.stdout.strip()[-500:],
        "stderr": completed.stderr.strip()[-1000:],
    }


def wait_for_ssh(args: argparse.Namespace) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    deadline: float | None
    if args.max_wait_minutes < 0:
        deadline = None
    else:
        deadline = time.monotonic() + (args.max_wait_minutes * 60)

    while True:
        attempt_number = len(attempts) + 1
        print(f"[{utc_now()}] SSH probe #{attempt_number} on {args.host}", flush=True)
        attempt = probe_ssh(args.host, args.ssh_connect_timeout)
        attempt["attempt"] = attempt_number
        attempt["checked_at"] = utc_now()
        attempts.append(attempt)
        if attempt["ok"]:
            return {"ok": True, "attempts": attempts}

        if args.max_wait_minutes == 0:
            return {"ok": False, "attempts": attempts, "reason": "single_check_failed"}
        if deadline is not None and time.monotonic() >= deadline:
            return {"ok": False, "attempts": attempts, "reason": "timeout"}

        sleep_seconds = args.poll_seconds
        if deadline is not None:
            sleep_seconds = min(sleep_seconds, max(1, int(deadline - time.monotonic())))
        print(
            f"[{utc_now()}] z2 not ready yet: {attempt['stderr'] or attempt['stdout'] or 'no output'}",
            flush=True,
        )
        time.sleep(sleep_seconds)


def build_launcher_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "launch_z2_rmse09_sequence_experiment.py"),
        "--host",
        args.host,
        "--ml-root",
        args.ml_root,
        "--training-run-id-prefix",
        args.training_run_id_prefix,
        "--ssh-connect-timeout",
        str(args.ssh_connect_timeout),
    ]
    if args.background:
        cmd.append("--background")
    else:
        cmd.append("--no-background")
    if args.include_lightgbm:
        cmd.append("--include-lightgbm")
    if args.include_moirai:
        cmd.append("--include-moirai")
    if args.require_fresh_training_features:
        cmd.append("--require-fresh-training-features")
    if args.require_ci_upper_below_threshold:
        cmd.append("--require-ci-upper-below-threshold")
    if args.sweep_suffix:
        cmd.extend(["--sweep-suffix", args.sweep_suffix])
    if args.assert_goal:
        cmd.append("--assert-goal")
    if args.max_cutoffs_per_spot is not None:
        cmd.extend(["--max-cutoffs-per-spot", str(args.max_cutoffs_per_spot)])
    if args.max_train_rows is not None:
        cmd.extend(["--max-train-rows", str(args.max_train_rows)])
    if args.max_training_features is not None:
        cmd.extend(["--max-training-features", str(args.max_training_features)])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.calibrator_n_jobs is not None:
        cmd.extend(["--calibrator-n-jobs", str(args.calibrator_n_jobs)])
    for family in args.calibrator_model_family:
        cmd.extend(["--calibrator-model-family", family])
    if args.force:
        cmd.append("--force")
    if args.launch_dry_run:
        cmd.extend(["--dry-run", "--remote-dry-run"])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="z2")
    parser.add_argument("--ml-root", default="/srv/data/corsewind/ml_dataset")
    parser.add_argument("--training-run-id-prefix", default="residual_windsup_sst_prev")
    parser.add_argument("--ssh-connect-timeout", type=int, default=8)
    parser.add_argument(
        "--max-wait-minutes",
        type=int,
        default=60,
        help="Use 0 for a single probe, or a negative value to wait forever.",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--skip-readiness", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--launch-dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-lightgbm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-moirai", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-fresh-training-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-ci-upper-below-threshold", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sweep-suffix")
    parser.add_argument("--assert-goal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-cutoffs-per-spot", type=int)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-training-features", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--calibrator-n-jobs", type=int)
    parser.add_argument("--calibrator-model-family", action="append", default=[])
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary: dict[str, Any] = {
        "started_at": utc_now(),
        "host": args.host,
        "dry_run": args.dry_run,
        "launch_dry_run": args.launch_dry_run,
        "max_wait_minutes": args.max_wait_minutes,
    }

    if args.dry_run:
        summary["ssh_wait"] = {"ok": True, "skipped": True}
    else:
        ssh_wait = wait_for_ssh(args)
        summary["ssh_wait"] = ssh_wait
        if not ssh_wait["ok"]:
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
            raise SystemExit(2)

    if not args.skip_readiness:
        readiness_cmd = [sys.executable, str(SCRIPT_DIR / "check_rmse09_local_readiness.py")]
        readiness = run(readiness_cmd, check=False, dry_run=args.dry_run)
        summary["readiness"] = {
            "ok": True if readiness is None else readiness.returncode == 0,
            "returncode": None if readiness is None else readiness.returncode,
            "stdout_tail": None if readiness is None else readiness.stdout.strip()[-4000:],
            "stderr_tail": None if readiness is None else readiness.stderr.strip()[-4000:],
        }
        if readiness is not None and readiness.returncode != 0:
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
            raise SystemExit(readiness.returncode)

    launcher_cmd = build_launcher_command(args)
    launched = run(launcher_cmd, check=False, dry_run=args.dry_run)
    summary["launcher"] = {
        "command": launcher_cmd,
        "ok": True if launched is None else launched.returncode == 0,
        "returncode": None if launched is None else launched.returncode,
        "stdout_tail": None if launched is None else launched.stdout.strip()[-4000:],
        "stderr_tail": None if launched is None else launched.stderr.strip()[-4000:],
    }
    summary["finished_at"] = utc_now()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if launched is not None and launched.returncode != 0:
        raise SystemExit(launched.returncode)


if __name__ == "__main__":
    main()
