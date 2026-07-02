#!/usr/bin/env python3
"""Wait for z2 SSH, sync the sampled hPa resume scripts, and launch them."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent


SYNC_SCRIPTS = (
    "sample_residual_training_jsonl.py",
    "z2_resume_partial_hpa_sampled_signal.sh",
    "export_training_table_parquet.py",
    "train_residual_correction_parquet.py",
    "audit_tabular_rmse09_result.py",
    "analyze_tabular_rmse09_errors.py",
    "z2_repair_open_meteo_pressure_after_rate_limit.sh",
)


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def run(cmd: list[str], *, check: bool = False, dry_run: bool = False) -> subprocess.CompletedProcess[str] | None:
    print("$ " + shlex.join(cmd), flush=True)
    if dry_run:
        return None
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=check)


def ssh_cmd(args: argparse.Namespace, remote_script: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={args.ssh_connect_timeout}",
        args.host,
        remote_script,
    ]


def probe_ssh(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    completed = run(
        ssh_cmd(args, "printf corsewind-hpa-sampled-ready"),
        check=False,
        dry_run=False,
    )
    elapsed = round(time.monotonic() - started, 3)
    assert completed is not None
    return {
        "ok": completed.returncode == 0 and "corsewind-hpa-sampled-ready" in completed.stdout,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "stdout": completed.stdout.strip()[-500:],
        "stderr": completed.stderr.strip()[-1000:],
    }


def wait_for_ssh(args: argparse.Namespace) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    deadline = None if args.max_wait_minutes < 0 else time.monotonic() + args.max_wait_minutes * 60
    while True:
        attempt_number = len(attempts) + 1
        print(f"[{utc_now()}] SSH probe #{attempt_number} on {args.host}", flush=True)
        attempt = probe_ssh(args)
        attempt["attempt"] = attempt_number
        attempt["checked_at"] = utc_now()
        attempts.append(attempt)
        if attempt["ok"]:
            return {"ok": True, "attempts": attempts}
        if args.max_wait_minutes == 0:
            return {"ok": False, "reason": "single_check_failed", "attempts": attempts}
        if deadline is not None and time.monotonic() >= deadline:
            return {"ok": False, "reason": "timeout", "attempts": attempts}
        print(
            f"[{utc_now()}] z2 not ready: {attempt['stderr'] or attempt['stdout'] or 'no output'}",
            flush=True,
        )
        time.sleep(args.poll_seconds)


def sync_scripts(args: argparse.Namespace, *, dry_run: bool) -> list[dict[str, Any]]:
    remote_script_dir = f"{args.remote_root}/scripts/ml_dataset"
    mkdir = run(
        ssh_cmd(args, f"mkdir -p {shlex.quote(remote_script_dir)} {shlex.quote(args.remote_log_root)}"),
        check=False,
        dry_run=dry_run,
    )
    results: list[dict[str, Any]] = [
        {
            "step": "mkdir",
            "ok": True if mkdir is None else mkdir.returncode == 0,
            "returncode": None if mkdir is None else mkdir.returncode,
            "stderr_tail": None if mkdir is None else mkdir.stderr.strip()[-1000:],
        }
    ]
    if mkdir is not None and mkdir.returncode != 0:
        return results
    for script_name in SYNC_SCRIPTS:
        source = SCRIPT_DIR / script_name
        destination = f"{args.host}:{remote_script_dir}/{script_name}"
        completed = run(
            [
                "scp",
                "-o",
                f"ConnectTimeout={args.ssh_connect_timeout}",
                str(source),
                destination,
            ],
            check=False,
            dry_run=dry_run,
        )
        results.append(
            {
                "step": "scp",
                "script": script_name,
                "ok": True if completed is None else completed.returncode == 0,
                "returncode": None if completed is None else completed.returncode,
                "stderr_tail": None if completed is None else completed.stderr.strip()[-1000:],
            }
        )
        if completed is not None and completed.returncode != 0:
            break
    chmod = run(
        ssh_cmd(args, f"chmod +x {shlex.quote(remote_script_dir)}/z2_resume_partial_hpa_sampled_signal.sh"),
        check=False,
        dry_run=dry_run,
    )
    results.append(
        {
            "step": "chmod",
            "ok": True if chmod is None else chmod.returncode == 0,
            "returncode": None if chmod is None else chmod.returncode,
            "stderr_tail": None if chmod is None else chmod.stderr.strip()[-1000:],
        }
    )
    return results


def launch_remote(args: argparse.Namespace, *, dry_run: bool) -> dict[str, Any]:
    log_path = f"{args.remote_log_root}/partial_hpa_sampled_signal_launcher.log"
    pid_path = f"{args.remote_log_root}/partial_hpa_sampled_signal.pid"
    env = {
        "ML_ROOT": args.ml_root,
        "SAMPLE_MAX_TRAIN_ROWS": str(args.sample_max_train_rows),
        "SAMPLE_MAX_TEST_ROWS": str(args.sample_max_test_rows),
        "STOP_REPAIR_DURING_SAMPLE": "1" if args.stop_repair_during_sample else "0",
        "RESTART_REPAIR_AFTER_SAMPLE": "1" if args.restart_repair_after_sample else "0",
        "MEMORY_MIN_AVAILABLE_KB": str(args.memory_min_available_kb),
    }
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))
    remote = (
        f"mkdir -p {shlex.quote(args.remote_log_root)} && "
        f"cd {shlex.quote(args.remote_root)} && "
        f"rm -f {shlex.quote(log_path)} && "
        f"setsid bash -lc {shlex.quote(env_prefix + ' bash scripts/ml_dataset/z2_resume_partial_hpa_sampled_signal.sh')} "
        f"> {shlex.quote(log_path)} 2>&1 < /dev/null & "
        f"pid=$!; echo \"$pid\" > {shlex.quote(pid_path)}; "
        f"sleep 2; "
        f"ps -p \"$pid\" -o pid,ppid,etime,%mem,%cpu,cmd || true; "
        f"echo pid_path={shlex.quote(pid_path)} log_path={shlex.quote(log_path)}"
    )
    completed = run(ssh_cmd(args, remote), check=False, dry_run=dry_run)
    return {
        "ok": True if completed is None else completed.returncode == 0,
        "returncode": None if completed is None else completed.returncode,
        "stdout_tail": None if completed is None else completed.stdout.strip()[-4000:],
        "stderr_tail": None if completed is None else completed.stderr.strip()[-2000:],
        "pid_path": pid_path,
        "log_path": log_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="z2")
    parser.add_argument("--remote-root", default="/srv/data/corsewind/backfill_runner")
    parser.add_argument("--ml-root", default="/srv/data/corsewind/ml_dataset")
    parser.add_argument("--remote-log-root", default="/srv/data/corsewind/ml_dataset/backfill_logs")
    parser.add_argument("--ssh-connect-timeout", type=int, default=8)
    parser.add_argument("--max-wait-minutes", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--sample-max-train-rows", type=int, default=60000)
    parser.add_argument("--sample-max-test-rows", type=int, default=40000)
    parser.add_argument("--memory-min-available-kb", type=int, default=2200000)
    parser.add_argument("--stop-repair-during-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restart-repair-after-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--launch-dry-run", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary: dict[str, Any] = {
        "started_at_utc": utc_now(),
        "args": vars(args),
    }
    if args.dry_run:
        summary["ssh_wait"] = {"ok": True, "skipped": True}
    else:
        wait = wait_for_ssh(args)
        summary["ssh_wait"] = wait
        if not wait.get("ok"):
            summary["finished_at_utc"] = utc_now()
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
            raise SystemExit(2)

    sync = sync_scripts(args, dry_run=args.dry_run)
    summary["sync"] = sync
    if not all(item.get("ok") for item in sync):
        summary["finished_at_utc"] = utc_now()
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        raise SystemExit(3)

    launch = launch_remote(args, dry_run=args.dry_run or args.launch_dry_run)
    summary["launch"] = launch
    summary["finished_at_utc"] = utc_now()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    if not launch.get("ok"):
        raise SystemExit(4)


if __name__ == "__main__":
    main()
