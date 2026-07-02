#!/usr/bin/env python3
"""Wait for the z2 training-shard rebuild, audit it, then launch RMSE09."""

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


def run(cmd: list[str], *, check: bool = False, dry_run: bool = False) -> subprocess.CompletedProcess[str] | None:
    print("$ " + shlex.join(cmd), flush=True)
    if dry_run:
        return None
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=check)


def ssh(host: str, timeout: int, remote_script: str, *, dry_run: bool = False) -> subprocess.CompletedProcess[str] | None:
    return run(["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}", host, remote_script], dry_run=dry_run)


def rebuild_probe(args: argparse.Namespace) -> dict[str, Any]:
    script = (
        "set +e\n"
        "pids=$(ps -eo pid,cmd | grep -E 'run_monthly_training|run_training_backfill|build_spot_feature|build_residual|export_training' | grep -v grep || true)\n"
        "status=$(cat /srv/data/corsewind/ml_dataset/run_logs/rebuild_training_shards.status 2>/dev/null || echo missing)\n"
        "months=$(find "
        + shlex.quote(f"{args.ml_root}/training_tables")
        + " -maxdepth 1 -type f -name impossible 2>/dev/null; find "
        + shlex.quote(f"{args.ml_root}/training_tables")
        + " -maxdepth 2 -type f -name training_rows.parquet 2>/dev/null | wc -l)\n"
        "echo '{\"processes\":'$(python3 - <<'PY'\nimport json,os\nprint(json.dumps(os.environ.get('pids','')))\nPY\n)' }' >/dev/null\n"
        "python3 - <<'PY'\nimport json, os, subprocess\npids = subprocess.run(\"ps -eo pid,cmd | grep -E 'run_monthly_training|run_training_backfill|build_spot_feature|build_residual|export_training' | grep -v grep || true\", shell=True, text=True, capture_output=True).stdout.strip()\nstatus = subprocess.run('cat /srv/data/corsewind/ml_dataset/run_logs/rebuild_training_shards.status 2>/dev/null || echo missing', shell=True, text=True, capture_output=True).stdout.strip()\nmonths = subprocess.run(\"find "
        + shlex.quote(f"{args.ml_root}/training_tables")
        + " -maxdepth 2 -type f -name training_rows.parquet 2>/dev/null | wc -l\", shell=True, text=True, capture_output=True).stdout.strip()\nprint(json.dumps({'running': bool(pids), 'processes': pids[-4000:], 'status': status, 'parquet_shard_count': int(months or 0)}))\nPY"
    )
    completed = ssh(args.host, args.ssh_connect_timeout, script)
    if completed is None:
        return {"ok": True, "running": False, "dry_run": True}
    payload: dict[str, Any] = {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[-5000:],
        "stderr": completed.stderr.strip()[-2000:],
    }
    if completed.returncode == 0 and completed.stdout.strip():
        try:
            payload.update(json.loads(completed.stdout.strip().splitlines()[-1]))
        except json.JSONDecodeError:
            payload["parse_error"] = completed.stdout.strip()[-1000:]
    return payload


def wait_for_rebuild(args: argparse.Namespace) -> dict[str, Any]:
    attempts = []
    deadline = None if args.max_wait_minutes < 0 else time.monotonic() + args.max_wait_minutes * 60
    while True:
        probe = rebuild_probe(args)
        probe["checked_at"] = utc_now()
        probe["attempt"] = len(attempts) + 1
        attempts.append(probe)
        print(json.dumps(probe, indent=2, sort_keys=True), flush=True)
        if probe.get("ok") and not probe.get("running"):
            return {"ok": True, "attempts": attempts}
        if args.max_wait_minutes == 0:
            return {"ok": False, "reason": "single_check_still_running", "attempts": attempts}
        if deadline is not None and time.monotonic() >= deadline:
            return {"ok": False, "reason": "timeout", "attempts": attempts}
        time.sleep(args.poll_seconds)


def audit_command(args: argparse.Namespace) -> str:
    return shlex.join([
        "/home/z2/corsewind-ml-smoke/.venv/bin/python",
        "scripts/ml_dataset/audit_training_table_features.py",
        "--training-table-root",
        f"{args.ml_root}/training_tables",
        "--run-id-prefix",
        args.training_run_id_prefix,
        "--start-month",
        args.start_month,
        "--end-month",
        args.end_month,
        "--output-json",
        f"{args.ml_root}/training_tables/fresh_full_feature_audit.json",
        "--output-md",
        f"{args.ml_root}/training_tables/fresh_full_feature_audit.md",
        "--fail-on-non-pass",
    ])


def launch_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "launch_z2_rmse09_sequence_experiment.py"),
        "--host",
        args.host,
        "--ml-root",
        args.ml_root,
        "--training-run-id-prefix",
        args.training_run_id_prefix,
        "--background",
        "--require-fresh-training-features",
        "--assert-goal",
        "--max-cutoffs-per-spot",
        str(args.max_cutoffs_per_spot),
        "--batch-size",
        str(args.batch_size),
        "--max-train-rows",
        str(args.max_train_rows),
        "--max-training-features",
        str(args.max_training_features),
        "--calibrator-n-jobs",
        str(args.calibrator_n_jobs),
        "--sweep-suffix",
        args.sweep_suffix,
    ]
    if args.include_lightgbm:
        cmd.append("--include-lightgbm")
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
    parser.add_argument("--ml-root", default="/srv/data/corsewind/ml_dataset_z2_rebuild")
    parser.add_argument("--training-run-id-prefix", default="residual_windsup_sst_prev_fresh")
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument("--end-month", default="2026-06")
    parser.add_argument("--ssh-connect-timeout", type=int, default=12)
    parser.add_argument("--max-wait-minutes", type=int, default=-1)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-cutoffs-per-spot", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-train-rows", type=int, default=120000)
    parser.add_argument("--max-training-features", type=int, default=900)
    parser.add_argument("--calibrator-n-jobs", type=int, default=1)
    parser.add_argument("--calibrator-model-family", action="append", default=["ridge", "hist_gradient_boosting"])
    parser.add_argument("--include-lightgbm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sweep-suffix", default="context_fresh_v1")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--launch-dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary: dict[str, Any] = {"started_at": utc_now(), "args": vars(args)}
    wait = {"ok": True, "skipped": True} if args.dry_run else wait_for_rebuild(args)
    summary["wait"] = wait
    if not wait.get("ok"):
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        raise SystemExit(2)

    audit_remote = f"cd /srv/data/corsewind/backfill_runner && {audit_command(args)}"
    audit = ssh(args.host, args.ssh_connect_timeout, audit_remote, dry_run=args.dry_run)
    summary["audit"] = {
        "ok": True if audit is None else audit.returncode == 0,
        "returncode": None if audit is None else audit.returncode,
        "stdout_tail": None if audit is None else audit.stdout.strip()[-4000:],
        "stderr_tail": None if audit is None else audit.stderr.strip()[-4000:],
    }
    if audit is not None and audit.returncode != 0:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        raise SystemExit(audit.returncode)

    launched = run(launch_command(args), dry_run=args.dry_run)
    summary["launcher"] = {
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
