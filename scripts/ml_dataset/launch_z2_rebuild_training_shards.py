#!/usr/bin/env python3
"""Sync ML scripts to z2 and rebuild monthly training shards with current features."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def run(cmd: list[str], *, dry_run: bool) -> None:
    print("$ " + shlex.join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def run_optional(cmd: list[str], *, dry_run: bool) -> subprocess.CompletedProcess[bytes] | None:
    print("$ " + shlex.join(cmd), flush=True)
    if dry_run:
        return None
    return subprocess.run(cmd, check=False)


def remote_shell(remote_root: str, command: list[str]) -> str:
    return f"cd {shlex.quote(remote_root)} && {shlex.join(command)}"


def background_shell(remote_root: str, command: list[str], log_path: str, pid_path: str, status_path: str) -> str:
    inner = (
        f"mkdir -p {shlex.quote(str(Path(log_path).parent))} {shlex.quote(str(Path(pid_path).parent))} "
        f"{shlex.quote(str(Path(status_path).parent))} && "
        f"rm -f {shlex.quote(status_path)} && "
        f"cd {shlex.quote(remote_root)} && "
        f"nohup setsid bash -lc {shlex.quote(shlex.join(command) + ' ; echo $? > ' + shlex.quote(status_path))} "
        f"> {shlex.quote(log_path)} 2>&1 < /dev/null & "
        f"pid=$!; echo \"$pid\" > {shlex.quote(pid_path)}; echo \"$pid\"; "
        f"disown \"$pid\" 2>/dev/null || true; exit 0"
    )
    return inner


def scp_options(connect_timeout: int) -> list[str]:
    return ["-o", f"ConnectTimeout={connect_timeout}"]


def sync_directory(source: Path, host: str, remote_destination: str, connect_timeout: int, *, dry_run: bool) -> None:
    rsync_cmd = [
        "rsync",
        "-az",
        "-e",
        f"ssh -o ConnectTimeout={connect_timeout}",
        str(source) + "/",
        f"{host}:{remote_destination}/",
    ]
    completed = run_optional(rsync_cmd, dry_run=dry_run)
    if dry_run or completed is None or completed.returncode == 0:
        return
    print(f"# rsync failed with exit {completed.returncode}; falling back to scp", flush=True)
    run(["ssh", "-o", f"ConnectTimeout={connect_timeout}", host, f"mkdir -p {shlex.quote(remote_destination)}"], dry_run=dry_run)
    run(["scp", *scp_options(connect_timeout), "-r", str(source) + "/.", f"{host}:{remote_destination}/"], dry_run=dry_run)


def sync_file(source: Path, host: str, remote_destination: str, connect_timeout: int, *, dry_run: bool) -> None:
    remote_parent = str(Path(remote_destination).parent)
    rsync_cmd = [
        "rsync",
        "-az",
        "-e",
        f"ssh -o ConnectTimeout={connect_timeout}",
        str(source),
        f"{host}:{remote_destination}",
    ]
    completed = run_optional(rsync_cmd, dry_run=dry_run)
    if dry_run or completed is None or completed.returncode == 0:
        return
    print(f"# rsync failed with exit {completed.returncode}; falling back to scp", flush=True)
    run(["ssh", "-o", f"ConnectTimeout={connect_timeout}", host, f"mkdir -p {shlex.quote(remote_parent)}"], dry_run=dry_run)
    run(["scp", *scp_options(connect_timeout), str(source), f"{host}:{remote_destination}"], dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="z2")
    parser.add_argument("--remote-root", default="/srv/data/corsewind/backfill_runner")
    parser.add_argument("--ml-root", default="/srv/data/corsewind/ml_dataset")
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument("--end-month", default="2026-06")
    parser.add_argument("--run-id-prefix", default="residual_windsup_sst_prev")
    parser.add_argument("--ssh-connect-timeout", type=int, default=12)
    parser.add_argument("--command-timeout-sec", type=int, default=7200)
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--parquet-batch-size", type=int, default=25000)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--remote-dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-preflight-fail", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--remote-log-root", default="/srv/data/corsewind/ml_dataset/run_logs")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    ssh = ["ssh", "-o", f"ConnectTimeout={args.ssh_connect_timeout}", args.host]

    sync_targets = [
        (repo_root / "scripts" / "ml_dataset", f"{args.remote_root}/scripts/ml_dataset"),
        (repo_root / "configs", f"{args.remote_root}/configs"),
        (repo_root / "docs" / "ml_nowcasting", f"{args.remote_root}/docs/ml_nowcasting"),
    ]
    for source, destination in sync_targets:
        sync_directory(source, args.host, destination, args.ssh_connect_timeout, dry_run=args.dry_run)
    sync_file(
        repo_root / "requirements-ml-dataset.txt",
        args.host,
        f"{args.remote_root}/requirements-ml-dataset.txt",
        args.ssh_connect_timeout,
        dry_run=args.dry_run,
    )

    preflight_cmd = [
        "/home/z2/corsewind-ml-smoke/.venv/bin/python",
        "scripts/ml_dataset/preflight_rmse09_environment.py",
        "--repo-root",
        args.remote_root,
        "--ml-root",
        args.ml_root,
        "--output-json",
        "/srv/data/corsewind/ml_dataset/benchmarks/rmse09_environment_preflight.json",
    ]
    if not args.allow_preflight_fail:
        preflight_cmd.append("--fail-on-non-pass")
    run([*ssh, remote_shell(args.remote_root, preflight_cmd)], dry_run=args.dry_run)

    remote_cmd = [
        "/home/z2/corsewind-ml-smoke/.venv/bin/python",
        "scripts/ml_dataset/run_monthly_training_shards.py",
        "--ml-root",
        args.ml_root,
        "--registry",
        f"{args.remote_root}/configs/ml_spots.json",
        "--context-registry",
        f"{args.remote_root}/configs/ml_context_stations.json",
        "--start-month",
        args.start_month,
        "--end-month",
        args.end_month,
        "--run-id-prefix",
        args.run_id_prefix,
        "--start-hour-utc",
        "8",
        "--end-hour-utc",
        "17",
        "--lead-minutes",
        "15,30,45,60,120,180,360",
        "--command-timeout-sec",
        str(args.command_timeout_sec),
        "--chunk-days",
        str(args.chunk_days),
        "--parquet-batch-size",
        str(args.parquet_batch_size),
        "--no-skip-existing-parquet",
    ]
    if args.continue_on_error:
        remote_cmd.append("--continue-on-error")
    if args.remote_dry_run:
        remote_cmd.append("--dry-run")
    if args.background:
        log_root = Path(args.remote_log_root)
        run([*ssh, background_shell(
            args.remote_root,
            remote_cmd,
            str(log_root / "rebuild_training_shards.log"),
            str(log_root / "rebuild_training_shards.pid"),
            str(log_root / "rebuild_training_shards.status"),
        )], dry_run=args.dry_run)
    else:
        run([*ssh, remote_shell(args.remote_root, remote_cmd)], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
