#!/usr/bin/env python3
"""Sync ML scripts to z2 and launch the RMSE 0.9 sequence experiment."""

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
        f"setsid bash -lc {shlex.quote(shlex.join(command) + ' ; echo $? > ' + shlex.quote(status_path))} "
        f"> {shlex.quote(log_path)} 2>&1 < /dev/null & "
        f"pid=$!; echo \"$pid\" > {shlex.quote(pid_path)}; echo \"$pid\"; exit 0"
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
    parser.add_argument("--training-run-id-prefix", default="residual_windsup_sst_prev")
    parser.add_argument("--ssh-connect-timeout", type=int, default=12)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--remote-dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-preflight-fail", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--remote-log-root", default="/srv/data/corsewind/ml_dataset/run_logs")
    parser.add_argument("--include-moirai", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-lightgbm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-training-table-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-fresh-training-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-ci-upper-below-threshold", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sweep-suffix")
    parser.add_argument("--selected-run-name")
    parser.add_argument("--assert-goal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-cutoffs-per-spot", type=int)
    parser.add_argument("--max-training-features", type=int)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--calibrator-n-jobs", type=int)
    parser.add_argument(
        "--calibrator-model-family",
        action="append",
        default=[],
        help="Forward one or more model families to the calibrator sweep.",
    )
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
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
        "scripts/ml_dataset/run_rmse09_sequence_experiment.py",
        "--repo-root",
        args.remote_root,
        "--ml-root",
        args.ml_root,
        "--training-run-id-prefix",
        args.training_run_id_prefix,
    ]
    if args.remote_dry_run:
        remote_cmd.append("--dry-run")
    if args.include_moirai:
        remote_cmd.append("--include-moirai")
    if args.include_lightgbm:
        remote_cmd.append("--include-lightgbm")
    if not args.include_training_table_features:
        remote_cmd.append("--no-include-training-table-features")
    if args.require_fresh_training_features:
        remote_cmd.append("--require-fresh-training-features")
    if args.require_ci_upper_below_threshold:
        remote_cmd.append("--require-ci-upper-below-threshold")
    if args.sweep_suffix:
        remote_cmd.extend(["--sweep-suffix", args.sweep_suffix])
    if args.selected_run_name:
        remote_cmd.extend(["--selected-run-name", args.selected_run_name])
    if args.assert_goal:
        remote_cmd.append("--assert-goal")
    if args.max_cutoffs_per_spot is not None:
        remote_cmd.extend(["--max-cutoffs-per-spot", str(args.max_cutoffs_per_spot)])
    if args.max_training_features is not None:
        remote_cmd.extend(["--max-training-features", str(args.max_training_features)])
    if args.max_train_rows is not None:
        remote_cmd.extend(["--max-train-rows", str(args.max_train_rows)])
    if args.batch_size is not None:
        remote_cmd.extend(["--batch-size", str(args.batch_size)])
    if args.calibrator_n_jobs is not None:
        remote_cmd.extend(["--calibrator-n-jobs", str(args.calibrator_n_jobs)])
    for family in args.calibrator_model_family:
        remote_cmd.extend(["--calibrator-model-family", family])
    if args.force:
        remote_cmd.append("--force")
    if args.background:
        log_root = Path(args.remote_log_root)
        run([*ssh, background_shell(
            args.remote_root,
            remote_cmd,
            str(log_root / "rmse09_sequence_experiment.log"),
            str(log_root / "rmse09_sequence_experiment.pid"),
            str(log_root / "rmse09_sequence_experiment.status"),
        )], dry_run=args.dry_run)
    else:
        run([*ssh, remote_shell(args.remote_root, remote_cmd)], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
