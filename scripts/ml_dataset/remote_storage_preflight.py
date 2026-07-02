#!/usr/bin/env python3
"""Check a remote SSH storage path before large ML backfills."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import PurePosixPath
from typing import Any


def gib(kib: int) -> float:
    return round(kib / (1024**2), 3)


def ssh(host: str, command: str, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", f"ConnectTimeout={timeout}", host, command],
        check=False,
        text=True,
        capture_output=True,
    )


def remote_check(args: argparse.Namespace) -> dict[str, Any]:
    path = str(PurePosixPath(args.remote_path))
    quoted = shlex.quote(path)
    create_cmd = f"mkdir -p {quoted}" if args.create else f"test -d {quoted}"
    create_result = ssh(args.host, create_cmd, args.timeout_sec)
    if create_result.returncode != 0:
        return {
            "ok": False,
            "host": args.host,
            "remote_path": path,
            "stage": "create_or_exists",
            "returncode": create_result.returncode,
            "stderr": create_result.stderr.strip(),
            "stdout": create_result.stdout.strip(),
        }

    df_result = ssh(args.host, f"df -Pk {quoted} | tail -1", args.timeout_sec)
    if df_result.returncode != 0:
        return {
            "ok": False,
            "host": args.host,
            "remote_path": path,
            "stage": "df",
            "returncode": df_result.returncode,
            "stderr": df_result.stderr.strip(),
            "stdout": df_result.stdout.strip(),
        }

    parts = df_result.stdout.split()
    if len(parts) < 6:
        return {
            "ok": False,
            "host": args.host,
            "remote_path": path,
            "stage": "parse_df",
            "stdout": df_result.stdout.strip(),
        }

    filesystem, total_kib, used_kib, free_kib, capacity, mount = parts[:6]
    free_kib_int = int(free_kib)
    required_kib = int(args.min_free_gb * 1024 * 1024)
    ok = free_kib_int >= required_kib
    return {
        "ok": ok,
        "host": args.host,
        "remote_path": path,
        "filesystem": filesystem,
        "mount": mount,
        "capacity": capacity,
        "total_gib": gib(int(total_kib)),
        "used_gib": gib(int(used_kib)),
        "free_gib": gib(free_kib_int),
        "required_free_gib": args.min_free_gb,
        "export": f"ML_DATASET_ROOT={args.host}:{path}",
        "recommendation": "OK for remote backfills." if ok else "Choose a larger remote volume before backfills.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="z2")
    parser.add_argument("--remote-path", default="/data/corsewind/ml_dataset")
    parser.add_argument("--min-free-gb", type=float, default=250)
    parser.add_argument("--timeout-sec", type=int, default=10)
    parser.add_argument("--create", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = remote_check(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
