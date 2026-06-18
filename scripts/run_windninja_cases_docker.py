#!/usr/bin/env python3
"""Run prepared WindNinja cases through the Katana Docker image."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


IMAGE = "usdaarsnwrc/katana:latest"
REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_docker_config() -> Path:
    config_dir = Path("/tmp/corsewind-docker-config")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.json"
    if not config_file.exists():
        config_file.write_text('{"auths":{}}\n', encoding="utf-8")
    return config_dir


def docker_host_path(path: Path) -> Path:
    """Map an in-container repo path to the host path seen by the Docker daemon."""
    host_root = os.environ.get("CORSEWIND_HOST_ROOT")
    if not host_root:
        return path.resolve()

    container_root = Path(os.environ.get("CORSEWIND_CONTAINER_ROOT", str(REPO_ROOT))).resolve()
    resolved = path.resolve()
    try:
        relative_path = resolved.relative_to(container_root)
    except ValueError:
        relative_path = resolved.relative_to(REPO_ROOT)
    return Path(host_root).expanduser() / relative_path


def run_case(case_dir: Path, platform: str, config_name: str, output_dir_name: str) -> None:
    output_dir = case_dir / output_dir_name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["DOCKER_CONFIG"] = str(ensure_docker_config())
    cmd = [
        "docker",
        "run",
        "--rm",
        "--platform",
        platform,
        "--entrypoint",
        "/bin/bash",
        "-v",
        f"{docker_host_path(case_dir)}:/case",
        "-w",
        "/case",
        IMAGE,
        "-lc",
        f"WindNinja_cli {config_name}",
    ]
    print(f"running {case_dir}")
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, help="Run one explicit WindNinja case directory.")
    parser.add_argument("--case-root", type=Path, default=Path("data/processed/physics/solver_cases"))
    parser.add_argument("--pattern", default="rank*")
    parser.add_argument("--config-name", default="windninja_candidate.cfg")
    parser.add_argument("--output-dir-name", default="windninja_output")
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--pull", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env["DOCKER_CONFIG"] = str(ensure_docker_config())
    if args.pull:
        subprocess.run(["docker", "pull", IMAGE], check=True, env=env)

    if args.case_dir:
        case_dirs = [args.case_dir]
    else:
        case_dirs = sorted(path for path in args.case_root.glob(args.pattern) if (path / args.config_name).exists())
    if not case_dirs:
        raise SystemExit(f"No case dirs found under {args.case_root} matching {args.pattern}")
    missing_configs = [path for path in case_dirs if not (path / args.config_name).exists()]
    if missing_configs:
        joined = ", ".join(str(path / args.config_name) for path in missing_configs)
        raise SystemExit(f"Missing WindNinja config(s): {joined}")
    for case_dir in case_dirs:
        run_case(case_dir, args.platform, args.config_name, args.output_dir_name)


if __name__ == "__main__":
    main()
