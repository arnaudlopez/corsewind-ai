#!/usr/bin/env python3
"""Run local readiness checks for the RMSE09 proof pipeline."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SECRET_PATTERN_PARTS = (
    ("ARfmTD", "WOlZHMLN", "_QLngDINaG2kYa"),
    ("EAmbLO", "cpBB_wf0D", "TEfzzipxAFFUa"),
    ("Arnaud", "2a34"),
    (r"k856@uc@", r"gJHDL\*K"),
    ("alopez", "1234567"),
    (r"u\*bBu", "Uup3U6hT6m"),
    ("d1b36a8d-", "3bf4-4afc-", "807f-0d1656d3813f"),
    ("Tmhe6iZX", "qe9CtaMM", "Cb8IrySU"),
)


PY_COMPILE_TARGETS = [
    "audit_rmse09_results.py",
    "analyze_rmse09_errors.py",
    "assert_rmse09_goal.py",
    "check_z2_rmse09_status.py",
    "launch_z2_rebuild_training_shards.py",
    "launch_z2_rmse09_sequence_experiment.py",
    "preflight_rmse09_environment.py",
    "run_rmse09_sequence_experiment.py",
    "select_sequence_calibrator_run.py",
    "smoke_test_rmse09_pipeline.py",
    "summarize_rmse09_decision.py",
    "sweep_sequence_calibrators.py",
    "train_sequence_calibrator.py",
    "wait_for_z2_and_launch_rmse09.py",
    "wait_for_z2_rebuild_and_launch_rmse09.py",
]


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=check)


def clean_pycache() -> None:
    for path in SCRIPT_DIR.rglob("__pycache__"):
        shutil.rmtree(path, ignore_errors=True)


def check_py_compile() -> dict[str, Any]:
    cmd = [sys.executable, "-m", "py_compile", *[str(SCRIPT_DIR / item) for item in PY_COMPILE_TARGETS]]
    completed = run(cmd, check=False)
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip()[-4000:],
    }


def check_smoke() -> dict[str, Any]:
    completed = run([sys.executable, str(SCRIPT_DIR / "smoke_test_rmse09_pipeline.py")], check=False)
    payload = {}
    if completed.stdout.strip():
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = {"parse_error": completed.stdout.strip()[-4000:]}
    return {
        "ok": completed.returncode == 0 and payload.get("status") == "ok",
        "returncode": completed.returncode,
        "payload": payload,
        "stderr": completed.stderr.strip()[-4000:],
    }


def check_contains(name: str, cmd: list[str], expected: list[str]) -> dict[str, Any]:
    completed = run(cmd, check=False)
    output = completed.stdout + completed.stderr
    missing = [item for item in expected if item not in output]
    return {
        "name": name,
        "ok": completed.returncode == 0 and not missing,
        "returncode": completed.returncode,
        "missing": missing,
        "stdout_tail": completed.stdout.strip()[-4000:],
        "stderr_tail": completed.stderr.strip()[-4000:],
    }


def check_secret_scan() -> dict[str, Any]:
    if shutil.which("rg") is None:
        return {"ok": False, "reason": "rg is required for the secret scan"}
    patterns = ["".join(parts) for parts in SECRET_PATTERN_PARTS]
    completed = run([
        "rg",
        "-n",
        "|".join(patterns),
        ".",
        "--glob",
        "!tmp/**",
        "--glob",
        "!data/**",
    ], check=False)
    return {
        "ok": completed.returncode == 1,
        "returncode": completed.returncode,
        "matches": completed.stdout.strip()[-4000:],
        "stderr": completed.stderr.strip()[-4000:],
    }


def main() -> None:
    checks = {
        "py_compile": check_py_compile(),
        "smoke_test": check_smoke(),
        "experiment_dry_run": check_contains(
            "experiment_dry_run",
            [
                sys.executable,
                str(SCRIPT_DIR / "run_rmse09_sequence_experiment.py"),
                "--dry-run",
                "--assert-goal",
                "--repo-root",
                "/srv/data/corsewind/backfill_runner",
            ],
            [
                "sequence_2025_windsurf_1h_rmse09_v1",
                "sequence_2026_windsurf_1h_rmse09_v1",
                "rmse09_run_manifest.json",
                "assert_rmse09_goal.py",
                "--max-training-features 1400",
            ],
        ),
        "launcher_dry_run": check_contains(
            "launcher_dry_run",
            [
                sys.executable,
                str(SCRIPT_DIR / "launch_z2_rmse09_sequence_experiment.py"),
                "--dry-run",
                "--remote-dry-run",
                "--background",
                "--assert-goal",
            ],
            ["--assert-goal", "run_rmse09_sequence_experiment.py", "setsid"],
        ),
        "status_dry_run": check_contains(
            "status_dry_run",
            [sys.executable, str(SCRIPT_DIR / "check_z2_rmse09_status.py"), "--dry-run"],
            ["rmse09_decision.json", "rmse09_run_manifest.json", "final_assert_command"],
        ),
        "wait_launcher_dry_run": check_contains(
            "wait_launcher_dry_run",
            [
                sys.executable,
                str(SCRIPT_DIR / "wait_for_z2_and_launch_rmse09.py"),
                "--dry-run",
                "--launch-dry-run",
            ],
            ["launch_z2_rmse09_sequence_experiment.py", "--include-lightgbm", "--assert-goal"],
        ),
        "rebuild_wait_launcher_dry_run": check_contains(
            "rebuild_wait_launcher_dry_run",
            [
                sys.executable,
                str(SCRIPT_DIR / "wait_for_z2_rebuild_and_launch_rmse09.py"),
                "--dry-run",
                "--launch-dry-run",
            ],
            ["fresh_full_feature_audit.json", "context_fresh_v1", "launch_z2_rmse09_sequence_experiment.py"],
        ),
        "secret_scan": check_secret_scan(),
    }
    clean_pycache()
    ok = all(item.get("ok") for item in checks.values())
    result = {"status": "ok" if ok else "fail", "checks": checks}
    print(json.dumps(result, indent=2, sort_keys=True))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
