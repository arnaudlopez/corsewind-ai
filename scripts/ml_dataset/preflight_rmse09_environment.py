#!/usr/bin/env python3
"""Preflight local/remote environment before expensive RMSE-0.9 experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "configs/ml_spots.json",
    "configs/ml_context_stations.json",
    "scripts/ml_dataset/run_rmse09_sequence_experiment.py",
    "scripts/ml_dataset/train_sequence_calibrator.py",
    "scripts/ml_dataset/audit_training_table_features.py",
    "scripts/ml_dataset/audit_rmse09_results.py",
    "scripts/ml_dataset/analyze_rmse09_errors.py",
    "scripts/ml_dataset/summarize_rmse09_decision.py",
    "scripts/ml_dataset/assert_rmse09_goal.py",
    "scripts/ml_dataset/smoke_test_rmse09_pipeline.py",
    "scripts/ml_dataset/check_rmse09_local_readiness.py",
    "requirements-ml-dataset.txt",
]
OPTIONAL_PACKAGES = ["lightgbm"]
REQUIRED_PACKAGES = ["pandas", "pyarrow", "sklearn", "joblib", "numpy"]
VENV_IMPORTS = {
    "chronos_python": ["chronos", "torch", "pandas", "pyarrow"],
    "timesfm_python": ["timesfm", "torch", "pandas"],
    "moirai_python": ["uni2ts", "torch", "pandas"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_jsonable(cmd: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=20)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"ok": False, "error": str(exc), "command": cmd}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[-2000:],
        "stderr": completed.stderr.strip()[-2000:],
        "command": cmd,
    }


def import_check(python_path: str, modules: list[str]) -> dict[str, Any]:
    path = Path(python_path)
    if not path.exists():
        return {"ok": False, "exists": False, "path": python_path, "missing_modules": modules}
    code = (
        "import importlib.util, json; "
        f"mods={modules!r}; "
        "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
        "print(json.dumps({'missing_modules': missing, 'ok': not missing}))"
    )
    result = run_jsonable([python_path, "-c", code])
    payload: dict[str, Any] = {"path": python_path, "exists": True, "command_result": result}
    if result.get("ok"):
        try:
            payload.update(json.loads(str(result.get("stdout") or "{}")))
        except json.JSONDecodeError:
            payload.update({"ok": False, "missing_modules": modules, "parse_error": result.get("stdout")})
    else:
        payload.update({"ok": False, "missing_modules": modules})
    return payload


def inspect_environment(repo_root: Path, ml_root: Path, chronos_python: str, timesfm_python: str, moirai_python: str) -> dict[str, Any]:
    files = {
        path: {
            "exists": (repo_root / path).exists(),
            "size_bytes": (repo_root / path).stat().st_size if (repo_root / path).exists() else None,
        }
        for path in REQUIRED_FILES
    }
    packages = {
        name: module_available(name)
        for name in [*REQUIRED_PACKAGES, *OPTIONAL_PACKAGES]
    }
    venvs = {
        "current_python": sys.executable,
        "chronos_python": {
            "path": chronos_python,
            "exists": Path(chronos_python).exists(),
        },
        "timesfm_python": {
            "path": timesfm_python,
            "exists": Path(timesfm_python).exists(),
        },
        "moirai_python": {
            "path": moirai_python,
            "exists": Path(moirai_python).exists(),
        },
    }
    venv_imports = {
        name: import_check(venvs[name]["path"], modules)
        for name, modules in VENV_IMPORTS.items()
    }
    dataset_paths = {
        "ml_root": str(ml_root),
        "ml_root_exists": ml_root.exists(),
        "training_tables_exists": (ml_root / "training_tables").exists(),
        "benchmarks_exists": (ml_root / "benchmarks").exists(),
        "sequence_2025_rmse09_exists": (ml_root / "benchmarks/sequence_2025_windsurf_1h_rmse09_v1").exists(),
        "sequence_2026_rmse09_exists": (ml_root / "benchmarks/sequence_2026_windsurf_1h_rmse09_v1").exists(),
    }
    disk = shutil.disk_usage(ml_root if ml_root.exists() else repo_root)
    gpu = run_jsonable(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"])
    reasons = []
    for path, item in files.items():
        if not item["exists"]:
            reasons.append(f"Missing required file: {path}")
    for name in REQUIRED_PACKAGES:
        if not packages.get(name):
            reasons.append(f"Missing required Python package in current env: {name}")
    if not venvs["chronos_python"]["exists"]:
        reasons.append(f"Missing Chronos Python: {chronos_python}")
    for name, item in venv_imports.items():
        if name == "moirai_python":
            continue
        if not item.get("ok"):
            reasons.append(f"{name} missing imports: {item.get('missing_modules')}")
    if not dataset_paths["training_tables_exists"]:
        reasons.append("Missing training_tables directory.")
    return {
        "format": "corsewind.rmse09_environment_preflight.v1",
        "generated_at_utc": utc_now(),
        "repo_root": str(repo_root),
        "files": files,
        "packages": packages,
        "venvs": venvs,
        "venv_imports": venv_imports,
        "dataset_paths": dataset_paths,
        "disk": {
            "total_gb": round(disk.total / 1024**3, 3),
            "used_gb": round(disk.used / 1024**3, 3),
            "free_gb": round(disk.free / 1024**3, 3),
        },
        "gpu": gpu,
        "verdict": "pass" if not reasons else "fail",
        "reasons": reasons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ml-root", type=Path, default=Path("/srv/data/corsewind/ml_dataset"))
    parser.add_argument("--chronos-python", default="/home/z2/corsewind-ml-smoke/.venv/bin/python")
    parser.add_argument("--timesfm-python", default="/home/z2/corsewind-ml-smoke/.venv-timesfm/bin/python")
    parser.add_argument("--moirai-python", default="/home/z2/corsewind-ml-smoke/.venv-moirai/bin/python")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--fail-on-non-pass", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = inspect_environment(
        args.repo_root.resolve(),
        args.ml_root,
        args.chronos_python,
        args.timesfm_python,
        args.moirai_python,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if args.fail_on_non_pass and result["verdict"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
