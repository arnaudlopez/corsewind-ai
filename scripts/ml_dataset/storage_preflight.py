#!/usr/bin/env python3
"""Check ML dataset storage before running large backfills."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))


def bytes_to_gib(value: int) -> float:
    return round(value / (1024**3), 3)


def tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def resolve_storage_path(path: Path, create: bool) -> Path:
    resolved = path.expanduser()
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        return resolved.resolve()
    parent = resolved.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.resolve()


def status(args: argparse.Namespace) -> dict[str, Any]:
    target = args.ml_root.expanduser()
    usage_path = resolve_storage_path(target, args.create)
    usage = shutil.disk_usage(usage_path)
    target_size = tree_size(target)
    repo_usage = shutil.disk_usage(ROOT)
    inside_repo = ROOT in target.resolve().parents if target.exists() else ROOT in usage_path.parents
    min_free_bytes = int(args.min_free_gb * 1024**3)
    ok = usage.free >= min_free_bytes
    return {
        "ok": ok,
        "target": str(target),
        "usage_path": str(usage_path),
        "target_exists": target.exists(),
        "created": args.create and target.exists(),
        "inside_repo": inside_repo,
        "free_gib": bytes_to_gib(usage.free),
        "total_gib": bytes_to_gib(usage.total),
        "used_gib": bytes_to_gib(usage.used),
        "target_current_size_gib": bytes_to_gib(target_size),
        "required_free_gib": args.min_free_gb,
        "repo_free_gib": bytes_to_gib(repo_usage.free),
        "recommendation": (
            "OK for large ML dataset writes."
            if ok and not inside_repo
            else "Use an external disk path via ML_DATASET_ROOT before large backfills."
            if inside_repo
            else "Not enough free space for the requested backfill budget."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-root", type=Path, default=DEFAULT_ML_ROOT)
    parser.add_argument("--min-free-gb", type=float, default=float(os.getenv("ML_DATASET_MIN_FREE_GB", "250")))
    parser.add_argument("--create", action="store_true", help="Create the target directory before checking it.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = status(args)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
