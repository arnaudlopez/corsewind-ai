#!/usr/bin/env python3
"""Helpers for deleting temporary raw weather downloads after publication."""

from __future__ import annotations

import shutil
import argparse
from pathlib import Path
from typing import Any


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file() or child.is_symlink():
            total += child.stat().st_size
    return total


def cleanup_raw_dir(raw_dir: Path, root: Path) -> dict[str, Any]:
    """Remove a raw download directory, guarded to stay under repo data/raw."""

    repo_root = root.resolve()
    raw_root = (repo_root / "data/raw").resolve()
    target = (raw_dir if raw_dir.is_absolute() else repo_root / raw_dir).resolve()
    if target == raw_root or raw_root not in target.parents:
        raise SystemExit(f"Refusing to cleanup raw path outside data/raw: {target}")
    if not target.exists():
        return {"path": str(target), "removed": False, "bytes": 0}

    size_bytes = _path_size_bytes(target)
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
    else:
        target.unlink()

    return {"path": str(target), "removed": True, "bytes": size_bytes}


def cleanup_message(result: dict[str, Any]) -> str:
    size_mb = float(result.get("bytes") or 0) / (1024 * 1024)
    if not result.get("removed"):
        return f"cleanup raw skipped; path absent: {result.get('path')}"
    return f"cleanup raw removed {size_mb:.1f} MiB from {result.get('path')}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dirs", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    for raw_dir in args.raw_dirs:
        print(cleanup_message(cleanup_raw_dir(raw_dir, args.root)))


if __name__ == "__main__":
    main()
