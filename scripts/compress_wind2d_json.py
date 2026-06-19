#!/usr/bin/env python3
"""Write reproducible gzip companions for generated Wind2D JSON payloads."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOADS = (
    ROOT / "visualizations/wind2d/arome-corsica-latest.json",
    ROOT / "visualizations/wind2d/aromepi-corsica-latest.json",
    ROOT / "visualizations/wind2d/moloch-corsica-latest.json",
    ROOT / "visualizations/wind2d/icon2i-corsica-latest.json",
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def compress_file(path: Path, level: int) -> dict[str, Any]:
    source = resolve_path(path)
    if not source.exists():
        return {"path": str(source.relative_to(ROOT)), "status": "missing"}

    data = source.read_bytes()
    compressed = gzip.compress(data, compresslevel=level, mtime=0)
    target = source.with_suffix(source.suffix + ".gz")
    tmp = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    tmp.write_bytes(compressed)
    tmp.replace(target)
    return {
        "path": str(source.relative_to(ROOT)),
        "gzip_path": str(target.relative_to(ROOT)),
        "status": "compressed",
        "bytes": len(data),
        "gzip_bytes": len(compressed),
        "ratio": round(len(compressed) / len(data), 4) if data else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=DEFAULT_PAYLOADS)
    parser.add_argument("--level", type=int, default=9, choices=range(1, 10))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [compress_file(path, args.level) for path in args.paths]
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
