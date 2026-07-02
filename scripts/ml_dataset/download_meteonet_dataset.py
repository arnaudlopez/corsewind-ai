#!/usr/bin/env python3
"""Download selected MeteoNet archives from a generated inventory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_INVENTORY = DEFAULT_ML_ROOT / "source_inventories/meteonet_inventory.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_ML_ROOT / "research/meteonet/raw"
DEFAULT_MANIFEST = DEFAULT_ML_ROOT / "research/meteonet/download_manifest.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_curl(url: str, output_path: Path, timeout_sec: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "--fail",
        "--location",
        "--continue-at",
        "-",
        "--retry",
        "5",
        "--retry-delay",
        "10",
        "--connect-timeout",
        "30",
        "--max-time",
        str(timeout_sec),
        "--output",
        str(output_path),
        url,
    ]
    subprocess.run(cmd, check=True)


def run_python_download(url: str, output_path: Path, timeout_sec: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = output_path.stat().st_size if output_path.exists() else 0
    headers = {"User-Agent": "CorseWind.ai MeteoNet downloader"}
    mode = "wb"
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response, output_path.open(mode + "") as handle:
            if existing > 0 and response.status == 200:
                handle.seek(0)
                handle.truncate()
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    except Exception:
        if existing > 0:
            headers.pop("Range", None)
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout_sec) as response, output_path.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
        else:
            raise


def download_file(url: str, output_path: Path, timeout_sec: int) -> None:
    if shutil.which("curl"):
        run_curl(url, output_path, timeout_sec)
    else:
        run_python_download(url, output_path, timeout_sec)


def category_path(item: dict[str, Any]) -> Path:
    zone = str(item.get("zone") or "unknown")
    category = str(item.get("category") or "misc")
    return Path(zone) / category / str(item["filename"])


def manifest_row(item: dict[str, Any], output_path: Path, status: str) -> dict[str, Any]:
    size = output_path.stat().st_size if output_path.exists() else 0
    expected = item.get("size_bytes")
    return {
        "format": "corsewind.meteonet_download_manifest.v1",
        "downloaded_at_utc": utc_now(),
        "status": status,
        "url": item.get("url"),
        "filename": item.get("filename"),
        "zone": item.get("zone"),
        "category": item.get("category"),
        "year": item.get("year"),
        "expected_size_bytes": expected,
        "local_size_bytes": size,
        "complete": expected is not None and int(expected) == size,
        "path": str(output_path),
    }


def append_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--zone", default="SE")
    parser.add_argument("--category", default="ground_stations")
    parser.add_argument("--year", action="append", type=int, help="Optional year filter. Can be repeated.")
    parser.add_argument("--timeout-sec", type=int, default=12 * 60 * 60)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = read_json(args.inventory)
    selected = []
    years = set(args.year or [])
    for item in payload.get("files", []):
        if item.get("zone") != args.zone:
            continue
        if item.get("category") != args.category:
            continue
        if years and item.get("year") not in years:
            continue
        selected.append(item)

    rows = []
    for item in selected:
        output_path = args.output_root / category_path(item)
        expected = item.get("size_bytes")
        if output_path.exists() and expected is not None and output_path.stat().st_size == int(expected):
            rows.append(manifest_row(item, output_path, "already_complete"))
            continue
        if args.dry_run:
            rows.append(manifest_row(item, output_path, "dry_run"))
            continue
        download_file(str(item["url"]), output_path, args.timeout_sec)
        rows.append(manifest_row(item, output_path, "downloaded"))

    append_manifest(args.manifest, rows)
    print(json.dumps({
        "generated_at_utc": utc_now(),
        "selected_count": len(selected),
        "manifest": str(args.manifest),
        "rows": rows,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
