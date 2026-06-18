#!/usr/bin/env python3
"""Run prepared Corsica WindNinja overview tiles with a wall-clock budget."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "data/processed/physics/corsica_windninja_tile_plan.json"
STATUS_PATH = ROOT / "data/processed/diagnostics/corsica_windninja_batch_status.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def output_exists(case_dir: Path) -> bool:
    output_dir = case_dir / "windninja_corsica_output"
    return output_dir.exists() and any(output_dir.glob("*_vel.asc"))


def run_tile(tile: dict[str, Any]) -> dict[str, Any]:
    case_dir = ROOT / tile["case_dir"]
    cmd = [
        sys.executable,
        "scripts/run_windninja_cases_docker.py",
        "--case-dir",
        str(case_dir.relative_to(ROOT)),
        "--config-name",
        "windninja_corsica_tile.cfg",
        "--output-dir-name",
        "windninja_corsica_output",
    ]
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return {
        "tile_id": tile["tile_id"],
        "case_dir": tile["case_dir"],
        "cmd": " ".join(cmd),
        "status": "pass" if proc.returncode == 0 else "fail",
        "returncode": proc.returncode,
        "runtime_s": round(time.time() - started, 3),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--status-output", type=Path, default=STATUS_PATH)
    parser.add_argument("--max-runtime-min", type=float, default=30.0)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--max-tiles", type=int, default=0, help="Run at most N tiles; 0 means until time budget.")
    parser.add_argument("--force", action="store_true", help="Rerun tiles even if output exists.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    tiles = plan.get("tiles", [])
    if args.max_tiles:
        tiles = tiles[: args.max_tiles]

    candidates = []
    skipped_existing = []
    for tile in tiles:
        case_dir = ROOT / tile["case_dir"]
        if output_exists(case_dir) and not args.force:
            skipped_existing.append(tile["tile_id"])
        else:
            candidates.append(tile)

    if args.dry_run:
        print(f"dry-run: {len(candidates)} candidate tile(s), {len(skipped_existing)} already solved")
        for tile in candidates[:10]:
            print(tile["run_command"])
        return

    started = time.time()
    deadline = started + args.max_runtime_min * 60
    results: list[dict[str, Any]] = []
    submitted = 0

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
        futures = {}
        while (submitted < len(candidates) or futures) and time.time() < deadline:
            while submitted < len(candidates) and len(futures) < max(1, args.parallel) and time.time() < deadline:
                tile = candidates[submitted]
                futures[executor.submit(run_tile, tile)] = tile
                submitted += 1

            if not futures:
                break
            done, _ = wait(futures, timeout=max(1, min(10, deadline - time.time())), return_when=FIRST_COMPLETED)
            for future in done:
                tile = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"tile_id": tile.get("tile_id", "unknown"), "status": "fail", "error": str(exc)}
                results.append(result)
                print(f"{result.get('tile_id')} {result.get('status')} {result.get('runtime_s', '?')}s")

        for future in list(futures):
            result = future.result()
            results.append(result)
            print(f"{result.get('tile_id')} {result.get('status')} {result.get('runtime_s', '?')}s")

    status = {
        "format": "corsewind.windninja.corsica.batch_status.v1",
        "generated_at_utc": utc_now(),
        "plan": str(args.plan.relative_to(ROOT) if args.plan.is_absolute() else args.plan),
        "max_runtime_min": args.max_runtime_min,
        "parallel": args.parallel,
        "submitted": submitted,
        "completed": len(results),
        "passed": sum(1 for result in results if result.get("status") == "pass"),
        "failed": sum(1 for result in results if result.get("status") == "fail"),
        "skipped_existing": skipped_existing,
        "elapsed_s": round(time.time() - started, 3),
        "results": results,
    }
    status_path = args.status_output if args.status_output.is_absolute() else ROOT / args.status_output
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"wrote {status_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
