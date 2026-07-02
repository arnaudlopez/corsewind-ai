#!/usr/bin/env python3
"""Recompute score JSON files for an existing shadow suite.

This is useful when scoring metrics evolve, while predictions and observations
are already available. It does not rerun model inference.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


OBSERVATION_DATASETS = ("dpobs_station_infrahoraire_6m", "dpobs_station_horaire")


PREDICTION_SCORE_SPECS = (
    (
        "base",
        Path("predictions/predictions.parquet"),
        Path("hindcast_score.json"),
        Path("hindcast_scored_rows.parquet"),
    ),
    (
        "shadow_router_v1",
        Path("predictions/predictions_with_shadow_router_v1.parquet"),
        Path("hindcast_score_with_shadow_router_v1.json"),
        Path("hindcast_scored_rows_with_shadow_router_v1.parquet"),
    ),
    (
        "guarded_stacker_v1",
        Path("predictions/predictions_with_guarded_stacker_v1.parquet"),
        Path("hindcast_score_with_guarded_stacker_v1.json"),
        Path("hindcast_scored_rows_with_guarded_stacker_v1.parquet"),
    ),
    (
        "threshold_guard_v1",
        Path("predictions/predictions_with_threshold_guard_v1.parquet"),
        Path("hindcast_score_with_threshold_guard_v1.json"),
        Path("hindcast_scored_rows_with_threshold_guard_v1.parquet"),
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def observation_paths_for_case(case: dict[str, Any], ml_root: Path) -> list[Path]:
    issue = parse_time(str(case["issue_time_utc"]))
    target_end = parse_time(str(case["target_end_utc"]))
    start = issue + timedelta(minutes=15)
    paths: list[Path] = []
    current = start.date()
    while current <= target_end.date():
        for dataset in OBSERVATION_DATASETS:
            path = (
                ml_root
                / "observations"
                / "meteo_france"
                / f"source_dataset={dataset}"
                / f"date={current.isoformat()}"
                / "observations.jsonl"
            )
            if path.exists():
                paths.append(path)
        current += timedelta(days=1)
    return paths


def run_command(cmd: list[str], *, cwd: Path, dry_run: bool) -> int:
    print(f"$ {shlex.join(cmd)}", flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=cwd, check=False)
    return int(completed.returncode)


def build_score_command(
    *,
    python: str,
    case: dict[str, Any],
    case_root: Path,
    prediction_path: Path,
    output_json: Path,
    output_scored_parquet: Path,
    observations: list[Path],
    score_spots: str,
    tolerance_minutes: float,
) -> list[str]:
    cmd = [
        python,
        "scripts/ml_dataset/score_live_hindcast_predictions.py",
        "--predictions-parquet",
        str(prediction_path),
        "--output-json",
        str(output_json),
        "--output-scored-parquet",
        str(output_scored_parquet),
        "--spots",
        score_spots,
        "--target-start-utc",
        iso_z(parse_time(str(case["issue_time_utc"])) + timedelta(minutes=15)),
        "--target-end-utc",
        str(case["target_end_utc"]),
        "--tolerance-minutes",
        str(tolerance_minutes),
    ]
    for path in observations:
        cmd.extend(["--observations-jsonl", str(path)])
    return cmd


def rescore_case(args: argparse.Namespace, case: dict[str, Any], ml_root: Path, score_spots: str) -> dict[str, Any]:
    case_root = Path(str(case["output_root"]))
    observations = observation_paths_for_case(case, ml_root)
    case_report: dict[str, Any] = {
        "run_id": case.get("run_id"),
        "case_root": str(case_root),
        "observation_count": len(observations),
        "outputs": [],
    }
    if not observations:
        case_report["error"] = "no observation files found"
        return case_report

    for name, prediction_rel, score_rel, scored_rel in PREDICTION_SCORE_SPECS:
        prediction_path = case_root / prediction_rel
        if not prediction_path.exists():
            case_report["outputs"].append({"name": name, "exists": False, "skipped": True})
            continue
        output_json = case_root / score_rel
        output_scored = case_root / scored_rel
        if output_json.exists() and not args.overwrite:
            case_report["outputs"].append(
                {
                    "name": name,
                    "exists": True,
                    "skipped": True,
                    "reason": "score exists; use --overwrite",
                    "score_json": str(output_json),
                }
            )
            continue
        cmd = build_score_command(
            python=args.python,
            case=case,
            case_root=case_root,
            prediction_path=prediction_path,
            output_json=output_json,
            output_scored_parquet=output_scored,
            observations=observations,
            score_spots=score_spots,
            tolerance_minutes=args.tolerance_minutes,
        )
        code = run_command(cmd, cwd=args.repo_root, dry_run=args.dry_run)
        case_report["outputs"].append(
            {
                "name": name,
                "exists": True,
                "skipped": False,
                "returncode": code,
                "score_json": str(output_json),
                "scored_parquet": str(output_scored),
            }
        )
        if code != 0 and not args.continue_on_error:
            raise SystemExit(code)
    return case_report


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary = read_json(args.suite_summary)
    ml_root = args.ml_root or Path(str(summary.get("ml_root") or "/srv/data/corsewind/ml_dataset"))
    score_spots = args.score_spots or str(summary.get("score_spots") or "cap_corse,la_parata,lfkf,lfkj,lfks,lfvf,lfvh")
    report = {
        "format": "corsewind.shadow_suite_rescore.v1",
        "generated_at_utc": utc_now(),
        "suite_summary": str(args.suite_summary),
        "ml_root": str(ml_root),
        "score_spots": score_spots,
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "cases": [],
    }
    for case in summary.get("cases") or []:
        report["cases"].append(rescore_case(args, case, ml_root, score_spots))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-summary", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ml-root", type=Path)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--score-spots")
    parser.add_argument("--tolerance-minutes", type=float, default=8.0)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
