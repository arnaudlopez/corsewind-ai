#!/usr/bin/env python3
"""Run a pseudo-live AROME-PI hindcast and score it after observations arrive."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_LEADS = "15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345,360,375,390,405,420,435,450,465,480,495,510,525,540,555,570,585,600,615"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def minutes_between(left: datetime, right: datetime) -> int:
    return int(round((right - left).total_seconds() / 60.0))


def run_command(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    print(f"\n$ {shlex.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    issue_time = parse_time(args.issue_time_utc)
    run_time = parse_time(args.run_time_utc)
    target_end = parse_time(args.target_end_utc)
    if target_end <= issue_time:
        raise SystemExit("--target-end-utc must be after --issue-time-utc")
    start_lead = minutes_between(run_time, issue_time + args.step_minutes_delta)
    end_lead = minutes_between(run_time, target_end)
    if start_lead <= 0:
        raise SystemExit("Issue time must be after run time enough to produce positive forecast leads.")

    grid_path = output_root / "aromepi_grid_layer.json"
    feature_root = output_root / "feature_store"
    training_root = output_root / "training_rows"
    predictions_root = output_root / "predictions"
    score_json = output_root / "hindcast_score.json"
    scored_rows = output_root / "hindcast_scored_rows.parquet"

    python = args.python
    steps: list[dict[str, Any]] = []

    output_root.mkdir(parents=True, exist_ok=True)
    grid_cmd = [
        python,
        "scripts/ml_dataset/create_meteo_france_forecast_grid_layer.py",
        "--source",
        args.source,
        "--run-time-utc",
        args.run_time_utc,
        "--start-lead-minutes",
        str(start_lead),
        "--end-lead-minutes",
        str(end_lead),
        "--step-minutes",
        str(args.step_minutes),
        "--output",
        str(grid_path),
    ]
    run_command(grid_cmd, cwd=repo_root, dry_run=args.dry_run)
    steps.append({"name": "create_grid", "command": grid_cmd})

    feature_cmd = [
        python,
        "scripts/ml_dataset/build_spot_feature_store.py",
        "--ml-root",
        str(args.ml_root),
        "--registry",
        str(args.registry),
        "--context-registry",
        str(args.context_registry),
        "--spot-static-features",
        str(args.spot_static_features),
        "--output-root",
        str(feature_root),
        "--schema-doc",
        str(feature_root / "feature_store_schema.md"),
        "--start-datetime",
        args.issue_time_utc,
        "--end-datetime",
        args.target_end_utc,
        "--include-inference-grid",
        "--step-minutes",
        str(args.step_minutes),
        "--read-margin-days-before",
        str(args.read_margin_days_before),
        "--observation-history-max-age-minutes",
        str(args.observation_history_max_age_minutes),
        "--context-station-max-age-minutes",
        str(args.context_station_max_age_minutes),
    ]
    run_command(feature_cmd, cwd=repo_root, dry_run=args.dry_run)
    steps.append({"name": "build_feature_store", "command": feature_cmd})

    training_cmd = [
        python,
        "scripts/ml_dataset/build_residual_training_table.py",
        "--feature-store",
        str(feature_root / "spot_forecast_15min.jsonl"),
        "--output-root",
        str(training_root),
        "--lead-minutes",
        args.lead_minutes,
        "--issue-start-datetime",
        args.issue_time_utc,
        "--issue-end-datetime",
        args.issue_time_utc,
    ]
    run_command(training_cmd, cwd=repo_root, dry_run=args.dry_run)
    steps.append({"name": "build_training_rows", "command": training_cmd})

    export_cmd = [
        python,
        "scripts/ml_dataset/export_training_table_parquet.py",
        "--training-rows",
        str(training_root / "training_rows.jsonl"),
        "--output-parquet",
        str(training_root / "training_rows.parquet"),
    ]
    run_command(export_cmd, cwd=repo_root, dry_run=args.dry_run)
    steps.append({"name": "export_training_rows", "command": export_cmd})

    input_parquet = training_root / "training_rows.parquet"
    if args.with_foundation:
        foundation_root = output_root / "shadow_foundation"
        foundation_cmd = [
            python,
            "scripts/ml_dataset/run_live_foundation_shadow_pipeline.py",
            "--repo-root",
            str(repo_root),
            "--live-rows-parquet",
            str(input_parquet),
            "--output-root",
            str(foundation_root),
            "--run-id",
            args.run_id,
            "--prediction-length",
            str(args.foundation_prediction_length),
            "--limit-json-rows-per-spot",
            str(args.limit_json_rows_per_spot),
        ]
        for history_parquet in args.history_parquet:
            foundation_cmd.extend(["--history-parquet", history_parquet])
        for observation_glob in args.history_observations_jsonl:
            foundation_cmd.extend(["--history-observations-jsonl", observation_glob])
        if args.allow_empty_observed_context:
            foundation_cmd.append("--allow-empty-observed-context")
        run_command(foundation_cmd, cwd=repo_root, dry_run=args.dry_run)
        steps.append({"name": "foundation_shadow_pipeline", "command": foundation_cmd})
        predictions_parquet = foundation_root / "predictions" / "predictions.parquet"
    else:
        inference_cmd = [
            python,
            "scripts/ml_dataset/run_live_wind_and_gust_inference.py",
            "--input-parquet",
            str(input_parquet),
            "--output-root",
            str(predictions_root),
            "--limit-json-rows-per-spot",
            str(args.limit_json_rows_per_spot),
        ]
        run_command(inference_cmd, cwd=repo_root, dry_run=args.dry_run)
        steps.append({"name": "champion_inference", "command": inference_cmd})
        predictions_parquet = predictions_root / "predictions.parquet"

    score_cmd = [
        python,
        "scripts/ml_dataset/score_live_hindcast_predictions.py",
        "--predictions-parquet",
        str(predictions_parquet),
        "--output-json",
        str(score_json),
        "--output-scored-parquet",
        str(scored_rows),
        "--target-start-utc",
        iso_z(issue_time + args.step_minutes_delta),
        "--target-end-utc",
        args.target_end_utc,
        "--tolerance-minutes",
        str(args.score_tolerance_minutes),
    ]
    for observations in args.observations_jsonl:
        score_cmd.extend(["--observations-jsonl", observations])
    if args.score_spots:
        score_cmd.extend(["--spots", args.score_spots])
    run_command(score_cmd, cwd=repo_root, dry_run=args.dry_run)
    steps.append({"name": "score_predictions", "command": score_cmd})

    score_summary: dict[str, Any] = {}
    if not args.dry_run:
        score_summary = read_json(score_json)

    summary = {
        "format": "corsewind.aromepi_hindcast_evaluation.v1",
        "generated_at_utc": utc_now(),
        "run_id": args.run_id,
        "source": args.source,
        "run_time_utc": args.run_time_utc,
        "issue_time_utc": args.issue_time_utc,
        "target_end_utc": args.target_end_utc,
        "output_root": str(output_root),
        "predictions_parquet": str(predictions_parquet),
        "score_json": str(score_json),
        "score_summary": score_summary,
        "steps": steps,
    }
    if not args.dry_run:
        write_json(output_root / "hindcast_run_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ml-root", type=Path, default=Path("/srv/data/corsewind/ml_dataset"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source", default="aromepi")
    parser.add_argument("--run-time-utc", required=True)
    parser.add_argument("--issue-time-utc", required=True)
    parser.add_argument("--target-end-utc", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("configs/ml_spots.json"))
    parser.add_argument("--context-registry", type=Path, default=Path("configs/ml_context_stations.json"))
    parser.add_argument("--spot-static-features", type=Path, default=Path("configs/ml_spot_static_features.json"))
    parser.add_argument("--observations-jsonl", action="append", default=[], required=True)
    parser.add_argument("--score-spots")
    parser.add_argument("--lead-minutes", default=DEFAULT_LEADS)
    parser.add_argument("--step-minutes", type=int, default=15)
    parser.add_argument("--read-margin-days-before", type=int, default=5)
    parser.add_argument("--observation-history-max-age-minutes", type=float, default=360.0)
    parser.add_argument("--context-station-max-age-minutes", type=float, default=360.0)
    parser.add_argument("--limit-json-rows-per-spot", type=int, default=64)
    parser.add_argument("--score-tolerance-minutes", type=float, default=8.0)
    parser.add_argument("--with-foundation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--history-parquet", action="append", default=[])
    parser.add_argument("--history-observations-jsonl", action="append", default=[])
    parser.add_argument("--foundation-prediction-length", type=int, default=4)
    parser.add_argument("--allow-empty-observed-context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    args.step_minutes_delta = timedelta(minutes=args.step_minutes)
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
