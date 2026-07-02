#!/usr/bin/env python3
"""Run live foundation experts and champion shadow inference end-to-end."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FOUNDATION_COLUMNS = (
    "chronos2_univar_wind_mean_ms_mean",
    "chronos2_univar_wind_mean_ms_p10",
    "chronos2_univar_wind_mean_ms_p50",
    "chronos2_univar_wind_mean_ms_p90",
    "chronos2_univar_gust_ms_mean",
    "chronos2_univar_gust_ms_p10",
    "chronos2_univar_gust_ms_p50",
    "chronos2_univar_gust_ms_p90",
    "timesfm_wind_mean_ms_mean",
    "timesfm_wind_mean_ms_p10",
    "timesfm_wind_mean_ms_p50",
    "timesfm_wind_mean_ms_p90",
    "timesfm_gust_ms_mean",
    "timesfm_gust_ms_p10",
    "timesfm_gust_ms_p50",
    "timesfm_gust_ms_p90",
)


def import_dependencies() -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Missing pandas dependency. Run inside the CorseWind ML venv.") from exc
    return {"pd": pd}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_command(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    print(f"\n$ {shlex.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def normalize_join_keys(frame: Any, pd: Any) -> Any:
    out = frame.copy()
    out["spot_id"] = out["spot_id"].astype(str)
    out["issue_time_utc"] = (
        pd.to_datetime(out["issue_time_utc"], utc=True, errors="coerce")
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    out["lead_time_minutes"] = pd.to_numeric(out["lead_time_minutes"], errors="coerce").astype("Int64")
    return out


def merge_foundation_predictions(
    *,
    live_rows_parquet: Path,
    foundation_predictions_parquet: Path,
    output_parquet: Path,
    compression: str,
) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]
    live = pd.read_parquet(live_rows_parquet)
    foundation = pd.read_parquet(foundation_predictions_parquet)
    live = normalize_join_keys(live, pd)
    foundation = normalize_join_keys(foundation, pd)
    keep = ["spot_id", "issue_time_utc", "lead_time_minutes"]
    keep.extend(column for column in FOUNDATION_COLUMNS if column in foundation.columns)
    foundation = foundation[keep].drop_duplicates(["spot_id", "issue_time_utc", "lead_time_minutes"], keep="last")
    merged = live.merge(foundation, on=["spot_id", "issue_time_utc", "lead_time_minutes"], how="left")
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_parquet, index=False, compression=compression)
    return {
        "live_rows": int(len(live)),
        "foundation_rows": int(len(foundation)),
        "merged_rows": int(len(merged)),
        "foundation_columns": [column for column in FOUNDATION_COLUMNS if column in merged.columns],
        "non_null_foundation_counts": {
            column: int(merged[column].notna().sum())
            for column in FOUNDATION_COLUMNS
            if column in merged.columns
        },
        "output_parquet": str(output_parquet),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    sequence_root = output_root / "foundation_sequence"
    enriched_rows = output_root / "live_rows_with_foundation.parquet"
    predictions_root = output_root / "predictions"

    history_args = []
    for history in args.history_parquet:
        history_args.extend(["--history-parquet", history])
    observation_history_args = []
    for observation_history in args.history_observations_jsonl:
        observation_history_args.extend(["--history-observations-jsonl", observation_history])

    steps: list[dict[str, Any]] = []
    build_cmd = [
        args.inference_python,
        "scripts/ml_dataset/build_live_foundation_sequence_inputs.py",
        "--live-rows-parquet",
        str(args.live_rows_parquet),
        *history_args,
        *observation_history_args,
        "--output-root",
        str(sequence_root),
        "--context-length",
        str(args.context_length),
        "--freq-minutes",
        str(args.freq_minutes),
    ]
    run_command(build_cmd, cwd=repo_root, dry_run=args.dry_run)
    steps.append({"name": "build_sequence_inputs", "command": build_cmd})
    if not args.dry_run and not args.allow_empty_observed_context:
        manifest_path = sequence_root / "sequence_input_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed_items = int(
            manifest.get("past_context_coverage", {}).get("items_with_observed_context", 0)
        )
        if observed_items <= 0:
            raise SystemExit(
                "Foundation live context has zero observed items. "
                "Pass --allow-empty-observed-context to force the run."
            )

    chronos_output = sequence_root / "predictions_with_chronos2_univariate.parquet"
    if not args.skip_chronos:
        chronos_cmd = [
            args.chronos_python,
            "scripts/ml_dataset/benchmark_chronos2_saved_sequences.py",
            "--benchmark-root",
            str(sequence_root),
            "--predictions-file",
            "predictions.parquet",
            "--run-id",
            args.run_id + "_chronos2_live_shadow",
            "--context-length",
            str(args.context_length),
            "--prediction-length",
            str(args.prediction_length),
            "--batch-size",
            str(args.chronos_batch_size),
            "--cross-learning",
        ]
        run_command(chronos_cmd, cwd=repo_root, dry_run=args.dry_run)
        steps.append({"name": "chronos2_univariate", "command": chronos_cmd})

    timesfm_input = "predictions_with_chronos2_univariate.parquet" if chronos_output.exists() or not args.dry_run else "predictions.parquet"
    timesfm_output = sequence_root / "predictions_with_timesfm.parquet"
    if not args.skip_timesfm:
        timesfm_cmd = [
            args.timesfm_python,
            "scripts/ml_dataset/benchmark_timesfm_sequences.py",
            "--benchmark-root",
            str(sequence_root),
            "--predictions-file",
            timesfm_input,
            "--run-id",
            args.run_id + "_timesfm_live_shadow",
            "--context-length",
            str(args.context_length),
            "--prediction-length",
            str(args.prediction_length),
        ]
        run_command(timesfm_cmd, cwd=repo_root, dry_run=args.dry_run)
        steps.append({"name": "timesfm", "command": timesfm_cmd})

    foundation_predictions = args.foundation_predictions_parquet
    if foundation_predictions is None:
        if timesfm_output.exists() or not args.skip_timesfm:
            foundation_predictions = timesfm_output
        elif chronos_output.exists() or not args.skip_chronos:
            foundation_predictions = chronos_output
        else:
            foundation_predictions = sequence_root / "predictions.parquet"

    merge_summary: dict[str, Any] = {}
    if not args.dry_run:
        merge_summary = merge_foundation_predictions(
            live_rows_parquet=args.live_rows_parquet,
            foundation_predictions_parquet=foundation_predictions,
            output_parquet=enriched_rows,
            compression=args.compression,
        )
    steps.append({
        "name": "merge_foundation_predictions",
        "foundation_predictions_parquet": str(foundation_predictions),
        "output_parquet": str(enriched_rows),
    })

    if not args.skip_inference:
        inference_cmd = [
            args.inference_python,
            "scripts/ml_dataset/run_live_wind_and_gust_inference.py",
            "--input-parquet",
            str(enriched_rows),
            "--output-root",
            str(predictions_root),
            "--limit-json-rows-per-spot",
            str(args.limit_json_rows_per_spot),
        ]
        run_command(inference_cmd, cwd=repo_root, dry_run=args.dry_run)
        steps.append({"name": "champion_shadow_inference", "command": inference_cmd})

    summary = {
        "format": "corsewind.live_foundation_shadow_pipeline.v1",
        "generated_at_utc": utc_now(),
        "run_id": args.run_id,
        "repo_root": str(repo_root),
        "live_rows_parquet": str(args.live_rows_parquet),
        "output_root": str(output_root),
        "sequence_root": str(sequence_root),
        "enriched_rows_parquet": str(enriched_rows),
        "predictions_root": str(predictions_root),
        "foundation_predictions_parquet": str(foundation_predictions),
        "merge_summary": merge_summary,
        "steps": steps,
    }
    if not args.dry_run:
        write_json(output_root / "live_foundation_shadow_pipeline_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--live-rows-parquet", type=Path, required=True)
    parser.add_argument("--history-parquet", action="append", default=[], help="History parquet path or glob. Repeatable.")
    parser.add_argument("--history-observations-jsonl", action="append", default=[], help="Observation JSONL path or glob. Repeatable.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", default="live_foundation_shadow")
    parser.add_argument("--chronos-python", default="/home/z2/corsewind-ml-smoke/.venv/bin/python")
    parser.add_argument("--timesfm-python", default="/home/z2/corsewind-ml-smoke/.venv-timesfm/bin/python")
    parser.add_argument("--inference-python", default="/home/z2/corsewind-ml-smoke/.venv/bin/python")
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--prediction-length", type=int, default=4)
    parser.add_argument("--freq-minutes", type=int, default=15)
    parser.add_argument("--chronos-batch-size", type=int, default=64)
    parser.add_argument("--foundation-predictions-parquet", type=Path)
    parser.add_argument("--skip-chronos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-timesfm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-inference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-empty-observed-context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--limit-json-rows-per-spot", type=int, default=64)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
