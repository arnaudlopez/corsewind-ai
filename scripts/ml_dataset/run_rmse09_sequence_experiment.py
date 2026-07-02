#!/usr/bin/env python3
"""Run the 2025->2026 sequence experiment for the RMSE 0.9 objective."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    printable = shlex.join(cmd)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def command_output(cmd: list[str], cwd: Path) -> dict[str, object]:
    try:
        completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False, timeout=15)
    except Exception as exc:  # pragma: no cover - provenance diagnostic path
        return {"ok": False, "error": str(exc), "command": cmd}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[-4000:],
        "stderr": completed.stderr.strip()[-4000:],
        "command": cmd,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ml-root", type=Path, default=Path("/srv/data/corsewind/ml_dataset"))
    parser.add_argument("--chronos-python", default="/home/z2/corsewind-ml-smoke/.venv/bin/python")
    parser.add_argument("--timesfm-python", default="/home/z2/corsewind-ml-smoke/.venv-timesfm/bin/python")
    parser.add_argument("--moirai-python", default="/home/z2/corsewind-ml-smoke/.venv-moirai/bin/python")
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument("--end-month", default="2026-06")
    parser.add_argument("--training-run-id-prefix", default="residual_windsup_sst_prev")
    parser.add_argument("--train-year", type=int, default=2025)
    parser.add_argument("--eval-year", type=int, default=2026)
    parser.add_argument("--issue-hour-start", type=int, default=8)
    parser.add_argument("--issue-hour-end", type=int, default=17)
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--prediction-length", type=int, default=4)
    parser.add_argument("--max-cutoffs-per-spot", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--include-training-table-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-training-features", type=int, default=1400)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--calibrator-n-jobs", type=int, default=2)
    parser.add_argument(
        "--calibrator-model-family",
        action="append",
        choices=("ridge", "hist_gradient_boosting", "random_forest", "extra_trees", "error_selector_extra_trees", "lightgbm"),
        default=[],
    )
    parser.add_argument("--sweep-suffix")
    parser.add_argument("--selected-run-name")
    parser.add_argument("--require-fresh-training-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-ci-upper-below-threshold", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-lightgbm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-moirai", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--assert-goal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def provenance(repo_root: Path) -> dict[str, object]:
    return {
        "git_head": command_output(["git", "rev-parse", "HEAD"], repo_root),
        "git_branch": command_output(["git", "branch", "--show-current"], repo_root),
        "git_status_short": command_output(["git", "status", "--short"], repo_root),
    }


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    training_root = args.ml_root / "training_tables"
    benchmark_2025 = args.ml_root / "benchmarks" / f"sequence_{args.train_year}_windsurf_1h_rmse09_v1"
    benchmark_2026 = args.ml_root / "benchmarks" / f"sequence_{args.eval_year}_windsurf_1h_rmse09_v1"
    sweep_suffix = args.sweep_suffix or ("context_v1" if args.include_training_table_features else "v1")
    sweep_root = args.ml_root / "benchmarks" / f"calibrator_{args.train_year}_to_{args.eval_year}_sweep_{sweep_suffix}"

    steps = []

    def add_sequence_benchmark_steps(year: int, benchmark_root: Path) -> str:
        run_prefix = f"sequence_{year}_windsurf_1h_rmse09_v1"
        if args.force or not (benchmark_root / "predictions.parquet").exists():
            steps.append((
                f"chronos_covariate_{year}",
                [
                    args.chronos_python,
                    "scripts/ml_dataset/benchmark_chronos2_sequences.py",
                    "--training-table-root",
                    str(training_root),
                    "--run-id-prefix",
                    args.training_run_id_prefix,
                    "--start-month",
                    args.start_month,
                    "--end-month",
                    args.end_month,
                    "--eval-start",
                    f"{year}-01-01T00:00:00Z",
                    "--eval-end",
                    f"{year}-12-31T23:59:59Z",
                    "--context-length",
                    str(args.context_length),
                    "--prediction-length",
                    str(args.prediction_length),
                    "--issue-hour-start",
                    str(args.issue_hour_start),
                    "--issue-hour-end",
                    str(args.issue_hour_end),
                    "--max-cutoffs-per-spot",
                    str(args.max_cutoffs_per_spot),
                    "--skip-hgb",
                    "--output-root",
                    str(benchmark_root),
                    "--run-id",
                    f"{run_prefix}_chronos_covariate",
                    "--batch-size",
                    str(args.batch_size),
                ],
            ))

        if args.force or not (benchmark_root / "predictions_with_chronos2_univariate.parquet").exists():
            steps.append((
                f"chronos_univariate_{year}",
                [
                    args.chronos_python,
                    "scripts/ml_dataset/benchmark_chronos2_saved_sequences.py",
                    "--benchmark-root",
                    str(benchmark_root),
                    "--predictions-file",
                    "predictions.parquet",
                    "--run-id",
                    f"{run_prefix}_chronos2_univar_cross",
                    "--context-length",
                    str(args.context_length),
                    "--prediction-length",
                    str(args.prediction_length),
                    "--cross-learning",
                ],
            ))

        if args.force or not (benchmark_root / "predictions_with_timesfm.parquet").exists():
            steps.append((
                f"timesfm_{year}",
                [
                    args.timesfm_python,
                    "scripts/ml_dataset/benchmark_timesfm_sequences.py",
                    "--benchmark-root",
                    str(benchmark_root),
                    "--predictions-file",
                    "predictions_with_chronos2_univariate.parquet",
                    "--run-id",
                    f"{run_prefix}_timesfm",
                    "--context-length",
                    str(args.context_length),
                    "--prediction-length",
                    str(args.prediction_length),
                ],
            ))

        predictions_file = "predictions_with_timesfm.parquet"
        if args.include_moirai:
            predictions_file = "predictions_with_moirai.parquet"
            if args.force or not (benchmark_root / predictions_file).exists():
                steps.append((
                    f"moirai_{year}",
                    [
                        args.moirai_python,
                        "scripts/ml_dataset/benchmark_moirai_sequences.py",
                        "--benchmark-root",
                        str(benchmark_root),
                        "--predictions-file",
                        "predictions_with_timesfm.parquet",
                        "--run-id",
                        f"{run_prefix}_moirai",
                        "--context-length",
                        str(args.context_length),
                        "--prediction-length",
                        str(args.prediction_length),
                        "--num-samples",
                        "100",
                        "--batch-size",
                        "32",
                    ],
                ))
        return predictions_file

    predictions_file = add_sequence_benchmark_steps(args.train_year, benchmark_2025)
    eval_predictions_file = add_sequence_benchmark_steps(args.eval_year, benchmark_2026)
    if predictions_file != eval_predictions_file:
        raise SystemExit(f"Train/eval predictions file mismatch: {predictions_file} != {eval_predictions_file}")

    if args.include_training_table_features:
        feature_audit_json = sweep_root / "training_table_feature_audit.json"
        feature_audit_md = sweep_root / "training_table_feature_audit.md"
        feature_selection_audit_json = sweep_root / "calibrator_feature_selection_audit.json"
        feature_selection_audit_md = sweep_root / "calibrator_feature_selection_audit.md"
        audit_cmd = [
            args.chronos_python,
            "scripts/ml_dataset/audit_training_table_features.py",
            "--training-table-root",
            str(training_root),
            "--run-id-prefix",
            args.training_run_id_prefix,
            "--start-month",
            args.start_month,
            "--end-month",
            args.end_month,
            "--output-json",
            str(feature_audit_json),
            "--output-md",
            str(feature_audit_md),
        ]
        if args.require_fresh_training_features:
            audit_cmd.append("--fail-on-non-pass")
        steps.append(("training_table_feature_audit", audit_cmd))
        steps.append((
            "calibrator_feature_selection_audit",
            [
                args.chronos_python,
                "scripts/ml_dataset/audit_calibrator_feature_selection.py",
                "--training-table-root",
                str(training_root),
                "--run-id-prefix",
                args.training_run_id_prefix,
                "--start-month",
                args.start_month,
                "--end-month",
                args.end_month,
                "--max-training-features",
                str(args.max_training_features),
                "--output-json",
                str(feature_selection_audit_json),
                "--output-md",
                str(feature_selection_audit_md),
                "--fail-on-missing-required",
            ],
        ))

    steps.append((
        "calibrator_sweep",
        [
            args.chronos_python,
            "scripts/ml_dataset/sweep_sequence_calibrators.py",
            "--benchmark-root",
            str(benchmark_2025),
            "--benchmark-root",
            str(benchmark_2026),
            "--predictions-file",
            predictions_file,
            "--output-root",
            str(sweep_root),
            "--train-end",
            f"{args.eval_year}-01-01T00:00:00Z",
            "--eval-start",
            f"{args.eval_year}-01-01T00:00:00Z",
            "--target-mode",
            "residual",
            "--residual-baseline",
            "raw_wind_mean_ms",
            "--calibrator-n-jobs",
            str(args.calibrator_n_jobs),
        ],
    ))
    for family in args.calibrator_model_family:
        steps[-1][1].extend(["--model-family", family])
    if args.include_training_table_features:
        steps[-1][1].extend([
            "--training-table-root",
            str(training_root),
            "--training-run-id-prefix",
            args.training_run_id_prefix,
            "--start-month",
            args.start_month,
            "--end-month",
            args.end_month,
            "--max-training-features",
            str(args.max_training_features),
            "--max-train-rows",
            str(args.max_train_rows),
            "--require-selected-training-feature",
            "features__model_error_now_",
            "--require-selected-training-feature",
            "features__previous_run_open_meteo_best_match_day1_wind_speed_10m",
            "--require-selected-training-feature",
            "features__previous_run_open_meteo_best_match_day2_wind_speed_10m",
            "--require-selected-training-feature",
            "features__sst_c",
            "--require-selected-training-feature",
            "features__eumetsat_",
        ])
    if args.include_lightgbm:
        steps[-1][1].append("--include-lightgbm")
    audit_json = sweep_root / "rmse09_audit.json"
    audit_md = sweep_root / "rmse09_audit.md"
    analysis_json = sweep_root / "rmse09_error_analysis.json"
    analysis_md = sweep_root / "rmse09_error_analysis.md"
    decision_json = sweep_root / "rmse09_decision.json"
    decision_md = sweep_root / "rmse09_decision.md"
    manifest_json = sweep_root / "rmse09_run_manifest.json"
    steps.append((
        "rmse09_audit",
        [
            args.chronos_python,
            "scripts/ml_dataset/audit_rmse09_results.py",
            str(sweep_root / "sweep_results.json"),
            "--output-json",
            str(audit_json),
            "--output-md",
            str(audit_md),
            "--bootstrap-samples",
            "1000",
            "--ci-confidence",
            "0.95",
            "--bootstrap-unit",
            "issue_day",
            "--require-prediction-diagnostics",
        ],
    ))
    if args.selected_run_name:
        steps[-1][1].extend(["--selected-run-name", args.selected_run_name])
    if args.require_ci_upper_below_threshold:
        steps[-1][1].append("--require-ci-upper-below-threshold")
    steps.append((
        "rmse09_error_analysis",
        [
            args.chronos_python,
            "scripts/ml_dataset/analyze_rmse09_errors.py",
            str(sweep_root / "sweep_results.json"),
            "--audit-json",
            str(audit_json),
            "--output-json",
            str(analysis_json),
            "--output-md",
            str(analysis_md),
        ],
    ))
    steps.append((
        "rmse09_decision",
        [
            args.chronos_python,
            "scripts/ml_dataset/summarize_rmse09_decision.py",
            "--audit-json",
            str(audit_json),
            "--analysis-json",
            str(analysis_json),
            "--output-json",
            str(decision_json),
            "--output-md",
            str(decision_md),
        ],
    ))
    final_assert_command = [
        args.chronos_python,
        "scripts/ml_dataset/assert_rmse09_goal.py",
        "--audit-json",
        str(audit_json),
        "--decision-json",
        str(decision_json),
    ]

    manifest = {
        "format": "corsewind.rmse09_run_manifest.v1",
        "generated_at_utc": utc_now(),
        "repo_root": str(repo_root),
        "ml_root": str(args.ml_root),
        "run_options": {
            "start_month": args.start_month,
            "end_month": args.end_month,
            "training_run_id_prefix": args.training_run_id_prefix,
            "train_year": args.train_year,
            "eval_year": args.eval_year,
            "issue_hour_start": args.issue_hour_start,
            "issue_hour_end": args.issue_hour_end,
            "context_length": args.context_length,
            "prediction_length": args.prediction_length,
            "max_cutoffs_per_spot": args.max_cutoffs_per_spot,
            "batch_size": args.batch_size,
            "include_training_table_features": args.include_training_table_features,
            "max_training_features": args.max_training_features,
            "max_train_rows": args.max_train_rows,
            "calibrator_n_jobs": args.calibrator_n_jobs,
            "calibrator_model_families": args.calibrator_model_family,
            "require_fresh_training_features": args.require_fresh_training_features,
            "require_ci_upper_below_threshold": args.require_ci_upper_below_threshold,
            "include_lightgbm": args.include_lightgbm,
            "include_moirai": args.include_moirai,
            "assert_goal": args.assert_goal,
            "force": args.force,
            "dry_run": args.dry_run,
        },
        "python_envs": {
            "chronos_python": args.chronos_python,
            "timesfm_python": args.timesfm_python,
            "moirai_python": args.moirai_python,
        },
        "benchmark_train_root": str(benchmark_2025),
        "benchmark_eval_root": str(benchmark_2026),
        "sweep_root": str(sweep_root),
        "predictions_file": predictions_file,
        "expected_artifacts": {
            "train_predictions": str(benchmark_2025 / predictions_file),
            "eval_predictions": str(benchmark_2026 / predictions_file),
            "training_table_feature_audit_json": str(sweep_root / "training_table_feature_audit.json"),
            "calibrator_feature_selection_audit_json": str(sweep_root / "calibrator_feature_selection_audit.json"),
            "sweep_results_json": str(sweep_root / "sweep_results.json"),
            "audit_json": str(audit_json),
            "audit_md": str(audit_md),
            "analysis_json": str(analysis_json),
            "analysis_md": str(analysis_md),
            "decision_json": str(decision_json),
            "decision_md": str(decision_md),
            "manifest_json": str(manifest_json),
        },
        "steps": [{"name": name, "command": command} for name, command in steps],
        "final_assert_command": final_assert_command,
        "provenance": provenance(repo_root),
    }
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not args.dry_run:
        manifest_json.parent.mkdir(parents=True, exist_ok=True)
        manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    for _, command in steps:
        run(command, cwd=repo_root, dry_run=args.dry_run)
    if args.assert_goal:
        run(final_assert_command, cwd=repo_root, dry_run=args.dry_run)

    best = None
    audit = None
    sweep_results_path = sweep_root / "sweep_results.json"
    if not args.dry_run and sweep_results_path.exists():
        summary = json.loads(sweep_results_path.read_text(encoding="utf-8"))
        runs = summary.get("runs", [])
        if runs:
            best = {
                "model_family": runs[0].get("model_family"),
                "run_name": runs[0].get("run_name"),
                "fit_group": runs[0].get("fit_group"),
                "metric": runs[0].get("metrics", {}).get("calibrator", {}),
                "meets_rmse_0_9": runs[0].get("metrics", {}).get("calibrator", {}).get("rmse", float("inf")) < 0.9,
            }
    if not args.dry_run and audit_json.exists():
        audit = json.loads(audit_json.read_text(encoding="utf-8"))

    print(json.dumps({
        "audit": {
            "best_model_family": audit.get("best_model_family"),
            "best_run_name": audit.get("best_run_name"),
            "best_fit_group": audit.get("best_fit_group"),
            "best_metric": audit.get("best_metric"),
            "effective_rmse": audit.get("effective_rmse"),
            "effective_rmse_source": audit.get("effective_rmse_source"),
            "prediction_diagnostics": audit.get("prediction_diagnostics"),
            "reasons": audit.get("reasons"),
            "verdict": audit.get("verdict"),
            "warnings": audit.get("warnings"),
        } if audit else None,
        "best": best,
        "audit_json": str(audit_json),
        "audit_md": str(audit_md),
        "analysis_json": str(analysis_json),
        "analysis_md": str(analysis_md),
        "decision_json": str(decision_json),
        "decision_md": str(decision_md),
        "manifest_json": str(manifest_json),
        "final_assert_command": shlex.join(final_assert_command),
        "generated_at_utc": utc_now(),
        "sweep_results": str(sweep_root / "sweep_results.json"),
        "success_metric": "rmse09_audit.json verdict == pass, with prediction diagnostics available and no leakage reasons",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
