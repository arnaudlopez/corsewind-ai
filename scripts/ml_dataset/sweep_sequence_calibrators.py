#!/usr/bin/env python3
"""Run several leakage-safe sequence calibrators and summarize their metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODELS = ["ridge", "hist_gradient_boosting", "random_forest", "extra_trees", "error_selector_extra_trees"]
ALL_MODELS = [*DEFAULT_MODELS, "lightgbm"]
LEAD_STRATIFIED_MODELS = ["ridge", "hist_gradient_boosting", "extra_trees"]
SPOT_STRATIFIED_MODELS = ["ridge", "hist_gradient_boosting", "extra_trees"]
SELECTOR_STRATIFIED_MODELS = ["error_selector_extra_trees"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, action="append", required=True)
    parser.add_argument("--predictions-file", default="predictions_with_timesfm.parquet")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-end", default="2026-01-01T00:00:00Z")
    parser.add_argument("--eval-start", default="2026-01-01T00:00:00Z")
    parser.add_argument("--target-mode", choices=("direct", "residual"), default="residual")
    parser.add_argument("--residual-baseline", default="raw_wind_mean_ms")
    parser.add_argument("--training-table-root", type=Path)
    parser.add_argument("--training-run-id-prefix", default="residual_windsup_sst_prev")
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument("--end-month", default="2026-06")
    parser.add_argument("--training-feature-prefix", action="append", default=[])
    parser.add_argument("--require-selected-training-feature", action="append", default=[])
    parser.add_argument("--max-training-features", type=int, default=1400)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--calibrator-n-jobs", type=int, default=2)
    parser.add_argument("--model-family", action="append", choices=ALL_MODELS, default=[])
    parser.add_argument("--include-lightgbm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-lead-stratified", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-spot-stratified", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-spot-lead-stratified", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-group-train-rows", type=int, default=200)
    parser.add_argument("--continue-on-model-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--calibrator-script", type=Path, default=Path(__file__).with_name("train_sequence_calibrator.py"))
    return parser.parse_args()


def run_name(family: str, fit_group: list[str]) -> str:
    if not fit_group:
        return family
    return f"{family}_by_{'_'.join(fit_group)}"


def run_calibrator(args: argparse.Namespace, family: str, fit_group: list[str] | None = None) -> dict:
    fit_group = fit_group or []
    name = run_name(family, fit_group)
    run_root = args.output_root / name
    cmd = [
        args.python,
        str(args.calibrator_script),
        "--predictions-file",
        args.predictions_file,
        "--output-root",
        str(run_root),
        "--train-end",
        args.train_end,
        "--eval-start",
        args.eval_start,
        "--target-mode",
        args.target_mode,
        "--residual-baseline",
        args.residual_baseline,
        "--model-family",
        family,
        "--n-jobs",
        str(args.calibrator_n_jobs),
    ]
    for group in fit_group:
        cmd.extend(["--fit-group", group])
    if fit_group:
        cmd.extend(["--min-group-train-rows", str(args.min_group_train_rows)])
    for root in args.benchmark_root:
        cmd.extend(["--benchmark-root", str(root)])
    if args.training_table_root is not None:
        cmd.extend([
            "--training-table-root",
            str(args.training_table_root),
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
        ])
        for prefix in args.training_feature_prefix:
            cmd.extend(["--training-feature-prefix", prefix])
        for pattern in args.require_selected_training_feature:
            cmd.extend(["--require-selected-training-feature", pattern])
    if family == "hist_gradient_boosting":
        cmd.extend([
            "--max-iter",
            "160",
            "--learning-rate",
            "0.04",
            "--max-leaf-nodes",
            "15",
            "--l2-regularization",
            "0.2",
        ])
    if family == "random_forest":
        cmd.extend([
            "--n-estimators",
            "500",
            "--min-samples-leaf",
            "8",
        ])
    if family == "extra_trees":
        cmd.extend([
            "--n-estimators",
            "700",
            "--min-samples-leaf",
            "6",
        ])
    if family == "error_selector_extra_trees":
        cmd.extend([
            "--n-estimators",
            "240",
            "--min-samples-leaf",
            "10",
        ])
    if family == "lightgbm":
        cmd.extend([
            "--n-estimators",
            "600",
            "--learning-rate",
            "0.03",
            "--num-leaves",
            "31",
            "--min-samples-leaf",
            "12",
            "--l2-regularization",
            "0.2",
        ])
    try:
        subprocess.run(cmd, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        if args.continue_on_model_error:
            return {
                "run_name": name,
                "model_family": family,
                "fit_group": fit_group,
                "status": "error",
                "returncode": exc.returncode,
                "stdout_tail": (exc.stdout or "")[-4000:],
                "stderr_tail": (exc.stderr or "")[-4000:],
                "metrics": {},
                "train_rows": 0,
                "test_rows": 0,
            }
        raise
    result_path = run_root / "calibrator_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["run_name"] = name
    result["status"] = "ok"
    return result


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Sequence Calibrator Sweep",
        "",
        f"Generated: `{summary['generated_at_utc']}`",
        "",
        "| Model | Fit group | RMSE | MAE | Bias | Train rows | Test rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["runs"]:
        metric = item.get("metrics", {}).get("calibrator", {})
        fit_group = ", ".join(item.get("fit_group") or []) or "global"
        lines.append(
            f"| `{item.get('run_name', item['model_family'])}` | `{fit_group}` | {metric.get('rmse')} | {metric.get('mae')} | "
            f"{metric.get('bias')} | {item.get('train_rows')} | {item.get('test_rows')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    families = args.model_family or [*DEFAULT_MODELS, *(["lightgbm"] if args.include_lightgbm else [])]
    variants = [(family, []) for family in families]
    if args.include_lead_stratified:
        for family in [*LEAD_STRATIFIED_MODELS, *SELECTOR_STRATIFIED_MODELS, *(["lightgbm"] if args.include_lightgbm else [])]:
            if family in families:
                variants.append((family, ["lead_time_minutes"]))
    if args.include_spot_stratified:
        for family in [*SPOT_STRATIFIED_MODELS, *SELECTOR_STRATIFIED_MODELS, *(["lightgbm"] if args.include_lightgbm else [])]:
            if family in families:
                variants.append((family, ["spot_id"]))
    if args.include_spot_lead_stratified:
        for family in [*SPOT_STRATIFIED_MODELS, *SELECTOR_STRATIFIED_MODELS, *(["lightgbm"] if args.include_lightgbm else [])]:
            if family in families:
                variants.append((family, ["spot_id", "lead_time_minutes"]))
    variants = list(dict.fromkeys((family, tuple(fit_group)) for family, fit_group in variants))
    runs = []
    for family, fit_group in variants:
        runs.append(run_calibrator(args, family, list(fit_group)))
    runs = sorted(runs, key=lambda item: item.get("metrics", {}).get("calibrator", {}).get("rmse", float("inf")))
    summary = {
        "generated_at_utc": utc_now(),
        "benchmark_roots": [str(root) for root in args.benchmark_root],
        "predictions_file": args.predictions_file,
        "train_end": args.train_end,
        "eval_start": args.eval_start,
        "target_mode": args.target_mode,
        "residual_baseline": args.residual_baseline,
        "include_lead_stratified": args.include_lead_stratified,
        "include_spot_stratified": args.include_spot_stratified,
        "include_spot_lead_stratified": args.include_spot_lead_stratified,
        "min_group_train_rows": args.min_group_train_rows,
        "training_table_root": str(args.training_table_root) if args.training_table_root else None,
        "training_feature_prefixes": args.training_feature_prefix,
        "max_training_features": args.max_training_features,
        "max_train_rows": args.max_train_rows,
        "calibrator_n_jobs": args.calibrator_n_jobs,
        "continue_on_model_error": args.continue_on_model_error,
        "runs": runs,
    }
    (args.output_root / "sweep_results.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_markdown(args.output_root / "sweep_results.md", summary)
    best_ok = next((run for run in runs if run.get("metrics", {}).get("calibrator", {}).get("rmse") is not None), None)
    print(json.dumps({
        "best_model_family": best_ok["model_family"] if best_ok else None,
        "best_run_name": best_ok.get("run_name") if best_ok else None,
        "best_metric": best_ok.get("metrics", {}).get("calibrator", {}) if best_ok else None,
        "output_root": str(args.output_root),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
