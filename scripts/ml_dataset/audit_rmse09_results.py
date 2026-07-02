#!/usr/bin/env python3
"""Audit RMSE-0.9 sequence experiment results for leakage-safe completion."""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORBIDDEN_FEATURE_PATTERNS = (
    "labels__",
    "target_feature_sources__",
    "actual_",
    "calibrated_",
    "target_time",
    "timestamp",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def forbidden_feature_matches(columns: list[str]) -> list[str]:
    matches = []
    for column in columns:
        lowered = str(column).lower()
        if any(pattern in lowered for pattern in FORBIDDEN_FEATURE_PATTERNS):
            matches.append(str(column))
    return matches


def rmse_from_errors(errors: list[float]) -> float | None:
    if not errors:
        return None
    return math.sqrt(sum(error * error for error in errors) / len(errors))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_rmse(
    rng: random.Random,
    errors: list[float],
    valid: Any,
    *,
    samples: int,
    unit: str,
) -> tuple[list[float], str, int | None, str | None]:
    if not errors or samples <= 0:
        return [], unit, None, None
    if unit == "row":
        count = len(errors)
        boot = []
        for _ in range(samples):
            picked = [errors[rng.randrange(count)] for _ in range(count)]
            value = rmse_from_errors(picked)
            if value is not None:
                boot.append(value)
        return boot, "row", count, None

    required = "_issue_day" if unit == "issue_day" else "_spot_day"
    if required not in valid.columns:
        boot, _, count, _ = bootstrap_rmse(rng, errors, valid, samples=samples, unit="row")
        return boot, "row", count, f"requested bootstrap unit {unit!r} unavailable; fell back to row"

    groups = []
    for _, group in valid.groupby(required, dropna=True):
        group_errors = (
            group["calibrated_wind_mean_ms"].astype(float).to_numpy()
            - group["actual_wind_mean_ms"].astype(float).to_numpy()
        ).tolist()
        if group_errors:
            groups.append(group_errors)
    if not groups:
        boot, _, count, _ = bootstrap_rmse(rng, errors, valid, samples=samples, unit="row")
        return boot, "row", count, f"requested bootstrap unit {unit!r} had no groups; fell back to row"

    boot = []
    group_count = len(groups)
    for _ in range(samples):
        picked_errors = []
        for _ in range(group_count):
            picked_errors.extend(groups[rng.randrange(group_count)])
        value = rmse_from_errors(picked_errors)
        if value is not None:
            boot.append(value)
    return boot, unit, group_count, None


def prediction_diagnostics(
    path: Path,
    *,
    samples: int,
    confidence: float,
    bootstrap_unit: str,
    seed: int,
) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        return {"available": False, "reason": f"pandas unavailable: {exc}"}
    if not path.exists():
        return {"available": False, "path": str(path), "reason": "missing calibrator_predictions.parquet"}
    frame = pd.read_parquet(path)
    required = {"actual_wind_mean_ms", "calibrated_wind_mean_ms"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return {"available": False, "path": str(path), "reason": f"missing columns: {missing}"}
    optional = [column for column in ("issue_time_utc", "timestamp") if column in frame.columns]
    valid = frame[["actual_wind_mean_ms", "calibrated_wind_mean_ms", "spot_id", "lead_time_minutes", *optional]].dropna(
        subset=["actual_wind_mean_ms", "calibrated_wind_mean_ms"]
    ).copy()
    errors = (
        valid["calibrated_wind_mean_ms"].astype(float).to_numpy()
        - valid["actual_wind_mean_ms"].astype(float).to_numpy()
    ).tolist()
    point = rmse_from_errors(errors)
    rng = random.Random(seed)
    alpha = (1.0 - confidence) / 2.0
    by_spot = {}
    if "spot_id" in valid.columns:
        for spot, group in valid.groupby("spot_id", dropna=False):
            group_errors = (group["calibrated_wind_mean_ms"].astype(float) - group["actual_wind_mean_ms"].astype(float)).tolist()
            by_spot[str(spot)] = {"count": len(group_errors), "rmse": round(float(rmse_from_errors(group_errors) or 0.0), 6)}
    by_lead = {}
    if "lead_time_minutes" in valid.columns:
        for lead, group in valid.groupby("lead_time_minutes", dropna=False):
            group_errors = (group["calibrated_wind_mean_ms"].astype(float) - group["actual_wind_mean_ms"].astype(float)).tolist()
            by_lead[str(int(lead)) if not pd.isna(lead) else "nan"] = {
                "count": len(group_errors),
                "rmse": round(float(rmse_from_errors(group_errors) or 0.0), 6),
            }
    unique_days = None
    unique_months = None
    if "issue_time_utc" in valid.columns:
        issue_times = pd.to_datetime(valid["issue_time_utc"], utc=True, errors="coerce").dropna()
        valid["_issue_day"] = pd.to_datetime(valid["issue_time_utc"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
        if "spot_id" in valid.columns:
            valid["_spot_day"] = valid["spot_id"].astype(str) + "|" + valid["_issue_day"].astype(str)
        unique_days = int(issue_times.dt.date.nunique())
        unique_months = int(issue_times.dt.tz_convert(None).dt.to_period("M").nunique())
    boot, effective_bootstrap_unit, bootstrap_group_count, bootstrap_warning = bootstrap_rmse(
        rng,
        errors,
        valid,
        samples=samples,
        unit=bootstrap_unit,
    )
    return {
        "available": True,
        "path": str(path),
        "count": int(len(errors)),
        "unique_spots": int(valid["spot_id"].nunique(dropna=True)) if "spot_id" in valid.columns else None,
        "unique_leads": int(valid["lead_time_minutes"].nunique(dropna=True)) if "lead_time_minutes" in valid.columns else None,
        "unique_issue_days": unique_days,
        "unique_issue_months": unique_months,
        "rmse": round(float(point), 6) if point is not None else None,
        "bootstrap_samples": samples,
        "bootstrap_unit_requested": bootstrap_unit,
        "bootstrap_unit": effective_bootstrap_unit,
        "bootstrap_group_count": bootstrap_group_count,
        "bootstrap_warning": bootstrap_warning,
        "confidence": confidence,
        "rmse_ci_lower": round(float(percentile(boot, alpha)), 6) if boot else None,
        "rmse_ci_upper": round(float(percentile(boot, 1.0 - alpha)), 6) if boot else None,
        "worst_spots": sorted(by_spot.items(), key=lambda item: item[1]["rmse"], reverse=True)[:5],
        "by_lead": dict(sorted(by_lead.items(), key=lambda item: item[0])),
    }


def audit(
    path: Path,
    threshold: float,
    min_train_rows: int,
    min_test_rows: int,
    *,
    bootstrap_samples: int,
    ci_confidence: float,
    bootstrap_unit: str,
    require_prediction_diagnostics: bool,
    require_ci_upper_below_threshold: bool,
    min_train_spots: int,
    min_train_leads: int,
    min_train_days: int,
    min_test_spots: int,
    min_test_leads: int,
    min_test_days: int,
    random_seed: int,
    selected_run_name: str | None = None,
) -> dict[str, Any]:
    summary = load_json(path)
    reasons = []
    warnings = []
    runs = summary.get("runs", [])
    if not runs:
        reasons.append("No calibrator runs are present.")
    train_end = parse_utc(str(summary.get("train_end", "")))
    eval_start = parse_utc(str(summary.get("eval_start", "")))
    if train_end is None:
        reasons.append("Missing or invalid train_end timestamp.")
    if eval_start is None:
        reasons.append("Missing or invalid eval_start timestamp.")
    if train_end and eval_start and train_end > eval_start:
        reasons.append("Temporal split is invalid: train_end is after eval_start.")
    if summary.get("target_mode") != "residual":
        warnings.append("The sweep did not use residual target mode.")
    if summary.get("residual_baseline") != "raw_wind_mean_ms":
        warnings.append("The residual baseline is not raw_wind_mean_ms.")
    benchmark_roots = [str(item) for item in summary.get("benchmark_roots", [])]
    if len(benchmark_roots) < 2:
        reasons.append("At least two benchmark roots are required for temporal train/eval evidence.")
    if not any("2025" in root for root in benchmark_roots):
        warnings.append("No 2025 benchmark root is visible in the sweep metadata.")
    if not any("2026" in root for root in benchmark_roots):
        warnings.append("No 2026 benchmark root is visible in the sweep metadata.")
    candidate_run_count = sum(
        1
        for run in runs
        if run.get("metrics", {}).get("calibrator", {}).get("rmse") is not None
    )
    if candidate_run_count > 1 and not selected_run_name:
        warnings.append(
            f"Best run is selected from {candidate_run_count} candidate runs on this evaluation split; "
            "treat this as a model-comparison result unless the run family was preselected on a separate validation split."
        )

    best = None
    if selected_run_name:
        for run in runs:
            run_name = str(run.get("run_name") or run.get("model_family"))
            if run_name == selected_run_name:
                best = run
                break
        if best is None:
            reasons.append(f"Selected run name {selected_run_name!r} is not present in sweep results.")
    else:
        for run in runs:
            metric = run.get("metrics", {}).get("calibrator", {})
            rmse = metric.get("rmse")
            if rmse is None:
                continue
            if best is None or float(rmse) < float(best.get("metrics", {}).get("calibrator", {}).get("rmse", float("inf"))):
                best = run
    if best is None:
        reasons.append("No run contains a calibrator RMSE metric.")
    if selected_run_name:
        warnings.append(f"Audit is constrained to preselected run {selected_run_name!r}.")

    best_metric = best.get("metrics", {}).get("calibrator", {}) if best else {}
    best_run_name = str(best.get("run_name") or best.get("model_family")) if best else None
    train_rows = int(best.get("train_rows", 0)) if best else 0
    test_rows = int(best.get("test_rows", 0)) if best else 0
    best_rmse = float(best_metric.get("rmse", float("inf"))) if best_metric.get("rmse") is not None else float("inf")
    effective_rmse = best_rmse
    effective_rmse_source = "sweep_results"
    if train_rows < min_train_rows:
        reasons.append(f"Best run has too few train rows: {train_rows} < {min_train_rows}.")
    if test_rows < min_test_rows:
        reasons.append(f"Best run has too few test rows: {test_rows} < {min_test_rows}.")
    split_coverage = best.get("split_coverage", {}) if best else {}
    train_coverage = split_coverage.get("train", {}) if isinstance(split_coverage, dict) else {}
    if train_coverage:
        train_spots = train_coverage.get("unique_spots")
        train_leads = train_coverage.get("unique_leads")
        train_days = train_coverage.get("unique_issue_days")
        if train_spots is not None and int(train_spots) < min_train_spots:
            reasons.append(f"Training split covers too few spots: {train_spots} < {min_train_spots}.")
        if train_leads is not None and int(train_leads) < min_train_leads:
            reasons.append(f"Training split covers too few lead times: {train_leads} < {min_train_leads}.")
        if train_days is not None and int(train_days) < min_train_days:
            reasons.append(f"Training split covers too few issue days: {train_days} < {min_train_days}.")
    else:
        warnings.append("Best run does not expose split_coverage metadata; training coverage could not be verified.")
    if best and best.get("eval_start") != summary.get("eval_start"):
        warnings.append("Best run eval_start differs from sweep eval_start.")
    if best and best.get("train_end") != summary.get("train_end"):
        warnings.append("Best run train_end differs from sweep train_end.")
    best_features = []
    if best:
        best_features = [*best.get("numeric_features", []), *best.get("categorical_features", [])]
        forbidden_features = forbidden_feature_matches(best_features)
        if forbidden_features:
            reasons.append(f"Best run used leakage-prone feature columns: {forbidden_features}")

    prediction_diag = {"available": False, "reason": "no best run"}
    if best:
        prediction_path = path.parent / str(best_run_name) / "calibrator_predictions.parquet"
        prediction_diag = prediction_diagnostics(
            prediction_path,
            samples=bootstrap_samples,
            confidence=ci_confidence,
            bootstrap_unit=bootstrap_unit,
            seed=random_seed,
        )
        if prediction_diag.get("available"):
            if prediction_diag.get("bootstrap_warning"):
                warnings.append(str(prediction_diag["bootstrap_warning"]))
            recomputed_rmse = prediction_diag.get("rmse")
            if recomputed_rmse is not None:
                effective_rmse = float(recomputed_rmse)
                effective_rmse_source = "calibrator_predictions.parquet"
                if abs(effective_rmse - best_rmse) > 0.0005:
                    warnings.append(f"Recomputed prediction RMSE {recomputed_rmse} differs from sweep RMSE {best_rmse}.")
            if require_ci_upper_below_threshold:
                ci_upper = prediction_diag.get("rmse_ci_upper")
                if ci_upper is None:
                    reasons.append("CI upper bound is unavailable but --require-ci-upper-below-threshold is set.")
                elif float(ci_upper) >= threshold:
                    reasons.append(f"Bootstrap RMSE upper CI is {ci_upper}, not below threshold {threshold}.")
            unique_spots = prediction_diag.get("unique_spots")
            unique_leads = prediction_diag.get("unique_leads")
            unique_days = prediction_diag.get("unique_issue_days")
            if unique_spots is not None and int(unique_spots) < min_test_spots:
                reasons.append(f"Prediction diagnostics cover too few spots: {unique_spots} < {min_test_spots}.")
            if unique_leads is not None and int(unique_leads) < min_test_leads:
                reasons.append(f"Prediction diagnostics cover too few lead times: {unique_leads} < {min_test_leads}.")
            if unique_days is None and min_test_days > 0:
                reasons.append("Prediction diagnostics cannot verify issue-day coverage because issue_time_utc is missing.")
            elif unique_days is not None and int(unique_days) < min_test_days:
                reasons.append(f"Prediction diagnostics cover too few issue days: {unique_days} < {min_test_days}.")
        else:
            warnings.append(f"Prediction diagnostics unavailable: {prediction_diag.get('reason')}")
            if require_prediction_diagnostics:
                reasons.append(f"Prediction diagnostics are required but unavailable: {prediction_diag.get('reason')}")

    if reasons:
        verdict = "inconclusive"
    elif effective_rmse < threshold:
        verdict = "pass"
    else:
        verdict = "fail"
        reasons.append(f"Best effective RMSE is {effective_rmse}, not below threshold {threshold}.")

    return {
        "format": "corsewind.rmse09_audit.v1",
        "generated_at_utc": utc_now(),
        "input_path": str(path),
        "threshold_rmse": threshold,
        "min_train_rows": min_train_rows,
        "min_test_rows": min_test_rows,
        "min_train_spots": min_train_spots,
        "min_train_leads": min_train_leads,
        "min_train_days": min_train_days,
        "min_test_spots": min_test_spots,
        "min_test_leads": min_test_leads,
        "min_test_days": min_test_days,
        "verdict": verdict,
        "candidate_run_count": candidate_run_count,
        "selected_run_name": selected_run_name,
        "best_model_family": best.get("model_family") if best else None,
        "best_run_name": best_run_name,
        "best_fit_group": best.get("fit_group") if best else None,
        "best_metric": best_metric,
        "effective_rmse": round(float(effective_rmse), 6) if math.isfinite(effective_rmse) else None,
        "effective_rmse_source": effective_rmse_source,
        "best_train_rows": train_rows,
        "best_test_rows": test_rows,
        "best_split_coverage": split_coverage,
        "best_feature_count": len(best_features),
        "forbidden_feature_patterns": list(FORBIDDEN_FEATURE_PATTERNS),
        "prediction_diagnostics": prediction_diag,
        "require_prediction_diagnostics": require_prediction_diagnostics,
        "require_ci_upper_below_threshold": require_ci_upper_below_threshold,
        "bootstrap_unit": bootstrap_unit,
        "train_end": summary.get("train_end"),
        "eval_start": summary.get("eval_start"),
        "benchmark_roots": benchmark_roots,
        "reasons": reasons,
        "warnings": warnings,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# RMSE 0.9 Audit",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Verdict: `{result['verdict']}`",
        f"Best model: `{result['best_model_family']}`",
        f"Best run: `{result.get('best_run_name')}`",
        f"Best fit group: `{result.get('best_fit_group')}`",
        f"Best RMSE: `{result['best_metric'].get('rmse')}`",
        f"Effective RMSE: `{result.get('effective_rmse')}`",
        f"Effective RMSE source: `{result.get('effective_rmse_source')}`",
        f"Train rows: `{result['best_train_rows']}`",
        f"Test rows: `{result['best_test_rows']}`",
        f"Train coverage: `{result.get('best_split_coverage', {}).get('train')}`",
        f"Test coverage: `{result.get('best_split_coverage', {}).get('test')}`",
        f"Best feature count: `{result.get('best_feature_count')}`",
        f"Required train coverage: `{result.get('min_train_spots')}` spots, `{result.get('min_train_leads')}` leads, `{result.get('min_train_days')}` days",
        f"Required test coverage: `{result.get('min_test_spots')}` spots, `{result.get('min_test_leads')}` leads, `{result.get('min_test_days')}` days",
        "",
        "## Prediction Diagnostics",
        "",
    ]
    diagnostics = result.get("prediction_diagnostics", {})
    if diagnostics.get("available"):
        lines.extend([
            f"Recomputed RMSE: `{diagnostics.get('rmse')}`",
            f"Bootstrap CI: `{diagnostics.get('rmse_ci_lower')}` -> `{diagnostics.get('rmse_ci_upper')}`",
            f"Bootstrap unit: `{diagnostics.get('bootstrap_unit')}` (`{diagnostics.get('bootstrap_group_count')}` groups)",
            f"Coverage: `{diagnostics.get('unique_spots')}` spots, `{diagnostics.get('unique_leads')}` leads, `{diagnostics.get('unique_issue_days')}` issue days, `{diagnostics.get('unique_issue_months')}` issue months",
            "",
            "### Worst Spots",
            "",
            "| Spot | Count | RMSE |",
            "| --- | ---: | ---: |",
        ])
        for spot, item in diagnostics.get("worst_spots", []):
            lines.append(f"| `{spot}` | {item.get('count')} | {item.get('rmse')} |")
        lines.extend(["", "### By Lead", "", "| Lead | Count | RMSE |", "| --- | ---: | ---: |"])
        for lead, item in diagnostics.get("by_lead", {}).items():
            lines.append(f"| `{lead}` | {item.get('count')} | {item.get('rmse')} |")
    else:
        lines.append(f"- Unavailable: {diagnostics.get('reason')}")
    lines.extend([
        "",
        "## Reasons",
        "",
    ])
    if result["reasons"]:
        lines.extend(f"- {item}" for item in result["reasons"])
    else:
        lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    if result["warnings"]:
        lines.extend(f"- {item}" for item in result["warnings"])
    else:
        lines.append("- None.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_results", type=Path)
    parser.add_argument("--threshold-rmse", type=float, default=0.9)
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--min-test-rows", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--ci-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-unit", choices=("row", "issue_day", "spot_day"), default="issue_day")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--require-prediction-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-ci-upper-below-threshold", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-train-spots", type=int, default=3)
    parser.add_argument("--min-train-leads", type=int, default=4)
    parser.add_argument("--min-train-days", type=int, default=60)
    parser.add_argument("--min-test-spots", type=int, default=3)
    parser.add_argument("--min-test-leads", type=int, default=4)
    parser.add_argument("--min-test-days", type=int, default=20)
    parser.add_argument("--selected-run-name")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--fail-on-non-pass", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(
        args.sweep_results,
        args.threshold_rmse,
        args.min_train_rows,
        args.min_test_rows,
        bootstrap_samples=args.bootstrap_samples,
        ci_confidence=args.ci_confidence,
        bootstrap_unit=args.bootstrap_unit,
        require_prediction_diagnostics=args.require_prediction_diagnostics,
        require_ci_upper_below_threshold=args.require_ci_upper_below_threshold,
        min_train_spots=args.min_train_spots,
        min_train_leads=args.min_train_leads,
        min_train_days=args.min_train_days,
        min_test_spots=args.min_test_spots,
        min_test_leads=args.min_test_leads,
        min_test_days=args.min_test_days,
        random_seed=args.random_seed,
        selected_run_name=args.selected_run_name,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if args.fail_on_non_pass and result["verdict"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
