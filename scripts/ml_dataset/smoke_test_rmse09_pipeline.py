#!/usr/bin/env python3
"""Smoke-test RMSE09 audit, analysis, and decision scripts on synthetic data."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_json(cmd: list[str]) -> dict[str, Any]:
    completed = subprocess.run(cmd, text=True, capture_output=True, check=True)
    return json.loads(completed.stdout)


def run_assert(cmd: list[str], *, should_pass: bool) -> dict[str, Any]:
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if should_pass and completed.returncode != 0:
        raise AssertionError(f"assert command should pass but failed: {completed.stderr or completed.stdout}")
    if not should_pass and completed.returncode == 0:
        raise AssertionError("assert command should fail but passed")
    return json.loads(completed.stdout)


def prediction_rows(case: str) -> list[dict[str, Any]]:
    rows = []
    for day in range(20):
        for spot in ("a", "b", "c"):
            for lead in (15, 30, 45, 60):
                actual = 5.0 + lead / 60.0
                if case in {"achieved", "inconclusive"}:
                    calibrated_error = 0.2
                    raw_error = 1.2
                    chronos_error = 0.7
                    timesfm_error = 0.6
                elif case == "calibration_gap":
                    calibrated_error = 1.0
                    raw_error = 0.2 if spot == "a" else 1.8
                    chronos_error = 0.2 if spot == "b" else 1.8
                    timesfm_error = 0.2 if spot == "c" else 1.8
                elif case == "input_signal_gap":
                    calibrated_error = 1.2
                    raw_error = 1.0
                    chronos_error = 1.1
                    timesfm_error = 1.2
                else:
                    raise ValueError(case)
                rows.append({
                    "actual_wind_mean_ms": actual,
                    "raw_wind_mean_ms": actual + raw_error,
                    "calibrated_wind_mean_ms": actual + calibrated_error,
                    "chronos2_univar_wind_mean_ms_p50": actual + chronos_error,
                    "timesfm_wind_mean_ms_p50": actual + timesfm_error,
                    "spot_id": spot,
                    "lead_time_minutes": lead,
                    "issue_time_utc": f"2026-01-{day + 1:02d}T08:00:00Z",
                })
    return rows


def write_predictions(path: Path, case: str) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas/pyarrow are required for the RMSE09 smoke test.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows(case)).to_parquet(path, index=False)


def sweep_payload(case: str) -> dict[str, Any]:
    split_coverage = {
        "train": {"unique_spots": 3, "unique_leads": 4, "unique_issue_days": 60},
        "test": {"unique_spots": 3, "unique_leads": 4, "unique_issue_days": 20},
    }
    if case == "inconclusive":
        split_coverage["train"] = {"unique_spots": 1, "unique_leads": 1, "unique_issue_days": 1}
    metric_rmse = {
        "achieved": 0.2,
        "inconclusive": 0.2,
        "calibration_gap": 1.0,
        "input_signal_gap": 1.2,
    }[case]
    return {
        "benchmark_roots": [
            "/x/sequence_2025_windsurf_1h_rmse09_v1",
            "/x/sequence_2026_windsurf_1h_rmse09_v1",
        ],
        "train_end": "2026-01-01T00:00:00Z",
        "eval_start": "2026-01-01T00:00:00Z",
        "target_mode": "residual",
        "residual_baseline": "raw_wind_mean_ms",
        "runs": [{
            "model_family": "ridge",
            "train_rows": 500,
            "test_rows": 240,
            "split_coverage": split_coverage,
            "numeric_features": ["raw_wind_mean_ms"],
            "categorical_features": ["spot_id"],
            "train_end": "2026-01-01T00:00:00Z",
            "eval_start": "2026-01-01T00:00:00Z",
            "metrics": {"calibrator": {"rmse": metric_rmse, "mae": metric_rmse, "bias": metric_rmse}},
        }],
    }


def run_case(root: Path, case: str, expected_decision: str) -> dict[str, Any]:
    case_root = root / case
    predictions_path = case_root / "ridge" / "calibrator_predictions.parquet"
    write_json(case_root / "sweep_results.json", sweep_payload(case))
    write_predictions(predictions_path, case)

    audit_json = case_root / "rmse09_audit.json"
    analysis_json = case_root / "rmse09_error_analysis.json"
    decision_json = case_root / "rmse09_decision.json"

    audit = run_json([
        sys.executable,
        str(SCRIPT_DIR / "audit_rmse09_results.py"),
        str(case_root / "sweep_results.json"),
        "--output-json",
        str(audit_json),
        "--bootstrap-samples",
        "20",
        "--require-prediction-diagnostics",
    ])
    analysis = run_json([
        sys.executable,
        str(SCRIPT_DIR / "analyze_rmse09_errors.py"),
        str(case_root / "sweep_results.json"),
        "--audit-json",
        str(audit_json),
        "--output-json",
        str(analysis_json),
    ])
    decision = run_json([
        sys.executable,
        str(SCRIPT_DIR / "summarize_rmse09_decision.py"),
        "--audit-json",
        str(audit_json),
        "--analysis-json",
        str(analysis_json),
        "--output-json",
        str(decision_json),
    ])
    assertion = run_assert([
        sys.executable,
        str(SCRIPT_DIR / "assert_rmse09_goal.py"),
        "--audit-json",
        str(audit_json),
        "--decision-json",
        str(decision_json),
    ], should_pass=expected_decision == "achieved")
    if decision.get("decision") != expected_decision:
        raise AssertionError(f"{case}: expected {expected_decision}, got {decision.get('decision')}")
    return {
        "case": case,
        "assert_status": assertion.get("status"),
        "audit_verdict": audit.get("verdict"),
        "effective_rmse": audit.get("effective_rmse"),
        "overall_rmse": analysis.get("overall", {}).get("rmse"),
        "decision": decision.get("decision"),
    }


def main() -> None:
    cases = {
        "achieved": "achieved",
        "inconclusive": "inconclusive",
        "calibration_gap": "calibration_gap",
        "input_signal_gap": "input_signal_gap",
    }
    with tempfile.TemporaryDirectory(prefix="corsewind-rmse09-smoke-") as tmp:
        root = Path(tmp)
        results = [run_case(root, case, expected) for case, expected in cases.items()]
    print(json.dumps({"status": "ok", "cases": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
