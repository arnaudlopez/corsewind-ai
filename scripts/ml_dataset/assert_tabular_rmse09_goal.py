#!/usr/bin/env python3
"""Exit successfully only when the tabular RMSE09 selection proves the goal."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def check_selection(selection: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    if selection.get("format") != "corsewind.tabular_rmse09_selection.v1":
        reasons.append(f"selection format is {selection.get('format')!r}")
    if selection.get("decision") != "achieved":
        reasons.append(f"selection decision is {selection.get('decision')!r}, expected 'achieved'")
    audit_count = as_int(selection.get("audit_count"))
    if audit_count is None or audit_count < args.min_audit_count:
        reasons.append(f"audit_count {audit_count} is below required {args.min_audit_count}")
    best = selection.get("best") or {}
    if not best:
        reasons.append("selection best run is missing")
    best_rmse = as_float(best.get("corrected_rmse"))
    if best_rmse is None:
        reasons.append("best corrected_rmse is missing")
    elif best_rmse >= args.threshold_rmse:
        reasons.append(f"best corrected_rmse {best_rmse} is not below threshold {args.threshold_rmse}")
    best_audit_path = Path(str(best.get("path") or ""))
    audit: dict[str, Any] = {}
    if not best_audit_path.exists():
        reasons.append(f"best audit path is missing: {best_audit_path}")
    else:
        audit = load_json(best_audit_path)
        if audit.get("verdict") != "achieved":
            reasons.append(f"best audit verdict is {audit.get('verdict')!r}, expected 'achieved'")
        audit_rmse = as_float(audit.get("corrected_rmse"))
        if audit_rmse is None:
            reasons.append("best audit corrected_rmse is missing")
        elif audit_rmse >= args.threshold_rmse:
            reasons.append(f"best audit corrected_rmse {audit_rmse} is not below threshold {args.threshold_rmse}")
        if audit_rmse is not None and best_rmse is not None and abs(audit_rmse - best_rmse) > args.rmse_tolerance:
            reasons.append(f"selection/audit RMSE mismatch: selection={best_rmse}, audit={audit_rmse}")
        if audit.get("reasons"):
            reasons.append(f"best audit has reasons: {audit.get('reasons')}")
        source_parquet_count = as_int(audit.get("source_parquet_count"))
        if source_parquet_count is None or source_parquet_count < args.min_source_parquets:
            reasons.append(f"source_parquet_count {source_parquet_count} is below required {args.min_source_parquets}")
        train_rows = as_int(audit.get("train_row_count"))
        if train_rows is None or train_rows < args.min_train_rows:
            reasons.append(f"train_row_count {train_rows} is below required {args.min_train_rows}")
        test_rows = as_int(audit.get("test_row_count"))
        if test_rows is None or test_rows < args.min_test_rows:
            reasons.append(f"test_row_count {test_rows} is below required {args.min_test_rows}")
        metric_count = as_int(audit.get("metric_count"))
        if metric_count is None or metric_count < args.min_metric_rows:
            reasons.append(f"metric_count {metric_count} is below required {args.min_metric_rows}")
        split = audit.get("temporal_split_issue_time_utc")
        if split != args.require_split_time:
            reasons.append(f"temporal split is {split!r}, expected {args.require_split_time!r}")
    evidence = {
        "selection_decision": selection.get("decision"),
        "selection_audit_count": audit_count,
        "best_run_id": best.get("run_id"),
        "best_rmse": best_rmse,
        "best_audit_path": str(best_audit_path) if best_audit_path else None,
        "best_audit_verdict": audit.get("verdict") if audit else None,
        "best_audit_source_parquet_count": audit.get("source_parquet_count") if audit else None,
        "best_audit_train_row_count": audit.get("train_row_count") if audit else None,
        "best_audit_test_row_count": audit.get("test_row_count") if audit else None,
        "best_audit_metric_count": audit.get("metric_count") if audit else None,
        "best_audit_split": audit.get("temporal_split_issue_time_utc") if audit else None,
    }
    return not reasons, reasons, evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--threshold-rmse", type=float, default=0.9)
    parser.add_argument("--min-audit-count", type=int, default=1)
    parser.add_argument("--min-source-parquets", type=int, default=30)
    parser.add_argument("--min-train-rows", type=int, default=100000)
    parser.add_argument("--min-test-rows", type=int, default=10000)
    parser.add_argument("--min-metric-rows", type=int, default=10000)
    parser.add_argument("--require-split-time", default="2026-01-01T00:00:00Z")
    parser.add_argument("--rmse-tolerance", type=float, default=1e-9)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ok, reasons, evidence = check_selection(load_json(args.selection_json), args)
    result = {
        "format": "corsewind.tabular_rmse09_assertion.v1",
        "generated_at_utc": utc_now(),
        "status": "pass" if ok else "fail",
        "threshold_rmse": args.threshold_rmse,
        "selection_json": str(args.selection_json),
        "evidence": evidence,
        "reasons": reasons,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
