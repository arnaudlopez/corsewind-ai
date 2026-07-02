#!/usr/bin/env python3
"""Exit successfully only when the RMSE09 goal is proven achieved."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def check_goal(audit: dict[str, Any], decision: dict[str, Any], threshold: float) -> tuple[bool, list[str]]:
    reasons = []
    if audit.get("verdict") != "pass":
        reasons.append(f"audit verdict is {audit.get('verdict')!r}, expected 'pass'")
    if decision.get("decision") != "achieved":
        reasons.append(f"decision is {decision.get('decision')!r}, expected 'achieved'")
    effective_rmse = as_float(audit.get("effective_rmse"))
    if effective_rmse is None:
        reasons.append("audit effective_rmse is missing")
    elif effective_rmse >= threshold:
        reasons.append(f"audit effective_rmse {effective_rmse} is not below threshold {threshold}")
    decision_rmse = as_float(decision.get("audit_effective_rmse"))
    if decision_rmse is None:
        reasons.append("decision audit_effective_rmse is missing")
    elif decision_rmse >= threshold:
        reasons.append(f"decision audit_effective_rmse {decision_rmse} is not below threshold {threshold}")
    audit_reasons = audit.get("reasons") or []
    if audit_reasons:
        reasons.append(f"audit contains reasons: {audit_reasons}")
    diagnostics = audit.get("prediction_diagnostics") or {}
    if diagnostics.get("available") is not True:
        reasons.append(f"prediction diagnostics unavailable: {diagnostics.get('reason')}")
    return not reasons, reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--decision-json", type=Path, required=True)
    parser.add_argument("--threshold-rmse", type=float, default=0.9)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ok, reasons = check_goal(load_json(args.audit_json), load_json(args.decision_json), args.threshold_rmse)
    result = {
        "status": "pass" if ok else "fail",
        "threshold_rmse": args.threshold_rmse,
        "audit_json": str(args.audit_json),
        "decision_json": str(args.decision_json),
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
