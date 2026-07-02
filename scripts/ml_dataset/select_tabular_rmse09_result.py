#!/usr/bin/env python3
"""Select the best audited tabular RMSE09 result among multiple runs."""

from __future__ import annotations

import argparse
import json
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


def discover_audits(paths: list[Path], roots: list[Path]) -> list[Path]:
    found = list(paths)
    for root in roots:
        if root.exists():
            found.extend(sorted(root.glob("**/tabular_regime_v1_rmse09_audit.json")))
            found.extend(sorted(root.glob("**/tabular_rmse09_audit.json")))
    return sorted({path.resolve() for path in found if path.exists()})


def compact_audit(path: Path) -> dict[str, Any]:
    audit = load_json(path)
    return {
        "path": str(path),
        "run_id": audit.get("run_id"),
        "verdict": audit.get("verdict"),
        "metric_source": audit.get("metric_source"),
        "corrected_rmse": as_float(audit.get("corrected_rmse")),
        "corrected_mae": as_float(audit.get("corrected_mae")),
        "raw_rmse": as_float(audit.get("raw_rmse")),
        "rmse_gain_pct_vs_raw": as_float(audit.get("rmse_gain_pct_vs_raw")),
        "rmse_gain_pct_vs_previous_best": as_float(audit.get("rmse_gain_pct_vs_previous_best")),
        "rmse_gap_to_threshold": as_float(audit.get("rmse_gap_to_threshold")),
        "metric_count": audit.get("metric_count"),
        "source_parquet_count": audit.get("source_parquet_count"),
        "train_row_count": audit.get("train_row_count"),
        "test_row_count": audit.get("test_row_count"),
        "feature_column_count": audit.get("feature_column_count"),
        "temporal_split_issue_time_utc": audit.get("temporal_split_issue_time_utc"),
        "by_lead": audit.get("by_lead") or {},
        "worst_spots": audit.get("worst_spots") or [],
        "worst_spot_leads": audit.get("worst_spot_leads") or [],
        "reasons": audit.get("reasons") or [],
        "warnings": audit.get("warnings") or [],
    }


def select_best(audits: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    valid = [audit for audit in audits if audit.get("corrected_rmse") is not None and not audit.get("reasons")]
    best = min(valid, key=lambda item: item["corrected_rmse"]) if valid else None
    achieved = [audit for audit in valid if audit["corrected_rmse"] < threshold]
    decision = "achieved" if achieved else "not_achieved"
    if not valid:
        decision = "invalid"
    return {
        "format": "corsewind.tabular_rmse09_selection.v1",
        "generated_at_utc": utc_now(),
        "threshold_rmse": threshold,
        "decision": decision,
        "audit_count": len(audits),
        "valid_audit_count": len(valid),
        "best": best,
        "achieved_runs": sorted(achieved, key=lambda item: item["corrected_rmse"]),
        "runs": sorted(
            audits,
            key=lambda item: (
                item.get("corrected_rmse") is None,
                item.get("corrected_rmse") if item.get("corrected_rmse") is not None else 999.0,
                str(item.get("run_id")),
            ),
        ),
    }


def write_markdown(path: Path, selection: dict[str, Any]) -> None:
    best = selection.get("best") or {}
    lines = [
        "# Tabular RMSE09 Selection",
        "",
        f"Generated: `{selection['generated_at_utc']}`",
        f"Decision: `{selection['decision']}`",
        f"Threshold RMSE: `{selection['threshold_rmse']}`",
        f"Audit count: `{selection['audit_count']}`",
        f"Valid audit count: `{selection['valid_audit_count']}`",
        "",
        "## Best Run",
        "",
    ]
    if best:
        lines.extend([
            f"- Run: `{best.get('run_id')}`",
            f"- Audit: `{best.get('path')}`",
            f"- Corrected RMSE: `{best.get('corrected_rmse')}`",
            f"- Corrected MAE: `{best.get('corrected_mae')}`",
            f"- Raw RMSE: `{best.get('raw_rmse')}`",
            f"- Gain vs raw: `{best.get('rmse_gain_pct_vs_raw')}%`",
            f"- Gain vs previous best: `{best.get('rmse_gain_pct_vs_previous_best')}%`",
            f"- Gap to threshold: `{best.get('rmse_gap_to_threshold')}`",
        ])
    else:
        lines.append("- None.")
    lines.extend(["", "## Runs", "", "| Run | Verdict | RMSE | MAE | Raw RMSE | Gap | Audit |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"])
    for item in selection.get("runs") or []:
        lines.append(
            f"| `{item.get('run_id')}` | `{item.get('verdict')}` | `{item.get('corrected_rmse')}` | "
            f"`{item.get('corrected_mae')}` | `{item.get('raw_rmse')}` | `{item.get('rmse_gap_to_threshold')}` | `{item.get('path')}` |"
        )
    lines.extend(["", "## Best By Lead", "", "| Lead | Count | RMSE | MAE | Raw RMSE |", "| ---: | ---: | ---: | ---: | ---: |"])
    for lead, item in (best.get("by_lead") or {}).items():
        lines.append(f"| `{lead}` | `{item.get('count')}` | `{item.get('rmse')}` | `{item.get('mae')}` | `{item.get('raw_rmse')}` |")
    lines.extend(["", "## Best Worst Spots", "", "| Group | Count | RMSE | MAE | Bias |", "| --- | ---: | ---: | ---: | ---: |"])
    for item in best.get("worst_spots") or []:
        lines.append(f"| `{item.get('key')}` | `{item.get('count')}` | `{item.get('rmse')}` | `{item.get('mae')}` | `{item.get('bias')}` |")
    lines.extend(["", "## Best Worst Spot Leads", "", "| Group | Count | RMSE | MAE | Bias |", "| --- | ---: | ---: | ---: | ---: |"])
    for item in best.get("worst_spot_leads") or []:
        lines.append(f"| `{item.get('key')}` | `{item.get('count')}` | `{item.get('rmse')}` | `{item.get('mae')}` | `{item.get('bias')}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, action="append", default=[])
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument("--threshold-rmse", type=float, default=0.9)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = discover_audits(args.audit_json, args.search_root)
    selection = select_best([compact_audit(path) for path in paths], args.threshold_rmse)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, selection)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
