#!/usr/bin/env python3
"""Compute subgroup RMSE targets required to reach a global RMSE threshold."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def group_target(item: dict[str, Any], total_sse: float, target_sse: float) -> dict[str, Any]:
    rows = int(item.get("rows") or item.get("count") or 0)
    current_rmse = float(item.get("rmse") or 0.0)
    if "sse_share_pct" in item:
        group_sse = total_sse * float(item["sse_share_pct"]) / 100.0
    else:
        group_sse = current_rmse * current_rmse * rows
    non_group_sse = max(0.0, total_sse - group_sse)
    allowed_group_sse = target_sse - non_group_sse
    impossible_if_only_group = allowed_group_sse < 0 or rows <= 0
    required_rmse = None if impossible_if_only_group else math.sqrt(max(0.0, allowed_group_sse) / rows)
    required_reduction_pct = None
    if required_rmse is not None and current_rmse > 0:
        required_reduction_pct = (1.0 - required_rmse / current_rmse) * 100.0
    return {
        "rows": rows,
        "current_rmse": round(current_rmse, 6),
        "current_sse_share_pct": round((group_sse / total_sse * 100.0) if total_sse else 0.0, 3),
        "non_group_global_rmse": round(math.sqrt(non_group_sse / int(item.get("_total_rows", 1))) if item.get("_total_rows") else 0.0, 6),
        "global_target_possible_by_only_this_group": not impossible_if_only_group,
        "required_group_rmse_for_global_threshold": None if required_rmse is None else round(float(required_rmse), 6),
        "required_group_rmse_reduction_pct": None if required_reduction_pct is None else round(float(required_reduction_pct), 3),
    }


def summarize_group_rows(
    rows: list[dict[str, Any]],
    total_sse: float,
    target_sse: float,
    total_rows: int,
    limit: int,
) -> list[dict[str, Any]]:
    out = []
    for item in rows[:limit]:
        enriched = dict(item)
        enriched["_total_rows"] = total_rows
        out.append({
            "group": item.get("group"),
            **group_target(enriched, total_sse, target_sse),
        })
    return out


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# RMSE 0.9 Reduction Targets",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Audit source: `{result['audit_path']}`",
        f"Current RMSE: `{result['current_rmse']}`",
        f"Target RMSE: `{result['threshold_rmse']}`",
        f"MSE reduction needed: `{result['mse_reduction_needed_pct']}%`",
        "",
        "## Composite Masks",
        "",
        "| Mask | Rows | SSE share | Current RMSE | Required RMSE | Required reduction | Possible alone |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, item in result["composite_targets"].items():
        lines.append(
            f"| `{name}` | {item['rows']} | {item['current_sse_share_pct']}% | "
            f"{item['current_rmse']} | {item['required_group_rmse_for_global_threshold']} | "
            f"{item['required_group_rmse_reduction_pct']}% | {item['global_target_possible_by_only_this_group']} |"
        )
    for group_name, rows in result["group_targets"].items():
        lines.extend([
            "",
            f"## {group_name}",
            "",
            "| Group | Rows | SSE share | Current RMSE | Required RMSE | Required reduction | Possible alone |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for item in rows:
            lines.append(
                f"| `{item['group']}` | {item['rows']} | {item['current_sse_share_pct']}% | "
                f"{item['current_rmse']} | {item['required_group_rmse_for_global_threshold']} | "
                f"{item['required_group_rmse_reduction_pct']}% | {item['global_target_possible_by_only_this_group']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    total_sse = float(audit["tail"]["current_sse"])
    target_sse = float(audit["tail"]["target_sse_for_threshold"])
    total_rows = int(audit["tail"]["row_count"])
    composites = {}
    for name, item in sorted(audit.get("composite_counterfactuals", {}).items()):
        enriched = dict(item)
        enriched["rmse"] = math.sqrt((total_sse * float(item["sse_share_pct"]) / 100.0) / max(1, int(item["rows"])))
        enriched["_total_rows"] = total_rows
        composites[name] = group_target(enriched, total_sse, target_sse)
    group_targets = {}
    for group_name in args.group:
        rows = audit.get("groups", {}).get(group_name, [])
        if rows:
            group_targets[group_name] = summarize_group_rows(rows, total_sse, target_sse, total_rows, args.limit)
    result = {
        "format": "corsewind.rmse09_reduction_targets.v1",
        "generated_at_utc": utc_now(),
        "audit_path": str(args.audit_json),
        "threshold_rmse": float(audit.get("threshold_rmse", args.threshold_rmse)),
        "current_rmse": audit.get("overall", {}).get("rmse"),
        "row_count": total_rows,
        "current_sse": total_sse,
        "target_sse": target_sse,
        "mse_reduction_needed_pct": audit.get("tail", {}).get("mse_reduction_needed_pct"),
        "composite_targets": composites,
        "group_targets": group_targets,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--threshold-rmse", type=float, default=0.9)
    parser.add_argument("--group", action="append", default=["spot_id", "lead_time_minutes", "actual_wind_bin_ms", "spot_id+lead_time_minutes"])
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, result)
    print(json.dumps({
        "current_rmse": result["current_rmse"],
        "mse_reduction_needed_pct": result["mse_reduction_needed_pct"],
        "composite_targets": result["composite_targets"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
