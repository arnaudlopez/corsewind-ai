#!/usr/bin/env python3
"""Inventory AROME WCS coverages for a future AROME -> FastEddy coupling."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from meteo_france_client import coverage_ids, endpoint, load_dotenv, request_api


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS = ROOT / "benchmarks/fasteddy/arome_to_fasteddy_requirements.json"
DEFAULT_OUTPUT = ROOT / "data/processed/benchmarks/fasteddy/arome_fasteddy_inventory.json"
DEFAULT_REPORT = ROOT / "reports/fasteddy_arome_inventory.md"

RUN_RE = re.compile(r"^(?P<variable>.+)___(?P<date>\d{4}-\d{2}-\d{2})T(?P<hour>\d{2})\.00\.00Z$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_coverage_id(coverage_id: str) -> dict[str, Any]:
    match = RUN_RE.match(coverage_id)
    if not match:
        return {"coverage_id": coverage_id, "variable": coverage_id, "run_time_utc": None, "parsed": False}
    run_time = f"{match.group('date')}T{match.group('hour')}:00:00Z"
    return {
        "coverage_id": coverage_id,
        "variable": match.group("variable"),
        "run_time_utc": run_time,
        "parsed": True,
    }


def load_coverages(args: argparse.Namespace) -> list[str]:
    if args.coverage_file:
        return [
            line.strip()
            for line in args.coverage_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    load_dotenv(args.env_file)
    url = endpoint(args.product, args.resolution, "GetCapabilities")
    response = request_api(
        url,
        [("service", "WCS"), ("version", "2.0.1"), ("language", args.language)],
        args.auth_header,
    )
    if args.capabilities_output:
        args.capabilities_output.parent.mkdir(parents=True, exist_ok=True)
        args.capabilities_output.write_text(response.text, encoding="utf-8")
    return coverage_ids(response.text)


def matches_any(text: str, patterns: list[str]) -> bool:
    upper = text.upper()
    for pattern in patterns:
        normalized = pattern.upper()
        if "__" in normalized:
            if upper == normalized:
                return True
            continue
        if normalized in upper:
            return True
    return False


def matches_level(text: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    upper = text.upper()
    for pattern in patterns:
        normalized = pattern.upper()
        if normalized == "SURFACE":
            if "__SURFACE" in upper and "ISOBARIC_SURFACE" not in upper:
                return True
            continue
        if normalized in upper:
            return True
    return False


def classify_variable(variable: str, requirements: list[dict[str, Any]]) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for req in requirements:
        direct = matches_any(variable, req.get("coverage_patterns", []))
        level_ok = matches_level(variable, req.get("level_patterns", []))
        fallback = matches_any(variable, req.get("fallback_patterns", []))
        if direct and level_ok:
            matches.append({"requirement_id": req["id"], "match_type": "direct", "priority": req["priority"]})
        elif fallback:
            matches.append({"requirement_id": req["id"], "match_type": "fallback", "priority": req["priority"]})
    return matches


def build_inventory(coverage_list: list[str], requirements: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    parsed = [parse_coverage_id(item) for item in coverage_list]
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    variables: dict[str, dict[str, Any]] = {}
    for item in parsed:
        variable = item["variable"]
        variables.setdefault(variable, {"variable": variable, "coverage_count": 0, "runs": [], "matches": []})
        variables[variable]["coverage_count"] += 1
        if item["run_time_utc"]:
            variables[variable]["runs"].append(item["run_time_utc"])
            by_run[item["run_time_utc"]].append(item)

    req_items = requirements["requirements"]
    requirement_matches: dict[str, list[dict[str, str]]] = {req["id"]: [] for req in req_items}
    for variable, payload in variables.items():
        matches = classify_variable(variable, req_items)
        payload["matches"] = matches
        payload["runs"] = sorted(set(payload["runs"]))
        for match in matches:
            requirement_matches[match["requirement_id"]].append(
                {"variable": variable, "match_type": match["match_type"]}
            )

    run_summaries = []
    for run_time, items in sorted(by_run.items(), reverse=True):
        available_req_ids = {
            match["requirement_id"]
            for item in items
            for match in classify_variable(item["variable"], req_items)
            if match["match_type"] == "direct"
        }
        fallback_req_ids = {
            match["requirement_id"]
            for item in items
            for match in classify_variable(item["variable"], req_items)
            if match["match_type"] == "fallback"
        }
        required_ids = {req["id"] for req in req_items if req["priority"] == "required"}
        run_summaries.append(
            {
                "run_time_utc": run_time,
                "coverage_count": len(items),
                "direct_requirement_ids": sorted(available_req_ids),
                "fallback_requirement_ids": sorted(fallback_req_ids),
                "missing_required_ids": sorted(required_ids - available_req_ids),
                "production_ready_core": required_ids.issubset(available_req_ids),
            }
        )

    return {
        "format": "corsewind.fasteddy_arome_inventory.v1",
        "generated_at_utc": utc_now(),
        "source": {
            "product": args.product,
            "resolution": args.resolution,
            "coverage_file": display_path(args.coverage_file) if args.coverage_file else None,
        },
        "coverage_count": len(coverage_list),
        "run_count": len(by_run),
        "latest_run_time_utc": max(by_run) if by_run else None,
        "requirements_source": display_path(args.requirements),
        "requirements": [
            {
                "id": req["id"],
                "priority": req["priority"],
                "role": req["role"],
                "match_count": len(requirement_matches[req["id"]]),
                "matches": requirement_matches[req["id"]],
            }
            for req in req_items
        ],
        "runs": run_summaries,
        "variables": sorted(variables.values(), key=lambda item: item["variable"]),
    }


def write_report(inventory: dict[str, Any], path: Path) -> None:
    lines = [
        "# AROME -> FastEddy Inventory",
        "",
        f"- Generated: `{inventory['generated_at_utc']}`",
        f"- Product/resolution: `{inventory['source']['product']} {inventory['source']['resolution']}`",
        f"- Coverage count: `{inventory['coverage_count']}`",
        f"- Run count: `{inventory['run_count']}`",
        f"- Latest run: `{inventory['latest_run_time_utc']}`",
        "",
        "## Requirement Readiness",
        "",
        "| Requirement | Priority | Matches | Role |",
        "|---|---:|---:|---|",
    ]
    for req in inventory["requirements"]:
        lines.append(f"| `{req['id']}` | `{req['priority']}` | `{req['match_count']}` | {req['role']} |")
    lines.extend(["", "## Latest Runs", ""])
    for run in inventory["runs"][:10]:
        lines.append(
            f"- `{run['run_time_utc']}`: coverages `{run['coverage_count']}`, "
            f"production core ready `{run['production_ready_core']}`, "
            f"missing required `{', '.join(run['missing_required_ids']) or 'none'}`"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=["arome", "aromepi"], default="arome")
    parser.add_argument("--resolution", choices=["001", "0025"], default="0025")
    parser.add_argument("--auth-header", choices=["apikey", "bearer"], default="apikey")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--language", choices=["eng", "fre"], default="eng")
    parser.add_argument("--coverage-file", type=Path)
    parser.add_argument("--capabilities-output", type=Path)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.requirements = args.requirements if args.requirements.is_absolute() else ROOT / args.requirements
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    args.report_output = args.report_output if args.report_output.is_absolute() else ROOT / args.report_output
    requirements = json.loads(args.requirements.read_text(encoding="utf-8"))
    coverage_list = load_coverages(args)
    inventory = build_inventory(coverage_list, requirements, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    write_report(inventory, args.report_output)
    print(f"wrote {display_path(args.output)}")
    print(f"wrote {display_path(args.report_output)}")


if __name__ == "__main__":
    main()
