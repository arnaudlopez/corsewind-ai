#!/usr/bin/env python3
"""Prepare an AROME 3D -> FastEddy POC package and download plan."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from meteo_france_client import endpoint, load_dotenv, request_api
from sample_arome_tiff_at_stations import read_float64_tiff


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS = ROOT / "benchmarks/fasteddy/arome_to_fasteddy_requirements.json"
DEFAULT_INVENTORY = ROOT / "data/processed/benchmarks/fasteddy/arome_fasteddy_inventory.json"
DEFAULT_OUTPUT_ROOT = ROOT / "data/processed/benchmarks/fasteddy/arome_poc"
DEFAULT_REPORT = ROOT / "reports/fasteddy_arome_poc_readiness.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def select_run(inventory: dict[str, Any], requested: str | None) -> dict[str, Any] | None:
    runs = inventory.get("runs", [])
    if not runs:
        return None
    if requested:
        for run in runs:
            if run["run_time_utc"] == requested:
                return run
        raise SystemExit(f"Run {requested} not found in inventory.")
    return runs[0]


def select_variable(requirement: dict[str, Any], inventory_req: dict[str, Any]) -> dict[str, Any] | None:
    matches = inventory_req.get("matches", [])
    direct = [item for item in matches if item["match_type"] == "direct"]
    fallback = [item for item in matches if item["match_type"] == "fallback"]
    selected = (direct or fallback)
    if not selected:
        return None
    return {
        "variable": selected[0]["variable"],
        "match_type": selected[0]["match_type"],
        "priority": requirement["priority"],
    }


def request_subsets(bbox: list[float], valid_time: datetime, requirement_id: str, selected: dict[str, Any]) -> list[str]:
    min_lon, min_lat, max_lon, max_lat = bbox
    subsets = [
        f"long({min_lon},{max_lon})",
        f"lat({min_lat},{max_lat})",
        f"time({valid_time.isoformat().replace('+00:00', 'Z')})",
    ]
    variable = selected["variable"].upper()
    if "SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND" in variable:
        subsets.insert(2, "height(10)")
    elif any(marker in variable for marker in ("ISOBARIC", "PRESSURE")):
        subsets.insert(2, "pressure(100000,85000)")
    elif any(marker in variable for marker in ("MODEL_LEVEL", "HYBRID")):
        subsets.insert(2, "level(1,20)")
    elif requirement_id in {"surface_temperature", "surface_pressure"}:
        pass
    return subsets


def build_plan(
    inventory: dict[str, Any],
    requirements: dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    run = select_run(inventory, args.run_time_utc)
    if run is None:
        raise SystemExit("Inventory has no parsed AROME run.")
    run_time = parse_time(run["run_time_utc"])
    req_by_id = {item["id"]: item for item in requirements["requirements"]}
    inv_req_by_id = {item["id"]: item for item in inventory["requirements"]}
    steps = []
    for lead_hour in args.lead_hours:
        valid_time = run_time + timedelta(hours=int(lead_hour))
        step_dir = output_root / f"h{int(lead_hour):02d}"
        downloads = []
        for req_id, req in req_by_id.items():
            selected = select_variable(req, inv_req_by_id.get(req_id, {"matches": []}))
            if selected is None:
                downloads.append(
                    {
                        "requirement_id": req_id,
                        "priority": req["priority"],
                        "status": "missing_variable",
                        "role": req["role"],
                    }
                )
                continue
            coverage_id = f"{selected['variable']}___{run_time.strftime('%Y-%m-%dT%H.00.00Z')}"
            output_format = "image/tiff" if selected["match_type"] == "fallback" else "application/wmo-grib"
            suffix = "tiff" if output_format == "image/tiff" else "grib2"
            output = step_dir / f"{req_id}__{safe_name(selected['variable'])}.{suffix}"
            downloads.append(
                {
                    "requirement_id": req_id,
                    "priority": req["priority"],
                    "role": req["role"],
                    "status": "planned",
                    "selected_variable": selected["variable"],
                    "match_type": selected["match_type"],
                    "coverage_id": coverage_id,
                    "format": output_format,
                    "subsets": request_subsets(list(args.bbox), valid_time, req_id, selected),
                    "output": display_path(output),
                }
            )
        steps.append(
            {
                "lead_hour": int(lead_hour),
                "valid_time_utc": valid_time.isoformat().replace("+00:00", "Z"),
                "step_dir": display_path(step_dir),
                "downloads": downloads,
            }
        )
    return {
        "format": "corsewind.fasteddy_arome_poc_plan.v1",
        "generated_at_utc": utc_now(),
        "objective": "Prepare AROME parent data for a production-grade FastEddy IC/BC coupling POC.",
        "source": {
            "product": args.product,
            "resolution": args.resolution,
            "run_time_utc": run["run_time_utc"],
            "inventory": display_path(args.inventory),
            "requirements": display_path(args.requirements),
        },
        "target": {
            "bbox_wgs84": list(args.bbox),
            "zone_label": args.zone_label,
            "lead_hours": list(args.lead_hours),
        },
        "readiness": readiness_summary(steps),
        "steps": steps,
    }


def readiness_summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    required_missing: set[str] = set()
    required_fallback: set[str] = set()
    planned_required: set[str] = set()
    for step in steps:
        for item in step["downloads"]:
            if item["priority"] != "required":
                continue
            if item["status"] == "missing_variable":
                required_missing.add(item["requirement_id"])
            elif item.get("match_type") == "fallback":
                required_fallback.add(item["requirement_id"])
            elif item["status"] == "planned":
                planned_required.add(item["requirement_id"])
    return {
        "production_fasteddy_ready": not required_missing and not required_fallback,
        "poc_download_ready": not required_missing,
        "required_planned_direct": sorted(planned_required),
        "required_only_fallback": sorted(required_fallback),
        "required_missing": sorted(required_missing),
    }


def download_plan(plan: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    load_dotenv(args.env_file)
    url = endpoint(args.product, args.resolution, "GetCoverage")
    results = []
    for step in plan["steps"]:
        for item in step["downloads"]:
            if item["status"] != "planned":
                continue
            output = ROOT / item["output"]
            if output.exists() and not args.force:
                results.append({"requirement_id": item["requirement_id"], "status": "exists", "output": item["output"]})
                continue
            params = [
                ("service", "WCS"),
                ("version", "2.0.1"),
                ("coverageid", item["coverage_id"]),
                ("format", item["format"]),
            ]
            for subset in item["subsets"]:
                params.append(("subset", subset))
            try:
                response = request_api(url, params, args.auth_header)
            except SystemExit as exc:
                results.append(
                    {
                        "requirement_id": item["requirement_id"],
                        "coverage_id": item["coverage_id"],
                        "status": "failed",
                        "message": str(exc),
                    }
                )
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(response.content)
            results.append(
                {
                    "requirement_id": item["requirement_id"],
                    "coverage_id": item["coverage_id"],
                    "status": "downloaded",
                    "bytes": len(response.content),
                    "output": item["output"],
                }
            )
    return results


def write_parent_schema(plan: dict[str, Any], output_root: Path) -> None:
    schema = {
        "format": "corsewind.fasteddy_parent_schema.v1",
        "purpose": "Intermediate AROME parent package expected by the future FastEddy IC/BC converter.",
        "dimensions": {
            "time": "selected lead times",
            "z": "height above terrain or pressure/model levels converted to meters",
            "y": "south/north grid",
            "x": "west/east grid",
        },
        "variables": {
            "lat": "2D latitude",
            "lon": "2D longitude",
            "topography": "2D terrain elevation from Copernicus GLO-30",
            "landmask": "2D land/sea mask",
            "z0m": "2D momentum roughness length",
            "u": "4D eastward wind",
            "v": "4D northward wind",
            "w": "4D vertical velocity, or zero-filled with explicit flag for POC",
            "theta": "4D potential temperature",
            "qv": "4D water vapor mixing ratio/specific humidity",
            "pressure": "4D pressure",
        },
        "plan_readiness": plan["readiness"],
        "next_converter_outputs": [
            "GeoSpec-compatible GIS netCDF",
            "SimGrid config",
            "GenICBCs-compatible parent files or direct FE_interp/FE_Bndys files",
        ],
    }
    path = output_root / "fasteddy_parent_schema.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def write_report(plan: dict[str, Any], download_results: list[dict[str, Any]], path: Path) -> None:
    readiness = plan["readiness"]
    lines = [
        "# FastEddy AROME Coupling POC Readiness",
        "",
        f"- Generated: `{plan['generated_at_utc']}`",
        f"- Zone: `{plan['target']['zone_label']}`",
        f"- Run: `{plan['source']['run_time_utc']}`",
        f"- BBox: `{plan['target']['bbox_wgs84']}`",
        f"- Production FastEddy ready: `{readiness['production_fasteddy_ready']}`",
        f"- POC download ready: `{readiness['poc_download_ready']}`",
        "",
        "## Missing / Fallback Required Fields",
        "",
        f"- Missing required: `{', '.join(readiness['required_missing']) or 'none'}`",
        f"- Required only fallback: `{', '.join(readiness['required_only_fallback']) or 'none'}`",
        "",
        "## Download Plan",
        "",
    ]
    for step in plan["steps"]:
        lines.append(f"### H+{step['lead_hour']} `{step['valid_time_utc']}`")
        for item in step["downloads"]:
            lines.append(
                f"- `{item['requirement_id']}` {item['status']} "
                f"{item.get('match_type', '')} `{item.get('selected_variable', '')}`"
            )
        lines.append("")
    if download_results:
        lines.extend(["## Download Results", ""])
        for item in download_results:
            lines.append(f"- `{item['requirement_id']}`: `{item['status']}` {item.get('message', item.get('output', ''))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=["arome", "aromepi"], default="arome")
    parser.add_argument("--resolution", choices=["001", "0025"], default="001")
    parser.add_argument("--auth-header", choices=["apikey", "bearer"], default="apikey")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--zone-label", default="Ajaccio FastEddy AROME POC")
    parser.add_argument("--bbox", nargs=4, type=float, default=[8.62, 41.82, 8.9, 42.0])
    parser.add_argument("--lead-hours", nargs="+", type=int, default=[0])
    parser.add_argument("--run-time-utc")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.inventory = args.inventory if args.inventory.is_absolute() else ROOT / args.inventory
    args.requirements = args.requirements if args.requirements.is_absolute() else ROOT / args.requirements
    args.output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    args.report_output = args.report_output if args.report_output.is_absolute() else ROOT / args.report_output
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    requirements = json.loads(args.requirements.read_text(encoding="utf-8"))
    plan = build_plan(inventory, requirements, args, args.output_root)
    plan_path = args.output_root / "arome_fasteddy_poc_download_plan.json"
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    write_parent_schema(plan, args.output_root)
    download_results = download_plan(plan, args) if args.download else []
    if download_results:
        (args.output_root / "download_status.json").write_text(json.dumps(download_results, indent=2), encoding="utf-8")
    write_report(plan, download_results, args.report_output)
    print(f"wrote {display_path(plan_path)}")
    print(f"wrote {display_path(args.output_root / 'fasteddy_parent_schema.json')}")
    print(f"wrote {display_path(args.report_output)}")
    print(f"production_fasteddy_ready={plan['readiness']['production_fasteddy_ready']}")


if __name__ == "__main__":
    main()
