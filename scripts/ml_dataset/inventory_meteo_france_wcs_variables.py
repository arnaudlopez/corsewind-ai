#!/usr/bin/env python3
"""Inventory Météo-France AROME/AROME-PI WCS variables for ML feature planning."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from meteo_france_client import coverage_ids, endpoint, load_dotenv, request_api  # noqa: E402


DEFAULT_OUTPUT = DEFAULT_ML_ROOT / "source_inventories/meteo_france_wcs_variables.json"

FAMILIES = {
    "wind": ("WIND", "GUST", "U_COMPONENT", "V_COMPONENT", "FF__", "FF_RAF", "U_RAF", "V_RAF"),
    "temperature": ("TEMPERATURE", "TMAX", "TMIN", "TPW_27315", "T__GROUND", "T__HEIGHT", "T__ISOBARIC"),
    "pressure": ("PRESSURE", "MSL", "P__GROUND", "P__SEA", "P__HEIGHT", "P__ISOBARIC"),
    "humidity": ("HUMIDITY", "DEW", "HU__"),
    "precipitation": ("PRECIPITATION", "PRECIP", "RAIN", "SNOW", "WATER_PRECIPITATION", "GRAUPEL", "GRELE", "NEIGE"),
    "cloud": ("CLOUD", "FOG", "CEILING", "NUAGE", "NEB", "NEBUL", "BASE_NUAGE", "PLAFOND"),
    "radiation": ("RADIATION", "SOLAR", "SHORT_WAVE", "LONG_WAVE", "FLSOLAIRE", "FLTHERM"),
    "stability": ("BOUNDARY_LAYER", "PLANETARY_BOUNDARY", "CAPE", "CIN", "LIFTED"),
    "topography": ("GEOMETRIC_HEIGHT", "OROGRAPHY"),
}

RUN_PATTERN = re.compile(r"^(?P<variable>.+)___(?P<run>\d{4}-\d{2}-\d{2}T\d{2}\.\d{2}\.\d{2}Z)(?P<period>_.+)?$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def split_coverage_id(coverage_id: str) -> tuple[str, str | None, str | None]:
    match = RUN_PATTERN.match(coverage_id)
    if not match:
        return coverage_id, None, None
    return match.group("variable"), match.group("run").replace(".", ":"), match.group("period")


def family_for_variable(variable: str) -> list[str]:
    upper = variable.upper()
    return [family for family, needles in FAMILIES.items() if any(needle in upper for needle in needles)]


def summarize_coverages(ids: list[str], sample_limit: int) -> dict[str, Any]:
    variables: dict[str, dict[str, Any]] = {}
    families: dict[str, dict[str, Any]] = defaultdict(lambda: {"coverage_count": 0, "variables": set()})
    runs: set[str] = set()

    for coverage_id in ids:
        variable, run_time, period = split_coverage_id(coverage_id)
        if run_time:
            runs.add(run_time)
        item = variables.setdefault(variable, {
            "coverage_count": 0,
            "run_times": set(),
            "periods": set(),
            "families": family_for_variable(variable),
            "sample_coverages": [],
        })
        item["coverage_count"] += 1
        if run_time:
            item["run_times"].add(run_time)
        if period:
            item["periods"].add(period.lstrip("_"))
        if len(item["sample_coverages"]) < sample_limit:
            item["sample_coverages"].append(coverage_id)
        for family in item["families"]:
            families[family]["coverage_count"] += 1
            families[family]["variables"].add(variable)

    variable_rows = []
    for variable, item in variables.items():
        variable_rows.append({
            "variable": variable,
            "coverage_count": item["coverage_count"],
            "families": sorted(item["families"]),
            "run_count": len(item["run_times"]),
            "first_run_time": min(item["run_times"]) if item["run_times"] else None,
            "last_run_time": max(item["run_times"]) if item["run_times"] else None,
            "periods": sorted(item["periods"]),
            "sample_coverages": item["sample_coverages"],
        })

    family_rows = {
        family: {
            "coverage_count": item["coverage_count"],
            "variable_count": len(item["variables"]),
            "variables": sorted(item["variables"]),
        }
        for family, item in sorted(families.items())
    }

    return {
        "coverage_count": len(ids),
        "variable_count": len(variables),
        "run_count": len(runs),
        "first_run_time": min(runs) if runs else None,
        "last_run_time": max(runs) if runs else None,
        "families": family_rows,
        "variables": sorted(variable_rows, key=lambda row: (-row["coverage_count"], row["variable"])),
    }


def inventory_service(product: str, resolution: str, auth_header: str, sample_limit: int) -> dict[str, Any]:
    url = endpoint(product, resolution, "GetCapabilities")
    response = request_api(url, [("service", "WCS"), ("version", "2.0.1"), ("language", "eng")], auth_header)
    ids = coverage_ids(response.text)
    return {
        "product": product,
        "resolution": resolution,
        "service_url": url,
        **summarize_coverages(ids, sample_limit),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--auth-header", choices=["apikey", "bearer"], default="apikey")
    parser.add_argument("--product", action="append", choices=["arome", "aromepi"], default=[])
    parser.add_argument("--resolution", action="append", choices=["001", "0025"], default=[])
    parser.add_argument("--sample-limit", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    products = args.product or ["arome", "aromepi"]
    resolutions = args.resolution or ["001", "0025"]
    services = []
    for product in products:
        for resolution in resolutions:
            services.append(inventory_service(product, resolution, args.auth_header, args.sample_limit))

    payload = {
        "format": "corsewind.meteo_france_wcs_variable_inventory.v1",
        "generated_at_utc": utc_now(),
        "services": services,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "generated_at_utc": payload["generated_at_utc"],
        "output": str(args.output),
        "service_count": len(services),
        "coverage_count": sum(service["coverage_count"] for service in services),
        "variable_count": sum(service["variable_count"] for service in services),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
