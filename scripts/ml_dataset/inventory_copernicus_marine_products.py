#!/usr/bin/env python3
"""Inventory Copernicus Marine datasets relevant to CorseWind ML features."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_TMP_PYTHONPATH = ROOT / "tmp/copernicusmarine_test_pkgs"
DEFAULT_OUTPUT_JSON = DEFAULT_ML_ROOT / "source_inventories/copernicus_marine_products.json"
DEFAULT_OUTPUT_MD = ROOT / "docs/ml_nowcasting/copernicus_marine_product_inventory.md"

CANDIDATES = [
    {
        "priority": "P1",
        "decision": "integrate_now",
        "feature_family": "sea_surface_temperature",
        "product_id": "SST_MED_PHY_SUBSKIN_L4_NRT_010_036",
        "dataset_id": "cmems_obs-sst_med_phy-sst_nrt_diurnal-oi-0.0625deg_PT1H-m",
        "target_variables": ["analysed_sst"],
        "why": "Hourly Mediterranean subskin SST, useful for land-sea thermal contrast.",
    },
    {
        "priority": "P2",
        "decision": "test_after_sst",
        "feature_family": "ocean_current",
        "product_id": "MEDSEA_ANALYSISFORECAST_PHY_006_013",
        "dataset_id": "cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m",
        "target_variables": ["uo", "vo"],
        "why": "Hourly 2D surface current forecast. Indirect for wind, useful for marine context and validation.",
    },
    {
        "priority": "P2",
        "decision": "test_after_sst",
        "feature_family": "mixed_layer",
        "product_id": "MEDSEA_ANALYSISFORECAST_PHY_006_013",
        "dataset_id": "cmems_mod_med_phy-mld_anfc_4.2km-2D_PT1H-m",
        "target_variables": ["mlotst"],
        "why": "Mixed layer depth may help characterize coastal water inertia, but impact on 15 min wind is uncertain.",
    },
    {
        "priority": "P2",
        "decision": "test_after_sst",
        "feature_family": "waves",
        "product_id": "MEDSEA_ANALYSISFORECAST_WAV_006_017",
        "dataset_id": "cmems_mod_med_wav_anfc_4.2km_PT1H-i",
        "target_variables": ["VHM0", "VMDR", "VTPK"],
        "why": "Hourly wave forecast can help explain sea state and observation quality, not the primary thermal driver.",
    },
    {
        "priority": "P3",
        "decision": "backtest_only",
        "feature_family": "satellite_sea_wind",
        "product_id": "WIND_GLO_PHY_L4_NRT_012_004",
        "dataset_id": "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
        "target_variables": ["eastward_wind", "northward_wind"],
        "why": "Hourly gridded sea-surface wind, too coarse for spot correction but useful as an independent large-scale check.",
    },
    {
        "priority": "P3",
        "decision": "backtest_only",
        "feature_family": "sar_sea_wind",
        "product_id": "WIND_MED_PHY_HR_L3_NRT_012_104",
        "dataset_id": "cmems_obs-wind_med_phy_nrt_l3-s1a-sar-asc-0.01deg_P1D-i",
        "target_variables": ["wind_speed", "wind_to_dir", "eastward_wind", "northward_wind"],
        "why": "High-resolution SAR sea wind is episodic, so useful for audits/backtests more than operational nowcast.",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def find_copernicusmarine_bin(explicit_path: Path | None) -> Path:
    if explicit_path:
        candidate = resolve_path(explicit_path)
        if candidate.exists():
            return candidate
        raise SystemExit(f"copernicusmarine binary not found: {candidate}")
    from_path = shutil.which("copernicusmarine")
    if from_path:
        return Path(from_path)
    local_bin = DEFAULT_TMP_PYTHONPATH / "bin/copernicusmarine"
    if local_bin.exists():
        return local_bin
    raise SystemExit(
        "copernicusmarine CLI not found. Install copernicusmarine or keep "
        "tmp/copernicusmarine_test_pkgs/bin/copernicusmarine available."
    )


def copernicus_env(binary: Path) -> dict[str, str]:
    env = os.environ.copy()
    if str(binary).startswith(str(DEFAULT_TMP_PYTHONPATH)) and DEFAULT_TMP_PYTHONPATH.exists():
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(DEFAULT_TMP_PYTHONPATH)
            if not existing
            else f"{DEFAULT_TMP_PYTHONPATH}{os.pathsep}{existing}"
        )
    return env


def describe_dataset(binary: Path, dataset_id: str, cache_dir: Path, refresh: bool) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{dataset_id.replace('/', '_').replace('-', '_').replace('.', '_')}.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    cmd = [
        str(binary),
        "describe",
        "--dataset-id",
        dataset_id,
        "--return-fields",
        "all",
        "--disable-progress-bar",
        "--log-level",
        "QUIET",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=copernicus_env(binary))
    cache_path.write_text(result.stdout, encoding="utf-8")
    return json.loads(result.stdout)


def service_variables(part: dict[str, Any]) -> list[dict[str, Any]]:
    variables_by_name: dict[str, dict[str, Any]] = {}
    for service in part.get("services", []) or []:
        for variable in service.get("variables", []) or []:
            short_name = variable.get("short_name")
            if not short_name:
                continue
            item = variables_by_name.setdefault(short_name, {
                "short_name": short_name,
                "standard_name": variable.get("standard_name"),
                "units": variable.get("units"),
                "bbox": variable.get("bbox"),
                "coordinates": variable.get("coordinates", []),
                "services": [],
            })
            if not item.get("coordinates") and variable.get("coordinates"):
                item["coordinates"] = variable.get("coordinates", [])
            item["services"].append(service.get("service_short_name") or service.get("service_name"))
    return sorted(variables_by_name.values(), key=lambda item: item["short_name"])


def summarize_candidate(candidate: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    products = metadata.get("products", []) or []
    product = products[0] if products else {}
    dataset = (product.get("datasets", []) or [{}])[0]
    version = (dataset.get("versions", []) or [{}])[0]
    part = (version.get("parts", []) or [{}])[0]
    services = part.get("services", []) or []
    variables = service_variables(part)
    available_names = {item["short_name"] for item in variables}
    return {
        **candidate,
        "dataset_name": dataset.get("dataset_name"),
        "version": version.get("label"),
        "part": part.get("name"),
        "released_date": part.get("released_date"),
        "arco_updated_date": part.get("arco_updated_date"),
        "service_short_names": sorted({service.get("service_short_name") for service in services if service.get("service_short_name")}),
        "variables": variables,
        "target_variables_available": {
            variable: variable in available_names
            for variable in candidate.get("target_variables", [])
        },
    }


def coordinate_summary(variable: dict[str, Any]) -> str:
    parts = []
    for coord in variable.get("coordinates", []) or []:
        coord_id = coord.get("coordinate_id")
        if coord_id in {"time", "latitude", "longitude", "depth"}:
            step = coord.get("step")
            parts.append(f"{coord_id}: step={step}")
    return ", ".join(parts)


def write_markdown(path: Path, inventory: dict[str, Any]) -> None:
    lines = [
        "# Copernicus Marine Product Inventory",
        "",
        f"Generated at: `{inventory['generated_at_utc']}`",
        "",
        "## Decision Summary",
        "",
        "| Priority | Decision | Feature | Dataset | Target variables | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in inventory["candidates"]:
        targets = ", ".join(
            f"`{name}`{' OK' if ok else ' missing'}"
            for name, ok in item["target_variables_available"].items()
        )
        lines.append(
            f"| `{item['priority']}` | `{item['decision']}` | `{item['feature_family']}` | "
            f"`{item['dataset_id']}` | {targets} | {item['why']} |"
        )
    lines.extend(["", "## Dataset Details", ""])
    for item in inventory["candidates"]:
        lines.extend([
            f"### `{item['dataset_id']}`",
            "",
            f"- product: `{item['product_id']}`",
            f"- version: `{item.get('version')}`",
            f"- services: {', '.join(f'`{value}`' for value in item.get('service_short_names', []))}",
            f"- released: `{item.get('released_date')}`",
            f"- arco updated: `{item.get('arco_updated_date')}`",
            "",
            "| Variable | Standard name | Units | BBox | Coordinates |",
            "| --- | --- | --- | --- | --- |",
        ])
        for variable in item.get("variables", []):
            lines.append(
                f"| `{variable['short_name']}` | `{variable.get('standard_name')}` | "
                f"`{variable.get('units')}` | `{variable.get('bbox')}` | {coordinate_summary(variable)} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copernicusmarine-bin", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "tmp/copernicusmarine_describe_cache")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    binary = find_copernicusmarine_bin(args.copernicusmarine_bin)
    cache_dir = resolve_path(args.cache_dir)
    candidates = []
    for candidate in CANDIDATES:
        metadata = describe_dataset(binary, candidate["dataset_id"], cache_dir, args.refresh)
        candidates.append(summarize_candidate(candidate, metadata))
    inventory = {
        "format": "corsewind.copernicus_marine_product_inventory.v1",
        "generated_at_utc": utc_now(),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    output_json = resolve_path(args.output_json)
    output_md = resolve_path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(output_md, inventory)
    print(json.dumps({
        "generated_at_utc": inventory["generated_at_utc"],
        "candidate_count": len(candidates),
        "output_json": str(output_json),
        "output_md": str(output_md),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
