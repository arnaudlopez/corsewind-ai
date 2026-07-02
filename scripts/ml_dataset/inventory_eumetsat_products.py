#!/usr/bin/env python3
"""Inventory EUMETSAT datasets relevant to CorseWind ML features."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_OUTPUT_JSON = DEFAULT_ML_ROOT / "source_inventories/eumetsat_products.json"
DEFAULT_OUTPUT_MD = ROOT / "docs/ml_nowcasting/eumetsat_product_inventory.md"
DEFAULT_CATALOG_KEYWORD_JSON = DEFAULT_ML_ROOT / "source_inventories/eumetsat_catalog_keyword_inventory.json"
EUMETSAT_COLLECTION_BASE = "https://api.eumetsat.int/data/browse/1.0.0/collections"

CANDIDATES = [
    {
        "priority": "P1",
        "decision": "integrate_after_access",
        "feature_family": "cloud_mask",
        "collection_id": "EO:EUM:DAT:0678",
        "name": "Cloud Mask (netCDF) - MTG - 0 degree",
        "target_features": ["cloud_fraction_satellite", "clear_sky_fraction", "dust_or_ash_flag"],
        "why": "Best first satellite feature for thermal days: tells whether the ground can actually heat.",
    },
    {
        "priority": "P1",
        "decision": "integrated_spot_sampler",
        "feature_family": "cloud_type",
        "collection_id": "EO:EUM:DAT:0680",
        "name": "Cloud Type - MTG - 0 degree",
        "target_features": ["cloud_type_dominant", "low_cloud_fraction", "high_cloud_fraction"],
        "why": "Distinguishes low marine cloud, high cloud, and convective/cloud regimes.",
    },
    {
        "priority": "P2",
        "decision": "test_after_cloud_mask",
        "feature_family": "cloud_top",
        "collection_id": "EO:EUM:DAT:0681",
        "name": "Cloud Top Temperature and Height - MTG - 0 degree",
        "target_features": ["cloud_top_height_m", "cloud_top_temperature_c"],
        "why": "Useful for convection and cloud vertical development, less direct for pure thermal sea breeze.",
    },
    {
        "priority": "P2",
        "decision": "test_after_cloud_mask",
        "feature_family": "optimal_cloud_analysis",
        "collection_id": "EO:EUM:DAT:0684",
        "name": "Optimal Cloud Analysis - MTG - 0 degree",
        "target_features": ["cloud_phase", "cloud_optical_thickness", "cloud_effective_radius"],
        "why": "Richer cloud microphysics; probably too much for V1 but valuable for ablation tests.",
    },
    {
        "priority": "P2",
        "decision": "integrated_spot_sampler",
        "feature_family": "land_surface_temperature",
        "collection_id": "EO:EUM:DAT:1088",
        "name": "Land Surface Temperature - MTG",
        "target_features": ["land_surface_temperature_c", "land_minus_sea_surface_temperature_c"],
        "why": "Potentially excellent proxy for actual ground heating, complementing air temperature stations.",
    },
    {
        "priority": "P2",
        "decision": "test_for_convection",
        "feature_family": "precipitation_rate",
        "collection_id": "EO:EUM:DAT:1086",
        "name": "Precipitation rate at ground by blended FCI IR / LEO MW precipitation - MTG - 0 Degree",
        "target_features": ["satellite_precip_rate_mm_h", "convective_precip_flag"],
        "why": "Helps exclude disturbed/convection days where thermal wind behaves differently.",
    },
    {
        "priority": "P2",
        "decision": "integrated_spot_sampler",
        "feature_family": "global_instability_indices",
        "collection_id": "EO:EUM:DAT:0683",
        "name": "Global Instability Indices - MTG - 0 degree",
        "target_features": ["satellite_instability_index", "convective_potential_flag"],
        "why": "Potentially useful to separate clean thermal days from unstable convective regimes.",
    },
    {
        "priority": "P2",
        "decision": "test_for_convection",
        "feature_family": "lightning",
        "collection_id": "EO:EUM:DAT:0691",
        "name": "LI Lightning Flashes - MTG - 0 degree",
        "target_features": ["lightning_flash_count_nearby", "lightning_detected_radius"],
        "why": "Useful disturbed-day flag; not a normal thermal driver but important for exclusion and gust risk.",
    },
    {
        "priority": "P3",
        "decision": "context_or_backtest",
        "feature_family": "atmospheric_motion_vectors",
        "collection_id": "EO:EUM:DAT:0676",
        "name": "Atmospheric Motion Vectors (netCDF) - MTG - 0 degree",
        "target_features": ["upper_cloud_motion_wind", "midlevel_flow_context"],
        "why": "Large-scale/aloft wind context; probably weaker than NWP wind fields for V1.",
    },
    {
        "priority": "P3",
        "decision": "duplicate_check",
        "feature_family": "mtg_sea_surface_temperature",
        "collection_id": "EO:EUM:DAT:0694",
        "name": "FCI Level 3 Sea Surface Temperature - MTG",
        "target_features": ["mtg_sst_c"],
        "why": "Could cross-check Copernicus SST, but Copernicus remains the cleaner SST source for V1.",
    },
    {
        "priority": "P3",
        "decision": "historical_or_backtest",
        "feature_family": "surface_radiation",
        "collection_id": "EO:EUM:DAT:0863",
        "name": "Surface Radiation Data Set - Heliosat (SARAH) - Edition 3",
        "target_features": ["surface_solar_radiation", "daily_solar_energy"],
        "why": "Very useful for historical solar context, but less likely to be a low-latency nowcast feed.",
    },
    {
        "priority": "P3",
        "decision": "fallback_legacy",
        "feature_family": "msg_cloud_mask",
        "collection_id": "EO:EUM:DAT:MSG:CLM",
        "name": "Cloud Mask - MSG - 0 degree",
        "target_features": ["cloud_fraction_satellite_legacy"],
        "why": "MSG/SEVIRI fallback with long archive and 15 min cadence if MTG is awkward to process.",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def fetch_collection(collection_id: str, timeout: int) -> dict[str, Any]:
    encoded = quote(collection_id, safe="")
    url = f"{EUMETSAT_COLLECTION_BASE}/{encoded}?format=json"
    response = requests.get(url, headers={"Accept": "application/json"}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def summarize_collection(candidate: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    collection = payload.get("collection", {})
    properties = collection.get("properties", {})
    geometry = collection.get("geometry", {})
    return {
        **candidate,
        "title": properties.get("title"),
        "abstract": properties.get("abstract"),
        "date": properties.get("date"),
        "updated": properties.get("updated"),
        "rights": properties.get("rights", []),
        "geometry": geometry,
        "quicklooks": [
            link.get("href")
            for link in properties.get("links", [])
            if str(link.get("title", "")).lower() == "quicklook"
        ],
        "catalogue_url": f"https://data.eumetsat.int/product/{quote(candidate['collection_id'], safe='')}",
    }


def cached_catalog_records(path: Path = DEFAULT_CATALOG_KEYWORD_JSON) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: dict[str, dict[str, Any]] = {}
    for item in payload.get("top_matches", []):
        collection_id = item.get("collection_id")
        if collection_id:
            records[collection_id] = item
    for items in payload.get("groups", {}).values():
        for item in items:
            collection_id = item.get("collection_id")
            if collection_id:
                records.setdefault(collection_id, item)
    return records


def summarize_cached_collection(candidate: dict[str, Any], record: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        **candidate,
        "title": record.get("title") or candidate.get("name"),
        "abstract": record.get("abstract"),
        "date": record.get("date"),
        "updated": None,
        "rights": record.get("rights") or [],
        "geometry": None,
        "quicklooks": [],
        "catalogue_url": f"https://data.eumetsat.int/product/{quote(candidate['collection_id'], safe='')}",
        "metadata_source": "catalog_keyword_cache",
        "live_metadata_error": error,
    }


def short_abstract(text: str | None, limit: int = 220) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "..."


def write_markdown(path: Path, inventory: dict[str, Any]) -> None:
    lines = [
        "# EUMETSAT Product Inventory",
        "",
        f"Generated at: `{inventory['generated_at_utc']}`",
        "",
        "## Decision Summary",
        "",
        "| Priority | Decision | Feature | Collection | Target features | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in inventory["candidates"]:
        lines.append(
            f"| `{item['priority']}` | `{item['decision']}` | `{item['feature_family']}` | "
            f"`{item['collection_id']}` | {', '.join(f'`{value}`' for value in item['target_features'])} | "
            f"{item['why']} |"
        )
    lines.extend([
        "",
        "## Access Model",
        "",
        "- Public catalogue metadata is available through `https://api.eumetsat.int/data/browse/1.0.0/collections/<collection>?format=json`.",
        "- Product download requires an EUMETSAT account plus `EUMETSAT_CONSUMER_KEY` and `EUMETSAT_CONSUMER_SECRET`.",
        "- Operational download/prototyping should use `eumdac`; spatial tailoring will likely need Data Tailor for MTG products.",
        "",
        "## Dataset Details",
        "",
    ])
    for item in inventory["candidates"]:
        catalogue_url = item.get("catalogue_url") or f"https://data.eumetsat.int/product/{quote(item['collection_id'], safe='')}"
        lines.extend([
            f"### `{item['collection_id']}`",
            "",
            f"- title: `{item.get('title') or item.get('name')}`",
            f"- date: `{item.get('date')}`",
            f"- rights: {', '.join(f'`{value}`' for value in item.get('rights', []))}",
            f"- catalogue: {catalogue_url}",
            "",
            short_abstract(item.get("abstract")),
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--timeout-sec", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cached_records = cached_catalog_records()
    candidates = []
    errors = []
    for candidate in CANDIDATES:
        try:
            payload = fetch_collection(candidate["collection_id"], args.timeout_sec)
        except requests.RequestException as exc:
            error = str(exc)
            errors.append({"collection_id": candidate["collection_id"], "error": error})
            cached = cached_records.get(candidate["collection_id"])
            if cached:
                candidates.append(summarize_cached_collection(candidate, cached, error))
            else:
                candidates.append({
                    **candidate,
                    "title": candidate.get("name"),
                    "catalogue_url": f"https://data.eumetsat.int/product/{quote(candidate['collection_id'], safe='')}",
                    "error": error,
                })
        else:
            candidates.append(summarize_collection(candidate, payload))
    inventory = {
        "format": "corsewind.eumetsat_product_inventory.v1",
        "generated_at_utc": utc_now(),
        "candidate_count": len(candidates),
        "error_count": len(errors),
        "errors": errors,
        "candidates": candidates,
    }
    output_json = resolve_path(args.output_json)
    output_md = resolve_path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(output_md, inventory)
    print(json.dumps({
        "generated_at_utc": inventory["generated_at_utc"],
        "candidate_count": inventory["candidate_count"],
        "error_count": inventory["error_count"],
        "output_json": str(output_json),
        "output_md": str(output_md),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
