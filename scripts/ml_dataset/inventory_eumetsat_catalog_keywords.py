#!/usr/bin/env python3
"""Search the public EUMETSAT catalogue for CorseWind-relevant datasets."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_OUTPUT_JSON = DEFAULT_ML_ROOT / "source_inventories/eumetsat_catalog_keyword_inventory.json"
DEFAULT_OUTPUT_MD = ROOT / "docs/ml_nowcasting/eumetsat_catalog_keyword_inventory.md"
COLLECTIONS_URL = "https://api.eumetsat.int/data/browse/1.0.0/collections?format=json"
DETAIL_URL = "https://api.eumetsat.int/data/browse/1.0.0/collections/{collection_id}?format=json"
CURATED_COLLECTIONS = {
    "EO:EUM:DAT:0678",
    "EO:EUM:DAT:0680",
    "EO:EUM:DAT:0681",
    "EO:EUM:DAT:0684",
    "EO:EUM:DAT:1088",
    "EO:EUM:DAT:1086",
    "EO:EUM:DAT:0863",
    "EO:EUM:DAT:MSG:CLM",
}

KEYWORD_GROUPS = {
    "cloud": ["cloud", "clm", "cloud mask", "cloud type", "cloud top", "oca"],
    "land_heating": ["land surface temperature", "surface temperature", "lst", "surface albedo"],
    "radiation": ["radiation", "irradiance", "heliosat", "solar", "surface incoming"],
    "precipitation_convection": ["precipitation", "rain", "convective", "convection", "instability", "lightning"],
    "atmospheric_wind": ["wind", "atmospheric motion", "amv", "motion vector"],
    "aerosol_dust_ash": ["aerosol", "dust", "ash", "volcanic"],
    "water_vapour": ["water vapour", "water vapor", "humidity", "moisture"],
    "sea_surface": ["sea surface temperature", "sst", "sea ice", "ocean"],
    "legacy_msg": ["msg", "seviri", "meteosat second generation"],
    "mtg_fci": ["mtg", "fci", "mti1"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def fetch_json(url: str, timeout: int) -> dict[str, Any]:
    response = requests.get(url, headers={"Accept": "application/json"}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def collection_links(timeout: int) -> list[dict[str, Any]]:
    payload = fetch_json(COLLECTIONS_URL, timeout)
    return payload.get("links", [])


def detail(collection_id: str, timeout: int) -> dict[str, Any] | None:
    try:
        return fetch_json(DETAIL_URL.format(collection_id=quote(collection_id, safe="")), timeout)
    except requests.RequestException:
        return None


def classify(text: str) -> dict[str, Any]:
    normalized = text.lower()
    matched: dict[str, list[str]] = {}
    score = 0
    for group, keywords in KEYWORD_GROUPS.items():
        hits = [keyword for keyword in keywords if keyword in normalized]
        if hits:
            matched[group] = hits
            score += len(hits)
    return {"score": score, "groups": matched}


def summarize_link(link: dict[str, Any], timeout: int, fetch_details: bool) -> dict[str, Any]:
    collection_id = link.get("title")
    title = link.get("datasetTitle") or ""
    details = detail(collection_id, timeout) if fetch_details and collection_id else None
    properties = (details or {}).get("collection", {}).get("properties", {})
    abstract = properties.get("abstract")
    text = " ".join(str(value or "") for value in [collection_id, title, abstract])
    classification = classify(text)
    return {
        "collection_id": collection_id,
        "title": title,
        "number_of_products": link.get("numberOfProducts"),
        "href": link.get("href"),
        "is_curated": collection_id in CURATED_COLLECTIONS,
        "score": classification["score"] + (5 if collection_id in CURATED_COLLECTIONS else 0),
        "matched_groups": classification["groups"],
        "date": properties.get("date"),
        "rights": properties.get("rights"),
        "abstract": abstract,
    }


def short(text: str | None, limit: int = 180) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "..."


def write_markdown(path: Path, inventory: dict[str, Any]) -> None:
    lines = [
        "# EUMETSAT Catalogue Keyword Inventory",
        "",
        f"Generated at: `{inventory['generated_at_utc']}`",
        "",
        f"Total collections scanned: `{inventory['collection_count']}`",
        f"Matched collections: `{inventory['matched_count']}`",
        "",
        "## Recommended Triage",
        "",
        "- P1 remains MTG Cloud Mask `EO:EUM:DAT:0678` because it is available every 10 minutes and directly measures clear/cloudy pixels.",
        "- Next strongest tests are Cloud Type `EO:EUM:DAT:0680` and Land Surface Temperature `EO:EUM:DAT:1088`.",
        "- Radiation and legacy MSG products are more useful for history/backtests than immediate nowcasting.",
        "",
        "## Matches By Group",
        "",
    ]
    for group, items in sorted(inventory["groups"].items()):
        lines.extend([f"### `{group}`", "", "| Score | Curated | Collection | Title | Products | Date |", "| ---: | --- | --- | --- | ---: | --- |"])
        for item in items[:30]:
            lines.append(
                f"| {item['score']} | `{item['is_curated']}` | `{item['collection_id']}` | "
                f"{item['title']} | {item.get('number_of_products') or ''} | `{item.get('date')}` |"
            )
        lines.append("")
    lines.extend(["## Top Ranked Matches", "", "| Score | Groups | Collection | Title | Notes |", "| ---: | --- | --- | --- | --- |"])
    for item in inventory["top_matches"][:50]:
        groups = ", ".join(f"`{group}`" for group in item.get("matched_groups", {}))
        lines.append(
            f"| {item['score']} | {groups} | `{item['collection_id']}` | {item['title']} | {short(item.get('abstract'))} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--timeout-sec", type=int, default=20)
    parser.add_argument("--fetch-details", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [
        summarize_link(link, args.timeout_sec, args.fetch_details)
        for link in collection_links(args.timeout_sec)
    ]
    matches = [item for item in records if item["score"] > 0]
    matches.sort(key=lambda item: (-item["score"], item.get("collection_id") or ""))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in matches:
        for group in item.get("matched_groups", {}):
            groups[group].append(item)
    for items in groups.values():
        items.sort(key=lambda item: (-item["score"], item.get("collection_id") or ""))
    inventory = {
        "format": "corsewind.eumetsat_catalog_keyword_inventory.v1",
        "generated_at_utc": utc_now(),
        "collection_count": len(records),
        "matched_count": len(matches),
        "top_matches": matches,
        "groups": dict(sorted(groups.items())),
    }
    output_json = resolve_path(args.output_json)
    output_md = resolve_path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(output_md, inventory)
    print(json.dumps({
        "generated_at_utc": inventory["generated_at_utc"],
        "collection_count": inventory["collection_count"],
        "matched_count": inventory["matched_count"],
        "output_json": str(output_json),
        "output_md": str(output_md),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
