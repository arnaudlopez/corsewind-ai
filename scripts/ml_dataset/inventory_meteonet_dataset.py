#!/usr/bin/env python3
"""Inventory public MeteoNet dataset files before downloading large archives."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_BASE_URL = "https://meteonet.umr-cnrm.fr/dataset/data/"
DEFAULT_OUTPUT_JSON = DEFAULT_ML_ROOT / "source_inventories/meteonet_inventory.json"
DEFAULT_OUTPUT_MD = ROOT / "docs/ml_nowcasting/meteonet_inventory.md"


class DirectoryListingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self._href: str | None = None
        self.rows: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attrs_dict = dict(attrs)
        self._href = attrs_dict.get("href")

    def handle_data(self, data: str) -> None:
        if self._href:
            text = data.strip()
            if text:
                self.links.append(self._href)
            self._href = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_text(url: str, timeout: int) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "CorseWind.ai MeteoNet inventory"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def list_links(url: str, timeout: int) -> list[str]:
    parser = DirectoryListingParser()
    parser.feed(fetch_text(url, timeout))
    links = []
    for href in parser.links:
        if href in {"../", "/"} or href.startswith("?"):
            continue
        links.append(urllib.parse.urljoin(url, href))
    return links


def head_size(url: str, timeout: int) -> int | None:
    try:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "CorseWind.ai MeteoNet inventory"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = response.headers.get("Content-Length")
    except Exception:
        return None
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def classify_url(url: str) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    item: dict[str, Any] = {"zone": None, "category": None, "year": None}
    if "data" in parts:
        idx = parts.index("data")
        if len(parts) > idx + 1:
            item["zone"] = parts[idx + 1]
        if len(parts) > idx + 2:
            item["category"] = "/".join(parts[idx + 2:-1])
    match = re.search(r"(2016|2017|2018)", Path(parsed.path).name)
    if match:
        item["year"] = int(match.group(1))
    return item


def inventory(base_url: str, timeout: int, zones: list[str]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for zone in zones:
        zone_url = urllib.parse.urljoin(base_url, f"{zone}/")
        for category_url in list_links(zone_url, timeout):
            if not category_url.endswith("/"):
                continue
            for nested_or_file in list_links(category_url, timeout):
                nested_links = []
                if nested_or_file.endswith("/"):
                    nested_links = list_links(nested_or_file, timeout)
                candidates = nested_links or [nested_or_file]
                for candidate in candidates:
                    if candidate.endswith("/"):
                        continue
                    if not re.search(r"\.(tar\.gz|csv|nc|npz|h5)$", candidate, re.IGNORECASE):
                        continue
                    size_bytes = head_size(candidate, timeout)
                    meta = classify_url(candidate)
                    files.append({
                        "url": candidate,
                        "filename": Path(urllib.parse.urlparse(candidate).path).name,
                        "size_bytes": size_bytes,
                        "size_gb": round(size_bytes / 1024**3, 3) if size_bytes is not None else None,
                        **meta,
                    })
    by_category: dict[str, dict[str, Any]] = {}
    for item in files:
        key = f"{item.get('zone')}/{item.get('category')}"
        summary = by_category.setdefault(key, {"file_count": 0, "size_bytes": 0, "years": set()})
        summary["file_count"] += 1
        summary["size_bytes"] += item.get("size_bytes") or 0
        if item.get("year"):
            summary["years"].add(item["year"])
    for summary in by_category.values():
        summary["years"] = sorted(summary["years"])
        summary["size_gb"] = round(summary["size_bytes"] / 1024**3, 3)
    return {
        "format": "corsewind.meteonet_inventory.v1",
        "generated_at_utc": utc_now(),
        "base_url": base_url,
        "zones": zones,
        "files": sorted(files, key=lambda item: item["url"]),
        "by_category": dict(sorted(by_category.items())),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MeteoNet Inventory",
        "",
        f"Generated at: `{payload['generated_at_utc']}`",
        "",
        "## Summary",
        "",
        "| Category | Files | Years | Size GiB |",
        "| --- | ---: | --- | ---: |",
    ]
    for key, item in payload["by_category"].items():
        lines.append(f"| `{key}` | {item['file_count']} | {', '.join(map(str, item['years']))} | {item['size_gb']} |")
    lines.extend(["", "## Files", "", "| File | Size GiB | URL |", "| --- | ---: | --- |"])
    for item in payload["files"]:
        lines.append(f"| `{item['filename']}` | {item.get('size_gb')} | {item['url']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--zones", default="SE", help="Comma-separated MeteoNet zones to inventory.")
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zones = [zone.strip() for zone in args.zones.split(",") if zone.strip()]
    payload = inventory(args.base_url, args.timeout_sec, zones)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(json.dumps({
        "generated_at_utc": payload["generated_at_utc"],
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
        "file_count": len(payload["files"]),
        "by_category": payload["by_category"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
