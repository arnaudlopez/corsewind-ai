#!/usr/bin/env python3
"""Small CLI for Météo-France public AROME WCS access.

The script avoids hard-coding secrets. Put the API token in `.env`:

    METEOFRANCE_API_KEY=...

Then start with:

    python3 scripts/meteo_france_client.py capabilities --product arome --resolution 001

Use `download` after selecting a coverage ID from capabilities output.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlencode

import requests
from requests import RequestException


PRODUCT_BASE_URLS = {
    "arome": "https://public-api.meteofrance.fr/public/arome/1.0",
    "aromepi": "https://public-api.meteofrance.fr/public/aromepi/1.0",
}

SERVICE_NAMES = {
    ("arome", "001"): "MF-NWP-HIGHRES-AROME-001-FRANCE-WCS",
    ("arome", "0025"): "MF-NWP-HIGHRES-AROME-0025-FRANCE-WCS",
    ("aromepi", "001"): "MF-NWP-HIGHRES-AROMEPI-001-FRANCE-WCS",
    ("aromepi", "0025"): "MF-NWP-HIGHRES-AROMEPI-0025-FRANCE-WCS",
}

WIND_PATTERNS = (
    "WIND__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
    "WIND_SPEED__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
    "WIND_SPEED_GUST__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
    "WIND_SPEED_GUST_MAX__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
    "WIND_SPEED_MAXIMUM_GUST__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
    "U_COMPONENT_OF_WIND__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
    "V_COMPONENT_OF_WIND__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def api_key() -> str:
    token = os.environ.get("METEOFRANCE_API_KEY") or os.environ.get("METEOFRANCE_TOKEN")
    if not token:
        raise SystemExit("Missing METEOFRANCE_API_KEY. Add it to .env or export it in the shell.")
    return token


def auth_headers(mode: str) -> dict[str, str]:
    token = api_key()
    if mode == "apikey":
        return {"apikey": token}
    if mode == "bearer":
        return {"Authorization": f"Bearer {token}"}
    raise ValueError(f"Unknown auth mode: {mode}")


def endpoint(product: str, resolution: str, operation: str) -> str:
    service = SERVICE_NAMES[(product, resolution)]
    return f"{PRODUCT_BASE_URLS[product]}/wcs/{service}/{operation}"


def request_api(url: str, params: list[tuple[str, str]], auth_header: str) -> requests.Response:
    max_attempts = int(os.environ.get("METEOFRANCE_MAX_ATTEMPTS", "5"))
    last_error: RequestException | None = None
    response: requests.Response | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            response = requests.get(url, params=params, headers=auth_headers(auth_header), timeout=120)
        except RequestException as exc:
            last_error = exc
            if attempt >= max_attempts:
                raise SystemExit(f"Request failed before receiving a response after {attempt} attempt(s): {exc}") from exc
            time.sleep(min(30.0, 2.0 * attempt))
            continue

        if response.status_code not in {429, 500, 502, 503, 504} or attempt >= max_attempts:
            break
        time.sleep(min(60.0, 3.0 * attempt))

    if response is None:
        raise SystemExit(f"Request failed before receiving a response: {last_error}")

    if response.status_code >= 400:
        redacted_url = f"{url}?{urlencode(params)}"
        raise SystemExit(
            f"Météo-France API returned HTTP {response.status_code} for {redacted_url}\n"
            f"Response preview: {response.text[:600]}"
        )
    return response


def coverage_ids(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    ids: list[str] = []
    for elem in root.iter():
        local_name = elem.tag.rsplit("}", 1)[-1]
        if local_name in {"CoverageId", "CoverageID"} and elem.text:
            ids.append(elem.text.strip())
    return ids


def print_capabilities(args: argparse.Namespace) -> None:
    url = endpoint(args.product, args.resolution, "GetCapabilities")
    params = [("service", "WCS"), ("version", "2.0.1"), ("language", args.language)]
    response = request_api(url, params, args.auth_header)
    ids = coverage_ids(response.text)
    if args.only_wind:
        ids = [item for item in ids if any(pattern in item for pattern in WIND_PATTERNS)]
    if args.contains:
        needle = args.contains.upper()
        ids = [item for item in ids if needle in item.upper()]
    for item in ids:
        print(item)
    print(f"# {len(ids)} coverage IDs", file=sys.stderr)


def describe_coverage(args: argparse.Namespace) -> None:
    url = endpoint(args.product, args.resolution, "DescribeCoverage")
    params = [
        ("service", "WCS"),
        ("version", "2.0.1"),
        ("coverageID", args.coverage_id),
    ]
    response = request_api(url, params, args.auth_header)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(response.text, encoding="utf-8")
    print(f"Wrote {args.output}")


def download_coverage(args: argparse.Namespace) -> None:
    url = endpoint(args.product, args.resolution, "GetCoverage")
    params = [
        ("service", "WCS"),
        ("version", "2.0.1"),
        ("coverageid", args.coverage_id),
        ("format", args.format),
    ]
    for subset in args.subset:
        params.append(("subset", subset))

    response = request_api(url, params, args.auth_header)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(response.content)
    print(f"Wrote {len(response.content)} bytes to {args.output}")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--product", choices=sorted(PRODUCT_BASE_URLS), default="arome")
    parser.add_argument("--resolution", choices=["001", "0025"], default="001")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--auth-header", choices=["bearer", "apikey"], default="apikey")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    capabilities = subparsers.add_parser("capabilities")
    add_common(capabilities)
    capabilities.add_argument("--language", choices=["eng", "fre"], default="eng")
    capabilities.add_argument("--only-wind", action="store_true")
    capabilities.add_argument("--contains", help="Case-insensitive coverage ID substring filter")
    capabilities.set_defaults(func=print_capabilities)

    describe = subparsers.add_parser("describe")
    add_common(describe)
    describe.add_argument("coverage_id")
    describe.add_argument("--output", type=Path, default=Path("data/raw/arome_describe_coverage.xml"))
    describe.set_defaults(func=describe_coverage)

    download = subparsers.add_parser("download")
    add_common(download)
    download.add_argument("coverage_id")
    download.add_argument("--subset", action="append", default=[], help="WCS subset expression, repeatable")
    download.add_argument("--format", choices=["application/wmo-grib", "image/tiff"], default="application/wmo-grib")
    download.add_argument("--output", type=Path, required=True)
    download.set_defaults(func=download_coverage)

    args = parser.parse_args()
    load_dotenv(args.env_file)
    args.func(args)


if __name__ == "__main__":
    main()
