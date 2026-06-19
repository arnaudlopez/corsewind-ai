#!/usr/bin/env python3
"""Small MeteoHub open-data helpers for public forecast bundles."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import requests


BASE_URL = "https://meteohub.agenziaitaliameteo.it"


@dataclass(frozen=True)
class OpenDataBundle:
    dataset_id: str
    date: str
    run: str
    filename: str
    vars: tuple[str, ...]

    @property
    def run_time_utc(self) -> datetime:
        return datetime.fromisoformat(f"{self.date}T{self.run}:00+00:00").astimezone(timezone.utc)

    @property
    def download_url(self) -> str:
        return f"{BASE_URL}/api/opendata/{self.filename}"

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "date": self.date,
            "run": self.run,
            "run_time_utc": self.run_time_utc.isoformat().replace("+00:00", "Z"),
            "filename": self.filename,
            "download_url": self.download_url,
            "vars": list(self.vars),
        }


def list_opendata_bundles(dataset_id: str, timeout_sec: int = 30) -> list[OpenDataBundle]:
    url = f"{BASE_URL}/api/datasets/{dataset_id}/opendata"
    response = requests.get(url, timeout=timeout_sec)
    response.raise_for_status()
    payload = response.json()
    bundles: list[OpenDataBundle] = []
    for item in payload:
        filename = str(item.get("filename") or "")
        if not filename:
            continue
        bundles.append(
            OpenDataBundle(
                dataset_id=dataset_id,
                date=str(item.get("date") or ""),
                run=str(item.get("run") or "00:00"),
                filename=filename,
                vars=tuple(str(value) for value in item.get("vars") or ()),
            )
        )
    return bundles


def has_required_vars(bundle: OpenDataBundle, required_vars: Iterable[str]) -> bool:
    haystack = " ".join(bundle.vars).lower()
    return all(required.lower() in haystack for required in required_vars)


def latest_opendata_bundle(
    dataset_id: str,
    required_vars: Iterable[str] = (),
    timeout_sec: int = 30,
) -> OpenDataBundle:
    bundles = list_opendata_bundles(dataset_id, timeout_sec=timeout_sec)
    required = tuple(required_vars)
    if required:
        bundles = [bundle for bundle in bundles if has_required_vars(bundle, required)]
    if not bundles:
        raise SystemExit(f"No MeteoHub open-data bundle found for {dataset_id}.")
    return max(bundles, key=lambda bundle: bundle.run_time_utc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id")
    parser.add_argument("--required-var", action="append", default=[])
    parser.add_argument("--timeout-sec", type=int, default=30)
    args = parser.parse_args()
    bundle = latest_opendata_bundle(args.dataset_id, args.required_var, args.timeout_sec)
    print(json.dumps(bundle.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
