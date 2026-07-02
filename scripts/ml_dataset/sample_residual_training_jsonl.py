#!/usr/bin/env python3
"""Deterministically sample a large residual-training JSONL before Parquet export."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def row_key(row: dict[str, Any]) -> str:
    return "|".join(
        str(row.get(key) or "")
        for key in ("spot_id", "issue_time_utc", "target_time_utc", "lead_time_minutes")
    )


def stable_hash(row: dict[str, Any], salt: str) -> int:
    digest = hashlib.blake2b(f"{salt}|{row_key(row)}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def parse_leads(values: list[int] | None) -> set[int] | None:
    if not values:
        return None
    return {int(value) for value in values}


def row_split(row: dict[str, Any], split_time_utc: str) -> str:
    issue_time = str(row.get("issue_time_utc") or "")
    return "train" if issue_time < split_time_utc else "test"


def row_allowed(row: dict[str, Any], split_time_utc: str, leads: set[int] | None) -> bool:
    if leads is not None:
        try:
            lead = int(row.get("lead_time_minutes"))
        except (TypeError, ValueError):
            return False
        if lead not in leads:
            return False
    issue_time = row.get("issue_time_utc")
    return bool(issue_time)


def count_rows(input_jsonl: Path, split_time_utc: str, leads: set[int] | None) -> dict[str, Any]:
    counts = {
        "total_rows": 0,
        "eligible_rows": 0,
        "train_rows": 0,
        "test_rows": 0,
        "rows_by_lead": Counter(),
        "rows_by_spot": Counter(),
    }
    for row in iter_jsonl(input_jsonl):
        counts["total_rows"] += 1
        if not row_allowed(row, split_time_utc, leads):
            continue
        split = row_split(row, split_time_utc)
        counts["eligible_rows"] += 1
        counts[f"{split}_rows"] += 1
        counts["rows_by_lead"].update([row.get("lead_time_minutes")])
        counts["rows_by_spot"].update([str(row.get("spot_id"))])
    return counts


def choose_cutoffs(
    input_jsonl: Path,
    split_time_utc: str,
    leads: set[int] | None,
    max_train_rows: int | None,
    max_test_rows: int | None,
    salt: str,
) -> dict[str, int | None]:
    limits = {"train": max_train_rows, "test": max_test_rows}
    hashes: dict[str, list[int]] = {"train": [], "test": []}
    for row in iter_jsonl(input_jsonl):
        if not row_allowed(row, split_time_utc, leads):
            continue
        split = row_split(row, split_time_utc)
        limit = limits[split]
        if limit is None or limit <= 0:
            continue
        hashes[split].append(stable_hash(row, salt))
    cutoffs: dict[str, int | None] = {"train": None, "test": None}
    for split, values in hashes.items():
        limit = limits[split]
        if limit is None or limit <= 0 or len(values) <= limit:
            continue
        values.sort()
        cutoffs[split] = values[limit - 1]
    return cutoffs


def sample_rows(
    input_jsonl: Path,
    output_jsonl: Path,
    split_time_utc: str,
    leads: set[int] | None,
    max_train_rows: int | None,
    max_test_rows: int | None,
    salt: str,
    cutoffs: dict[str, int | None],
) -> dict[str, Any]:
    limits = {"train": max_train_rows, "test": max_test_rows}
    written = {"train": 0, "test": 0}
    skipped_by_cutoff = {"train": 0, "test": 0}
    rows_by_lead: Counter[int | None] = Counter()
    rows_by_spot: Counter[str] = Counter()
    first_issue_time = None
    last_issue_time = None
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_jsonl.with_name(f".{output_jsonl.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(input_jsonl):
            if not row_allowed(row, split_time_utc, leads):
                continue
            split = row_split(row, split_time_utc)
            limit = limits[split]
            if limit is not None and limit > 0:
                if written[split] >= limit:
                    skipped_by_cutoff[split] += 1
                    continue
                cutoff = cutoffs.get(split)
                if cutoff is not None and stable_hash(row, salt) > cutoff:
                    skipped_by_cutoff[split] += 1
                    continue
            issue_time = row.get("issue_time_utc")
            if first_issue_time is None:
                first_issue_time = issue_time
            last_issue_time = issue_time
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            written[split] += 1
            rows_by_lead.update([row.get("lead_time_minutes")])
            rows_by_spot.update([str(row.get("spot_id"))])
    tmp_path.replace(output_jsonl)
    return {
        "output_jsonl": str(output_jsonl),
        "written_rows": written["train"] + written["test"],
        "written_train_rows": written["train"],
        "written_test_rows": written["test"],
        "skipped_by_cutoff": skipped_by_cutoff,
        "rows_by_lead": {str(int(key)): value for key, value in sorted(rows_by_lead.items()) if key is not None},
        "rows_by_spot": dict(sorted(rows_by_spot.items())),
        "first_issue_time_utc": first_issue_time,
        "last_issue_time_utc": last_issue_time,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-name", default="training_rows_sampled.jsonl")
    parser.add_argument("--split-time-utc", required=True)
    parser.add_argument("--include-lead-minute", type=int, action="append")
    parser.add_argument("--max-train-rows", type=int, default=150000)
    parser.add_argument("--max-test-rows", type=int, default=100000)
    parser.add_argument("--salt", default="corsewind_residual_training_sample_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    leads = parse_leads(args.include_lead_minute)
    counts = count_rows(args.input_jsonl, args.split_time_utc, leads)
    cutoffs = choose_cutoffs(
        args.input_jsonl,
        args.split_time_utc,
        leads,
        args.max_train_rows,
        args.max_test_rows,
        args.salt,
    )
    output_jsonl = args.output_root / args.output_name
    sample = sample_rows(
        args.input_jsonl,
        output_jsonl,
        args.split_time_utc,
        leads,
        args.max_train_rows,
        args.max_test_rows,
        args.salt,
        cutoffs,
    )
    profile = {
        "format": "corsewind.sampled_residual_training_jsonl.v1",
        "generated_at_utc": utc_now(),
        "input_jsonl": str(args.input_jsonl),
        "split_time_utc": args.split_time_utc,
        "include_lead_minutes": sorted(leads) if leads else None,
        "max_train_rows": args.max_train_rows,
        "max_test_rows": args.max_test_rows,
        "salt": args.salt,
        "pre_sample_counts": {
            key: value
            for key, value in counts.items()
            if key not in {"rows_by_lead", "rows_by_spot"}
        },
        "pre_sample_rows_by_lead": {
            str(int(key)): value for key, value in sorted(counts["rows_by_lead"].items()) if key is not None
        },
        "pre_sample_rows_by_spot": dict(sorted(counts["rows_by_spot"].items())),
        "hash_cutoffs": cutoffs,
        **sample,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "sample_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
