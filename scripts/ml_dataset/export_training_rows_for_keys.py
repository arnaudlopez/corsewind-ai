#!/usr/bin/env python3
"""Export full training-table rows matching a parquet of prediction keys."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KEY_COLUMNS = ("spot_id", "issue_time_utc", "lead_time_minutes")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def month_range(start_month: str, end_month: str) -> list[str]:
    start_year, start_m = [int(part) for part in start_month.split("-", 1)]
    end_year, end_m = [int(part) for part in end_month.split("-", 1)]
    months = []
    year, month = start_year, start_m
    while (year, month) <= (end_year, end_m):
        months.append(f"{year:04d}_{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return months


def discover_parquets(root: Path, prefix: str, start_month: str, end_month: str) -> list[Path]:
    paths = []
    for suffix in month_range(start_month, end_month):
        path = root / f"{prefix}_{suffix}" / "training_rows.parquet"
        if path.exists():
            paths.append(path)
    if not paths:
        raise SystemExit(f"No training_rows.parquet found under {root} for {prefix} {start_month}..{end_month}")
    return paths


def canonical_issue_time(series: Any, pd: Any) -> Any:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_lead(series: Any, pd: Any) -> Any:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def load_keys(path: Path, pd: Any) -> tuple[Any, set[tuple[str, str, int]]]:
    keys = pd.read_parquet(path)
    missing = [column for column in KEY_COLUMNS if column not in keys.columns]
    if missing:
        raise SystemExit(f"{path} missing required key columns: {missing}")
    keys = keys[list(KEY_COLUMNS)].copy()
    keys["spot_id"] = keys["spot_id"].astype(str)
    keys["issue_time_utc"] = canonical_issue_time(keys["issue_time_utc"], pd)
    keys["lead_time_minutes"] = canonical_lead(keys["lead_time_minutes"], pd)
    keys = keys.dropna(subset=list(KEY_COLUMNS)).drop_duplicates()
    key_set = set(zip(keys["spot_id"], keys["issue_time_utc"], keys["lead_time_minutes"].astype("int64"), strict=True))
    return keys, key_set


def filter_batch(frame: Any, wanted: set[tuple[str, str, int]], pd: Any) -> Any:
    frame = frame.copy()
    frame["spot_id"] = frame["spot_id"].astype(str)
    frame["issue_time_utc"] = canonical_issue_time(frame["issue_time_utc"], pd)
    frame["lead_time_minutes"] = canonical_lead(frame["lead_time_minutes"], pd)
    frame = frame.dropna(subset=list(KEY_COLUMNS))
    if frame.empty:
        return frame
    keys = list(zip(frame["spot_id"], frame["issue_time_utc"], frame["lead_time_minutes"].astype("int64"), strict=True))
    return frame[[key in wanted for key in keys]].copy()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import pandas as pd
    import pyarrow.parquet as pq

    keys, wanted = load_keys(args.keys_parquet, pd)
    paths = discover_parquets(args.training_table_root, args.run_id_prefix, args.start_month, args.end_month)
    frames = []
    path_summaries = []
    for path in paths:
        pf = pq.ParquetFile(path)
        schema = set(pf.schema.names)
        missing = [column for column in KEY_COLUMNS if column not in schema]
        if missing:
            path_summaries.append({"path": str(path), "status": "skipped_missing_keys", "missing": missing})
            continue
        columns = args.column or None
        if columns is not None:
            columns = list(dict.fromkeys([*KEY_COLUMNS, *columns]))
            columns = [column for column in columns if column in schema]
        matched_rows = 0
        batches = 0
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=columns):
            batches += 1
            selected = filter_batch(batch.to_pandas(), wanted, pd)
            if selected.empty:
                continue
            matched_rows += int(len(selected))
            frames.append(selected)
        path_summaries.append({"path": str(path), "status": "read", "batches": batches, "matched_rows": matched_rows})

    if frames:
        output = pd.concat(frames, ignore_index=True)
        output = output.drop_duplicates(subset=list(KEY_COLUMNS), keep="last")
        output = output.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    else:
        output = pd.DataFrame(columns=list(KEY_COLUMNS))

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output_parquet, index=False, compression=args.compression)
    matched = output[list(KEY_COLUMNS)].copy() if not output.empty else pd.DataFrame(columns=list(KEY_COLUMNS))
    if not matched.empty:
        matched["spot_id"] = matched["spot_id"].astype(str)
        matched["issue_time_utc"] = canonical_issue_time(matched["issue_time_utc"], pd)
        matched["lead_time_minutes"] = canonical_lead(matched["lead_time_minutes"], pd)
    matched_set = set(zip(matched["spot_id"], matched["issue_time_utc"], matched["lead_time_minutes"].astype("int64"), strict=True)) if not matched.empty else set()
    result = {
        "format": "corsewind.training_rows_for_keys.v1",
        "generated_at_utc": utc_now(),
        "keys_parquet": str(args.keys_parquet),
        "training_table_root": str(args.training_table_root),
        "run_id_prefix": args.run_id_prefix,
        "start_month": args.start_month,
        "end_month": args.end_month,
        "input_key_count": int(len(keys)),
        "matched_key_count": int(len(matched_set)),
        "missing_key_count": int(len(wanted - matched_set)),
        "output_row_count": int(len(output)),
        "output_column_count": int(len(output.columns)),
        "output_parquet": str(args.output_parquet),
        "paths": path_summaries,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "input_key_count": result["input_key_count"],
        "matched_key_count": result["matched_key_count"],
        "missing_key_count": result["missing_key_count"],
        "output_row_count": result["output_row_count"],
        "output_column_count": result["output_column_count"],
        "output_parquet": result["output_parquet"],
    }, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys-parquet", type=Path, required=True)
    parser.add_argument("--training-table-root", type=Path, required=True)
    parser.add_argument("--run-id-prefix", required=True)
    parser.add_argument("--start-month", required=True)
    parser.add_argument("--end-month", required=True)
    parser.add_argument("--column", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
