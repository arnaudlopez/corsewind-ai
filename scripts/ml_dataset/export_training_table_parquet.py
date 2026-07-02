#!/usr/bin/env python3
"""Export nested residual training JSONL rows to a flat Parquet dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STRING_HINTS = (
    "_id",
    "_ids",
    "_role",
    "_type",
    "_kind",
    "_source",
    "_dataset",
    "_project",
    "_model",
    "_utc",
    "format",
    "spot_name",
    "baseline_model",
)


def import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Missing pyarrow. Rebuild/install requirements-ml-dataset.txt or run "
            "inside the ml dataset runner image."
        ) from exc
    return pa, pq


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def finite_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def normalize_key(value: str) -> str:
    return (
        value.replace(".", "__")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(":", "_")
    )


def flatten_mapping(prefix: str, values: dict[str, Any], out: dict[str, Any]) -> None:
    for key, value in values.items():
        out[f"{prefix}__{normalize_key(str(key))}"] = value


def flatten_row(row: dict[str, Any], include_feature_sources: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "format",
        "spot_id",
        "spot_name",
        "spot_kind",
        "spot_source_type",
        "station_id",
        "latitude",
        "longitude",
        "issue_time_utc",
        "target_time_utc",
        "lead_time_minutes",
        "built_at_utc",
    ):
        out[key] = row.get(key)
    for group in ("features", "baselines", "labels"):
        values = row.get(group, {}) if isinstance(row.get(group), dict) else {}
        flatten_mapping(group, values, out)
    if include_feature_sources:
        for group in ("issue_feature_sources", "target_feature_sources"):
            values = row.get(group, {}) if isinstance(row.get(group), dict) else {}
            flatten_mapping(group, values, out)
    return out


def is_string_column(name: str, values: list[Any]) -> bool:
    lowered = name.lower()
    if any(hint in lowered for hint in STRING_HINTS):
        return True
    return any(isinstance(value, str) for value in values if value is not None)


def infer_schema(path: Path, include_feature_sources: bool):
    pa, _ = import_pyarrow()
    columns: dict[str, bool] = {}
    row_count = 0
    for row in iter_jsonl(path):
        row_count += 1
        flat = flatten_row(row, include_feature_sources)
        for key, value in flat.items():
            if value is None:
                columns.setdefault(key, False)
                continue
            columns[key] = columns.get(key, False) or isinstance(value, str)
    fields = []
    string_columns = set()
    for key, has_string_value in sorted(columns.items()):
        if any(hint in key.lower() for hint in STRING_HINTS) or has_string_value:
            fields.append(pa.field(key, pa.string()))
            string_columns.add(key)
        else:
            fields.append(pa.field(key, pa.float64()))
    return pa.schema(fields), row_count, string_columns


def coerce_record(record: dict[str, Any], schema, string_columns: set[str]) -> dict[str, Any]:
    out = {}
    for field in schema:
        value = record.get(field.name)
        if field.name in string_columns:
            out[field.name] = None if value in {None, ""} else str(value)
        elif isinstance(value, bool):
            out[field.name] = 1.0 if value else 0.0
        else:
            out[field.name] = finite_float(value)
    return out


def write_parquet(
    input_path: Path,
    output_path: Path,
    batch_size: int,
    compression: str,
    include_feature_sources: bool,
) -> dict[str, Any]:
    pa, pq = import_pyarrow()
    schema, row_count, string_columns = infer_schema(input_path, include_feature_sources)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    if tmp_output_path.exists():
        tmp_output_path.unlink()
    batch: list[dict[str, Any]] = []
    written = 0
    writer = pq.ParquetWriter(tmp_output_path, schema=schema, compression=compression)
    closed = False
    try:
        for row in iter_jsonl(input_path):
            batch.append(coerce_record(flatten_row(row, include_feature_sources), schema, string_columns))
            if len(batch) >= batch_size:
                table = pa.Table.from_pylist(batch, schema=schema)
                writer.write_table(table)
                written += len(batch)
                batch.clear()
        if batch:
            table = pa.Table.from_pylist(batch, schema=schema)
            writer.write_table(table)
            written += len(batch)
        writer.close()
        closed = True
        tmp_output_path.replace(output_path)
    finally:
        if not closed:
            writer.close()
            if tmp_output_path.exists():
                tmp_output_path.unlink()
    return {
        "format": "corsewind.residual_training_table_parquet_export.v1",
        "generated_at_utc": utc_now(),
        "input_jsonl": str(input_path),
        "output_parquet": str(output_path),
        "row_count": written,
        "input_row_count": row_count,
        "column_count": len(schema),
        "string_column_count": len(string_columns),
        "numeric_column_count": len(schema) - len(string_columns),
        "compression": compression,
        "include_feature_sources": include_feature_sources,
    }


def summarize_parquet(path: Path) -> dict[str, Any]:
    pa, pq = import_pyarrow()
    table = pq.read_table(path, columns=["spot_id", "lead_time_minutes", "labels__target_observation_source_dataset"])
    spots = Counter(table["spot_id"].to_pylist()) if "spot_id" in table.column_names else Counter()
    leads = Counter(table["lead_time_minutes"].to_pylist()) if "lead_time_minutes" in table.column_names else Counter()
    sources = (
        Counter(table["labels__target_observation_source_dataset"].to_pylist())
        if "labels__target_observation_source_dataset" in table.column_names
        else Counter()
    )
    return {
        "rows_by_spot": dict(sorted(spots.items())),
        "rows_by_lead": {str(int(key)): value for key, value in sorted(leads.items()) if key is not None},
        "rows_by_target_source_dataset": dict(sorted(sources.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-rows", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=25000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--include-feature-sources", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_parquet:
        output_parquet = args.output_parquet
        output_root = output_parquet.parent
    else:
        output_root = args.output_root or args.training_rows.parent
        output_parquet = output_root / "training_rows.parquet"
    result = write_parquet(
        args.training_rows,
        output_parquet,
        args.batch_size,
        args.compression,
        args.include_feature_sources,
    )
    result.update(summarize_parquet(output_parquet))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "parquet_export_profile.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
