#!/usr/bin/env python3
"""Audit a CorseWind SAPHIR-style sequence dataset export."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def import_deps() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Missing pandas/numpy/pyarrow dependencies.") from exc
    return {"np": np, "pd": pd, "pq": pq}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def table_summary(path: Path, pq: Any) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "rows": 0, "columns": 0}
    pf = pq.ParquetFile(path)
    return {
        "exists": True,
        "rows": int(pf.metadata.num_rows),
        "columns": len(pf.schema_arrow.names),
        "column_names": list(pf.schema_arrow.names),
    }


def metric(np: Any, frame: Any, prediction: str, actual: str) -> dict[str, Any]:
    import pandas as pd

    data = frame[[prediction, actual]].apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        return {"count": 0}
    errors = data[prediction].to_numpy(dtype=float) - data[actual].to_numpy(dtype=float)
    return {
        "count": int(len(errors)),
        "mae": round(float(np.mean(np.abs(errors))), 6),
        "rmse": round(float(np.sqrt(np.mean(errors * errors))), 6),
        "bias": round(float(np.mean(errors)), 6),
    }


def baseline_metrics(future: Any, deps: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    np = deps["np"]
    if future.empty:
        return out
    for split, split_frame in future.groupby("split", dropna=False):
        split_key = str(split)
        out[split_key] = {
            "overall": {},
            "by_lead": {},
            "by_spot": {},
        }
        pairs = [
            ("wind_mean", "baseline_wind_mean_ms", "target_wind_mean_ms"),
            ("gust", "baseline_gust_ms", "target_gust_ms"),
        ]
        for name, prediction, actual in pairs:
            if prediction in split_frame.columns and actual in split_frame.columns:
                out[split_key]["overall"][name] = metric(np, split_frame, prediction, actual)
        for lead, group in split_frame.groupby("lead_time_minutes", dropna=False):
            lead_key = str(int(lead)) if str(lead) != "nan" else "nan"
            out[split_key]["by_lead"][lead_key] = {}
            for name, prediction, actual in pairs:
                if prediction in group.columns and actual in group.columns:
                    out[split_key]["by_lead"][lead_key][name] = metric(np, group, prediction, actual)
        for spot, group in split_frame.groupby("spot_id", dropna=False):
            spot_key = str(spot)
            out[split_key]["by_spot"][spot_key] = {}
            for name, prediction, actual in pairs:
                if prediction in group.columns and actual in group.columns:
                    out[split_key]["by_spot"][spot_key][name] = metric(np, group, prediction, actual)
    return out


def temporal_checks(root: Path, deps: dict[str, Any]) -> dict[str, Any]:
    pd = deps["pd"]
    checks: dict[str, Any] = {}
    future_path = root / "future_targets.parquet"
    history_path = root / "station_history.parquet"
    if future_path.exists():
        future = pd.read_parquet(future_path, columns=["sample_id", "issue_time_utc", "target_time_utc", "lead_time_minutes"])
        issue = pd.to_datetime(future["issue_time_utc"], utc=True, errors="coerce")
        target = pd.to_datetime(future["target_time_utc"], utc=True, errors="coerce")
        lead = pd.to_numeric(future["lead_time_minutes"], errors="coerce")
        expected = issue + pd.to_timedelta(lead, unit="m")
        mismatch = (target - expected).dt.total_seconds().abs()
        checks["future_targets"] = {
            "rows": int(len(future)),
            "target_not_after_issue_rows": int((target <= issue).fillna(False).sum()),
            "lead_mismatch_rows": int((mismatch > 1.0).fillna(False).sum()),
            "max_lead_mismatch_seconds": None if mismatch.dropna().empty else round(float(mismatch.max()), 6),
        }
    if history_path.exists():
        history = pd.read_parquet(history_path, columns=["sample_id", "issue_time_utc", "timestamp_utc"])
        issue = pd.to_datetime(history["issue_time_utc"], utc=True, errors="coerce")
        timestamp = pd.to_datetime(history["timestamp_utc"], utc=True, errors="coerce")
        checks["station_history"] = {
            "rows": int(len(history)),
            "history_after_issue_rows": int((timestamp > issue).fillna(False).sum()),
            "max_timestamp": None if timestamp.dropna().empty else timestamp.max().isoformat(),
        }
    return checks


def coverage(root: Path, deps: dict[str, Any]) -> dict[str, Any]:
    pd = deps["pd"]
    out: dict[str, Any] = {}
    history_path = root / "station_history.parquet"
    if history_path.exists():
        history = pd.read_parquet(history_path)
        observed_cols = [column for column in history.columns if column.endswith("_observed")]
        out["station_history"] = {
            "rows": int(len(history)),
            "observed_ratios": {
                column: round(float(history[column].fillna(False).mean()), 6)
                for column in observed_cols
            },
        }
    context_path = root / "context_station_snapshot.parquet"
    if context_path.exists():
        context = pd.read_parquet(context_path)
        out["context_station_snapshot"] = {
            "rows": int(len(context)),
            "slots": dict(sorted(Counter(context.get("station_slot_name", [])).items())),
            "available_ratio": (
                None
                if "available" not in context.columns or context.empty
                else round(float(pd.to_numeric(context["available"], errors="coerce").fillna(0).mean()), 6)
            ),
        }
    vertical_path = root / "nwp_vertical_profile.parquet"
    if vertical_path.exists():
        vertical = pd.read_parquet(vertical_path)
        out["nwp_vertical_profile"] = {
            "rows": int(len(vertical)),
            "levels": dict(sorted(Counter(vertical.get("pressure_hpa", [])).items())),
        }
    offset_path = root / "nwp_surface_offsets.parquet"
    if offset_path.exists():
        offsets = pd.read_parquet(offset_path)
        out["nwp_surface_offsets"] = {
            "rows": int(len(offsets)),
            "offsets": dict(sorted(Counter(offsets.get("offset_name", [])).items())),
        }
    return out


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# CorseWind SAPHIR-Style Dataset Audit",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Dataset root: `{result['dataset_root']}`",
        f"Verdict: `{result['verdict']}`",
        "",
        "## Tables",
        "",
        "| Table | Exists | Rows | Columns |",
        "| --- | --- | ---: | ---: |",
    ]
    for name, item in result["tables"].items():
        lines.append(f"| `{name}` | `{item['exists']}` | {item['rows']} | {item['columns']} |")
    lines.extend([
        "",
        "## Temporal Checks",
        "",
        "```json",
        json.dumps(result["temporal_checks"], indent=2, sort_keys=True),
        "```",
        "",
        "## Baseline Metrics",
        "",
        "```json",
        json.dumps(result["baseline_metrics"], indent=2, sort_keys=True),
        "```",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_deps()
    pd = deps["pd"]
    pq = deps["pq"]
    root = args.dataset_root
    table_names = [
        "samples",
        "future_targets",
        "station_history",
        "static_context",
        "context_station_snapshot",
        "nwp_surface_offsets",
        "nwp_vertical_profile",
    ]
    tables = {name: table_summary(root / f"{name}.parquet", pq) for name in table_names}
    future = pd.read_parquet(root / "future_targets.parquet") if (root / "future_targets.parquet").exists() else pd.DataFrame()
    temporal = temporal_checks(root, deps)
    failures = []
    if temporal.get("future_targets", {}).get("target_not_after_issue_rows", 0):
        failures.append("future target timestamp is not after issue time")
    if temporal.get("future_targets", {}).get("lead_mismatch_rows", 0):
        failures.append("future target lead mismatch")
    if temporal.get("station_history", {}).get("history_after_issue_rows", 0):
        failures.append("station history contains rows after issue time")
    if not tables["samples"]["rows"] or not tables["future_targets"]["rows"]:
        failures.append("empty core dataset tables")
    result = {
        "format": "corsewind.saphir_style_sequence_dataset_audit.v1",
        "generated_at_utc": utc_now(),
        "dataset_root": str(root),
        "verdict": "fail" if failures else "pass",
        "failures": failures,
        "tables": {
            name: {key: value for key, value in item.items() if key != "column_names"}
            for name, item in tables.items()
        },
        "temporal_checks": temporal,
        "coverage": coverage(root, deps),
        "baseline_metrics": baseline_metrics(future, deps),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args)
    output_json = args.output_json or args.dataset_root / "dataset_audit.json"
    output_md = args.output_md or args.dataset_root / "dataset_audit.md"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_markdown(output_md, result)
    print(json.dumps({
        "dataset_root": result["dataset_root"],
        "verdict": result["verdict"],
        "tables": result["tables"],
        "failures": result["failures"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
