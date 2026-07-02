#!/usr/bin/env python3
"""Summarize CorseWind ML pipeline status without launching heavy work."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PREFIXES = [
    "residual_windsup_sst_prev_phys_v1",
    "residual_windsup_sst_prev_phys_v2_dem",
    "residual_windsup_sst_prev_phys_v3_dem_fetch",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_text_tail(path: Path, limit: int = 4000) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def import_pyarrow() -> Any | None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    return pq


def process_matches(needles: list[str]) -> list[dict[str, Any]]:
    proc = Path("/proc")
    if not proc.exists():
        return []
    rows = []
    self_pid = os.getpid()
    for child in proc.iterdir():
        if not child.name.isdigit():
            continue
        pid = int(child.name)
        if pid == self_pid:
            continue
        try:
            raw = (child / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore").strip()
        except OSError:
            continue
        if raw and any(needle in raw for needle in needles):
            rows.append({"pid": pid, "cmdline": raw[:1000]})
    return rows


def disk_summary(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_gb": round(usage.total / 1024**3, 3),
        "used_gb": round(usage.used / 1024**3, 3),
        "free_gb": round(usage.free / 1024**3, 3),
        "used_pct": round(usage.used / usage.total * 100.0, 3) if usage.total else None,
    }


def shard_summary(ml_root: Path, prefix: str, pq: Any | None) -> dict[str, Any]:
    root = ml_root / "training_tables"
    paths = sorted(root.glob(f"{prefix}_*/training_rows.parquet"))
    rows = []
    total_rows = 0
    total_bytes = 0
    for path in paths:
        item = {
            "run_id": path.parent.name,
            "path": str(path),
            "size_mb": round(path.stat().st_size / 1024**2, 3),
            "rows": None,
        }
        total_bytes += path.stat().st_size
        if pq is not None:
            try:
                item["rows"] = int(pq.ParquetFile(path).metadata.num_rows)
                total_rows += item["rows"]
            except Exception as exc:
                item["row_error"] = str(exc)
        rows.append(item)
    return {
        "prefix": prefix,
        "shard_count": len(paths),
        "total_rows": total_rows if pq is not None else None,
        "total_size_mb": round(total_bytes / 1024**2, 3),
        "latest_shard": rows[-1] if rows else None,
        "shards_tail": rows[-6:],
    }


def status_files(ml_root: Path) -> dict[str, Any]:
    log_root = ml_root / "run_logs"
    names = [
        "rebuild_phys_v1_2024_2026.status",
        "phys_v1_signal_audit_watcher.status",
        "phys_v1_post_rebuild_lowmem.status",
        "phys_v1_decision_report_watcher.status",
        "phys_v1_sequence_benchmarks.status",
        "rebuild_phys_v2_dem_2024_2026.status",
        "phys_v2_dem_signal_audit_watcher.status",
        "phys_v2_dem_post_rebuild_lowmem.status",
    ]
    return {
        name: {
            "path": str(log_root / name),
            "exists": (log_root / name).exists(),
            "tail": read_text_tail(log_root / name, 1000),
        }
        for name in names
    }


def artifact_summary(ml_root: Path) -> dict[str, Any]:
    paths = {
        "phys_v1_feature_audit": ml_root / "training_tables/phys_v1_required_feature_audit.json",
        "phys_v1_signal_coverage": ml_root / "training_tables/phys_v1_signal_coverage.json",
        "phys_v1_decision_report": ml_root / "benchmarks/phys_v1_decision_report.json",
        "phys_v2_dem_feature_audit": ml_root / "training_tables/phys_v2_dem_required_feature_audit.json",
    }
    out = {}
    for name, path in paths.items():
        payload = load_json(path)
        item = {"path": str(path), "exists": path.exists()}
        if payload:
            item["format"] = payload.get("format")
            item["verdict"] = payload.get("verdict") or (payload.get("decision") or {}).get("status")
            item["row_count"] = payload.get("row_count")
            item["summary_metrics"] = payload.get("summary_metrics")
        out[name] = item
    return out


def best_model_summary(ml_root: Path) -> dict[str, Any]:
    candidates = []
    for path in sorted((ml_root / "benchmarks").glob("**/calibration_results.json")):
        payload = load_json(path)
        if not payload:
            continue
        metrics = payload.get("calibrated_metrics") or {}
        rmse = metrics.get("rmse")
        if rmse is None:
            continue
        candidates.append({
            "run_id": path.parent.name,
            "path": str(path),
            "rmse": float(rmse),
            "mae": metrics.get("mae"),
            "count": metrics.get("count"),
            "verdict": payload.get("verdict"),
        })
    candidates.sort(key=lambda item: item["rmse"])
    return {"best": candidates[0] if candidates else None, "top": candidates[:10], "count": len(candidates)}


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    ml_root = args.ml_root
    pq = import_pyarrow()
    process_needles = [
        "run_monthly_training_shards.py",
        "run_training_backfill_pipeline.py",
        "collect_open_meteo_historical_forecast.py",
        "train_residual_correction_parquet.py",
        "train_prediction_residual_calibrator.py",
        "run_rmse09_sequence_experiment.py",
    ]
    return {
        "format": "corsewind.ml_pipeline_status.v1",
        "generated_at_utc": utc_now(),
        "ml_root": str(ml_root),
        "disk": disk_summary(args.disk_path or ml_root),
        "pyarrow_available": pq is not None,
        "processes": process_matches(process_needles),
        "statuses": status_files(ml_root),
        "shards": {prefix: shard_summary(ml_root, prefix, pq) for prefix in args.prefix},
        "artifacts": artifact_summary(ml_root),
        "best_model": best_model_summary(ml_root),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    disk = result["disk"]
    lines = [
        "# CorseWind ML Pipeline Status",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"ML root: `{result['ml_root']}`",
        "",
        "## Disk",
        "",
        f"- Path: `{disk['path']}`",
        f"- Free: `{disk['free_gb']} GB`",
        f"- Used: `{disk['used_gb']} GB` / `{disk['total_gb']} GB` (`{disk['used_pct']}%`)",
        "",
        "## Active Processes",
        "",
    ]
    if result["processes"]:
        for item in result["processes"]:
            lines.append(f"- `{item['pid']}` {item['cmdline']}")
    else:
        lines.append("- None detected.")
    lines.extend(["", "## Shards", "", "| Prefix | Shards | Rows | Size MB | Latest |", "| --- | ---: | ---: | ---: | --- |"])
    for prefix, item in result["shards"].items():
        latest = (item.get("latest_shard") or {}).get("run_id")
        lines.append(f"| `{prefix}` | {item['shard_count']} | {item['total_rows']} | {item['total_size_mb']} | `{latest}` |")
    lines.extend(["", "## Status Files", "", "| Name | Exists | Tail |", "| --- | ---: | --- |"])
    for name, item in result["statuses"].items():
        tail = (item.get("tail") or "").strip().replace("\n", "<br>")
        lines.append(f"| `{name}` | `{item['exists']}` | {tail} |")
    lines.extend(["", "## Artifacts", "", "| Artifact | Exists | Verdict | Rows |", "| --- | ---: | --- | ---: |"])
    for name, item in result["artifacts"].items():
        lines.append(f"| `{name}` | `{item['exists']}` | `{item.get('verdict')}` | `{item.get('row_count')}` |")
    best = result["best_model"].get("best")
    lines.extend(["", "## Best Model", ""])
    if best:
        lines.extend([
            f"- Run: `{best['run_id']}`",
            f"- RMSE: `{best['rmse']}`",
            f"- MAE: `{best.get('mae')}`",
            f"- Rows: `{best.get('count')}`",
        ])
    else:
        lines.append("- None detected.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-root", type=Path, default=Path("/srv/data/corsewind/ml_dataset"))
    parser.add_argument("--disk-path", type=Path)
    parser.add_argument("--prefix", action="append", default=[])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()
    if not args.prefix:
        args.prefix = list(DEFAULT_PREFIXES)
    return args


def main() -> None:
    args = parse_args()
    result = summarize(args)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, result)
    print(json.dumps({
        "disk_free_gb": result["disk"]["free_gb"],
        "active_process_count": len(result["processes"]),
        "shards": {prefix: item["shard_count"] for prefix, item in result["shards"].items()},
        "best_model": result["best_model"].get("best"),
    }, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
