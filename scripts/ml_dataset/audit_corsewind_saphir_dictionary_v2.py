#!/usr/bin/env python3
"""Audit a CorseWind SAPHIR dictionary V2 export."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def import_deps() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Missing pandas/numpy dependencies.") from exc
    return {"np": np, "pd": pd}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_parquet(root: Path, name: str, pd: Any) -> Any:
    path = root / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, details: dict[str, Any] | None = None) -> None:
    checks.append({"name": name, "ok": bool(ok), "details": details or {}})


def audit_dataset(root: Path, deps: dict[str, Any]) -> dict[str, Any]:
    pd = deps["pd"]
    np = deps["np"]
    checks: list[dict[str, Any]] = []
    profile_path = root / "dataset_profile.json"
    tensor_path = root / "dictionary_tensors.npz"
    profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else {}
    add_check(checks, "profile_exists", profile_path.exists(), {"path": str(profile_path)})
    add_check(checks, "tensor_file_exists", tensor_path.exists(), {"path": str(tensor_path)})

    samples = read_parquet(root, "samples", pd)
    future = read_parquet(root, "future_targets", pd)
    station = read_parquet(root, "station_sequence", pd)
    add_check(checks, "samples_not_empty", not samples.empty, {"rows": int(len(samples))})
    add_check(checks, "future_targets_not_empty", not future.empty, {"rows": int(len(future))})
    add_check(checks, "station_sequence_not_empty", not station.empty, {"rows": int(len(station))})

    if not samples.empty:
        add_check(checks, "sample_id_unique", samples["sample_id"].is_unique, {"rows": int(len(samples))})
        split_counts = samples["split"].astype(str).value_counts().to_dict() if "split" in samples.columns else {}
        add_check(checks, "has_train_and_eval", "train" in split_counts and any(k != "train" for k in split_counts), split_counts)

    if not future.empty:
        future_times = future.copy()
        future_times["issue_time"] = pd.to_datetime(future_times["issue_time_utc"], utc=True, errors="coerce")
        future_times["target_time"] = pd.to_datetime(future_times["target_time_utc"], utc=True, errors="coerce")
        valid_delta = future_times["target_time"] > future_times["issue_time"]
        delta_minutes = (future_times["target_time"] - future_times["issue_time"]).dt.total_seconds() / 60.0
        declared = pd.to_numeric(future_times["lead_time_minutes"], errors="coerce")
        add_check(checks, "future_targets_after_issue", bool(valid_delta.all()), {"bad_rows": int((~valid_delta).sum())})
        add_check(
            checks,
            "lead_minutes_match_target_delta",
            bool(np.allclose(delta_minutes.fillna(-1), declared.fillna(-2))),
            {"max_abs_delta": float(np.nanmax(np.abs(delta_minutes - declared))) if len(future_times) else None},
        )
        expected = set(profile.get("lead_minutes", []))
        if expected:
            counts = future.groupby("sample_id")["lead_time_minutes"].nunique()
            add_check(
                checks,
                "all_samples_have_all_leads",
                bool((counts == len(expected)).all()),
                {"expected_leads": sorted(expected), "bad_samples": int((counts != len(expected)).sum())},
            )

    if not station.empty:
        st = station.copy()
        st["issue_time"] = pd.to_datetime(st["issue_time_utc"], utc=True, errors="coerce")
        st["timestamp"] = pd.to_datetime(st["timestamp_utc"], utc=True, errors="coerce")
        no_future = st["timestamp"] <= st["issue_time"]
        add_check(checks, "station_sequence_not_future", bool(no_future.all()), {"bad_rows": int((~no_future).sum())})
        if "station_slot" in st.columns and "time_index" in st.columns:
            duplicate_keys = st.duplicated(["sample_id", "station_slot", "time_index"]).sum()
            add_check(checks, "station_sequence_unique_slot_time", duplicate_keys == 0, {"duplicates": int(duplicate_keys)})
        if not samples.empty:
            missing = set(samples["sample_id"].astype(str)) - set(st["sample_id"].astype(str))
            add_check(checks, "station_sequence_covers_all_samples", not missing, {"missing_samples": len(missing)})

    tensor_summary: dict[str, Any] = {}
    if tensor_path.exists():
        data = np.load(tensor_path, allow_pickle=True)
        tensor_summary = {key: list(value.shape) for key, value in data.items() if hasattr(value, "shape")}
        n_samples = len(samples) if not samples.empty else None
        if n_samples is not None:
            for key in ("station_tensor", "station_mask", "baseline_tensor", "y_actual", "y_residual", "static_tensor"):
                if key in data:
                    add_check(checks, f"{key}_sample_axis_matches", int(data[key].shape[0]) == n_samples, {"shape": list(data[key].shape), "samples": int(n_samples)})
        if "station_mask" in data:
            coverage = float(np.nanmean(data["station_mask"])) if data["station_mask"].size else 0.0
            add_check(checks, "station_mask_nonzero", coverage > 0.0, {"mean_mask": coverage})
        if "y_actual" in data:
            missing_targets = int(np.isnan(data["y_actual"]).sum())
            add_check(checks, "targets_no_nan", missing_targets == 0, {"nan_count": missing_targets})

    ok = all(check["ok"] for check in checks)
    return {
        "format": "corsewind.saphir_dictionary_v2_audit.v1",
        "generated_at_utc": utc_now(),
        "dataset_root": str(root),
        "ok": ok,
        "checks": checks,
        "tensor_summary": tensor_summary,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# CorseWind SAPHIR Dictionary V2 Audit",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Dataset root: `{result['dataset_root']}`",
        f"Verdict: `{'pass' if result['ok'] else 'fail'}`",
        "",
        "## Checks",
        "",
        "| Check | OK | Details |",
        "| --- | ---: | --- |",
    ]
    for check in result["checks"]:
        lines.append(f"| `{check['name']}` | `{check['ok']}` | `{json.dumps(check.get('details', {}), sort_keys=True)}` |")
    lines.extend(["", "## Tensor Summary", "", "```json", json.dumps(result.get("tensor_summary", {}), indent=2, sort_keys=True), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = import_deps()
    result = audit_dataset(args.dataset_root, deps)
    (args.dataset_root / "dataset_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_markdown(args.dataset_root / "dataset_audit.md", result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
