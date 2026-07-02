#!/usr/bin/env python3
"""Run and summarize multiple collector-backed pseudo-live hindcasts."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


KT_PER_MS = 1.9438444924406
DEFAULT_SCORE_SPOTS = "cap_corse,la_parata,lfkf,lfkj,lfks,lfvf,lfvh"
OBSERVATION_DATASETS = ("dpobs_station_infrahoraire_6m", "dpobs_station_horaire")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def parse_case(value: str) -> dict[str, str]:
    parts = [part.strip() for part in value.split("|")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--case must use run_id|run_time_utc|issue_time_utc|target_end_utc"
        )
    return {
        "run_id": parts[0],
        "run_time_utc": parts[1],
        "issue_time_utc": parts[2],
        "target_end_utc": parts[3],
    }


def load_cases(args: argparse.Namespace) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    cases.extend(args.case)
    if args.cases_json:
        payload = read_json(args.cases_json)
        items = payload.get("cases") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise SystemExit("--cases-json must contain a list or {'cases': [...]}.")
        for item in items:
            cases.append(
                {
                    "run_id": str(item["run_id"]),
                    "run_time_utc": str(item["run_time_utc"]),
                    "issue_time_utc": str(item["issue_time_utc"]),
                    "target_end_utc": str(item["target_end_utc"]),
                }
            )
    if not cases:
        raise SystemExit("Provide at least one --case or --cases-json.")
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for case in cases:
        if case["run_id"] in seen:
            raise SystemExit(f"Duplicate case run_id: {case['run_id']}")
        seen.add(case["run_id"])
        out.append(case)
    return out


def observation_paths_for_case(case: dict[str, str], ml_root: Path) -> list[Path]:
    issue = parse_time(case["issue_time_utc"])
    target_end = parse_time(case["target_end_utc"])
    start = issue + timedelta(minutes=15)
    dates: list[str] = []
    current = start.date()
    while current <= target_end.date():
        dates.append(current.isoformat())
        current += timedelta(days=1)
    paths: list[Path] = []
    for date in dates:
        for dataset in OBSERVATION_DATASETS:
            path = (
                ml_root
                / "observations"
                / "meteo_france"
                / f"source_dataset={dataset}"
                / f"date={date}"
                / "observations.jsonl"
            )
            if path.exists():
                paths.append(path)
    return paths


def run_command(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    print(f"\n$ {shlex.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def build_hindcast_command(
    *,
    args: argparse.Namespace,
    case: dict[str, str],
    case_root: Path,
    observation_paths: list[Path],
) -> list[str]:
    cmd = [
        args.python,
        "scripts/ml_dataset/run_aromepi_hindcast_evaluation.py",
        "--repo-root",
        str(args.repo_root),
        "--ml-root",
        str(args.ml_root),
        "--run-id",
        case["run_id"],
        "--source",
        args.source,
        "--run-time-utc",
        case["run_time_utc"],
        "--issue-time-utc",
        case["issue_time_utc"],
        "--target-end-utc",
        case["target_end_utc"],
        "--output-root",
        str(case_root),
        "--registry",
        str(args.registry),
        "--context-registry",
        str(args.context_registry),
        "--spot-static-features",
        str(args.spot_static_features),
        "--score-spots",
        args.score_spots,
        "--python",
        args.python,
        "--lead-minutes",
        args.lead_minutes,
        "--step-minutes",
        str(args.step_minutes),
        "--read-margin-days-before",
        str(args.read_margin_days_before),
        "--observation-history-max-age-minutes",
        str(args.observation_history_max_age_minutes),
        "--context-station-max-age-minutes",
        str(args.context_station_max_age_minutes),
    ]
    for path in observation_paths:
        cmd.extend(["--observations-jsonl", str(path)])
    if args.with_foundation:
        cmd.append("--with-foundation")
        if args.allow_empty_observed_context:
            cmd.append("--allow-empty-observed-context")
        for path in args.history_parquet:
            cmd.extend(["--history-parquet", path])
        for pattern in args.history_observations_jsonl:
            cmd.extend(["--history-observations-jsonl", pattern])
    return cmd


def build_apply_shadow_command(
    *,
    args: argparse.Namespace,
    case_root: Path,
) -> list[str]:
    if args.shadow_artifact is None:
        raise ValueError("shadow_artifact is required")
    input_parquet = case_root / "predictions" / "predictions.parquet"
    output_parquet = case_root / "predictions" / "predictions_with_shadow_router_v1.parquet"
    output_json = case_root / "predictions" / "shadow_router_v1_summary.json"
    cmd = [
        args.python,
        "scripts/ml_dataset/apply_hindcast_router_v1_artifact.py",
        "--input-parquet",
        str(input_parquet),
        "--artifact",
        str(args.shadow_artifact),
        "--output-parquet",
        str(output_parquet),
        "--output-json",
        str(output_json),
    ]
    if args.shadow_allow_missing_features:
        cmd.append("--allow-missing-features")
    return cmd


def build_shadow_score_command(
    *,
    args: argparse.Namespace,
    case: dict[str, str],
    case_root: Path,
    observation_paths: list[Path],
) -> list[str]:
    cmd = [
        args.python,
        "scripts/ml_dataset/score_live_hindcast_predictions.py",
        "--predictions-parquet",
        str(case_root / "predictions" / "predictions_with_shadow_router_v1.parquet"),
        "--output-json",
        str(case_root / "hindcast_score_with_shadow_router_v1.json"),
        "--output-scored-parquet",
        str(case_root / "hindcast_scored_rows_with_shadow_router_v1.parquet"),
        "--spots",
        args.score_spots,
        "--target-start-utc",
        iso_z(parse_time(case["issue_time_utc"]) + timedelta(minutes=15)),
        "--target-end-utc",
        case["target_end_utc"],
    ]
    for path in observation_paths:
        cmd.extend(["--observations-jsonl", str(path)])
    return cmd


def build_threshold_guard_command(*, args: argparse.Namespace, case_root: Path) -> list[str]:
    return [
        args.python,
        "scripts/ml_dataset/apply_threshold_guard_v1.py",
        "--input-parquet",
        str(case_root / "predictions" / "predictions_with_shadow_router_v1.parquet"),
        "--output-parquet",
        str(case_root / "predictions" / "predictions_with_threshold_guard_v1.parquet"),
        "--output-json",
        str(case_root / "predictions" / "threshold_guard_v1_summary.json"),
    ]


def build_threshold_guard_score_command(
    *,
    args: argparse.Namespace,
    case: dict[str, str],
    case_root: Path,
    observation_paths: list[Path],
) -> list[str]:
    cmd = [
        args.python,
        "scripts/ml_dataset/score_live_hindcast_predictions.py",
        "--predictions-parquet",
        str(case_root / "predictions" / "predictions_with_threshold_guard_v1.parquet"),
        "--output-json",
        str(case_root / "hindcast_score_with_threshold_guard_v1.json"),
        "--output-scored-parquet",
        str(case_root / "hindcast_scored_rows_with_threshold_guard_v1.parquet"),
        "--spots",
        args.score_spots,
        "--target-start-utc",
        iso_z(parse_time(case["issue_time_utc"]) + timedelta(minutes=15)),
        "--target-end-utc",
        case["target_end_utc"],
    ]
    for path in observation_paths:
        cmd.extend(["--observations-jsonl", str(path)])
    return cmd


def local_fallback_risk_audit(args: argparse.Namespace) -> Path | None:
    if args.local_fallback_risk_audit:
        return args.local_fallback_risk_audit if args.local_fallback_risk_audit.exists() else None
    candidate = (
        args.ml_root
        / "live_inference"
        / "shadow_rollups"
        / "shadow_rollup_latest"
        / "threshold_guard_impact_audit.json"
    )
    return candidate if candidate.exists() else None


def build_local_fallback_guard_command(*, args: argparse.Namespace, case_root: Path, risk_audit: Path) -> list[str]:
    return [
        args.python,
        "scripts/ml_dataset/apply_local_fallback_guard_v1.py",
        "--input-parquet",
        str(case_root / "predictions" / "predictions_with_threshold_guard_v1.parquet"),
        "--risk-audit-json",
        str(risk_audit),
        "--output-parquet",
        str(case_root / "predictions" / "predictions_with_local_fallback_guard_v1.parquet"),
        "--output-json",
        str(case_root / "predictions" / "local_fallback_guard_v1_summary.json"),
    ]


def build_local_fallback_guard_score_command(
    *,
    args: argparse.Namespace,
    case: dict[str, str],
    case_root: Path,
    observation_paths: list[Path],
) -> list[str]:
    cmd = [
        args.python,
        "scripts/ml_dataset/score_live_hindcast_predictions.py",
        "--predictions-parquet",
        str(case_root / "predictions" / "predictions_with_local_fallback_guard_v1.parquet"),
        "--output-json",
        str(case_root / "hindcast_score_with_local_fallback_guard_v1.json"),
        "--output-scored-parquet",
        str(case_root / "hindcast_scored_rows_with_local_fallback_guard_v1.parquet"),
        "--spots",
        args.score_spots,
        "--target-start-utc",
        iso_z(parse_time(case["issue_time_utc"]) + timedelta(minutes=15)),
        "--target-end-utc",
        case["target_end_utc"],
    ]
    for path in observation_paths:
        cmd.extend(["--observations-jsonl", str(path)])
    return cmd


def metric_ms(score: dict[str, Any], metric_name: str) -> dict[str, Any]:
    metric = (score.get("overall") or {}).get(metric_name) or {}
    if not metric or not metric.get("n"):
        return {"n": 0}
    out = {"n": int(metric["n"])}
    for key in ("rmse", "mae", "bias", "p50_abs_error", "p90_abs_error"):
        if metric.get(key) is not None:
            out[f"{key}_ms"] = float(metric[key]) / KT_PER_MS
    return out


def nested_metric_ms(group: dict[str, Any], metric_name: str) -> dict[str, Any]:
    metric = group.get(metric_name) or {}
    if not metric or not metric.get("n"):
        return {"n": 0}
    out = {"n": int(metric["n"])}
    for key in ("rmse", "mae", "bias"):
        if metric.get(key) is not None:
            out[f"{key}_ms"] = float(metric[key]) / KT_PER_MS
    return out


def improvement_ms(score: dict[str, Any], model_metric: str, raw_metric: str) -> float | None:
    model = metric_ms(score, model_metric)
    raw = metric_ms(score, raw_metric)
    if "rmse_ms" not in model or "rmse_ms" not in raw:
        return None
    return raw["rmse_ms"] - model["rmse_ms"]


def best_overall(score: dict[str, Any], metric_names: list[str]) -> dict[str, Any]:
    candidates = []
    for name in metric_names:
        item = metric_ms(score, name)
        if item.get("n") and item.get("rmse_ms") is not None:
            candidates.append({"name": name, **item})
    if not candidates:
        return {}
    return min(candidates, key=lambda item: item["rmse_ms"])


def threshold_item(score: dict[str, Any], name: str) -> dict[str, Any]:
    item = (score.get("thresholds") or {}).get(name) or {}
    return {
        "n": item.get("n", 0),
        "csi": item.get("csi"),
        "precision": item.get("precision"),
        "recall": item.get("recall"),
        "tp": item.get("tp"),
        "fp": item.get("fp"),
        "fn": item.get("fn"),
    }


def case_summary(case: dict[str, str], case_root: Path) -> dict[str, Any]:
    score_path = case_root / "hindcast_score.json"
    score = read_json(score_path)
    out = {
        "run_id": case["run_id"],
        "run_time_utc": case["run_time_utc"],
        "issue_time_utc": case["issue_time_utc"],
        "target_end_utc": case["target_end_utc"],
        "output_root": str(case_root),
        "score_json": str(score_path),
        "joined_rows": score.get("joined_rows"),
        "spot_count": score.get("spot_count"),
        "target_start_utc": score.get("target_start_utc"),
        "target_end_scored_utc": score.get("target_end_utc"),
        "overall_ms": {
            "wind_raw": metric_ms(score, "wind_raw_kt"),
            "wind_champion": metric_ms(score, "wind_ml_kt"),
            "wind_strong_gated": metric_ms(score, "wind_strong_gated_kt"),
            "gust_raw": metric_ms(score, "gust_raw_kt"),
            "gust_champion": metric_ms(score, "gust_ml_kt"),
            "gust_high": metric_ms(score, "gust_high_kt"),
            "gust_strong_gated": metric_ms(score, "gust_strong_gated_kt"),
        },
        "rmse_gain_vs_raw_ms": {
            "wind_champion": improvement_ms(score, "wind_ml_kt", "wind_raw_kt"),
            "wind_strong_gated": improvement_ms(score, "wind_strong_gated_kt", "wind_raw_kt"),
            "gust_champion": improvement_ms(score, "gust_ml_kt", "gust_raw_kt"),
            "gust_high": improvement_ms(score, "gust_high_kt", "gust_raw_kt"),
            "gust_strong_gated": improvement_ms(score, "gust_strong_gated_kt", "gust_raw_kt"),
        },
        "best": {
            "wind": best_overall(score, ["wind_raw_kt", "wind_ml_kt", "wind_strong_gated_kt"]),
            "gust": best_overall(score, ["gust_raw_kt", "gust_ml_kt", "gust_high_kt", "gust_strong_gated_kt"]),
        },
        "thresholds": {
            "wind_15kt_raw": threshold_item(score, "wind_15kt_raw"),
            "wind_15kt_champion": threshold_item(score, "wind_15kt_ml"),
            "wind_15kt_strong_gated": threshold_item(score, "wind_15kt_strong_gated"),
            "gust_20kt_raw": threshold_item(score, "gust_20kt_raw"),
            "gust_20kt_champion": threshold_item(score, "gust_20kt_ml"),
            "gust_20kt_high": threshold_item(score, "gust_20kt_high"),
            "gust_20kt_strong_gated": threshold_item(score, "gust_20kt_strong_gated"),
            "gust_25kt_raw": threshold_item(score, "gust_25kt_raw"),
            "gust_25kt_champion": threshold_item(score, "gust_25kt_ml"),
            "gust_25kt_high": threshold_item(score, "gust_25kt_high"),
            "gust_25kt_strong_gated": threshold_item(score, "gust_25kt_strong_gated"),
        },
        "actual_wind_regime_ms": {
            regime: {
                "n": group.get("n"),
                "raw": nested_metric_ms(group, "wind_raw_kt"),
                "champion": nested_metric_ms(group, "wind_ml_kt"),
                "strong_gated": nested_metric_ms(group, "wind_strong_gated_kt"),
            }
            for regime, group in (score.get("by_actual_wind_regime_kt") or {}).items()
        },
        "actual_gust_regime_ms": {
            regime: {
                "n": group.get("n"),
                "raw": nested_metric_ms(group, "gust_raw_kt"),
                "champion": nested_metric_ms(group, "gust_ml_kt"),
                "high": nested_metric_ms(group, "gust_high_kt"),
                "strong_gated": nested_metric_ms(group, "gust_strong_gated_kt"),
            }
            for regime, group in (score.get("by_actual_gust_regime_kt") or {}).items()
        },
    }
    shadow_score_path = case_root / "hindcast_score_with_shadow_router_v1.json"
    if shadow_score_path.exists():
        shadow_score = read_json(shadow_score_path)
        out["shadow_router_v1"] = {
            "score_json": str(shadow_score_path),
            "overall_ms": {
                "wind_router": metric_ms(shadow_score, "wind_shadow_router_v1_kt"),
                "wind_stacker": metric_ms(shadow_score, "wind_shadow_stacker_v1_kt"),
                "wind_guarded_stacker": metric_ms(shadow_score, "wind_shadow_guarded_stacker_v1_kt"),
                "gust_router": metric_ms(shadow_score, "gust_shadow_router_v1_kt"),
                "gust_stacker": metric_ms(shadow_score, "gust_shadow_stacker_v1_kt"),
                "gust_guarded_stacker": metric_ms(shadow_score, "gust_shadow_guarded_stacker_v1_kt"),
            },
            "thresholds": {
                "wind_15kt_router": threshold_item(shadow_score, "wind_15kt_shadow_router_v1"),
                "wind_15kt_stacker": threshold_item(shadow_score, "wind_15kt_shadow_stacker_v1"),
                "wind_15kt_guarded_stacker": threshold_item(shadow_score, "wind_15kt_shadow_guarded_stacker_v1"),
                "gust_20kt_router": threshold_item(shadow_score, "gust_20kt_shadow_router_v1"),
                "gust_20kt_stacker": threshold_item(shadow_score, "gust_20kt_shadow_stacker_v1"),
                "gust_20kt_guarded_stacker": threshold_item(shadow_score, "gust_20kt_shadow_guarded_stacker_v1"),
                "gust_25kt_router": threshold_item(shadow_score, "gust_25kt_shadow_router_v1"),
                "gust_25kt_stacker": threshold_item(shadow_score, "gust_25kt_shadow_stacker_v1"),
                "gust_25kt_guarded_stacker": threshold_item(shadow_score, "gust_25kt_shadow_guarded_stacker_v1"),
            },
        }
    local_fallback_score_path = case_root / "hindcast_score_with_local_fallback_guard_v1.json"
    if local_fallback_score_path.exists():
        local_fallback_score = read_json(local_fallback_score_path)
        shadow = out.setdefault("shadow_router_v1", {"score_json": str(local_fallback_score_path)})
        shadow.setdefault("overall_ms", {})
        shadow["overall_ms"]["gust_local_fallback_guard"] = metric_ms(
            local_fallback_score, "gust_local_fallback_guard_v1_kt"
        )
        shadow.setdefault("thresholds", {})
        for level in (20, 25):
            shadow["thresholds"][f"gust_{level}kt_local_fallback_guard"] = threshold_item(
                local_fallback_score, f"gust_{level}kt_local_fallback_guard_v1"
            )
    return out


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def metric_value(item: dict[str, Any], key: str = "rmse_ms") -> Any:
    return item.get(key) if item else None


def render_markdown(summary: dict[str, Any]) -> str:
    has_shadow = any(case.get("shadow_router_v1") for case in summary["cases"])
    lines = [
        "# Collector Hindcast Suite",
        "",
        f"- generated: `{summary['generated_at_utc']}`",
        f"- cases: `{len(summary['cases'])}`",
        f"- output root: `{summary['output_root']}`",
        f"- shadow artifact: `{summary.get('shadow_artifact') or ''}`",
        "",
        "## Overall RMSE",
        "",
        "| Case | Rows | Wind raw | Wind champion | Wind strong | Gust raw | Gust champion | Gust high | Gust strong |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["cases"]:
        overall = case["overall_ms"]
        lines.append(
            "| "
            + " | ".join(
                [
                    case["run_id"],
                    fmt(case.get("joined_rows"), 0),
                    fmt(metric_value(overall["wind_raw"])),
                    fmt(metric_value(overall["wind_champion"])),
                    fmt(metric_value(overall["wind_strong_gated"])),
                    fmt(metric_value(overall["gust_raw"])),
                    fmt(metric_value(overall["gust_champion"])),
                    fmt(metric_value(overall["gust_high"])),
                    fmt(metric_value(overall["gust_strong_gated"])),
                ]
            )
            + " |"
        )
    if has_shadow:
        lines.extend(
            [
                "",
                "## Shadow Router v1 RMSE",
                "",
                "These values are in m/s. In-sample shadow runs only prove plumbing; promotion still requires unseen fresh-day validation.",
                "",
                "| Case | Wind router | Wind stacker | Wind guarded | Gust router | Gust stacker | Gust guarded |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for case in summary["cases"]:
            shadow = (case.get("shadow_router_v1") or {}).get("overall_ms") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        case["run_id"],
                        fmt(metric_value(shadow.get("wind_router") or {})),
                        fmt(metric_value(shadow.get("wind_stacker") or {})),
                        fmt(metric_value(shadow.get("wind_guarded_stacker") or {})),
                        fmt(metric_value(shadow.get("gust_router") or {})),
                        fmt(metric_value(shadow.get("gust_stacker") or {})),
                        fmt(metric_value(shadow.get("gust_guarded_stacker") or {})),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## RMSE Gain Versus Raw",
            "",
            "Positive means the candidate beats raw NWP on RMSE.",
            "",
            "| Case | Wind champion | Wind strong | Gust champion | Gust high | Gust strong |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in summary["cases"]:
        gain = case["rmse_gain_vs_raw_ms"]
        lines.append(
            "| "
            + " | ".join(
                [
                    case["run_id"],
                    fmt(gain["wind_champion"]),
                    fmt(gain["wind_strong_gated"]),
                    fmt(gain["gust_champion"]),
                    fmt(gain["gust_high"]),
                    fmt(gain["gust_strong_gated"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Threshold CSI",
            "",
            "| Case | Wind >=15 raw | Wind >=15 strong | Gust >=20 raw | Gust >=20 strong | Gust >=25 raw | Gust >=25 high | Gust >=25 strong |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in summary["cases"]:
        thresholds = case["thresholds"]
        lines.append(
            "| "
            + " | ".join(
                [
                    case["run_id"],
                    fmt(thresholds["wind_15kt_raw"].get("csi")),
                    fmt(thresholds["wind_15kt_strong_gated"].get("csi")),
                    fmt(thresholds["gust_20kt_raw"].get("csi")),
                    fmt(thresholds["gust_20kt_strong_gated"].get("csi")),
                    fmt(thresholds["gust_25kt_raw"].get("csi")),
                    fmt(thresholds["gust_25kt_high"].get("csi")),
                    fmt(thresholds["gust_25kt_strong_gated"].get("csi")),
                ]
            )
            + " |"
        )
    if has_shadow:
        lines.extend(
            [
                "",
                "## Shadow Router v1 Threshold CSI",
                "",
                "| Case | Wind >=15 router | Wind >=15 stacker | Wind >=15 guarded | Gust >=20 router | Gust >=20 stacker | Gust >=20 guarded | Gust >=25 router | Gust >=25 stacker | Gust >=25 guarded |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for case in summary["cases"]:
            thresholds = (case.get("shadow_router_v1") or {}).get("thresholds") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        case["run_id"],
                        fmt((thresholds.get("wind_15kt_router") or {}).get("csi")),
                        fmt((thresholds.get("wind_15kt_stacker") or {}).get("csi")),
                        fmt((thresholds.get("wind_15kt_guarded_stacker") or {}).get("csi")),
                        fmt((thresholds.get("gust_20kt_router") or {}).get("csi")),
                        fmt((thresholds.get("gust_20kt_stacker") or {}).get("csi")),
                        fmt((thresholds.get("gust_20kt_guarded_stacker") or {}).get("csi")),
                        fmt((thresholds.get("gust_25kt_router") or {}).get("csi")),
                        fmt((thresholds.get("gust_25kt_stacker") or {}).get("csi")),
                        fmt((thresholds.get("gust_25kt_guarded_stacker") or {}).get("csi")),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- Use this suite as the oracle before promoting any live champion.",
            "- A candidate should beat raw/champion on average and avoid large regressions in calm regimes.",
            "- For strong-wind work, prioritize `wind >=15 kt`, `gust >=20 kt`, and especially `gust >=25 kt` CSI/recall.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    cases = load_cases(args)
    output_root.mkdir(parents=True, exist_ok=True)

    case_outputs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in cases:
        case_root = output_root / case["run_id"]
        score_path = case_root / "hindcast_score.json"
        observation_paths = observation_paths_for_case(case, args.ml_root)
        if not observation_paths:
            message = f"No Meteo-France observation files found for case {case['run_id']}"
            if args.continue_on_error:
                failures.append({"run_id": case["run_id"], "error": message})
                continue
            raise SystemExit(message)
        if not (args.reuse_existing and score_path.exists()):
            cmd = build_hindcast_command(
                args=args,
                case=case,
                case_root=case_root,
                observation_paths=observation_paths,
            )
            try:
                run_command(cmd, cwd=repo_root, dry_run=args.dry_run)
            except subprocess.CalledProcessError as exc:
                if args.continue_on_error:
                    failures.append({"run_id": case["run_id"], "error": str(exc)})
                    continue
                raise
        if args.shadow_artifact and not args.dry_run:
            shadow_score_path = case_root / "hindcast_score_with_shadow_router_v1.json"
            threshold_score_path = case_root / "hindcast_score_with_threshold_guard_v1.json"
            local_fallback_score_path = case_root / "hindcast_score_with_local_fallback_guard_v1.json"
            risk_audit = local_fallback_risk_audit(args)
            if not (
                args.reuse_existing
                and shadow_score_path.exists()
                and threshold_score_path.exists()
                and (risk_audit is None or local_fallback_score_path.exists())
            ):
                try:
                    if not (args.reuse_existing and shadow_score_path.exists()):
                        run_command(build_apply_shadow_command(args=args, case_root=case_root), cwd=repo_root, dry_run=args.dry_run)
                        run_command(
                            build_shadow_score_command(
                                args=args,
                                case=case,
                                case_root=case_root,
                                observation_paths=observation_paths,
                            ),
                            cwd=repo_root,
                            dry_run=args.dry_run,
                        )
                    if not (args.reuse_existing and threshold_score_path.exists()):
                        run_command(build_threshold_guard_command(args=args, case_root=case_root), cwd=repo_root, dry_run=args.dry_run)
                        run_command(
                            build_threshold_guard_score_command(
                                args=args,
                                case=case,
                                case_root=case_root,
                                observation_paths=observation_paths,
                            ),
                            cwd=repo_root,
                            dry_run=args.dry_run,
                        )
                    if risk_audit is not None and not (args.reuse_existing and local_fallback_score_path.exists()):
                        run_command(
                            build_local_fallback_guard_command(args=args, case_root=case_root, risk_audit=risk_audit),
                            cwd=repo_root,
                            dry_run=args.dry_run,
                        )
                        run_command(
                            build_local_fallback_guard_score_command(
                                args=args,
                                case=case,
                                case_root=case_root,
                                observation_paths=observation_paths,
                            ),
                            cwd=repo_root,
                            dry_run=args.dry_run,
                        )
                except subprocess.CalledProcessError as exc:
                    if args.continue_on_error:
                        failures.append({"run_id": case["run_id"], "error": f"shadow_router_threshold_or_local_fallback_v1: {exc}"})
                    else:
                        raise
        if args.dry_run:
            case_outputs.append(
                {
                    "run_id": case["run_id"],
                    "output_root": str(case_root),
                    "observations_jsonl": [str(path) for path in observation_paths],
                    "dry_run": True,
                }
            )
        else:
            case_outputs.append(case_summary(case, case_root))

    summary = {
        "format": "corsewind.collector_hindcast_suite.v1",
        "generated_at_utc": utc_now(),
        "repo_root": str(repo_root),
        "ml_root": str(args.ml_root),
        "output_root": str(output_root),
        "source": args.source,
        "score_spots": args.score_spots,
        "shadow_artifact": None if args.shadow_artifact is None else str(args.shadow_artifact),
        "cases": case_outputs,
        "failures": failures,
    }
    write_json(output_root / "suite_summary.json", summary)
    if not args.dry_run:
        (output_root / "suite_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ml-root", type=Path, default=Path("/srv/data/corsewind/ml_dataset"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case", type=parse_case, action="append", default=[])
    parser.add_argument("--cases-json", type=Path)
    parser.add_argument("--source", default="aromepi")
    parser.add_argument("--registry", type=Path, default=Path("configs/ml_spots.json"))
    parser.add_argument("--context-registry", type=Path, default=Path("configs/ml_context_stations.json"))
    parser.add_argument("--spot-static-features", type=Path, default=Path("configs/ml_spot_static_features.json"))
    parser.add_argument("--score-spots", default=DEFAULT_SCORE_SPOTS)
    parser.add_argument("--lead-minutes", default="15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345,360,375,390,405,420,435,450,465,480,495,510,525,540,555,570,585,600,615")
    parser.add_argument("--step-minutes", type=int, default=15)
    parser.add_argument("--read-margin-days-before", type=int, default=5)
    parser.add_argument("--observation-history-max-age-minutes", type=float, default=360.0)
    parser.add_argument("--context-station-max-age-minutes", type=float, default=360.0)
    parser.add_argument("--with-foundation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--history-parquet", action="append", default=[])
    parser.add_argument("--history-observations-jsonl", action="append", default=[])
    parser.add_argument("--allow-empty-observed-context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shadow-artifact", type=Path)
    parser.add_argument("--shadow-allow-missing-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local-fallback-risk-audit", type=Path)
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
