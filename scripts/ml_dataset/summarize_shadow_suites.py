#!/usr/bin/env python3
"""Aggregate collector hindcast suite summaries across validation days."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MS_PER_KT = 1.0 / 1.9438444924406
WIND_THRESHOLDS_KT = (12, 15, 20, 25)
GUST_THRESHOLDS_KT = (12, 15, 20, 25)

METRIC_KEYS = {
    "wind_raw": ("overall_ms", "wind_raw"),
    "wind_champion": ("overall_ms", "wind_champion"),
    "wind_strong_gated": ("overall_ms", "wind_strong_gated"),
    "wind_router": ("shadow_router_v1", "overall_ms", "wind_router"),
    "wind_stacker": ("shadow_router_v1", "overall_ms", "wind_stacker"),
    "wind_guarded_stacker": ("shadow_router_v1", "overall_ms", "wind_guarded_stacker"),
    "wind_threshold_guard": ("shadow_router_v1", "overall_ms", "wind_threshold_guard"),
    "wind_high_event_guard": ("shadow_router_v1", "overall_ms", "wind_high_event_guard"),
    "wind_gust_floor_guard": ("shadow_router_v1", "overall_ms", "wind_gust_floor_guard"),
    "gust_raw": ("overall_ms", "gust_raw"),
    "gust_champion": ("overall_ms", "gust_champion"),
    "gust_quantile_q50": ("overall_ms", "gust_quantile_q50"),
    "gust_quantile_q60": ("overall_ms", "gust_quantile_q60"),
    "gust_quantile_q75": ("overall_ms", "gust_quantile_q75"),
    "gust_quantile_q90": ("overall_ms", "gust_quantile_q90"),
    "gust_high": ("overall_ms", "gust_high"),
    "gust_strong_gated": ("overall_ms", "gust_strong_gated"),
    "gust_router": ("shadow_router_v1", "overall_ms", "gust_router"),
    "gust_stacker": ("shadow_router_v1", "overall_ms", "gust_stacker"),
    "gust_guarded_stacker": ("shadow_router_v1", "overall_ms", "gust_guarded_stacker"),
    "gust_threshold_guard": ("shadow_router_v1", "overall_ms", "gust_threshold_guard"),
    "gust_local_fallback_guard": ("shadow_router_v1", "overall_ms", "gust_local_fallback_guard"),
    "gust_probability_event_guard": ("shadow_router_v1", "overall_ms", "gust_probability_event_guard"),
    "gust_recall_floor_guard": ("shadow_router_v1", "overall_ms", "gust_recall_floor_guard"),
}

def build_threshold_keys() -> dict[str, tuple[str, ...]]:
    keys: dict[str, tuple[str, ...]] = {}
    wind_rails = {
        "raw": ("thresholds", "wind_{level}kt_raw"),
        "champion": ("thresholds", "wind_{level}kt_ml"),
        "strong_gated": ("thresholds", "wind_{level}kt_strong_gated"),
        "router": ("shadow_router_v1", "thresholds", "wind_{level}kt_router"),
        "stacker": ("shadow_router_v1", "thresholds", "wind_{level}kt_stacker"),
        "guarded_stacker": ("shadow_router_v1", "thresholds", "wind_{level}kt_guarded_stacker"),
        "threshold_guard": ("shadow_router_v1", "thresholds", "wind_{level}kt_threshold_guard"),
        "high_event_guard": ("shadow_router_v1", "thresholds", "wind_{level}kt_high_event_guard"),
        "gust_floor_guard": ("shadow_router_v1", "thresholds", "wind_{level}kt_gust_floor_guard"),
    }
    gust_rails = {
        "raw": ("thresholds", "gust_{level}kt_raw"),
        "champion": ("thresholds", "gust_{level}kt_ml"),
        "quantile_q50": ("thresholds", "gust_{level}kt_quantile_q50"),
        "quantile_q60": ("thresholds", "gust_{level}kt_quantile_q60"),
        "quantile_q75": ("thresholds", "gust_{level}kt_quantile_q75"),
        "quantile_q90": ("thresholds", "gust_{level}kt_quantile_q90"),
        "high": ("thresholds", "gust_{level}kt_high"),
        "strong_gated": ("thresholds", "gust_{level}kt_strong_gated"),
        "router": ("shadow_router_v1", "thresholds", "gust_{level}kt_router"),
        "stacker": ("shadow_router_v1", "thresholds", "gust_{level}kt_stacker"),
        "guarded_stacker": ("shadow_router_v1", "thresholds", "gust_{level}kt_guarded_stacker"),
        "threshold_guard": ("shadow_router_v1", "thresholds", "gust_{level}kt_threshold_guard"),
        "local_fallback_guard": ("shadow_router_v1", "thresholds", "gust_{level}kt_local_fallback_guard"),
        "probability_event_guard": ("shadow_router_v1", "thresholds", "gust_{level}kt_probability_event_guard"),
        "recall_floor_guard": ("shadow_router_v1", "thresholds", "gust_{level}kt_recall_floor_guard"),
    }
    for level in WIND_THRESHOLDS_KT:
        for rail, path_template in wind_rails.items():
            keys[f"wind_{level}kt_{rail}"] = tuple(part.format(level=level) for part in path_template)
    for level in GUST_THRESHOLDS_KT:
        for rail, path_template in gust_rails.items():
            keys[f"gust_{level}kt_{rail}"] = tuple(part.format(level=level) for part in path_template)
    return keys


THRESHOLD_KEYS = build_threshold_keys()

SCORE_METRIC_KEYS = {
    "wind_raw": "wind_raw_kt",
    "wind_champion": "wind_ml_kt",
    "wind_strong_gated": "wind_strong_gated_kt",
    "wind_router": "wind_shadow_router_v1_kt",
    "wind_stacker": "wind_shadow_stacker_v1_kt",
    "wind_guarded_stacker": "wind_shadow_guarded_stacker_v1_kt",
    "wind_threshold_guard": "wind_threshold_guard_v1_kt",
    "wind_high_event_guard": "wind_high_event_guard_v1_kt",
    "wind_gust_floor_guard": "wind_gust_floor_guard_v1_kt",
    "gust_raw": "gust_raw_kt",
    "gust_champion": "gust_ml_kt",
    "gust_quantile_q50": "gust_quantile_q50_kt",
    "gust_quantile_q60": "gust_quantile_q60_kt",
    "gust_quantile_q75": "gust_quantile_q75_kt",
    "gust_quantile_q90": "gust_quantile_q90_kt",
    "gust_high": "gust_high_kt",
    "gust_strong_gated": "gust_strong_gated_kt",
    "gust_router": "gust_shadow_router_v1_kt",
    "gust_stacker": "gust_shadow_stacker_v1_kt",
    "gust_guarded_stacker": "gust_shadow_guarded_stacker_v1_kt",
    "gust_threshold_guard": "gust_threshold_guard_v1_kt",
    "gust_local_fallback_guard": "gust_local_fallback_guard_v1_kt",
    "gust_probability_event_guard": "gust_probability_event_guard_v1_kt",
    "gust_recall_floor_guard": "gust_recall_floor_guard_v1_kt",
}

BASELINE_RAILS = ("raw", "champion")

CANDIDATE_RAILS = {
    "wind": (
        "strong_gated",
        "router",
        "stacker",
        "guarded_stacker",
        "threshold_guard",
        "high_event_guard",
        "gust_floor_guard",
    ),
    "gust": (
        "quantile_q50",
        "quantile_q60",
        "quantile_q75",
        "quantile_q90",
        "high",
        "strong_gated",
        "router",
        "stacker",
        "guarded_stacker",
        "threshold_guard",
        "local_fallback_guard",
        "probability_event_guard",
        "recall_floor_guard",
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def score_metric_ms(score: dict[str, Any], key: str) -> dict[str, Any]:
    metric = (score.get("overall") or {}).get(key) or {}
    if not metric:
        return {}
    return {
        "n": metric.get("n"),
        "rmse_ms": None if metric.get("rmse") is None else float(metric["rmse"]) * MS_PER_KT,
        "mae_ms": None if metric.get("mae") is None else float(metric["mae"]) * MS_PER_KT,
        "bias_ms": None if metric.get("bias") is None else float(metric["bias"]) * MS_PER_KT,
        "prediction_std_ms": None
        if metric.get("prediction_std") is None
        else float(metric["prediction_std"]) * MS_PER_KT,
        "observation_std_ms": None
        if metric.get("observation_std") is None
        else float(metric["observation_std"]) * MS_PER_KT,
        "variance_ratio": metric.get("variance_ratio"),
    }


def score_group_metric_ms(group: dict[str, Any], key: str) -> dict[str, Any]:
    metric = group.get(key) or {}
    if not metric:
        return {}
    return {
        "n": metric.get("n"),
        "rmse_ms": None if metric.get("rmse") is None else float(metric["rmse"]) * MS_PER_KT,
        "mae_ms": None if metric.get("mae") is None else float(metric["mae"]) * MS_PER_KT,
        "bias_ms": None if metric.get("bias") is None else float(metric["bias"]) * MS_PER_KT,
        "prediction_std_ms": None
        if metric.get("prediction_std") is None
        else float(metric["prediction_std"]) * MS_PER_KT,
        "observation_std_ms": None
        if metric.get("observation_std") is None
        else float(metric["observation_std"]) * MS_PER_KT,
        "variance_ratio": metric.get("variance_ratio"),
    }


def score_path_for_case(case: dict[str, Any]) -> Path | None:
    output_root = case.get("output_root")
    if not output_root:
        return None
    case_root = Path(str(output_root))
    candidates = [
        case_root / "hindcast_score_with_wind_gust_floor_guard_v1.json",
        case_root / "hindcast_score_with_gust_recall_floor_guard_v1.json",
        case_root / "hindcast_score_with_probability_event_guard_v1.json",
        case_root / "hindcast_score_with_local_fallback_guard_v1.json",
        case_root / "hindcast_score_with_threshold_guard_v1.json",
        case_root / "hindcast_score_with_guarded_stacker_v1.json",
        case_root / "hindcast_score_with_shadow_router_v1.json",
        case_root / "hindcast_score.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def augment_shadow_from_score_file(case: dict[str, Any]) -> None:
    score_path = score_path_for_case(case)
    if score_path is None:
        return
    score = read_json(score_path)
    shadow = case.setdefault("shadow_router_v1", {})
    shadow.setdefault("score_json", str(score_path))
    overall = shadow.setdefault("overall_ms", {})
    overall.update(
        {
            "wind_router": score_metric_ms(score, "wind_shadow_router_v1_kt"),
            "wind_stacker": score_metric_ms(score, "wind_shadow_stacker_v1_kt"),
            "wind_guarded_stacker": score_metric_ms(score, "wind_shadow_guarded_stacker_v1_kt"),
            "wind_threshold_guard": score_metric_ms(score, "wind_threshold_guard_v1_kt"),
            "wind_high_event_guard": score_metric_ms(score, "wind_high_event_guard_v1_kt"),
            "wind_gust_floor_guard": score_metric_ms(score, "wind_gust_floor_guard_v1_kt"),
            "gust_router": score_metric_ms(score, "gust_shadow_router_v1_kt"),
            "gust_quantile_q50": score_metric_ms(score, "gust_quantile_q50_kt"),
            "gust_quantile_q60": score_metric_ms(score, "gust_quantile_q60_kt"),
            "gust_quantile_q75": score_metric_ms(score, "gust_quantile_q75_kt"),
            "gust_quantile_q90": score_metric_ms(score, "gust_quantile_q90_kt"),
            "gust_stacker": score_metric_ms(score, "gust_shadow_stacker_v1_kt"),
            "gust_guarded_stacker": score_metric_ms(score, "gust_shadow_guarded_stacker_v1_kt"),
            "gust_threshold_guard": score_metric_ms(score, "gust_threshold_guard_v1_kt"),
            "gust_local_fallback_guard": score_metric_ms(score, "gust_local_fallback_guard_v1_kt"),
            "gust_probability_event_guard": score_metric_ms(score, "gust_probability_event_guard_v1_kt"),
            "gust_recall_floor_guard": score_metric_ms(score, "gust_recall_floor_guard_v1_kt"),
        }
    )
    thresholds = shadow.setdefault("thresholds", {})
    score_thresholds = score.get("thresholds") or {}
    case_thresholds = case.setdefault("thresholds", {})
    for level in WIND_THRESHOLDS_KT:
        case_thresholds[f"wind_{level}kt_raw"] = score_thresholds.get(f"wind_{level}kt_raw") or {}
        case_thresholds[f"wind_{level}kt_ml"] = score_thresholds.get(f"wind_{level}kt_ml") or {}
        case_thresholds[f"wind_{level}kt_strong_gated"] = score_thresholds.get(f"wind_{level}kt_strong_gated") or {}
    for level in GUST_THRESHOLDS_KT:
        case_thresholds[f"gust_{level}kt_raw"] = score_thresholds.get(f"gust_{level}kt_raw") or {}
        case_thresholds[f"gust_{level}kt_ml"] = score_thresholds.get(f"gust_{level}kt_ml") or {}
        case_thresholds[f"gust_{level}kt_quantile_q50"] = score_thresholds.get(f"gust_{level}kt_quantile_q50") or {}
        case_thresholds[f"gust_{level}kt_quantile_q60"] = score_thresholds.get(f"gust_{level}kt_quantile_q60") or {}
        case_thresholds[f"gust_{level}kt_quantile_q75"] = score_thresholds.get(f"gust_{level}kt_quantile_q75") or {}
        case_thresholds[f"gust_{level}kt_quantile_q90"] = score_thresholds.get(f"gust_{level}kt_quantile_q90") or {}
        case_thresholds[f"gust_{level}kt_high"] = score_thresholds.get(f"gust_{level}kt_high") or {}
        case_thresholds[f"gust_{level}kt_strong_gated"] = score_thresholds.get(f"gust_{level}kt_strong_gated") or {}
    for level in WIND_THRESHOLDS_KT:
        thresholds[f"wind_{level}kt_router"] = score_thresholds.get(f"wind_{level}kt_shadow_router_v1") or {}
        thresholds[f"wind_{level}kt_stacker"] = score_thresholds.get(f"wind_{level}kt_shadow_stacker_v1") or {}
        thresholds[f"wind_{level}kt_guarded_stacker"] = score_thresholds.get(
            f"wind_{level}kt_shadow_guarded_stacker_v1"
        ) or {}
        thresholds[f"wind_{level}kt_threshold_guard"] = score_thresholds.get(f"wind_{level}kt_threshold_guard_v1") or {}
        thresholds[f"wind_{level}kt_high_event_guard"] = score_thresholds.get(
            f"wind_{level}kt_wind_high_event_guard_v1"
        ) or {}
        thresholds[f"wind_{level}kt_gust_floor_guard"] = score_thresholds.get(
            f"wind_{level}kt_wind_gust_floor_guard_v1"
        ) or {}
    for level in GUST_THRESHOLDS_KT:
        thresholds[f"gust_{level}kt_router"] = score_thresholds.get(f"gust_{level}kt_shadow_router_v1") or {}
        thresholds[f"gust_{level}kt_stacker"] = score_thresholds.get(f"gust_{level}kt_shadow_stacker_v1") or {}
        thresholds[f"gust_{level}kt_guarded_stacker"] = score_thresholds.get(
            f"gust_{level}kt_shadow_guarded_stacker_v1"
        ) or {}
        thresholds[f"gust_{level}kt_threshold_guard"] = score_thresholds.get(f"gust_{level}kt_threshold_guard_v1") or {}
        thresholds[f"gust_{level}kt_local_fallback_guard"] = score_thresholds.get(
            f"gust_{level}kt_local_fallback_guard_v1"
        ) or {}
        thresholds[f"gust_{level}kt_probability_event_guard"] = score_thresholds.get(
            f"gust_{level}kt_probability_event_guard_v1"
        ) or {}
        thresholds[f"gust_{level}kt_recall_floor_guard"] = score_thresholds.get(
            f"gust_{level}kt_gust_recall_floor_guard_v1"
        ) or {}


def add_metric(acc: dict[str, dict[str, float]], name: str, metric: dict[str, Any] | None) -> None:
    if not metric:
        return
    n = int(metric.get("n") or 0)
    rmse = metric.get("rmse_ms")
    mae = metric.get("mae_ms")
    bias = metric.get("bias_ms")
    if n <= 0 or rmse is None:
        return
    item = acc.setdefault(
        name,
        {
            "n": 0.0,
            "sse": 0.0,
            "abs_sum": 0.0,
            "err_sum": 0.0,
            "pred_var_sum": 0.0,
            "obs_var_sum": 0.0,
        },
    )
    item["n"] += n
    item["sse"] += float(rmse) ** 2 * n
    if mae is not None:
        item["abs_sum"] += float(mae) * n
    if bias is not None:
        item["err_sum"] += float(bias) * n
    if metric.get("prediction_std_ms") is not None:
        item["pred_var_sum"] += float(metric["prediction_std_ms"]) ** 2 * n
    if metric.get("observation_std_ms") is not None:
        item["obs_var_sum"] += float(metric["observation_std_ms"]) ** 2 * n


def finish_metric(item: dict[str, float]) -> dict[str, Any]:
    n = int(item["n"])
    if n <= 0:
        return {"n": 0}
    return {
        "n": n,
        "rmse_ms": math.sqrt(item["sse"] / n),
        "mae_ms": item["abs_sum"] / n,
        "bias_ms": item["err_sum"] / n,
        "prediction_std_ms": math.sqrt(item["pred_var_sum"] / n) if item.get("pred_var_sum") else None,
        "observation_std_ms": math.sqrt(item["obs_var_sum"] / n) if item.get("obs_var_sum") else None,
        "variance_ratio": None
        if not item.get("obs_var_sum")
        else item.get("pred_var_sum", 0.0) / item["obs_var_sum"],
    }


def add_threshold(acc: dict[str, dict[str, int]], name: str, item: dict[str, Any] | None) -> None:
    if not item:
        return
    current = acc.setdefault(name, {"n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0})
    for key in ("n", "tp", "fp", "fn", "tn"):
        current[key] += int(item.get(key) or 0)


def finish_threshold(item: dict[str, int]) -> dict[str, Any]:
    tp = item["tp"]
    fp = item["fp"]
    fn = item["fn"]
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    csi = None if tp + fp + fn == 0 else tp / (tp + fp + fn)
    false_alarm_ratio = None if tp + fp == 0 else fp / (tp + fp)
    false_alarm_rate = None if fp + item["tn"] == 0 else fp / (fp + item["tn"])
    return {
        **item,
        "precision": precision,
        "recall": recall,
        "csi": csi,
        "false_alarm_ratio": false_alarm_ratio,
        "false_alarm_rate": false_alarm_rate,
    }


def add_score_regimes(
    acc: dict[str, dict[str, dict[str, dict[str, float]]]],
    *,
    score: dict[str, Any],
    score_key: str,
    output_key: str,
) -> None:
    groups = score.get(score_key) or {}
    for regime, group in groups.items():
        regime_acc = acc.setdefault(output_key, {}).setdefault(regime, {})
        for rail, metric_key in SCORE_METRIC_KEYS.items():
            add_metric(regime_acc, rail, score_group_metric_ms(group, metric_key))


def finish_regimes(acc: dict[str, dict[str, dict[str, dict[str, float]]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group_name, regimes in sorted(acc.items()):
        out[group_name] = {}
        for regime, rails in sorted(regimes.items()):
            out[group_name][regime] = {rail: finish_metric(item) for rail, item in sorted(rails.items())}
    return out


def metric_n(overall: dict[str, Any], target: str, rail: str) -> int:
    metric = overall.get(f"{target}_{rail}") or {}
    return int(metric.get("n") or 0)


def baseline_coverage_summary(overall: dict[str, Any]) -> dict[str, Any]:
    by_target: dict[str, Any] = {}
    for target, candidates in CANDIDATE_RAILS.items():
        baseline_counts = {
            baseline: metric_n(overall, target, baseline)
            for baseline in BASELINE_RAILS
        }
        candidate_items = {}
        failing_candidates = []
        for candidate in candidates:
            candidate_count = metric_n(overall, target, candidate)
            if candidate_count <= 0:
                continue
            baselines = {}
            pass_candidate = True
            for baseline, baseline_count in baseline_counts.items():
                missing_rows = max(0, candidate_count - baseline_count)
                coverage_ratio = None if candidate_count <= 0 else baseline_count / candidate_count
                baseline_pass = baseline_count >= candidate_count
                if not baseline_pass:
                    pass_candidate = False
                baselines[baseline] = {
                    "baseline_n": baseline_count,
                    "candidate_n": candidate_count,
                    "coverage_ratio": coverage_ratio,
                    "missing_rows_vs_candidate": missing_rows,
                    "pass": baseline_pass,
                }
            candidate_items[candidate] = {
                "candidate_n": candidate_count,
                "baselines": baselines,
                "pass": pass_candidate,
            }
            if not pass_candidate:
                failing_candidates.append(candidate)
        by_target[target] = {
            "required_baselines": list(BASELINE_RAILS),
            "baseline_counts": baseline_counts,
            "candidates": candidate_items,
            "failing_candidates": failing_candidates,
            "pass": not failing_candidates,
        }
    return by_target


def load_cases(paths: list[Path]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        for case in payload.get("cases") or []:
            augment_shadow_from_score_file(case)
            if case.get("score_json") or case.get("shadow_router_v1"):
                cases.append({"suite_summary": str(path), **case})
    return cases


def aggregate(paths: list[Path]) -> dict[str, Any]:
    cases = load_cases(paths)
    metric_acc: dict[str, dict[str, float]] = {}
    threshold_acc: dict[str, dict[str, int]] = {}
    regime_acc: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    case_rows = []
    for case in cases:
        case_rows.append(
            {
                "suite_summary": case["suite_summary"],
                "run_id": case.get("run_id"),
                "issue_time_utc": case.get("issue_time_utc"),
                "target_start_utc": case.get("target_start_utc"),
                "target_end_scored_utc": case.get("target_end_scored_utc"),
                "joined_rows": case.get("joined_rows"),
                "has_shadow": bool(case.get("shadow_router_v1")),
            }
        )
        for name, keys in METRIC_KEYS.items():
            add_metric(metric_acc, name, nested_get(case, keys))
        for name, keys in THRESHOLD_KEYS.items():
            add_threshold(threshold_acc, name, nested_get(case, keys))
        score_path = score_path_for_case(case)
        if score_path is not None:
            score = read_json(score_path)
            add_score_regimes(
                regime_acc,
                score=score,
                score_key="by_actual_wind_regime_kt",
                output_key="actual_wind",
            )
            add_score_regimes(
                regime_acc,
                score=score,
                score_key="by_actual_gust_regime_kt",
                output_key="actual_gust",
            )
    overall = {name: finish_metric(item) for name, item in sorted(metric_acc.items())}
    thresholds = {name: finish_threshold(item) for name, item in sorted(threshold_acc.items())}
    return {
        "format": "corsewind.shadow_suite_aggregate.v1",
        "generated_at_utc": utc_now(),
        "suite_summaries": [str(path) for path in paths],
        "case_count": len(cases),
        "shadow_case_count": sum(1 for case in cases if case.get("shadow_router_v1")),
        "joined_rows": sum(int(case.get("joined_rows") or 0) for case in cases),
        "cases": case_rows,
        "overall_ms": overall,
        "baseline_coverage": baseline_coverage_summary(overall),
        "thresholds": thresholds,
        "regimes_ms": finish_regimes(regime_acc),
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Shadow Suite Aggregate",
        "",
        f"- generated: `{summary['generated_at_utc']}`",
        f"- suite summaries: `{len(summary['suite_summaries'])}`",
        f"- scored cases: `{summary['case_count']}`",
        f"- shadow cases: `{summary['shadow_case_count']}`",
        f"- joined rows: `{summary['joined_rows']}`",
        "",
        "## Overall RMSE",
        "",
        "| Rail | n | RMSE m/s | MAE m/s | Bias m/s | Variance ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metric in summary["overall_ms"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    fmt(metric.get("n"), 0),
                    fmt(metric.get("rmse_ms")),
                    fmt(metric.get("mae_ms")),
                    fmt(metric.get("bias_ms")),
                    fmt(metric.get("variance_ratio")),
                ]
            )
            + " |"
        )
    baseline_coverage = summary.get("baseline_coverage") or {}
    if baseline_coverage:
        lines.extend(
            [
                "",
                "## Baseline Coverage",
                "",
                "| Target | Candidate | Candidate n | Raw n | Champion n | Pass |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for target, target_item in baseline_coverage.items():
            for candidate, item in (target_item.get("candidates") or {}).items():
                baselines = item.get("baselines") or {}
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{target}`",
                            f"`{candidate}`",
                            fmt(item.get("candidate_n"), 0),
                            fmt((baselines.get("raw") or {}).get("baseline_n"), 0),
                            fmt((baselines.get("champion") or {}).get("baseline_n"), 0),
                            f"`{item.get('pass')}`",
                        ]
                    )
                    + " |"
                )
    lines.extend([
        "",
        "## Threshold CSI",
        "",
        "| Rail | n | CSI | Precision | Recall | FAR | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name, item in summary["thresholds"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    fmt(item.get("n"), 0),
                    fmt(item.get("csi")),
                    fmt(item.get("precision")),
                    fmt(item.get("recall")),
                    fmt(item.get("false_alarm_ratio")),
                    fmt(item.get("tp"), 0),
                    fmt(item.get("fp"), 0),
                    fmt(item.get("fn"), 0),
                ]
            )
            + " |"
        )
    regimes = summary.get("regimes_ms") or {}
    for group_name in ("actual_wind", "actual_gust"):
        if not regimes.get(group_name):
            continue
        lines.extend(
            [
                "",
                f"## {group_name} Regimes",
                "",
                "| Regime | Rail | n | RMSE m/s | MAE m/s | Bias m/s |",
                "| Regime | Rail | n | RMSE m/s | MAE m/s | Bias m/s | Variance ratio |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for regime, rails in regimes[group_name].items():
            for rail, metric in rails.items():
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{regime}`",
                            f"`{rail}`",
                            fmt(metric.get("n"), 0),
                            fmt(metric.get("rmse_ms")),
                            fmt(metric.get("mae_ms")),
                            fmt(metric.get("bias_ms")),
                            fmt(metric.get("variance_ratio")),
                        ]
                    )
                    + " |"
                )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary = aggregate(args.suite_summary)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-summary", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
