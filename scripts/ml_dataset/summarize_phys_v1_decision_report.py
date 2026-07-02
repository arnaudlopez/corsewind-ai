#!/usr/bin/env python3
"""Build a compact decision report for the phys_v1 benchmark iteration."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def metric(prediction: Any, observation: Any, np: Any) -> dict[str, Any]:
    valid = ~(np.isnan(prediction) | np.isnan(observation))
    prediction = prediction[valid]
    observation = observation[valid]
    if len(prediction) == 0:
        return {"count": 0}
    error = prediction - observation
    return {
        "count": int(len(error)),
        "mae": round(float(np.mean(np.abs(error))), 6),
        "rmse": round(float(math.sqrt(float(np.mean(error * error)))), 6),
        "bias": round(float(np.mean(error)), 6),
        "p90_abs_error": round(float(np.quantile(np.abs(error), 0.90)), 6),
    }


def metric_frame(frame: Any, prediction_col: str, observation_col: str, np: Any) -> dict[str, Any]:
    valid = frame[[prediction_col, observation_col]].dropna()
    return metric(
        valid[prediction_col].astype(float).to_numpy(),
        valid[observation_col].astype(float).to_numpy(),
        np,
    )


def grouped_metrics(
    frame: Any,
    group_columns: list[str],
    prediction_col: str,
    observation_col: str,
    np: Any,
    pd: Any,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if any(column not in frame.columns for column in group_columns):
        return []
    rows = []
    valid = frame.dropna(subset=[prediction_col, observation_col])
    for raw_key, group in valid.groupby(group_columns, dropna=False):
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        rows.append({
            "group": {
                column: json_scalar(pd, value)
                for column, value in zip(group_columns, values, strict=True)
            },
            **metric_frame(group, prediction_col, observation_col, np),
        })
    rows.sort(key=lambda item: (item.get("rmse") is None, item.get("rmse", -1)), reverse=True)
    return rows[:limit]


def json_scalar(pd: Any, value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def delta(reference: float | None, value: float | None) -> float | None:
    if reference is None or value is None:
        return None
    return round(value - reference, 6)


def gain_pct(reference: float | None, value: float | None) -> float | None:
    if reference is None or value is None or reference == 0:
        return None
    return round((reference - value) / reference * 100.0, 3)


def compact_metric_block(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metrics:
        return None
    return {
        "count": as_int(metrics.get("count")),
        "rmse": as_float(metrics.get("rmse")),
        "mae": as_float(metrics.get("mae")),
        "bias": as_float(metrics.get("bias")),
        "p90_abs_error": as_float(metrics.get("p90_abs_error")),
    }


def feature_audit_summary(feature_audit: dict[str, Any] | None) -> dict[str, Any]:
    if not feature_audit:
        return {"available": False}
    return {
        "available": True,
        "verdict": feature_audit.get("verdict"),
        "existing_shard_count": feature_audit.get("existing_shard_count"),
        "shard_count": feature_audit.get("shard_count"),
        "stale_shard_count": feature_audit.get("stale_shard_count"),
        "required_patterns": feature_audit.get("required_patterns") or [],
        "reasons": feature_audit.get("reasons") or [],
    }


def signal_coverage_summary(signal_coverage: dict[str, Any] | None) -> dict[str, Any]:
    if not signal_coverage:
        return {"available": False}
    columns = signal_coverage.get("columns") or []
    key_columns = []
    for item in columns:
        key_columns.append({
            "column": item.get("column"),
            "present": item.get("present"),
            "coverage_pct": item.get("coverage_pct"),
            "nonnull": item.get("nonnull"),
        })
    key_columns.sort(key=lambda item: (item.get("coverage_pct") is None, item.get("column") or ""))
    return {
        "available": True,
        "row_count": signal_coverage.get("row_count"),
        "columns": key_columns,
    }


def calibration_summary(calibration: dict[str, Any] | None) -> dict[str, Any]:
    if not calibration:
        return {"available": False}
    return {
        "available": True,
        "verdict": calibration.get("verdict"),
        "model_family": calibration.get("model_family"),
        "lead_minutes": calibration.get("lead_minutes"),
        "calibration_row_count": calibration.get("calibration_row_count"),
        "evaluation_row_count": calibration.get("evaluation_row_count"),
        "feature_column_count": calibration.get("feature_column_count"),
        "base_metrics": compact_metric_block(calibration.get("base_metrics")),
        "calibrated_metrics": compact_metric_block(calibration.get("calibrated_metrics")),
        "rmse_gain_pct_vs_base": calibration.get("rmse_gain_pct_vs_base"),
        "rmse_gap_to_threshold": calibration.get("rmse_gap_to_threshold"),
        "selected_scale": ((calibration.get("scale_selection") or {}).get("selected_scale")),
        "calibrated_by_lead": calibration.get("calibrated_by_lead") or [],
        "calibrated_worst_spots": calibration.get("calibrated_worst_spots") or [],
        "calibrated_worst_spot_leads": calibration.get("calibrated_worst_spot_leads") or [],
    }


def base_audit_summary(base_audit: dict[str, Any] | None) -> dict[str, Any]:
    if not base_audit:
        return {"available": False}
    return {
        "available": True,
        "verdict": base_audit.get("verdict"),
        "run_id": base_audit.get("run_id"),
        "corrected_rmse": as_float(base_audit.get("corrected_rmse")),
        "corrected_mae": as_float(base_audit.get("corrected_mae")),
        "raw_rmse": as_float(base_audit.get("raw_rmse")),
        "raw_mae": as_float(base_audit.get("raw_mae")),
        "metric_count": base_audit.get("metric_count"),
        "feature_column_count": base_audit.get("feature_column_count"),
        "by_lead": base_audit.get("by_lead") or {},
        "worst_spots": base_audit.get("worst_spots") or [],
        "worst_spot_leads": base_audit.get("worst_spot_leads") or [],
        "warnings": base_audit.get("warnings") or [],
        "reasons": base_audit.get("reasons") or [],
    }


def prediction_diagnostics(path: Path | None, limit: int) -> dict[str, Any]:
    if not path:
        return {"available": False}
    if not path.exists():
        return {"available": False, "path": str(path), "reason": "missing predictions parquet"}
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        return {"available": False, "path": str(path), "reason": f"missing dependency: {exc}"}

    frame = pd.read_parquet(path)
    prediction_col = "calibrated_wind_mean_ms" if "calibrated_wind_mean_ms" in frame.columns else "corrected_wind_mean_ms"
    required = {prediction_col, "actual_wind_mean_ms"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return {"available": False, "path": str(path), "reason": f"missing columns: {missing}"}

    out = frame.copy()
    if "issue_time_utc" in out.columns:
        issue_time = pd.to_datetime(out["issue_time_utc"], utc=True, errors="coerce")
        out["issue_month"] = issue_time.dt.strftime("%Y-%m")
        out["target_hour_local_approx"] = ((issue_time.dt.hour + 2) % 24).astype("Int64")
    out["actual_wind_bin_ms"] = pd.cut(
        out["actual_wind_mean_ms"].astype(float),
        bins=[-0.001, 2.0, 4.0, 6.0, 8.0, 10.0, 999.0],
        labels=["0-2", "2-4", "4-6", "6-8", "8-10", "10+"],
    ).astype(str)
    thermal_candidates = [
        "features__thermal_air_minus_sst_c",
        "features__thermal_land_minus_sst_c",
        "features__thermal_inland_minus_coastal_temperature_c",
    ]
    thermal_column = next((column for column in thermal_candidates if column in out.columns), None)
    if thermal_column:
        values = pd.to_numeric(out[thermal_column], errors="coerce")
        if int(values.notna().sum()) >= 100 and int(values.dropna().nunique()) >= 4:
            out["thermal_signal_quartile"] = pd.qcut(values, q=4, duplicates="drop").astype(str)

    group_specs = {
        "by_lead": ["lead_time_minutes"],
        "worst_spots": ["spot_id"],
        "worst_spot_leads": ["spot_id", "lead_time_minutes"],
        "by_actual_wind_bin": ["actual_wind_bin_ms"],
        "by_month": ["issue_month"],
        "by_target_hour_local_approx": ["target_hour_local_approx"],
    }
    if "thermal_signal_quartile" in out.columns:
        group_specs["by_thermal_signal_quartile"] = ["thermal_signal_quartile"]

    result = {
        "available": True,
        "path": str(path),
        "prediction_column": prediction_col,
        "row_count": int(len(out)),
        "overall": metric_frame(out, prediction_col, "actual_wind_mean_ms", np),
    }
    for key, columns in group_specs.items():
        result[key] = grouped_metrics(out, columns, prediction_col, "actual_wind_mean_ms", np, pd, limit=limit)
    return result


def build_decision(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    calibrated = ((result.get("calibration") or {}).get("calibrated_metrics") or {}).get("rmse")
    base = (result.get("base_audit") or {}).get("corrected_rmse")
    best = min(value for value in (calibrated, base) if value is not None) if any(
        value is not None for value in (calibrated, base)
    ) else None
    feature_verdict = (result.get("feature_audit") or {}).get("verdict")
    reasons = []
    if best is None:
        return {
            "status": "incomplete",
            "reasons": ["Missing base/calibrated RMSE metrics."],
            "next_action": "Wait for benchmark artifacts or inspect watcher logs.",
        }
    if feature_verdict not in {None, "pass"}:
        reasons.append("Feature audit is not passing; do not compare RMSE until feature coverage is fixed.")
    if best < args.threshold_rmse:
        reasons.append(f"Best phys_v1 RMSE {best:.6f} is below target {args.threshold_rmse:.6f}.")
        return {
            "status": "target_achieved_candidate",
            "reasons": reasons,
            "next_action": "Run the formal assertion gate and inspect leakage/audit evidence before promotion.",
        }
    champion_delta = delta(args.current_champion_rmse, best)
    if champion_delta is not None and champion_delta <= -args.min_rmse_improvement:
        reasons.append(
            f"phys_v1 improves champion RMSE by {-champion_delta:.6f}, above minimum {args.min_rmse_improvement:.6f}."
        )
        return {
            "status": "promote_candidate",
            "reasons": reasons,
            "next_action": "Review spot/horizon regressions and promote only if critical spots do not regress.",
        }
    if champion_delta is not None and champion_delta < 0:
        reasons.append(f"phys_v1 improves champion RMSE by {-champion_delta:.6f}, but below promotion margin.")
        return {
            "status": "small_improvement",
            "reasons": reasons,
            "next_action": "Inspect hard regimes; keep as evidence for phys_v2 rather than promote automatically.",
        }
    reasons.append(f"Best phys_v1 RMSE {best:.6f} does not beat champion {args.current_champion_rmse:.6f}.")
    return {
        "status": "not_improved",
        "reasons": reasons,
        "next_action": "Use the diagnostics to identify whether vertical/offset signals are missing, noisy, or useful only by regime.",
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    base = base_audit_summary(load_json(args.base_audit))
    calibration = calibration_summary(load_json(args.calibration_results))
    feature_audit = feature_audit_summary(load_json(args.feature_audit))
    signal_coverage = signal_coverage_summary(load_json(args.signal_coverage))
    predictions = prediction_diagnostics(args.predictions, args.group_limit)
    calibrated_rmse = ((calibration.get("calibrated_metrics") or {}).get("rmse") if calibration.get("available") else None)
    base_rmse = base.get("corrected_rmse") if base.get("available") else None
    best = min(value for value in (base_rmse, calibrated_rmse) if value is not None) if any(
        value is not None for value in (base_rmse, calibrated_rmse)
    ) else None
    result = {
        "format": "corsewind.phys_v1_decision_report.v1",
        "generated_at_utc": utc_now(),
        "run_suffix": args.run_suffix,
        "threshold_rmse": args.threshold_rmse,
        "current_champion_rmse": args.current_champion_rmse,
        "current_champion_mae": args.current_champion_mae,
        "artifacts": {
            "base_audit": str(args.base_audit) if args.base_audit else None,
            "calibration_results": str(args.calibration_results) if args.calibration_results else None,
            "feature_audit": str(args.feature_audit) if args.feature_audit else None,
            "signal_coverage": str(args.signal_coverage) if args.signal_coverage else None,
            "predictions": str(args.predictions) if args.predictions else None,
        },
        "summary_metrics": {
            "base_rmse": base_rmse,
            "base_mae": base.get("corrected_mae") if base.get("available") else None,
            "calibrated_rmse": calibrated_rmse,
            "calibrated_mae": ((calibration.get("calibrated_metrics") or {}).get("mae") if calibration.get("available") else None),
            "best_phys_v1_rmse": best,
            "delta_vs_champion_rmse": delta(args.current_champion_rmse, best),
            "gain_pct_vs_champion_rmse": gain_pct(args.current_champion_rmse, best),
            "gap_to_target_rmse": delta(args.threshold_rmse, best),
        },
        "base_audit": base,
        "calibration": calibration,
        "feature_audit": feature_audit,
        "signal_coverage": signal_coverage,
        "prediction_diagnostics": predictions,
    }
    result["decision"] = build_decision(args, result)
    return result


def table_group_label(group: dict[str, Any] | None) -> str:
    if not group:
        return ""
    return " / ".join(f"{key}={value}" for key, value in group.items())


def write_group_table(lines: list[str], title: str, rows: list[dict[str, Any]], *, limit: int = 12) -> None:
    lines.extend(["", f"## {title}", "", "| Group | Count | RMSE | MAE | Bias | P90 abs |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    if not rows:
        lines.append("| None |  |  |  |  |  |")
        return
    for item in rows[:limit]:
        lines.append(
            f"| `{table_group_label(item.get('group'))}` | {item.get('count')} | {item.get('rmse')} | "
            f"{item.get('mae')} | {item.get('bias')} | {item.get('p90_abs_error')} |"
        )


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    metrics = result["summary_metrics"]
    decision = result["decision"]
    calibration = result["calibration"]
    base = result["base_audit"]
    coverage = result["signal_coverage"]
    diagnostics = result["prediction_diagnostics"]
    lines = [
        "# Phys V1 Decision Report",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Decision: `{decision['status']}`",
        "",
        "## Headline",
        "",
        f"- Champion RMSE: `{result['current_champion_rmse']}`",
        f"- Base phys_v1 RMSE: `{metrics['base_rmse']}`",
        f"- Calibrated phys_v1 RMSE: `{metrics['calibrated_rmse']}`",
        f"- Best phys_v1 RMSE: `{metrics['best_phys_v1_rmse']}`",
        f"- Delta vs champion: `{metrics['delta_vs_champion_rmse']}`",
        f"- Gain vs champion: `{metrics['gain_pct_vs_champion_rmse']}%`",
        f"- Gap to target `{result['threshold_rmse']}`: `{metrics['gap_to_target_rmse']}`",
        "",
        "## Decision Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in decision.get("reasons") or ["None."])
    lines.extend(["", "## Next Action", "", decision.get("next_action") or "None."])
    lines.extend([
        "",
        "## Feature/Audit Gates",
        "",
        f"- Feature audit available: `{result['feature_audit'].get('available')}`",
        f"- Feature audit verdict: `{result['feature_audit'].get('verdict')}`",
        f"- Signal coverage available: `{coverage.get('available')}`",
        f"- Signal coverage rows: `{coverage.get('row_count')}`",
        f"- Base audit verdict: `{base.get('verdict')}`",
        f"- Calibration verdict: `{calibration.get('verdict')}`",
        f"- Calibration selected scale: `{calibration.get('selected_scale')}`",
        f"- Calibration rows: `{calibration.get('calibration_row_count')}`",
        f"- Evaluation rows: `{calibration.get('evaluation_row_count')}`",
        f"- Feature columns: `{calibration.get('feature_column_count')}`",
    ])
    if coverage.get("columns"):
        lines.extend(["", "## Physical Signal Coverage", "", "| Column | Present | Non-null | Coverage |", "| --- | ---: | ---: | ---: |"])
        for item in coverage["columns"]:
            lines.append(
                f"| `{item.get('column')}` | `{item.get('present')}` | {item.get('nonnull')} | {item.get('coverage_pct')}% |"
            )
    if calibration.get("calibrated_by_lead"):
        write_group_table(lines, "Calibrated By Lead", calibration["calibrated_by_lead"])
    if calibration.get("calibrated_worst_spots"):
        write_group_table(lines, "Worst Calibrated Spots", calibration["calibrated_worst_spots"])
    if calibration.get("calibrated_worst_spot_leads"):
        write_group_table(lines, "Worst Calibrated Spot/Lead", calibration["calibrated_worst_spot_leads"])
    if diagnostics.get("available"):
        lines.extend(["", "## Prediction Diagnostics", "", f"- Prediction file rows: `{diagnostics.get('row_count')}`", f"- Overall RMSE: `{(diagnostics.get('overall') or {}).get('rmse')}`"])
        for title, key in [
            ("Diagnostics By Actual Wind Bin", "by_actual_wind_bin"),
            ("Diagnostics By Local Target Hour Approx", "by_target_hour_local_approx"),
            ("Diagnostics By Thermal Signal", "by_thermal_signal_quartile"),
        ]:
            if diagnostics.get(key):
                write_group_table(lines, title, diagnostics[key])
    else:
        lines.extend(["", "## Prediction Diagnostics", "", f"- Unavailable: `{diagnostics.get('reason')}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-suffix", default="phys_v1")
    parser.add_argument("--base-audit", type=Path)
    parser.add_argument("--calibration-results", type=Path)
    parser.add_argument("--feature-audit", type=Path)
    parser.add_argument("--signal-coverage", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--threshold-rmse", type=float, default=0.9)
    parser.add_argument("--current-champion-rmse", type=float, default=1.268019)
    parser.add_argument("--current-champion-mae", type=float, default=0.930465)
    parser.add_argument("--min-rmse-improvement", type=float, default=0.005)
    parser.add_argument("--group-limit", type=int, default=20)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


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
        "decision": result["decision"]["status"],
        "best_phys_v1_rmse": result["summary_metrics"]["best_phys_v1_rmse"],
        "delta_vs_champion_rmse": result["summary_metrics"]["delta_vs_champion_rmse"],
        "gap_to_target_rmse": result["summary_metrics"]["gap_to_target_rmse"],
        "next_action": result["decision"]["next_action"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
