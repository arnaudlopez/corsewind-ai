#!/usr/bin/env python3
"""Apply production-safe foundation blends on same-key prediction tables."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_WIND_OUTPUT = "guarded_foundation_wind_mean_ms"
DEFAULT_GUST_OUTPUT = "guarded_foundation_gust_ms"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def metric(frame: Any, prediction_column: str, actual_column: str, np: Any) -> dict[str, Any]:
    if prediction_column not in frame.columns or actual_column not in frame.columns:
        return {"count": 0}
    valid = frame[[prediction_column, actual_column]].dropna()
    if valid.empty:
        return {"count": 0}
    errors = valid[prediction_column].to_numpy(dtype=float) - valid[actual_column].to_numpy(dtype=float)
    return {
        "count": int(len(errors)),
        "rmse": round(float(math.sqrt(float(np.mean(errors * errors)))), 6),
        "mae": round(float(np.mean(np.abs(errors))), 6),
        "bias": round(float(np.mean(errors)), 6),
    }


def gain_pct(champion: dict[str, Any], candidate: dict[str, Any]) -> float | None:
    champion_rmse = champion.get("rmse")
    candidate_rmse = candidate.get("rmse")
    if champion_rmse in (None, 0) or candidate_rmse is None:
        return None
    return round(float(100.0 * (champion_rmse - candidate_rmse) / champion_rmse), 6)


def json_scalar(pd: Any, value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def grouped_metrics(frame: Any, prediction_column: str, actual_column: str, group_column: str, np: Any, pd: Any) -> list[dict[str, Any]]:
    if group_column not in frame.columns:
        return []
    rows = []
    for raw_key, group in frame.groupby(group_column, dropna=False):
        rows.append({
            "group": json_scalar(pd, raw_key),
            **metric(group, prediction_column, actual_column, np),
        })
    rows.sort(key=lambda item: (item.get("rmse") is None, item.get("rmse", -1)), reverse=True)
    return rows


def add_blend(
    frame: Any,
    *,
    champion_column: str,
    expert_column: str,
    output_column: str,
    alpha: float,
    cap_delta_ms: float | None,
    pd: Any,
) -> Any:
    if champion_column not in frame.columns:
        raise SystemExit(f"Missing champion column: {champion_column}")
    if expert_column not in frame.columns:
        raise SystemExit(f"Missing foundation expert column: {expert_column}")

    champion = pd.to_numeric(frame[champion_column], errors="coerce")
    expert = pd.to_numeric(frame[expert_column], errors="coerce")
    raw_delta = alpha * (expert - champion)
    capped_delta = raw_delta
    if cap_delta_ms is not None:
        capped_delta = raw_delta.clip(lower=-cap_delta_ms, upper=cap_delta_ms)

    used = champion.notna() & expert.notna()
    output = champion + capped_delta
    output = output.where(used, champion)

    prefix = output_column.removesuffix("_ms")
    frame[output_column] = output
    frame[f"{prefix}_raw_delta_ms"] = raw_delta
    frame[f"{prefix}_delta_ms"] = output - champion
    frame[f"{prefix}_expert_available"] = expert.notna()
    frame[f"{prefix}_used_foundation"] = used & raw_delta.notna() & (output != champion)
    frame[f"{prefix}_delta_was_capped"] = used & raw_delta.notna() & (capped_delta != raw_delta)
    return frame


def summarize_target(
    frame: Any,
    *,
    target_name: str,
    champion_column: str,
    output_column: str,
    actual_column: str | None,
    np: Any,
    pd: Any,
) -> dict[str, Any]:
    prefix = output_column.removesuffix("_ms")
    summary: dict[str, Any] = {
        "target": target_name,
        "champion_column": champion_column,
        "output_column": output_column,
        "expert_available_count": int(frame[f"{prefix}_expert_available"].sum()) if f"{prefix}_expert_available" in frame.columns else None,
        "used_foundation_count": int(frame[f"{prefix}_used_foundation"].sum()) if f"{prefix}_used_foundation" in frame.columns else None,
        "capped_delta_count": int(frame[f"{prefix}_delta_was_capped"].sum()) if f"{prefix}_delta_was_capped" in frame.columns else None,
    }
    if actual_column and actual_column in frame.columns:
        champion_metric = metric(frame, champion_column, actual_column, np)
        blend_metric = metric(frame, output_column, actual_column, np)
        summary.update({
            "actual_column": actual_column,
            "champion": champion_metric,
            "guarded_blend": blend_metric,
            "gain_pct_vs_champion": gain_pct(champion_metric, blend_metric),
            "by_lead": grouped_metrics(frame, output_column, actual_column, "lead_time_minutes", np, pd),
            "by_spot": grouped_metrics(frame, output_column, actual_column, "spot_id", np, pd),
        })
    return summary


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Foundation Blend Guardrail Result",
        "",
        f"Generated at UTC: `{result['generated_at_utc']}`",
        "",
        f"Input: `{result['input_parquet']}`",
        "",
        f"Output: `{result['output_parquet']}`",
        "",
        "| Target | Champion RMSE | Guarded RMSE | Guarded MAE | Bias | Gain | Used foundation | Capped deltas |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target in result["targets"]:
        champion = target.get("champion", {})
        guarded = target.get("guarded_blend", {})
        gain = target.get("gain_pct_vs_champion")
        lines.append(
            "| {target} | {champion_rmse} | {guarded_rmse} | {guarded_mae} | {guarded_bias} | {gain} | {used} | {capped} |".format(
                target=target["target"],
                champion_rmse=champion.get("rmse", ""),
                guarded_rmse=guarded.get("rmse", ""),
                guarded_mae=guarded.get("mae", ""),
                guarded_bias=guarded.get("bias", ""),
                gain="" if gain is None else f"{gain}%",
                used=target.get("used_foundation_count", ""),
                capped=target.get("capped_delta_count", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    frame = pd.read_parquet(args.input_parquet)
    if args.enable_wind:
        frame = add_blend(
            frame,
            champion_column=args.wind_champion_column,
            expert_column=args.wind_expert_column,
            output_column=args.wind_output_column,
            alpha=args.wind_alpha,
            cap_delta_ms=args.wind_cap_delta_ms,
            pd=pd,
        )
    if args.enable_gust:
        frame = add_blend(
            frame,
            champion_column=args.gust_champion_column,
            expert_column=args.gust_expert_column,
            output_column=args.gust_output_column,
            alpha=args.gust_alpha,
            cap_delta_ms=args.gust_cap_delta_ms,
            pd=pd,
        )

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_parquet, index=False, compression=args.compression)

    targets = []
    if args.enable_wind:
        targets.append(summarize_target(
            frame,
            target_name="wind_mean",
            champion_column=args.wind_champion_column,
            output_column=args.wind_output_column,
            actual_column=args.wind_actual_column,
            np=np,
            pd=pd,
        ))
    if args.enable_gust:
        targets.append(summarize_target(
            frame,
            target_name="gust",
            champion_column=args.gust_champion_column,
            output_column=args.gust_output_column,
            actual_column=args.gust_actual_column,
            np=np,
            pd=pd,
        ))

    result = {
        "format": "corsewind.foundation_blend_guardrails.v1",
        "generated_at_utc": utc_now(),
        "input_parquet": str(args.input_parquet),
        "output_parquet": str(args.output_parquet),
        "compression": args.compression,
        "config": {
            "wind": {
                "enabled": args.enable_wind,
                "alpha": args.wind_alpha,
                "cap_delta_ms": args.wind_cap_delta_ms,
                "champion_column": args.wind_champion_column,
                "expert_column": args.wind_expert_column,
                "output_column": args.wind_output_column,
            },
            "gust": {
                "enabled": args.enable_gust,
                "alpha": args.gust_alpha,
                "cap_delta_ms": args.gust_cap_delta_ms,
                "champion_column": args.gust_champion_column,
                "expert_column": args.gust_expert_column,
                "output_column": args.gust_output_column,
            },
        },
        "row_count": int(len(frame)),
        "targets": targets,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_markdown, result)
    print(json.dumps({
        "row_count": result["row_count"],
        "output_parquet": result["output_parquet"],
        "targets": [
            {
                "target": target["target"],
                "gain_pct_vs_champion": target.get("gain_pct_vs_champion"),
                "guarded_rmse": target.get("guarded_blend", {}).get("rmse"),
            }
            for target in targets
        ],
    }, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--compression", default="zstd")

    parser.add_argument("--enable-wind", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wind-champion-column", default="wind_champion_prediction_ms")
    parser.add_argument("--wind-expert-column", default="chronos2_univar_wind_mean_ms_mean")
    parser.add_argument("--wind-actual-column", default="actual_wind_mean_ms")
    parser.add_argument("--wind-output-column", default=DEFAULT_WIND_OUTPUT)
    parser.add_argument("--wind-alpha", type=float, default=0.10)
    parser.add_argument("--wind-cap-delta-ms", type=float, default=0.50)

    parser.add_argument("--enable-gust", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gust-champion-column", default="gust_champion_prediction_ms")
    parser.add_argument("--gust-expert-column", default="timesfm_gust_ms_mean")
    parser.add_argument("--gust-actual-column", default="actual_gust_ms")
    parser.add_argument("--gust-output-column", default=DEFAULT_GUST_OUTPUT)
    parser.add_argument("--gust-alpha", type=float, default=0.10)
    parser.add_argument("--gust-cap-delta-ms", type=float, default=0.25)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
