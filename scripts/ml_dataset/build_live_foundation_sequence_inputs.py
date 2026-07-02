#!/usr/bin/env python3
"""Build saved-sequence inputs for live Chronos/TimesFM shadow inference."""

from __future__ import annotations

import argparse
import glob
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_COLUMNS = {
    "wind_mean_ms": ("actual_wind_mean_ms", "labels__target_wind_mean_ms", "target_wind_mean_ms"),
    "gust_ms": ("actual_gust_ms", "labels__target_gust_ms", "target_gust_ms"),
}
BASELINE_COLUMNS = {
    "wind_mean_ms": ("raw_wind_mean_ms", "baselines__baseline_wind_mean_ms", "baseline_wind_mean_ms"),
    "gust_ms": ("raw_gust_ms", "baselines__baseline_gust_ms", "baseline_gust_ms"),
    "temperature_2m_c": ("baselines__baseline_temperature_2m_c", "baseline_temperature_2m_c"),
    "pressure_msl_hpa": ("baselines__baseline_pressure_msl_hpa", "baseline_pressure_msl_hpa"),
    "cloud_cover_pct": ("baselines__baseline_cloud_cover_pct", "baseline_cloud_cover_pct"),
    "shortwave_radiation": ("baselines__baseline_shortwave_radiation", "baseline_shortwave_radiation"),
}
PAST_COLUMNS = {
    "wind_mean_ms": TARGET_COLUMNS["wind_mean_ms"],
    "gust_ms": TARGET_COLUMNS["gust_ms"],
    "nwp_wind_mean_ms": BASELINE_COLUMNS["wind_mean_ms"],
    "nwp_gust_ms": BASELINE_COLUMNS["gust_ms"],
    "nwp_temperature_2m_c": BASELINE_COLUMNS["temperature_2m_c"],
    "nwp_pressure_msl_hpa": BASELINE_COLUMNS["pressure_msl_hpa"],
    "nwp_cloud_cover_pct": BASELINE_COLUMNS["cloud_cover_pct"],
    "nwp_shortwave_radiation": BASELINE_COLUMNS["shortwave_radiation"],
}
OBSERVATION_COLUMNS = {
    "wind_mean_ms": ("wind_mean_ms", "avg_wind_ms", "ff"),
    "gust_ms": ("gust_ms", "gust_wind_ms", "gust"),
}


def import_dependencies() -> dict[str, Any]:
    try:
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Missing pandas/pyarrow dependencies. Run inside the CorseWind ML venv.") from exc
    return {"pd": pd, "pq": pq}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return sorted(dict.fromkeys(path for path in paths if path.exists()))


def first_present(frame: Any, candidates: tuple[str, ...], pd: Any) -> Any:
    for column in candidates:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series([math.nan] * len(frame), index=frame.index, dtype="float64")


def normalize_time_column(frame: Any, column: str, pd: Any) -> Any:
    values = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return values


def iso_z_series(values: Any) -> Any:
    return values.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def sample_id_for(spot_id: str, issue_time_utc: str) -> str:
    return f"{spot_id}|{issue_time_utc}"


def read_parquet_subset(paths: list[Path], wanted_columns: list[str], deps: dict[str, Any]) -> Any:
    pd = deps["pd"]
    pq = deps["pq"]
    frames = []
    for path in paths:
        pf = pq.ParquetFile(path)
        available = [column for column in wanted_columns if column in pf.schema_arrow.names]
        required = {"spot_id", "target_time_utc"}
        if not required.issubset(set(available)):
            continue
        frames.append(pf.read(columns=available).to_pandas().reindex(columns=wanted_columns))
    if not frames:
        return pd.DataFrame(columns=wanted_columns)
    return pd.concat(frames, ignore_index=True)


def read_observation_series(paths: list[Path], deps: dict[str, Any]) -> Any:
    pd = deps["pd"]
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                spot_id = row.get("spot_id")
                timestamp = row.get("timestamp_utc")
                if not spot_id or not timestamp:
                    continue
                item = {"spot_id": str(spot_id), "timestamp": timestamp}
                for output, candidates in OBSERVATION_COLUMNS.items():
                    for candidate in candidates:
                        if candidate in row:
                            item[output] = row.get(candidate)
                            break
                rows.append(item)
    if not rows:
        return pd.DataFrame(columns=["spot_id", "timestamp", *PAST_COLUMNS])
    observations = pd.DataFrame(rows)
    observations["timestamp"] = pd.to_datetime(observations["timestamp"], utc=True, errors="coerce")
    observations["spot_id"] = observations["spot_id"].astype(str)
    for column in PAST_COLUMNS:
        if column not in observations.columns:
            observations[column] = math.nan
        observations[column] = pd.to_numeric(observations[column], errors="coerce")
    observations = observations.dropna(subset=["spot_id", "timestamp"])
    observations = observations.dropna(subset=list(OBSERVATION_COLUMNS), how="all")
    observations = observations[["spot_id", "timestamp", *PAST_COLUMNS]]
    return observations.sort_values(["spot_id", "timestamp"]).drop_duplicates(["spot_id", "timestamp"], keep="last")


def wanted_history_columns() -> list[str]:
    columns = ["spot_id", "target_time_utc", "lead_time_minutes"]
    for candidates in [*TARGET_COLUMNS.values(), *BASELINE_COLUMNS.values()]:
        columns.extend(candidates)
    return list(dict.fromkeys(columns))


def build_prediction_table(live: Any, pd: Any) -> Any:
    live = live.copy()
    live["issue_time"] = normalize_time_column(live, "issue_time_utc", pd)
    live["target_time"] = normalize_time_column(live, "target_time_utc", pd)
    live["issue_time_utc"] = iso_z_series(live["issue_time"])
    live["target_time_utc"] = iso_z_series(live["target_time"])
    live["lead_time_minutes"] = pd.to_numeric(live["lead_time_minutes"], errors="coerce").astype("Int64")
    live = live.dropna(subset=["spot_id", "issue_time", "target_time", "lead_time_minutes"])
    live["spot_id"] = live["spot_id"].astype(str)
    live["item_id"] = [
        sample_id_for(spot, issue)
        for spot, issue in zip(live["spot_id"], live["issue_time_utc"])
    ]

    predictions = live[["item_id", "spot_id", "issue_time_utc", "target_time", "lead_time_minutes"]].copy()
    predictions = predictions.rename(columns={"target_time": "timestamp"})
    predictions["timestamp"] = predictions["timestamp"].dt.tz_localize(None)
    predictions["lead_time_minutes"] = predictions["lead_time_minutes"].astype(float)
    predictions["actual_wind_mean_ms"] = first_present(live, TARGET_COLUMNS["wind_mean_ms"], pd)
    predictions["actual_gust_ms"] = first_present(live, TARGET_COLUMNS["gust_ms"], pd)
    predictions["raw_wind_mean_ms"] = first_present(live, BASELINE_COLUMNS["wind_mean_ms"], pd)
    predictions["raw_gust_ms"] = first_present(live, BASELINE_COLUMNS["gust_ms"], pd)
    predictions["hgb_wind_mean_ms"] = first_present(live, ("hgb_wind_mean_ms", "calibrated_wind_mean_ms"), pd)
    predictions["hgb_gust_ms"] = first_present(live, ("hgb_gust_ms", "calibrated_gust_ms"), pd)
    return live, predictions.sort_values(["item_id", "timestamp"]).reset_index(drop=True)


def build_history_series(history: Any, live: Any, pd: Any) -> Any:
    if history.empty:
        return pd.DataFrame(columns=["spot_id", "timestamp", *PAST_COLUMNS])
    history = history.copy()
    history["spot_id"] = history["spot_id"].astype(str)
    history["timestamp"] = normalize_time_column(history, "target_time_utc", pd)
    if "lead_time_minutes" in history.columns:
        history["lead_time_minutes"] = pd.to_numeric(history["lead_time_minutes"], errors="coerce")
        history = history[history["lead_time_minutes"].isna() | (history["lead_time_minutes"] == 15)]
    max_issue = live["issue_time"].max()
    if pd.notna(max_issue):
        history = history[history["timestamp"] <= max_issue]

    out = history[["spot_id", "timestamp"]].copy()
    for output, candidates in PAST_COLUMNS.items():
        out[output] = first_present(history, candidates, pd)
    out = out.dropna(subset=["spot_id", "timestamp"])
    out = out.sort_values(["spot_id", "timestamp"]).drop_duplicates(["spot_id", "timestamp"], keep="last")
    return out


def merge_history_sources(history_series: Any, observation_series: Any, pd: Any) -> Any:
    if observation_series.empty:
        return history_series
    if history_series.empty:
        return observation_series
    combined = pd.concat([history_series, observation_series], ignore_index=True, sort=False)
    combined = combined.sort_values(["spot_id", "timestamp"])
    value_columns = [column for column in PAST_COLUMNS if column in combined.columns]
    merged = (
        combined.groupby(["spot_id", "timestamp"], as_index=False)[value_columns]
        .last()
        .sort_values(["spot_id", "timestamp"])
        .reset_index(drop=True)
    )
    return merged


def build_past_context(
    live: Any,
    history_series: Any,
    *,
    context_length: int,
    freq_minutes: int,
    max_interpolate_steps: int,
    max_ffill_steps: int,
    max_bfill_steps: int,
    pd: Any,
) -> tuple[Any, dict[str, int]]:
    freq = f"{freq_minutes}min"
    value_columns = list(PAST_COLUMNS)
    by_spot = {
        str(spot): group.set_index("timestamp").sort_index()
        for spot, group in history_series.groupby("spot_id")
    }
    rows = []
    skipped = {"missing_history_spot": 0}
    for sample in live[["item_id", "spot_id", "issue_time_utc", "issue_time"]].drop_duplicates().itertuples(index=False):
        group = by_spot.get(str(sample.spot_id))
        if group is None or group.empty:
            skipped["missing_history_spot"] += 1
            context = pd.DataFrame(index=pd.date_range(end=sample.issue_time, periods=context_length, freq=freq, tz="UTC"))
        else:
            index = pd.date_range(end=sample.issue_time, periods=context_length, freq=freq, tz="UTC")
            context = group.reindex(index)
        for column in value_columns:
            if column not in context.columns:
                context[column] = math.nan
        context[value_columns] = context[value_columns].apply(pd.to_numeric, errors="coerce")
        context[value_columns] = (
            context[value_columns]
            .interpolate(method="time", limit=max_interpolate_steps, limit_direction="both")
            .ffill(limit=max_ffill_steps)
            .bfill(limit=max_bfill_steps)
        )
        for timestamp, values in context[value_columns].iterrows():
            row = {"item_id": sample.item_id, "timestamp": timestamp.tz_localize(None)}
            row.update({column: values[column] for column in value_columns})
            rows.append(row)
    return pd.DataFrame(rows), skipped


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def context_coverage(past_context: Any) -> dict[str, Any]:
    if past_context.empty:
        return {"rows": 0, "non_null_counts": {}, "items_with_observed_context": 0}
    value_columns = [column for column in PAST_COLUMNS if column in past_context.columns]
    observed_columns = [column for column in ("wind_mean_ms", "gust_ms") if column in past_context.columns]
    return {
        "rows": int(len(past_context)),
        "non_null_counts": {
            column: int(past_context[column].notna().sum())
            for column in value_columns
        },
        "items_with_observed_context": int(
            past_context.groupby("item_id")[observed_columns].apply(lambda group: bool(group.notna().any().any())).sum()
        ) if observed_columns else 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]
    pq = deps["pq"]
    live = pq.read_table(args.live_rows_parquet).to_pandas()
    if live.empty:
        raise SystemExit(f"Live rows parquet is empty: {args.live_rows_parquet}")
    live, predictions = build_prediction_table(live, pd)

    history_paths = expand_paths(args.history_parquet)
    observation_paths = expand_paths(args.history_observations_jsonl)
    history = read_parquet_subset(history_paths, wanted_history_columns(), deps) if history_paths else pd.DataFrame()
    history_series = build_history_series(history, live, pd)
    observation_series = read_observation_series(observation_paths, deps) if observation_paths else pd.DataFrame()
    if not observation_series.empty:
        max_issue = live["issue_time"].max()
        if pd.notna(max_issue):
            observation_series = observation_series[observation_series["timestamp"] <= max_issue]
        history_series = merge_history_sources(history_series, observation_series, pd)
    past_context, skipped = build_past_context(
        live,
        history_series,
        context_length=args.context_length,
        freq_minutes=args.freq_minutes,
        max_interpolate_steps=args.max_history_interpolate_steps,
        max_ffill_steps=args.max_history_ffill_steps,
        max_bfill_steps=args.max_history_bfill_steps,
        pd=pd,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_root / "predictions.parquet"
    past_path = args.output_root / "past_context.parquet"
    predictions.to_parquet(predictions_path, index=False, compression=args.compression)
    past_context.to_parquet(past_path, index=False, compression=args.compression)
    manifest = {
        "format": "corsewind.live_foundation_sequence_inputs.v1",
        "generated_at_utc": utc_now(),
        "live_rows_parquet": str(args.live_rows_parquet),
        "history_parquet": [str(path) for path in history_paths],
        "history_observations_jsonl": [str(path) for path in observation_paths],
        "prediction_rows": int(len(predictions)),
        "item_count": int(predictions["item_id"].nunique()),
        "spot_count": int(predictions["spot_id"].nunique()),
        "history_rows": int(len(history)),
        "history_observation_rows": int(len(observation_series)),
        "history_series_rows": int(len(history_series)),
        "past_context_rows": int(len(past_context)),
        "past_context_coverage": context_coverage(past_context),
        "context_length": args.context_length,
        "freq_minutes": args.freq_minutes,
        "skipped": skipped,
        "predictions_parquet": str(predictions_path),
        "past_context_parquet": str(past_path),
    }
    write_manifest(args.output_root / "sequence_input_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-rows-parquet", type=Path, required=True)
    parser.add_argument("--history-parquet", action="append", default=[], help="History parquet path or glob. Repeatable.")
    parser.add_argument("--history-observations-jsonl", action="append", default=[], help="Observation JSONL path or glob. Repeatable.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--freq-minutes", type=int, default=15)
    parser.add_argument("--max-history-interpolate-steps", type=int, default=4)
    parser.add_argument("--max-history-ffill-steps", type=int, default=24)
    parser.add_argument("--max-history-bfill-steps", type=int, default=8)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
