#!/usr/bin/env python3
"""Export a CorseWind SAPHIR-style structured sequence dataset from Parquet shards."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_COLUMNS = {
    "wind_mean_ms": "labels__target_wind_mean_ms",
    "gust_ms": "labels__target_gust_ms",
    "wind_direction_deg": "labels__target_wind_direction_deg",
}
RESIDUAL_COLUMNS = {
    "wind_mean_ms": "labels__residual_wind_mean_ms",
    "gust_ms": "labels__residual_gust_ms",
}
BASELINE_COLUMNS = {
    "wind_mean_ms": "baselines__baseline_wind_mean_ms",
    "gust_ms": "baselines__baseline_gust_ms",
    "wind_direction_deg": "baselines__baseline_wind_direction_deg",
    "temperature_2m_c": "baselines__baseline_temperature_2m_c",
    "pressure_msl_hpa": "baselines__baseline_pressure_msl_hpa",
    "surface_pressure_hpa": "baselines__baseline_surface_pressure_hpa",
    "shortwave_radiation": "baselines__baseline_shortwave_radiation",
    "cloud_cover_pct": "baselines__baseline_cloud_cover_pct",
    "cape": "baselines__baseline_cape",
}
PAST_NWP_COLUMNS = {
    "nwp_wind_mean_ms": "baselines__baseline_wind_mean_ms",
    "nwp_gust_ms": "baselines__baseline_gust_ms",
    "nwp_temperature_2m_c": "baselines__baseline_temperature_2m_c",
    "nwp_pressure_msl_hpa": "baselines__baseline_pressure_msl_hpa",
    "nwp_cloud_cover_pct": "baselines__baseline_cloud_cover_pct",
    "nwp_shortwave_radiation": "baselines__baseline_shortwave_radiation",
}

CONTEXT_SLOT_RE = re.compile(
    r"^features__(context_(?:nearest|coastal|inland|relief|regional|global_(?:nearest|coastal|inland|relief|regional))_\d+)_(.+)$"
)
NWP_OFFSET_RE = re.compile(r"^features__nwp_offset_([a-z]\d+)_(.+)$")
VERTICAL_LEVEL_RE = re.compile(
    r"^features__model_open_meteo_meteofrance_arome_france_(.+)_(1000|950|925|900|850)hPa$"
)


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


def discover_parquet_paths(root: Path, prefix: str, start_month: str, end_month: str) -> list[Path]:
    paths = []
    for suffix in month_range(start_month, end_month):
        path = root / f"{prefix}_{suffix}" / "training_rows.parquet"
        if path.exists():
            paths.append(path)
    if not paths:
        raise SystemExit(f"No training_rows.parquet shards found under {root} for prefix {prefix}")
    return paths


def parse_int_list(value: str) -> list[int]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return sorted(set(out))


def iso_z_from_timestamp(value: Any, pd: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def sample_id_for(spot_id: str, issue_time_utc: str) -> str:
    return f"{spot_id}|{issue_time_utc}"


def schema_columns(paths: list[Path], pq: Any) -> list[str]:
    columns: set[str] = set()
    for path in paths:
        columns.update(pq.ParquetFile(path).schema_arrow.names)
    return sorted(columns)


def read_parquet_columns(paths: list[Path], columns: list[str], deps: dict[str, Any]):
    pd = deps["pd"]
    pq = deps["pq"]
    frames = []
    wanted = list(dict.fromkeys(columns))
    for path in paths:
        pf = pq.ParquetFile(path)
        available = [column for column in wanted if column in pf.schema_arrow.names]
        if not available:
            continue
        frames.append(pf.read(columns=available).to_pandas().reindex(columns=wanted))
    if not frames:
        return pd.DataFrame(columns=wanted)
    return pd.concat(frames, ignore_index=True)


def add_time_columns(frame: Any, pd: Any) -> Any:
    frame = frame.copy()
    frame["issue_time"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    frame["target_time"] = pd.to_datetime(frame["target_time_utc"], utc=True, errors="coerce")
    frame["issue_time_utc"] = frame["issue_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    frame["target_time_utc"] = frame["target_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return frame


def thin_columns() -> list[str]:
    return [
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
        *TARGET_COLUMNS.values(),
        *RESIDUAL_COLUMNS.values(),
        *BASELINE_COLUMNS.values(),
        "labels__target_observation_source_dataset",
        "labels__target_observation_source_project",
        "labels__target_observation_source_type",
        "labels__target_observation_station_id",
        "labels__target_observation_distance_minutes",
        "labels__target_observation_source_resolution_minutes",
    ]


def build_thin_frame(paths: list[Path], deps: dict[str, Any]):
    pd = deps["pd"]
    frame = read_parquet_columns(paths, thin_columns(), deps)
    if frame.empty:
        raise SystemExit("No thin identity/target rows could be read from training shards.")
    frame = add_time_columns(frame, pd)
    frame["lead_time_minutes"] = pd.to_numeric(frame["lead_time_minutes"], errors="coerce")
    frame = frame.dropna(subset=["spot_id", "issue_time", "target_time", "lead_time_minutes"])
    frame["spot_id"] = frame["spot_id"].astype(str)
    frame["lead_time_minutes"] = frame["lead_time_minutes"].astype(int)
    return frame


def select_complete_samples(frame: Any, args: argparse.Namespace, deps: dict[str, Any]):
    pd = deps["pd"]
    leads = parse_int_list(args.lead_minutes)
    data = frame[frame["lead_time_minutes"].isin(leads)].copy()
    data = data.dropna(subset=[TARGET_COLUMNS["wind_mean_ms"], BASELINE_COLUMNS["wind_mean_ms"]])
    if args.require_gust:
        data = data.dropna(subset=[TARGET_COLUMNS["gust_ms"], BASELINE_COLUMNS["gust_ms"]])
    if args.issue_hour_start is not None:
        data = data[data["issue_time"].dt.hour >= args.issue_hour_start]
    if args.issue_hour_end is not None:
        data = data[data["issue_time"].dt.hour <= args.issue_hour_end]
    if args.spot_id:
        data = data[data["spot_id"].isin(set(args.spot_id))]
    if args.eval_start:
        # Keep samples before eval_start too, because they are the training split.
        data = data[data["issue_time"].notna()]

    data = data.sort_values(["spot_id", "issue_time", "lead_time_minutes"])
    data = data.drop_duplicates(["spot_id", "issue_time_utc", "lead_time_minutes"], keep="last")

    groups = []
    required = set(leads)
    for (spot_id, issue_time_utc), group in data.groupby(["spot_id", "issue_time_utc"], sort=False):
        available = set(int(value) for value in group["lead_time_minutes"].dropna().tolist())
        if not required.issubset(available):
            continue
        issue_time = group["issue_time"].iloc[0]
        split = "train"
        if args.train_end and issue_time >= pd.Timestamp(args.train_end, tz="UTC"):
            split = "eval"
        if args.eval_start and issue_time >= pd.Timestamp(args.eval_start, tz="UTC"):
            split = "eval"
        groups.append({
            "spot_id": str(spot_id),
            "issue_time_utc": str(issue_time_utc),
            "issue_time": issue_time,
            "split": split,
        })

    cases = pd.DataFrame(groups)
    if cases.empty:
        raise SystemExit("No complete multi-lead samples selected.")

    selected_parts = []
    for (spot_id, split), group in cases.sort_values("issue_time").groupby(["spot_id", "split"], sort=False):
        if args.max_samples_per_spot and args.max_samples_per_spot > 0 and len(group) > args.max_samples_per_spot:
            if args.max_samples_per_spot == 1:
                indexes = [len(group) - 1]
            else:
                indexes = [
                    round(index * (len(group) - 1) / (args.max_samples_per_spot - 1))
                    for index in range(args.max_samples_per_spot)
                ]
            group = group.iloc[sorted(set(int(index) for index in indexes))]
        selected_parts.append(group)
    selected = pd.concat(selected_parts, ignore_index=True).sort_values(["spot_id", "issue_time"])
    if args.max_samples and args.max_samples > 0 and len(selected) > args.max_samples:
        selected = selected.head(args.max_samples)
    selected["sample_id"] = [
        sample_id_for(row.spot_id, row.issue_time_utc)
        for row in selected.itertuples(index=False)
    ]
    selected_keys = set(zip(selected["spot_id"], selected["issue_time_utc"]))
    future = data[
        [
            (spot, issue) in selected_keys
            for spot, issue in zip(data["spot_id"], data["issue_time_utc"])
        ]
    ].copy()
    future["sample_id"] = [
        sample_id_for(row.spot_id, row.issue_time_utc)
        for row in future.itertuples(index=False)
    ]
    return selected.reset_index(drop=True), future.reset_index(drop=True), leads


def build_samples_table(samples: Any, future: Any, leads: list[int], deps: dict[str, Any]):
    pd = deps["pd"]
    rows = []
    first_future = future.sort_values("lead_time_minutes").drop_duplicates("sample_id", keep="first")
    meta = first_future.set_index("sample_id")
    for sample in samples.itertuples(index=False):
        row = {
            "sample_id": sample.sample_id,
            "spot_id": sample.spot_id,
            "issue_time_utc": sample.issue_time_utc,
            "split": sample.split,
            "available_leads": ",".join(str(lead) for lead in leads),
        }
        if sample.sample_id in meta.index:
            item = meta.loc[sample.sample_id]
            for column in ("spot_name", "spot_kind", "spot_source_type", "station_id", "latitude", "longitude"):
                row[column] = item.get(column)
        group = future[future["sample_id"] == sample.sample_id]
        by_lead = {int(item.lead_time_minutes): item for item in group.itertuples(index=False)}
        for lead in leads:
            item = by_lead.get(lead)
            for alias, column in TARGET_COLUMNS.items():
                row[f"target_{alias}_lead_{lead}"] = None if item is None else getattr(item, column)
            for alias, column in BASELINE_COLUMNS.items():
                row[f"baseline_{alias}_lead_{lead}"] = None if item is None else getattr(item, column)
            for alias, column in RESIDUAL_COLUMNS.items():
                row[f"residual_{alias}_lead_{lead}"] = None if item is None else getattr(item, column)
        rows.append(row)
    return pd.DataFrame(rows)


def build_future_table(future: Any, samples: Any, deps: dict[str, Any]):
    keep = [
        "sample_id",
        "spot_id",
        "issue_time_utc",
        "target_time_utc",
        "lead_time_minutes",
        *TARGET_COLUMNS.values(),
        *RESIDUAL_COLUMNS.values(),
        *BASELINE_COLUMNS.values(),
        "labels__target_observation_source_dataset",
        "labels__target_observation_source_project",
        "labels__target_observation_source_type",
        "labels__target_observation_station_id",
        "labels__target_observation_distance_minutes",
        "labels__target_observation_source_resolution_minutes",
    ]
    out = future[[column for column in keep if column in future.columns]].copy()
    split_map = samples.set_index("sample_id")["split"].to_dict()
    out["split"] = out["sample_id"].map(split_map)
    rename = {
        TARGET_COLUMNS["wind_mean_ms"]: "target_wind_mean_ms",
        TARGET_COLUMNS["gust_ms"]: "target_gust_ms",
        TARGET_COLUMNS["wind_direction_deg"]: "target_wind_direction_deg",
        RESIDUAL_COLUMNS["wind_mean_ms"]: "residual_wind_mean_ms",
        RESIDUAL_COLUMNS["gust_ms"]: "residual_gust_ms",
    }
    rename.update({column: f"baseline_{alias}" for alias, column in BASELINE_COLUMNS.items()})
    return out.rename(columns=rename)


def build_actual_series(frame: Any, deps: dict[str, Any]):
    series = frame[frame["lead_time_minutes"] == 15].copy()
    series = series.rename(columns={
        TARGET_COLUMNS["wind_mean_ms"]: "wind_mean_ms",
        TARGET_COLUMNS["gust_ms"]: "gust_ms",
        TARGET_COLUMNS["wind_direction_deg"]: "wind_direction_deg",
        **{source: alias for alias, source in PAST_NWP_COLUMNS.items()},
    })
    keep = ["spot_id", "target_time", "target_time_utc", "wind_mean_ms", "gust_ms", "wind_direction_deg", *PAST_NWP_COLUMNS]
    series = series[[column for column in keep if column in series.columns]].drop_duplicates(["spot_id", "target_time"], keep="last")
    return series.sort_values(["spot_id", "target_time"])


def build_station_history(samples: Any, actual_series: Any, args: argparse.Namespace, deps: dict[str, Any]):
    pd = deps["pd"]
    freq = f"{args.freq_minutes}min"
    value_columns = ["wind_mean_ms", "gust_ms", "wind_direction_deg", *PAST_NWP_COLUMNS]
    by_spot = {spot: group.set_index("target_time").sort_index() for spot, group in actual_series.groupby("spot_id")}
    rows = []
    skipped = Counter()
    for sample in samples.itertuples(index=False):
        group = by_spot.get(sample.spot_id)
        if group is None:
            skipped["missing_spot_series"] += 1
            continue
        index = pd.date_range(end=sample.issue_time, periods=args.context_length, freq=freq, tz="UTC")
        context = group.reindex(index)
        for column in value_columns:
            if column not in context.columns:
                context[column] = math.nan
        observed_mask = context[value_columns].notna()
        context[value_columns] = context[value_columns].apply(pd.to_numeric, errors="coerce")
        context[value_columns] = (
            context[value_columns]
            .interpolate(method="time", limit=args.max_history_interpolate_steps)
            .ffill(limit=args.max_history_ffill_steps)
        )
        if args.require_complete_history and context[["wind_mean_ms"]].isna().any().any():
            skipped["incomplete_required_history"] += 1
            continue
        for step_index, timestamp in enumerate(index):
            item = {
                "sample_id": sample.sample_id,
                "spot_id": sample.spot_id,
                "issue_time_utc": sample.issue_time_utc,
                "split": sample.split,
                "station_slot": 0,
                "station_slot_name": "target",
                "time_index": step_index,
                "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
                "minutes_before_issue": int((sample.issue_time - timestamp).total_seconds() // 60),
            }
            for column in value_columns:
                item[column] = context.at[timestamp, column] if column in context.columns else None
                item[f"{column}_observed"] = bool(observed_mask.at[timestamp, column]) if column in observed_mask.columns else False
            rows.append(item)
    return pd.DataFrame(rows), dict(skipped)


def feature_column_groups(columns: list[str]) -> dict[str, list[str]]:
    context = []
    offsets = []
    vertical = []
    static = []
    for column in columns:
        if CONTEXT_SLOT_RE.match(column):
            context.append(column)
            continue
        if VERTICAL_LEVEL_RE.match(column):
            vertical.append(column)
            continue
        offset_match = NWP_OFFSET_RE.match(column)
        if offset_match and not column.startswith("features__nwp_offset_gradient_"):
            offsets.append(column)
            continue
        if column.startswith((
            "features__spot_static_",
            "features__thermal_",
            "features__open_meteo_vertical_",
            "features__context_agg_",
            "features__obs_",
            "features__model_error_now_",
            "features__previous_run_open_meteo_",
            "features__nwp_offset_gradient_",
            "features__model_open_meteo_meteofrance_arome_france_",
            "features__sst_",
            "features__eumetsat_",
        )):
            static.append(column)
    return {
        "context": sorted(context),
        "offsets": sorted(offsets),
        "vertical": sorted(vertical),
        "static": sorted(set(static) - set(vertical)),
    }


def load_issue_feature_rows(
    paths: list[Path],
    selected_samples: Any,
    feature_columns: list[str],
    issue_lead_minutes: int,
    deps: dict[str, Any],
):
    pd = deps["pd"]
    pq = deps["pq"]
    selected_keys = set(
        f"{row.spot_id}|{row.issue_time_utc}|{issue_lead_minutes}"
        for row in selected_samples.itertuples(index=False)
    )
    wanted = ["spot_id", "issue_time_utc", "lead_time_minutes", *feature_columns]
    frames = []
    for path in paths:
        pf = pq.ParquetFile(path)
        available = [column for column in wanted if column in pf.schema_arrow.names]
        if not {"spot_id", "issue_time_utc", "lead_time_minutes"}.issubset(available):
            continue
        for batch in pf.iter_batches(batch_size=50000, columns=available):
            frame = batch.to_pandas().reindex(columns=wanted)
            frame["lead_time_minutes"] = pd.to_numeric(frame["lead_time_minutes"], errors="coerce").astype("Int64")
            keys = (
                frame["spot_id"].astype(str)
                + "|"
                + frame["issue_time_utc"].astype(str)
                + "|"
                + frame["lead_time_minutes"].astype(str)
            )
            keep = frame[keys.isin(selected_keys)]
            if not keep.empty:
                frames.append(keep)
    if not frames:
        return pd.DataFrame(columns=wanted)
    out = pd.concat(frames, ignore_index=True)
    out["sample_id"] = out.apply(lambda row: sample_id_for(str(row["spot_id"]), str(row["issue_time_utc"])), axis=1)
    return out.drop_duplicates(["sample_id"], keep="last")


def short_feature_name(column: str) -> str:
    return column.removeprefix("features__")


def build_static_context(issue_features: Any, static_columns: list[str], samples: Any, deps: dict[str, Any]):
    keep = ["sample_id", "spot_id", "issue_time_utc", *static_columns]
    out = issue_features[[column for column in keep if column in issue_features.columns]].copy()
    out = out.rename(columns={column: short_feature_name(column) for column in static_columns if column in out.columns})
    split_map = samples.set_index("sample_id")["split"].to_dict()
    out["split"] = out["sample_id"].map(split_map)
    return out


def build_context_station_snapshot(issue_features: Any, context_columns: list[str], samples: Any, deps: dict[str, Any]):
    pd = deps["pd"]
    slots: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for column in context_columns:
        match = CONTEXT_SLOT_RE.match(column)
        if match:
            slots[match.group(1)].append((column, match.group(2)))
    split_map = samples.set_index("sample_id")["split"].to_dict()
    rows = []
    for source in issue_features.itertuples(index=False):
        row = source._asdict()
        for slot, pairs in slots.items():
            item = {
                "sample_id": row["sample_id"],
                "spot_id": row["spot_id"],
                "issue_time_utc": row["issue_time_utc"],
                "split": split_map.get(row["sample_id"]),
                "station_slot_name": slot,
            }
            non_empty = False
            for column, field in pairs:
                value = row.get(column)
                item[field] = value
                if value not in {None, ""} and not (isinstance(value, float) and math.isnan(value)):
                    non_empty = True
            if non_empty:
                rows.append(item)
    return pd.DataFrame(rows)


def build_nwp_surface_offsets(issue_features: Any, offset_columns: list[str], samples: Any, deps: dict[str, Any]):
    pd = deps["pd"]
    slots: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for column in offset_columns:
        match = NWP_OFFSET_RE.match(column)
        if match:
            slots[match.group(1)].append((column, match.group(2)))
    split_map = samples.set_index("sample_id")["split"].to_dict()
    rows = []
    for source in issue_features.itertuples(index=False):
        row = source._asdict()
        for offset_name, pairs in slots.items():
            item = {
                "sample_id": row["sample_id"],
                "spot_id": row["spot_id"],
                "issue_time_utc": row["issue_time_utc"],
                "split": split_map.get(row["sample_id"]),
                "offset_name": offset_name,
            }
            non_empty = False
            for column, field in pairs:
                value = row.get(column)
                item[field] = value
                if value not in {None, ""} and not (isinstance(value, float) and math.isnan(value)):
                    non_empty = True
            if non_empty:
                rows.append(item)
    return pd.DataFrame(rows)


def build_vertical_profile(issue_features: Any, vertical_columns: list[str], samples: Any, deps: dict[str, Any]):
    pd = deps["pd"]
    by_level: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for column in vertical_columns:
        match = VERTICAL_LEVEL_RE.match(column)
        if match:
            by_level[int(match.group(2))].append((column, match.group(1)))
    split_map = samples.set_index("sample_id")["split"].to_dict()
    rows = []
    for source in issue_features.itertuples(index=False):
        row = source._asdict()
        for level, pairs in sorted(by_level.items(), reverse=True):
            item = {
                "sample_id": row["sample_id"],
                "spot_id": row["spot_id"],
                "issue_time_utc": row["issue_time_utc"],
                "split": split_map.get(row["sample_id"]),
                "pressure_hpa": level,
            }
            non_empty = False
            for column, field in pairs:
                value = row.get(column)
                item[field] = value
                if value not in {None, ""} and not (isinstance(value, float) and math.isnan(value)):
                    non_empty = True
            if non_empty:
                rows.append(item)
    out = pd.DataFrame(rows)
    if not out.empty and {"wind_speed", "wind_direction"}.issubset(out.columns):
        speed = deps["pd"].to_numeric(out["wind_speed"], errors="coerce")
        direction = deps["pd"].to_numeric(out["wind_direction"], errors="coerce")
        radians = deps["np"].deg2rad(direction)
        out["wind_u_ms"] = -speed * deps["np"].sin(radians)
        out["wind_v_ms"] = -speed * deps["np"].cos(radians)
    return out


def numeric_stats(frame: Any, pd: Any) -> dict[str, dict[str, float | int]]:
    stats = {}
    if frame.empty or "split" not in frame.columns:
        return stats
    train = frame[frame["split"] == "train"]
    if train.empty:
        return stats
    for column in train.columns:
        if column in {"sample_id", "spot_id", "issue_time_utc", "target_time_utc", "timestamp_utc", "split"}:
            continue
        values = pd.to_numeric(train[column], errors="coerce").dropna()
        if values.empty:
            continue
        std = float(values.std(ddof=0))
        stats[column] = {
            "count": int(len(values)),
            "mean": round(float(values.mean()), 10),
            "std": round(std if std > 0 else 1.0, 10),
        }
    return stats


def metric_rows(frame: Any, pd: Any, np: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if frame.empty:
        return out
    for split, split_frame in frame.groupby("split"):
        split_out = {}
        for target, baseline in [
            ("target_wind_mean_ms", "baseline_wind_mean_ms"),
            ("target_gust_ms", "baseline_gust_ms"),
        ]:
            if target not in split_frame.columns or baseline not in split_frame.columns:
                continue
            valid = split_frame[[target, baseline]].apply(pd.to_numeric, errors="coerce").dropna()
            if valid.empty:
                split_out[baseline] = {"count": 0}
                continue
            errors = valid[baseline].to_numpy(dtype=float) - valid[target].to_numpy(dtype=float)
            split_out[baseline] = {
                "count": int(len(errors)),
                "mae": round(float(np.mean(np.abs(errors))), 6),
                "rmse": round(float(np.sqrt(np.mean(errors * errors))), 6),
                "bias": round(float(np.mean(errors)), 6),
            }
        out[str(split)] = split_out
    return out


def write_frame(frame: Any, path: Path, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression=compression)


def write_markdown(path: Path, profile: dict[str, Any]) -> None:
    lines = [
        "# CorseWind SAPHIR-Style Sequence Dataset",
        "",
        f"Generated: `{profile['generated_at_utc']}`",
        f"Output root: `{profile['output_root']}`",
        "",
        "## Tables",
        "",
        "| Table | Rows | Columns |",
        "| --- | ---: | ---: |",
    ]
    for table, item in profile["tables"].items():
        lines.append(f"| `{table}` | {item['rows']} | {item['columns']} |")
    lines.extend([
        "",
        "## Splits",
        "",
        "| Split | Samples |",
        "| --- | ---: |",
    ])
    for split, count in sorted(profile["sample_split_counts"].items()):
        lines.append(f"| `{split}` | {count} |")
    lines.extend([
        "",
        "## Baseline Metrics",
        "",
        "```json",
        json.dumps(profile.get("baseline_metrics", {}), indent=2, sort_keys=True),
        "```",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-table-root", type=Path, required=True)
    parser.add_argument("--run-id-prefix", default="residual_windsup_sst_prev_phys_v3_dem_fetch")
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument("--end-month", default="2026-06")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lead-minutes", default="15,30,45,60")
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--freq-minutes", type=int, default=15)
    parser.add_argument("--train-end", default="2026-01-01T00:00:00Z")
    parser.add_argument("--eval-start", default="2026-01-01T00:00:00Z")
    parser.add_argument("--issue-hour-start", type=int, default=8)
    parser.add_argument("--issue-hour-end", type=int, default=17)
    parser.add_argument("--spot-id", action="append", default=[])
    parser.add_argument("--max-samples-per-spot", type=int, default=240)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-history-interpolate-steps", type=int, default=4)
    parser.add_argument("--max-history-ffill-steps", type=int, default=8)
    parser.add_argument("--require-complete-history", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-gust", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = import_deps()
    pd = deps["pd"]
    np = deps["np"]
    pq = deps["pq"]

    paths = discover_parquet_paths(args.training_table_root, args.run_id_prefix, args.start_month, args.end_month)
    columns = schema_columns(paths, pq)
    thin = build_thin_frame(paths, deps)
    samples, future_rows, leads = select_complete_samples(thin, args, deps)
    samples_table = build_samples_table(samples, future_rows, leads, deps)
    future_table = build_future_table(future_rows, samples, deps)
    actual_series = build_actual_series(thin, deps)
    station_history, station_history_skipped = build_station_history(samples, actual_series, args, deps)

    groups = feature_column_groups(columns)
    feature_columns = sorted(set(groups["context"] + groups["offsets"] + groups["vertical"] + groups["static"]))
    issue_features = load_issue_feature_rows(paths, samples, feature_columns, min(leads), deps)
    static_context = build_static_context(issue_features, groups["static"], samples, deps)
    context_snapshot = build_context_station_snapshot(issue_features, groups["context"], samples, deps)
    nwp_offsets = build_nwp_surface_offsets(issue_features, groups["offsets"], samples, deps)
    vertical_profile = build_vertical_profile(issue_features, groups["vertical"], samples, deps)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    tables = {
        "samples": samples_table,
        "future_targets": future_table,
        "station_history": station_history,
        "static_context": static_context,
        "context_station_snapshot": context_snapshot,
        "nwp_surface_offsets": nwp_offsets,
        "nwp_vertical_profile": vertical_profile,
    }
    for name, frame in tables.items():
        write_frame(frame, output_root / f"{name}.parquet", args.compression)

    normalization = {
        name: numeric_stats(frame, pd)
        for name, frame in tables.items()
    }
    (output_root / "normalization_stats.json").write_text(
        json.dumps(normalization, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    profile = {
        "format": "corsewind.saphir_style_sequence_dataset.v1",
        "generated_at_utc": utc_now(),
        "output_root": str(output_root),
        "training_table_root": str(args.training_table_root),
        "run_id_prefix": args.run_id_prefix,
        "start_month": args.start_month,
        "end_month": args.end_month,
        "lead_minutes": leads,
        "context_length": args.context_length,
        "freq_minutes": args.freq_minutes,
        "source_shards": [str(path) for path in paths],
        "source_row_count": int(len(thin)),
        "selected_sample_count": int(len(samples_table)),
        "sample_split_counts": dict(sorted(Counter(samples_table["split"]).items())),
        "sample_spot_counts": dict(sorted(Counter(samples_table["spot_id"]).items())),
        "station_history_skipped": station_history_skipped,
        "feature_column_groups": {key: len(value) for key, value in groups.items()},
        "tables": {
            name: {"rows": int(len(frame)), "columns": int(len(frame.columns))}
            for name, frame in tables.items()
        },
        "baseline_metrics": metric_rows(future_table, pd, np),
        "settings": {
            "issue_hour_start": args.issue_hour_start,
            "issue_hour_end": args.issue_hour_end,
            "max_samples_per_spot": args.max_samples_per_spot,
            "max_samples": args.max_samples,
            "require_complete_history": args.require_complete_history,
            "require_gust": args.require_gust,
        },
    }
    (output_root / "dataset_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_root / "dataset_summary.md", profile)
    print(json.dumps({
        "output_root": str(output_root),
        "selected_sample_count": profile["selected_sample_count"],
        "tables": profile["tables"],
        "baseline_metrics": profile["baseline_metrics"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
