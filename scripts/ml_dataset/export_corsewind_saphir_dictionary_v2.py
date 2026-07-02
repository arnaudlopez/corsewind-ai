#!/usr/bin/env python3
"""Export a SAPHIR-inspired tensor dictionary for CorseWind.

This V2 export keeps one sample per (spot_id, issue_time) and stores multi-horizon
targets. The key addition versus the earlier sequence export is a true station
sequence table/tensor: target history plus context station histories.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from export_corsewind_saphir_sequence_dataset import (
    BASELINE_COLUMNS,
    CONTEXT_SLOT_RE,
    NWP_OFFSET_RE,
    RESIDUAL_COLUMNS,
    TARGET_COLUMNS,
    VERTICAL_LEVEL_RE,
    add_time_columns,
    build_context_station_snapshot,
    build_future_table,
    build_samples_table,
    build_static_context,
    build_thin_frame,
    discover_parquet_paths,
    feature_column_groups,
    import_deps,
    parse_int_list,
    read_parquet_columns,
    sample_id_for,
    schema_columns,
    select_complete_samples,
    short_feature_name,
    write_frame,
)


TARGET_HISTORY_COLUMNS = {
    "wind_mean_ms": TARGET_COLUMNS["wind_mean_ms"],
    "gust_ms": TARGET_COLUMNS["gust_ms"],
    "wind_direction_deg": TARGET_COLUMNS["wind_direction_deg"],
}
TARGET_HISTORY_NWP_COLUMNS = {
    "nwp_wind_mean_ms": BASELINE_COLUMNS["wind_mean_ms"],
    "nwp_gust_ms": BASELINE_COLUMNS["gust_ms"],
    "nwp_temperature_2m_c": BASELINE_COLUMNS["temperature_2m_c"],
    "nwp_pressure_msl_hpa": BASELINE_COLUMNS["pressure_msl_hpa"],
    "nwp_cloud_cover_pct": BASELINE_COLUMNS["cloud_cover_pct"],
    "nwp_shortwave_radiation": BASELINE_COLUMNS["shortwave_radiation"],
}
STATION_VALUE_COLUMNS = [
    "wind_mean_ms",
    "gust_ms",
    "wind_direction_deg",
    "wind_u_ms",
    "wind_v_ms",
    "temperature_c",
    "pressure_hpa",
    "humidity_pct",
    "age_minutes",
    "available",
    "nwp_wind_mean_ms",
    "nwp_gust_ms",
    "nwp_temperature_2m_c",
    "nwp_pressure_msl_hpa",
    "nwp_cloud_cover_pct",
    "nwp_shortwave_radiation",
    "wind_mean_error_ms",
    "gust_error_ms",
]
FUTURE_BASELINE_FEATURES = [
    "baseline_wind_mean_ms",
    "baseline_gust_ms",
    "baseline_wind_direction_deg",
    "baseline_temperature_2m_c",
    "baseline_pressure_msl_hpa",
    "baseline_surface_pressure_hpa",
    "baseline_shortwave_radiation",
    "baseline_cloud_cover_pct",
    "baseline_cape",
]
STATIC_BLOCKED_PREFIXES = ("target_", "residual_", "baseline_")
STATIC_BLOCKED_COLUMNS = {"sample_id", "spot_id", "issue_time_utc", "split"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def numeric(value: Any, pd: Any) -> Any:
    return pd.to_numeric(value, errors="coerce")


def wind_uv(speed: Any, direction_deg: Any, np: Any) -> tuple[Any, Any]:
    radians = np.deg2rad(direction_deg)
    return -speed * np.sin(radians), -speed * np.cos(radians)


def thin_history_columns() -> list[str]:
    return [
        "spot_id",
        "station_id",
        "issue_time_utc",
        "target_time_utc",
        "lead_time_minutes",
        *TARGET_HISTORY_COLUMNS.values(),
        *TARGET_HISTORY_NWP_COLUMNS.values(),
    ]


def build_target_observation_series(paths: list[Path], deps: dict[str, Any]):
    pd = deps["pd"]
    np = deps["np"]
    frame = read_parquet_columns(paths, thin_history_columns(), deps)
    if frame.empty:
        return pd.DataFrame()
    frame = add_time_columns(frame, pd)
    frame["lead_time_minutes"] = numeric(frame["lead_time_minutes"], pd).astype("Int64")
    frame = frame[frame["lead_time_minutes"] == 15].copy()
    rename = {source: alias for alias, source in TARGET_HISTORY_COLUMNS.items()}
    rename.update({source: alias for alias, source in TARGET_HISTORY_NWP_COLUMNS.items()})
    frame = frame.rename(columns=rename)
    frame["spot_id"] = frame["spot_id"].astype(str)
    frame["timestamp"] = frame["target_time"]
    if {"wind_mean_ms", "wind_direction_deg"}.issubset(frame.columns):
        speed = numeric(frame["wind_mean_ms"], pd)
        direction = numeric(frame["wind_direction_deg"], pd)
        frame["wind_u_ms"], frame["wind_v_ms"] = wind_uv(speed, direction, np)
    if {"wind_mean_ms", "nwp_wind_mean_ms"}.issubset(frame.columns):
        frame["wind_mean_error_ms"] = numeric(frame["wind_mean_ms"], pd) - numeric(frame["nwp_wind_mean_ms"], pd)
    if {"gust_ms", "nwp_gust_ms"}.issubset(frame.columns):
        frame["gust_error_ms"] = numeric(frame["gust_ms"], pd) - numeric(frame["nwp_gust_ms"], pd)
    keep = ["spot_id", "station_id", "timestamp", *STATION_VALUE_COLUMNS]
    keep = [column for column in keep if column in frame.columns]
    return frame[keep].drop_duplicates(["spot_id", "timestamp"], keep="last").sort_values(["spot_id", "timestamp"])


def context_slots_from_columns(columns: list[str]) -> dict[str, list[tuple[str, str]]]:
    slots: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for column in columns:
        match = CONTEXT_SLOT_RE.match(column)
        if match:
            slots[match.group(1)].append((column, match.group(2)))
    return dict(slots)


def load_lead_feature_rows(
    paths: list[Path],
    selected_samples: Any,
    feature_columns: list[str],
    lead_minutes: list[int],
    deps: dict[str, Any],
):
    pd = deps["pd"]
    pq = deps["pq"]
    selected = {
        f"{row.spot_id}|{row.issue_time_utc}|{lead}"
        for row in selected_samples.itertuples(index=False)
        for lead in lead_minutes
    }
    wanted = ["spot_id", "issue_time_utc", "lead_time_minutes", *feature_columns]
    frames = []
    for path in paths:
        pf = pq.ParquetFile(path)
        available = [column for column in wanted if column in pf.schema_arrow.names]
        if not {"spot_id", "issue_time_utc", "lead_time_minutes"}.issubset(available):
            continue
        for batch in pf.iter_batches(batch_size=50000, columns=available):
            frame = batch.to_pandas().reindex(columns=wanted)
            frame["lead_time_minutes"] = numeric(frame["lead_time_minutes"], pd).astype("Int64")
            keys = (
                frame["spot_id"].astype(str)
                + "|"
                + frame["issue_time_utc"].astype(str)
                + "|"
                + frame["lead_time_minutes"].astype(str)
            )
            keep = frame[keys.isin(selected)]
            if not keep.empty:
                frames.append(keep)
    if not frames:
        return pd.DataFrame(columns=wanted)
    out = pd.concat(frames, ignore_index=True)
    out["sample_id"] = out.apply(lambda row: sample_id_for(str(row["spot_id"]), str(row["issue_time_utc"])), axis=1)
    return out.drop_duplicates(["sample_id", "lead_time_minutes"], keep="last")


def load_all_context_issue_rows(
    paths: list[Path],
    context_columns: list[str],
    issue_lead_minutes: int,
    wanted_station_ids: set[str],
    deps: dict[str, Any],
):
    pd = deps["pd"]
    pq = deps["pq"]
    slots = context_slots_from_columns(context_columns)
    wanted = ["spot_id", "issue_time_utc", "lead_time_minutes", *context_columns]
    frames = []
    for path in paths:
        pf = pq.ParquetFile(path)
        available = [column for column in wanted if column in pf.schema_arrow.names]
        if not {"spot_id", "issue_time_utc", "lead_time_minutes"}.issubset(available):
            continue
        for batch in pf.iter_batches(batch_size=30000, columns=available):
            frame = batch.to_pandas().reindex(columns=wanted)
            frame["lead_time_minutes"] = numeric(frame["lead_time_minutes"], pd)
            frame = frame[frame["lead_time_minutes"] == issue_lead_minutes]
            if frame.empty:
                continue
            frame["issue_time"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
            base = frame[["issue_time"]].rename(columns={"issue_time": "timestamp"})
            for slot_name, pairs in slots.items():
                source_columns = [column for column, _field in pairs if column in frame.columns]
                if not source_columns:
                    continue
                rename = {column: field for column, field in pairs if column in frame.columns}
                slot_frame = pd.concat([base.reset_index(drop=True), frame[source_columns].rename(columns=rename).reset_index(drop=True)], axis=1)
                if "station_id" not in slot_frame.columns:
                    continue
                slot_frame = slot_frame[slot_frame["station_id"].notna()].copy()
                if slot_frame.empty:
                    continue
                slot_frame["station_id"] = slot_frame["station_id"].astype(str)
                slot_frame = slot_frame[slot_frame["station_id"].ne("")]
                if wanted_station_ids:
                    slot_frame = slot_frame[slot_frame["station_id"].isin(wanted_station_ids)]
                if slot_frame.empty:
                    continue
                slot_frame["station_slot_name"] = slot_name
                frames.append(slot_frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["station_id"] = out["station_id"].astype(str)
    for column in STATION_VALUE_COLUMNS:
        if column in out.columns:
            out[column] = numeric(out[column], pd)
    if {"wind_mean_ms", "wind_direction_deg"}.issubset(out.columns):
        out["wind_u_ms"], out["wind_v_ms"] = wind_uv(
            numeric(out["wind_mean_ms"], pd),
            numeric(out["wind_direction_deg"], pd),
            deps["np"],
        )
    if {"available", "wind_mean_ms"}.issubset(out.columns):
        out["available"] = numeric(out["available"], pd)
        # Treat a non-empty current wind as available even if the provider flag is missing.
        out["available"] = out["available"].fillna(out["wind_mean_ms"].notna().astype(float))
    numeric_columns = [column for column in STATION_VALUE_COLUMNS if column in out.columns]
    meta_columns = [column for column in ("role", "latitude", "longitude", "altitude_m", "distance_km") if column in out.columns]
    agg = out.groupby(["station_id", "timestamp"], as_index=False)[numeric_columns].mean(numeric_only=True)
    if meta_columns:
        meta = out.groupby(["station_id", "timestamp"], as_index=False)[meta_columns].last()
        agg = agg.merge(meta, on=["station_id", "timestamp"], how="left")
    return agg.sort_values(["station_id", "timestamp"])


def select_context_slots_for_sample(context_snapshot: Any, sample_id: str, max_context_stations: int, pd: Any) -> list[dict[str, Any]]:
    if context_snapshot.empty:
        return []
    group = context_snapshot[context_snapshot["sample_id"].astype(str).eq(str(sample_id))].copy()
    if group.empty or "station_id" not in group.columns:
        return []
    group["station_id"] = group["station_id"].astype(str)
    group = group[group["station_id"].notna() & group["station_id"].ne("")]
    if group.empty:
        return []
    if "available" in group.columns:
        group["_available_sort"] = numeric(group["available"], pd).fillna(0.0)
    else:
        group["_available_sort"] = 0.0
    if "distance_km" in group.columns:
        group["_distance_sort"] = numeric(group["distance_km"], pd).fillna(999999.0)
    else:
        group["_distance_sort"] = 999999.0
    if "upwind_score_from_target_wind" in group.columns:
        group["_upwind_sort"] = numeric(group["upwind_score_from_target_wind"], pd).fillna(-999999.0)
    else:
        group["_upwind_sort"] = -999999.0
    group = group.sort_values(["_available_sort", "_upwind_sort", "_distance_sort"], ascending=[False, False, True])
    group = group.drop_duplicates("station_id", keep="first").head(max_context_stations)
    slots = []
    for row in group.itertuples(index=False):
        data = row._asdict()
        slots.append({
            "station_id": str(data.get("station_id")),
            "station_slot_name": str(data.get("station_slot_name", "context")),
            "role": data.get("role"),
            "distance_km": data.get("distance_km"),
            "bearing_from_spot_deg": data.get("bearing_from_spot_deg"),
            "altitude_m": data.get("altitude_m"),
        })
    return slots


def build_context_slots_by_sample(context_snapshot: Any, max_context_stations: int, pd: Any) -> dict[str, list[dict[str, Any]]]:
    if context_snapshot.empty:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for sample_id in context_snapshot["sample_id"].astype(str).drop_duplicates():
        out[str(sample_id)] = select_context_slots_for_sample(context_snapshot, str(sample_id), max_context_stations, pd)
    return out


def build_station_sequence(
    samples: Any,
    target_series: Any,
    context_slots_by_sample: dict[str, list[dict[str, Any]]],
    context_series: Any,
    args: argparse.Namespace,
    deps: dict[str, Any],
):
    pd = deps["pd"]
    freq = f"{args.freq_minutes}min"
    rows = []
    skipped = Counter()
    target_by_spot = {
        spot: group.set_index("timestamp").sort_index()
        for spot, group in target_series.groupby("spot_id")
    } if not target_series.empty else {}
    context_by_station = {
        station: group.set_index("timestamp").sort_index()
        for station, group in context_series.groupby("station_id")
    } if not context_series.empty else {}
    value_columns = [column for column in STATION_VALUE_COLUMNS if column in set(target_series.columns) | set(context_series.columns)]
    for sample in samples.itertuples(index=False):
        index = pd.date_range(end=sample.issue_time, periods=args.context_length, freq=freq, tz="UTC")
        slot_specs = [{
            "station_slot": 0,
            "station_slot_name": "target",
            "source_kind": "target",
            "station_id": getattr(sample, "spot_id"),
            "role": "target",
            "distance_km": 0.0,
        }]
        for slot_index, spec in enumerate(context_slots_by_sample.get(str(sample.sample_id), []), start=1):
            slot_specs.append({"station_slot": slot_index, "source_kind": "context", **spec})
        for slot in slot_specs:
            if slot["source_kind"] == "target":
                source = target_by_spot.get(str(sample.spot_id))
            else:
                source = context_by_station.get(str(slot.get("station_id")))
            if source is None:
                skipped[f"missing_{slot['source_kind']}_series"] += 1
                source = pd.DataFrame(index=index)
            context = source.reindex(index)
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
            for step_index, timestamp in enumerate(index):
                item = {
                    "sample_id": sample.sample_id,
                    "spot_id": sample.spot_id,
                    "issue_time_utc": sample.issue_time_utc,
                    "split": sample.split,
                    "station_slot": int(slot["station_slot"]),
                    "station_slot_name": slot.get("station_slot_name"),
                    "source_kind": slot["source_kind"],
                    "station_id": slot.get("station_id"),
                    "role": slot.get("role"),
                    "distance_km": slot.get("distance_km"),
                    "bearing_from_spot_deg": slot.get("bearing_from_spot_deg"),
                    "altitude_m": slot.get("altitude_m"),
                    "time_index": step_index,
                    "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
                    "minutes_before_issue": int((sample.issue_time - timestamp).total_seconds() // 60),
                }
                for column in value_columns:
                    item[column] = context.at[timestamp, column] if column in context.columns else None
                    item[f"{column}_observed"] = bool(observed_mask.at[timestamp, column]) if column in observed_mask.columns else False
                rows.append(item)
    return pd.DataFrame(rows), dict(skipped)


def build_nwp_surface_offsets(issue_features: Any, offset_columns: list[str], samples: Any, deps: dict[str, Any]):
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
                "lead_time_minutes": row["lead_time_minutes"],
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
    return deps["pd"].DataFrame(rows)


def build_vertical_profile(issue_features: Any, vertical_columns: list[str], samples: Any, deps: dict[str, Any]):
    pd = deps["pd"]
    np = deps["np"]
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
                "lead_time_minutes": row["lead_time_minutes"],
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
        speed = numeric(out["wind_speed"], pd)
        direction = numeric(out["wind_direction"], pd)
        out["wind_u_ms"], out["wind_v_ms"] = wind_uv(speed, direction, np)
    return out


def static_candidates(static: Any, pd: Any) -> list[str]:
    out = []
    for column in static.columns:
        if column in STATIC_BLOCKED_COLUMNS:
            continue
        if any(column.startswith(prefix) for prefix in STATIC_BLOCKED_PREFIXES):
            continue
        values = numeric(static[column], pd)
        if values.notna().sum() > 0 and values.nunique(dropna=True) > 1:
            out.append(column)
    return out


def build_npz_tensors(
    output_root: Path,
    samples_table: Any,
    future_table: Any,
    station_sequence: Any,
    static_context: Any,
    args: argparse.Namespace,
    deps: dict[str, Any],
) -> dict[str, Any]:
    pd = deps["pd"]
    np = deps["np"]
    sample_ids = samples_table["sample_id"].astype(str).tolist()
    sample_to_idx = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    leads = parse_int_list(args.lead_minutes)
    lead_to_idx = {lead: idx for idx, lead in enumerate(leads)}
    station_features = [column for column in STATION_VALUE_COLUMNS if column in station_sequence.columns]
    n_slots = args.max_context_stations + 1
    station_tensor = np.full((len(sample_ids), args.context_length, n_slots, len(station_features)), np.nan, dtype="float32")
    station_mask = np.zeros((len(sample_ids), args.context_length, n_slots), dtype="float32")
    if not station_sequence.empty:
        for row in station_sequence.itertuples(index=False):
            data = row._asdict()
            sidx = sample_to_idx.get(str(data.get("sample_id")))
            tidx = int(data.get("time_index", -1))
            slot = int(data.get("station_slot", -1))
            if sidx is None or tidx < 0 or tidx >= args.context_length or slot < 0 or slot >= n_slots:
                continue
            station_mask[sidx, tidx, slot] = 1.0
            for fidx, column in enumerate(station_features):
                value = data.get(column)
                try:
                    station_tensor[sidx, tidx, slot, fidx] = float(value)
                except (TypeError, ValueError):
                    station_tensor[sidx, tidx, slot, fidx] = np.nan
    future_features = [column for column in FUTURE_BASELINE_FEATURES if column in future_table.columns]
    baseline_tensor = np.full((len(sample_ids), len(leads), len(future_features)), np.nan, dtype="float32")
    y_actual = np.full((len(sample_ids), len(leads), 2), np.nan, dtype="float32")
    y_residual = np.full((len(sample_ids), len(leads), 2), np.nan, dtype="float32")
    for row in future_table.itertuples(index=False):
        data = row._asdict()
        sidx = sample_to_idx.get(str(data.get("sample_id")))
        lidx = lead_to_idx.get(int(data.get("lead_time_minutes")))
        if sidx is None or lidx is None:
            continue
        for fidx, column in enumerate(future_features):
            try:
                baseline_tensor[sidx, lidx, fidx] = float(data.get(column))
            except (TypeError, ValueError):
                pass
        for tidx, column in enumerate(("target_wind_mean_ms", "target_gust_ms")):
            try:
                y_actual[sidx, lidx, tidx] = float(data.get(column))
            except (TypeError, ValueError):
                pass
        for tidx, column in enumerate(("residual_wind_mean_ms", "residual_gust_ms")):
            try:
                y_residual[sidx, lidx, tidx] = float(data.get(column))
            except (TypeError, ValueError):
                pass
    static_columns = static_candidates(static_context, pd)[: args.max_static_features]
    static_tensor = np.full((len(sample_ids), len(static_columns)), np.nan, dtype="float32")
    if static_columns and not static_context.empty:
        for row in static_context[["sample_id", *static_columns]].itertuples(index=False):
            data = row._asdict()
            sidx = sample_to_idx.get(str(data.get("sample_id")))
            if sidx is None:
                continue
            for fidx, column in enumerate(static_columns):
                try:
                    static_tensor[sidx, fidx] = float(data.get(column))
                except (TypeError, ValueError):
                    pass
    split = samples_table["split"].astype(str).to_numpy()
    issue_time = samples_table["issue_time_utc"].astype(str).to_numpy()
    spot_id = samples_table["spot_id"].astype(str).to_numpy()
    np.savez_compressed(
        output_root / "dictionary_tensors.npz",
        sample_id=np.asarray(sample_ids),
        spot_id=spot_id,
        issue_time_utc=issue_time,
        split=split,
        lead_minutes=np.asarray(leads, dtype="int32"),
        station_tensor=station_tensor,
        station_mask=station_mask,
        baseline_tensor=baseline_tensor,
        y_actual=y_actual,
        y_residual=y_residual,
        static_tensor=static_tensor,
    )
    manifest = {
        "tensor_file": "dictionary_tensors.npz",
        "station_shape": list(station_tensor.shape),
        "station_features": station_features,
        "baseline_shape": list(baseline_tensor.shape),
        "baseline_features": future_features,
        "static_shape": list(static_tensor.shape),
        "static_features": static_columns,
        "target_shape": list(y_actual.shape),
        "lead_minutes": leads,
    }
    (output_root / "tensor_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def metric_rows(frame: Any, pd: Any, np: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split, group in frame.groupby("split"):
        split_out = {}
        for actual, baseline in [("target_wind_mean_ms", "baseline_wind_mean_ms"), ("target_gust_ms", "baseline_gust_ms")]:
            valid = group[[actual, baseline]].apply(pd.to_numeric, errors="coerce").dropna()
            if valid.empty:
                split_out[baseline] = {"count": 0}
                continue
            err = valid[baseline].to_numpy(dtype=float) - valid[actual].to_numpy(dtype=float)
            split_out[baseline] = {
                "count": int(len(err)),
                "mae": round(float(np.mean(np.abs(err))), 6),
                "rmse": round(float(np.sqrt(np.mean(err * err))), 6),
                "bias": round(float(np.mean(err)), 6),
            }
        out[str(split)] = split_out
    return out


def write_summary(path: Path, profile: dict[str, Any]) -> None:
    lines = [
        "# CorseWind SAPHIR Dictionary V2",
        "",
        f"Generated: `{profile['generated_at_utc']}`",
        f"Output root: `{profile['output_root']}`",
        "",
        "## Tables",
        "",
        "| Table | Rows | Columns |",
        "| --- | ---: | ---: |",
    ]
    for name, item in profile["tables"].items():
        lines.append(f"| `{name}` | {item['rows']} | {item['columns']} |")
    lines.extend(["", "## Tensor Shapes", "", "```json", json.dumps(profile["tensor_manifest"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Baseline Metrics", "", "```json", json.dumps(profile["baseline_metrics"], indent=2, sort_keys=True), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-table-root", type=Path, required=True)
    parser.add_argument("--run-id-prefix", default="residual_windsup_sst_prev_phys_v3_dem_fetch")
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument("--end-month", default="2026-06")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lead-minutes", default="15,30,45,60")
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--freq-minutes", type=int, default=15)
    parser.add_argument("--max-context-stations", type=int, default=10)
    parser.add_argument("--max-static-features", type=int, default=256)
    parser.add_argument("--train-end", default="2026-01-01T00:00:00Z")
    parser.add_argument("--eval-start", default="2026-01-01T00:00:00Z")
    parser.add_argument("--issue-hour-start", type=int, default=8)
    parser.add_argument("--issue-hour-end", type=int, default=17)
    parser.add_argument("--spot-id", action="append", default=[])
    parser.add_argument("--max-samples-per-spot", type=int, default=400)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-history-interpolate-steps", type=int, default=4)
    parser.add_argument("--max-history-ffill-steps", type=int, default=8)
    parser.add_argument("--require-gust", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = import_deps()
    pd = deps["pd"]
    np = deps["np"]
    pq = deps["pq"]
    leads = parse_int_list(args.lead_minutes)
    paths = discover_parquet_paths(args.training_table_root, args.run_id_prefix, args.start_month, args.end_month)
    columns = schema_columns(paths, pq)
    thin = build_thin_frame(paths, deps)
    samples, future_rows, leads = select_complete_samples(thin, args, deps)
    samples_table = build_samples_table(samples, future_rows, leads, deps)
    future_table = build_future_table(future_rows, samples, deps)

    groups = feature_column_groups(columns)
    context_columns = groups["context"]
    selected_feature_columns = sorted(set(groups["context"] + groups["static"]))
    lead_feature_columns = sorted(set(groups["offsets"] + groups["vertical"]))
    issue_features = load_lead_feature_rows(paths, samples, selected_feature_columns, [min(leads)], deps)
    static_context = build_static_context(issue_features, groups["static"], samples, deps)
    context_snapshot = build_context_station_snapshot(issue_features, context_columns, samples, deps)
    context_slots_by_sample = build_context_slots_by_sample(context_snapshot, args.max_context_stations, pd)
    wanted_station_ids = {
        str(slot["station_id"])
        for slots in context_slots_by_sample.values()
        for slot in slots
        if slot.get("station_id") not in {None, ""}
    }

    target_series = build_target_observation_series(paths, deps)
    context_series = load_all_context_issue_rows(paths, context_columns, min(leads), wanted_station_ids, deps)
    station_sequence, station_sequence_skipped = build_station_sequence(
        samples,
        target_series,
        context_slots_by_sample,
        context_series,
        args,
        deps,
    )

    lead_features = load_lead_feature_rows(paths, samples, lead_feature_columns, leads, deps)
    nwp_offsets = build_nwp_surface_offsets(lead_features, groups["offsets"], samples, deps)
    vertical_profile = build_vertical_profile(lead_features, groups["vertical"], samples, deps)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    tables = {
        "samples": samples_table,
        "future_targets": future_table,
        "station_sequence": station_sequence,
        "context_station_snapshot": context_snapshot,
        "context_station_source_series": context_series,
        "nwp_surface_offsets": nwp_offsets,
        "nwp_vertical_profile": vertical_profile,
        "static_context": static_context,
    }
    for name, frame in tables.items():
        write_frame(frame, output_root / f"{name}.parquet", args.compression)
    tensor_manifest = build_npz_tensors(output_root, samples_table, future_table, station_sequence, static_context, args, deps)
    profile = {
        "format": "corsewind.saphir_dictionary.v2",
        "generated_at_utc": utc_now(),
        "output_root": str(output_root),
        "training_table_root": str(args.training_table_root),
        "run_id_prefix": args.run_id_prefix,
        "start_month": args.start_month,
        "end_month": args.end_month,
        "lead_minutes": leads,
        "context_length": args.context_length,
        "freq_minutes": args.freq_minutes,
        "max_context_stations": args.max_context_stations,
        "source_shards": [str(path) for path in paths],
        "source_row_count": int(len(thin)),
        "selected_sample_count": int(len(samples_table)),
        "sample_split_counts": dict(sorted(Counter(samples_table["split"]).items())),
        "sample_spot_counts": dict(sorted(Counter(samples_table["spot_id"]).items())),
        "wanted_context_station_count": len(wanted_station_ids),
        "feature_column_groups": {key: len(value) for key, value in groups.items()},
        "station_sequence_skipped": station_sequence_skipped,
        "tensor_manifest": tensor_manifest,
        "tables": {name: {"rows": int(len(frame)), "columns": int(len(frame.columns))} for name, frame in tables.items()},
        "baseline_metrics": metric_rows(future_table, pd, np),
        "settings": vars(args),
    }
    (output_root / "dataset_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_summary(output_root / "dataset_summary.md", profile)
    print(json.dumps({
        "output_root": str(output_root),
        "selected_sample_count": profile["selected_sample_count"],
        "tables": profile["tables"],
        "tensor_manifest": tensor_manifest,
        "baseline_metrics": profile["baseline_metrics"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
