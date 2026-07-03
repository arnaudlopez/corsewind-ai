#!/usr/bin/env python3
"""Run the autonomous raw-first ML data collector."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/ml_dataset"))

from meteo_france_client import coverage_ids, endpoint, load_dotenv, request_api  # noqa: E402
import collect_meteo_france_nwp_spot_features as nwp_surface  # noqa: E402
import collect_meteo_france_vertical_profiles as nwp_profiles  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/ml_data_collector_sources.json"
DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))


class CollectorError(RuntimeError):
    pass


class SourceTemporarilyUnavailable(CollectorError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def split_csv(values: list[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        items.extend(item.strip() for item in value.split(","))
    return [item for item in items if item]


def config_sources(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["key"]: item for item in config.get("sources", [])}


def env_is_set(name: str) -> bool:
    return bool(os.getenv(name))


def source_required_env_missing(source: dict[str, Any]) -> list[str]:
    return [name for name in source.get("required_env", []) if not env_is_set(name)]


def storage_free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def command_to_string(command: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def run_command(command: list[str], dry_run: bool, timeout_sec: int | None = None) -> dict[str, Any]:
    started = time.time()
    printable = command_to_string(command)
    if dry_run:
        return {
            "cmd": printable,
            "status": "dry_run",
            "elapsed_s": 0.0,
            "returncode": None,
        }
    completed = subprocess.run(
        [str(item) for item in command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    result = {
        "cmd": printable,
        "status": "ok" if completed.returncode == 0 else "failed",
        "elapsed_s": round(time.time() - started, 3),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_tail": completed.stdout[-8000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        raise CollectorError(
            f"Command failed: {printable}\n"
            f"STDOUT:\n{completed.stdout[-2000:]}\n"
            f"STDERR:\n{completed.stderr[-2000:]}"
        )
    return result


def parse_command_json(result: dict[str, Any]) -> dict[str, Any] | None:
    text = str(result.get("stdout") or result.get("stdout_tail") or "").strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def load_state(path: Path) -> dict[str, Any]:
    return read_json(path) or {
        "format": "corsewind.ml_data_collector.state.v1",
        "created_at_utc": utc_now(),
        "sources": {},
        "history": [],
    }


def source_state(state: dict[str, Any], key: str) -> dict[str, Any]:
    sources = state.setdefault("sources", {})
    item = sources.setdefault(key, {})
    item.setdefault("last_success_at_utc", None)
    item.setdefault("last_failure_at_utc", None)
    item.setdefault("last_error", None)
    item.setdefault("consecutive_failures", 0)
    item.setdefault("last_run_key", None)
    return item


def validate_selection(config: dict[str, Any], selected: list[str], mode: str, ml_root: Path, dry_run: bool) -> list[str]:
    sources = config_sources(config)
    errors: list[str] = []
    unknown = sorted(set(selected) - set(sources))
    if unknown:
        errors.append(f"Unknown source(s): {', '.join(unknown)}")
        return errors
    min_free_gb = float(config.get("defaults", {}).get("minimum_free_gb") or 0)
    free_gb = storage_free_gb(ml_root)
    if free_gb < min_free_gb and not (dry_run and mode != "official"):
        errors.append(f"Not enough free space at {ml_root}: {free_gb:.1f} GiB < {min_free_gb:.1f} GiB")
    if mode != "official":
        return errors

    required = [item["key"] for item in config.get("sources", []) if item.get("required_for_official")]
    missing_required = sorted(set(required) - set(selected))
    if missing_required:
        errors.append(f"Official mode must include all required sources: {', '.join(missing_required)}")
    for key in selected:
        source = sources[key]
        if source.get("required_for_official") and not source.get("official_ready"):
            errors.append(f"{key} is required but not official_ready: {source.get('blocking_reason') or 'no reason recorded'}")
        missing_env = source_required_env_missing(source)
        if missing_env:
            errors.append(f"{key} missing required env: {', '.join(missing_env)}")
    if "beacon_live_observations" in selected:
        if not (os.getenv("BEACON_LIVE_WEATHER_STATE_SSH") or os.getenv("BEACON_LIVE_WEATHER_STATE")):
            errors.append("beacon_live_observations requires BEACON_LIVE_WEATHER_STATE_SSH or BEACON_LIVE_WEATHER_STATE")
    return errors


def get_capabilities(product: str, resolution: str, auth_header: str) -> list[str]:
    response = request_api(
        endpoint(product, resolution, "GetCapabilities"),
        [("service", "WCS"), ("version", "2.0.1"), ("language", "eng")],
        auth_header,
    )
    return coverage_ids(response.text)


def latest_common_surface_run(source: dict[str, Any], auth_header: str) -> datetime:
    model = str(source["source"])
    resolution = str(source["resolution"])
    capabilities = get_capabilities(model, resolution, auth_header)
    features = nwp_surface.selected_features(model, set(source.get("features") or []))
    run_feature_counts: dict[datetime, int] = {}
    for feature in features:
        feature_runs = {
            parsed
            for coverage in capabilities
            if (parsed := nwp_surface.coverage_run_time(coverage, feature["prefix"], str(feature.get("suffix") or "")))
        }
        for run_time in feature_runs:
            run_feature_counts[run_time] = run_feature_counts.get(run_time, 0) + 1
    if not run_feature_counts:
        raise SourceTemporarilyUnavailable(
            f"No {model} {resolution} run found for configured surface features."
        )
    full_runs = [run_time for run_time, count in run_feature_counts.items() if count == len(features)]
    if full_runs:
        return max(full_runs)
    minimum_count = int(source.get("minimum_run_feature_count") or max(1, len(features) // 2))
    partial_runs = [run_time for run_time, count in run_feature_counts.items() if count >= minimum_count]
    if partial_runs:
        return max(partial_runs)
    best_count = max(run_feature_counts.values())
    best_runs = [run_time for run_time, count in run_feature_counts.items() if count == best_count]
    return max(best_runs)


def latest_common_profile_run(auth_header: str) -> datetime:
    capabilities = get_capabilities("arome", "0025", auth_header)
    runs: set[datetime] | None = None
    for feature in nwp_profiles.FEATURES.values():
        feature_runs = {
            parsed
            for coverage in capabilities
            if (parsed := nwp_profiles.coverage_run_time(coverage, feature["prefix"]))
        }
        runs = feature_runs if runs is None else runs & feature_runs
    if not runs:
        raise SourceTemporarilyUnavailable("No common AROME 0.025 run found for vertical profile features.")
    return max(runs)


def generated_valid_times(run_time: datetime, key: str, max_steps: int) -> list[str]:
    if key == "meteo_france_aromepi_surface_fields":
        return [iso_z(run_time + timedelta(minutes=15 * (index + 1))) for index in range(max_steps)]
    return [iso_z(run_time + timedelta(hours=index)) for index in range(max_steps)]


def window_end_start(now: datetime, *, hours: int = 0, minutes: int = 0, end_lag_hours: int = 0, end_lag_minutes: int = 0) -> tuple[datetime, datetime]:
    end = now - timedelta(hours=end_lag_hours, minutes=end_lag_minutes)
    end = end.replace(second=0, microsecond=0)
    if hours:
        end = end.replace(minute=0)
        return end - timedelta(hours=hours - 1), end
    return end - timedelta(minutes=minutes), end


def base_output(ml_root: Path, relative: str) -> str:
    return str(ml_root / relative)


def build_command(source: dict[str, Any], args: argparse.Namespace, now: datetime) -> tuple[list[str], str | None]:
    key = source["key"]
    py = sys.executable
    bbox = [str(item) for item in args.bbox]
    common = ["--registry", str(args.registry)]
    if key == "beacon_live_observations":
        command = [
            py,
            "scripts/ml_dataset/import_beacon_live_observations.py",
            *common,
            "--output-root",
            base_output(args.ml_root, "observations/beacon_live"),
        ]
        if os.getenv("BEACON_LIVE_WEATHER_STATE_SSH"):
            command.extend(["--weather-state-ssh", os.environ["BEACON_LIVE_WEATHER_STATE_SSH"]])
            if os.getenv("BEACON_LIVE_WEATHER_STATE_SSH_IDENTITY_FILE"):
                command.extend(["--weather-state-ssh-identity-file", os.environ["BEACON_LIVE_WEATHER_STATE_SSH_IDENTITY_FILE"]])
        elif os.getenv("BEACON_LIVE_WEATHER_STATE"):
            command.extend(["--weather-state", os.environ["BEACON_LIVE_WEATHER_STATE"]])
        return command, None

    if key == "meteo_france_live_observations":
        command = [
            py,
            "scripts/ml_dataset/collect_meteo_france_observations.py",
            *common,
            "--output-root",
            base_output(args.ml_root, "observations/meteo_france"),
        ]
        for mode in source.get("modes", []):
            command.extend(["--mode", mode])
        return command, None

    if key == "remote_wind2d_model_layers":
        remote_container = os.getenv("ML_REMOTE_MODEL_CONTAINER_SSH")
        if not remote_container:
            raise CollectorError("remote_wind2d_model_layers requires ML_REMOTE_MODEL_CONTAINER_SSH")
        command = [
            py,
            "scripts/ml_dataset/collect_remote_wind2d_model_layers.py",
            "--remote-container-ssh",
            remote_container,
            *common,
            "--model-runs-root",
            base_output(args.ml_root, "model_runs"),
            "--model-samples-root",
            base_output(args.ml_root, "model_samples"),
            "--sample-method",
            str(source.get("sample_method") or "bilinear"),
        ]
        for source_name in source.get("sources", []):
            command.extend(["--source", str(source_name)])
        identity_file = os.getenv("ML_REMOTE_MODEL_SSH_IDENTITY_FILE")
        if identity_file:
            command.extend(["--ssh-identity-file", identity_file])
        if source.get("include_context_spots", args.include_context_spots):
            command.append("--include-context-spots")
        else:
            command.append("--no-include-context-spots")
        if bool(source.get("skip_existing_runs", True)):
            command.append("--skip-existing-runs")
        else:
            command.append("--no-skip-existing-runs")
        return command, "remote_wind2d_model_layers"

    if source.get("kind") == "native_nwp_surface_fields":
        run_time = (
            now.replace(minute=0, second=0, microsecond=0)
            if args.dry_run and source_required_env_missing(source)
            else latest_common_surface_run(source, args.auth_header)
        )
        valid_times = generated_valid_times(run_time, key, int(source.get("max_steps") or 1))
        command = [
            py,
            "scripts/ml_dataset/collect_meteo_france_nwp_spot_features.py",
            "--source",
            source["source"],
            "--resolution",
            source["resolution"],
            "--run-time-utc",
            iso_z(run_time),
            "--bbox",
            *bbox,
            *common,
            "--raw-root",
            base_output(args.ml_root, "meteo_france_nwp/raw/extra_fields"),
            "--output-root",
            base_output(args.ml_root, "meteo_france_nwp/extra_field_samples"),
            "--max-steps",
            str(source.get("max_steps") or 1),
            "--request-sleep-sec",
            str(args.request_sleep_sec),
        ]
        for valid_time in valid_times:
            command.extend(["--valid-time-utc", valid_time])
        for feature in source.get("features", []):
            command.extend(["--feature", feature])
        if source.get("include_context_spots", args.include_context_spots):
            command.append("--include-context-spots")
        return command, f"{source['source']}:{iso_z(run_time)}"

    if key == "meteo_france_arome_vertical_profiles":
        run_time = (
            now.replace(minute=0, second=0, microsecond=0)
            if args.dry_run and source_required_env_missing(source)
            else latest_common_profile_run(args.auth_header)
        )
        valid_times = generated_valid_times(run_time, key, int(source.get("max_steps") or 1))
        command = [
            py,
            "scripts/ml_dataset/collect_meteo_france_vertical_profiles.py",
            "--run-time-utc",
            iso_z(run_time),
            "--bbox",
            *bbox,
            *common,
            "--raw-root",
            base_output(args.ml_root, "meteo_france_nwp/raw/vertical_profiles"),
            "--output-root",
            base_output(args.ml_root, "meteo_france_nwp/vertical_profiles"),
            "--max-steps",
            str(source.get("max_steps") or 1),
            "--request-sleep-sec",
            str(args.request_sleep_sec),
        ]
        for valid_time in valid_times:
            command.extend(["--valid-time-utc", valid_time])
        for level in source.get("pressure_levels_hpa", []):
            command.extend(["--pressure-level-hpa", str(level)])
        if source.get("include_context_spots", args.include_context_spots):
            command.append("--include-context-spots")
        return command, f"arome_profiles:{iso_z(run_time)}"

    if key == "copernicus_marine_sst":
        start, end = window_end_start(
            now,
            hours=int(source.get("window_hours") or 1),
            end_lag_hours=int(source.get("end_lag_hours") or 0),
        )
        command = [
            py,
            "scripts/ml_dataset/collect_copernicus_marine_sst.py",
            "--start-datetime",
            start.strftime("%Y-%m-%dT%H:%M:%S"),
            "--end-datetime",
            end.strftime("%Y-%m-%dT%H:%M:%S"),
            *common,
            "--raw-root",
            base_output(args.ml_root, "copernicus_marine/raw/sst"),
            "--output-root",
            base_output(args.ml_root, "copernicus_marine/sst_samples"),
            "--output-filename",
            f"sst_corse_{start:%Y%m%dT%H}_{end:%Y%m%dT%H}.nc",
        ]
        if args.include_context_spots:
            command.append("--include-context-spots")
        return command, f"{start:%Y%m%dT%H}_{end:%Y%m%dT%H}"

    if key == "eumetsat_cloud_mask":
        start, end = window_end_start(
            now,
            minutes=int(source.get("window_minutes") or 120),
            end_lag_minutes=int(source.get("end_lag_minutes") or 5),
        )
        command = [
            py,
            "scripts/ml_dataset/collect_eumetsat_cloud_mask.py",
            "--start-datetime",
            iso_z(start),
            "--end-datetime",
            iso_z(end),
            "--bbox",
            ",".join(bbox),
            *common,
            "--raw-root",
            base_output(args.ml_root, "eumetsat/raw/cloud_mask"),
            "--output-root",
            base_output(args.ml_root, "eumetsat/cloud_mask_samples"),
            "--max-products",
            str(source.get("max_products") or 6),
        ]
        if args.include_context_spots:
            command.append("--include-context-spots")
        return command, f"{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}"

    if str(source.get("key", "")).startswith("eumetsat_") and source.get("product"):
        product = str(source["product"])
        start, end = window_end_start(
            now,
            minutes=int(source.get("window_minutes") or 180),
            end_lag_minutes=int(source.get("end_lag_minutes") or 10),
        )
        output_name = {
            "cloud_type": "cloud_type",
            "land_surface_temperature": "land_surface_temperature",
            "global_instability_indices": "global_instability_indices",
        }[product]
        command = [
            py,
            "scripts/ml_dataset/collect_eumetsat_spot_product.py",
            "--product",
            product,
            "--start-datetime",
            iso_z(start),
            "--end-datetime",
            iso_z(end),
            "--bbox",
            ",".join(bbox),
            *common,
            "--raw-root",
            base_output(args.ml_root, f"eumetsat/raw/{output_name}"),
            "--output-root",
            base_output(args.ml_root, f"eumetsat/{output_name}_samples"),
            "--max-products",
            str(source.get("max_products") or 6),
        ]
        if args.include_context_spots:
            command.append("--include-context-spots")
        return command, f"{product}:{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}"

    raise CollectorError(f"No runner implemented for source {key}")


def run_source(source: dict[str, Any], args: argparse.Namespace, state: dict[str, Any], now: datetime) -> dict[str, Any]:
    key = source["key"]
    item_state = source_state(state, key)
    status: dict[str, Any] = {
        "key": key,
        "provider": source.get("provider"),
        "kind": source.get("kind"),
        "required_for_official": bool(source.get("required_for_official")),
    }
    try:
        command, run_key = build_command(source, args, now)
        status["run_key"] = run_key
        can_skip_last_run = (
            bool(source.get("collector_skip_last_run", True))
            and args.skip_existing_runs
            and
            "last_success_row_count" in item_state
            and int(item_state.get("last_success_row_count") or 0) > 0
        )
        if run_key and item_state.get("last_run_key") == run_key and can_skip_last_run:
            status["status"] = "skipped_existing"
            return status
        result = run_command(command, args.dry_run, args.command_timeout_sec)
        parsed = parse_command_json(result) or {}
        status.update({
            "status": result.get("status"),
            "command": {
                "cmd": result.get("cmd"),
                "elapsed_s": result.get("elapsed_s"),
                "returncode": result.get("returncode"),
            },
            "parsed": parsed,
        })
        if not args.dry_run:
            item_state["last_success_at_utc"] = utc_now()
            item_state["last_failure_at_utc"] = None
            item_state["last_error"] = None
            item_state["last_unavailable_at_utc"] = None
            item_state["last_unavailable_reason"] = None
            item_state["consecutive_failures"] = 0
            item_state["last_success_row_count"] = parsed.get("row_count")
            if run_key:
                item_state["last_run_key"] = run_key
        return status
    except SourceTemporarilyUnavailable as exc:
        if not args.dry_run:
            item_state["last_unavailable_at_utc"] = utc_now()
            item_state["last_unavailable_reason"] = str(exc)
            item_state["last_error"] = None
            item_state["consecutive_failures"] = 0
        status.update({
            "status": "temporarily_unavailable",
            "reason": str(exc),
            "retriable": True,
        })
        return status
    except Exception as exc:
        if not args.dry_run:
            item_state["last_failure_at_utc"] = utc_now()
            item_state["last_error"] = str(exc)
            item_state["consecutive_failures"] = int(item_state.get("consecutive_failures") or 0) + 1
        status.update({"status": "failed", "error": str(exc)})
        if args.fail_fast:
            raise
        return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=["official", "smoke", "debug", "bridge"], default="official")
    parser.add_argument("--source", action="append", default=[], help="Source key to run. Repeatable or comma-separated. Defaults to official required sources.")
    parser.add_argument("--ml-root", type=Path, default=DEFAULT_ML_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--bbox", nargs=4, type=float, default=None, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--auth-header", choices=["apikey", "bearer"], default="apikey")
    parser.add_argument("--request-sleep-sec", type=float, default=0.2)
    parser.add_argument("--command-timeout-sec", type=int, default=1800)
    parser.add_argument("--include-context-spots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing-runs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    args.config = resolve_path(args.config)
    args.ml_root = resolve_path(args.ml_root)
    args.registry = resolve_path(args.registry)
    args.state_file = resolve_path(args.state_file) if args.state_file else args.ml_root / "collector_state.json"
    args.status_file = resolve_path(args.status_file) if args.status_file else args.ml_root / "collector_status.json"
    config = read_json(args.config)
    if not config:
        raise SystemExit(f"Collector config not found: {args.config}")
    sources_by_key = config_sources(config)
    selected = split_csv(args.source)
    if not selected:
        selected = [item["key"] for item in config.get("sources", []) if item.get("required_for_official")]
    args.bbox = args.bbox or config.get("defaults", {}).get("bbox_wgs84") or [8.45, 41.25, 9.75, 43.1]

    validation_errors = validate_selection(config, selected, args.mode, args.ml_root, args.dry_run)
    state = load_state(args.state_file)
    started = time.time()
    now = datetime.now(timezone.utc)
    status: dict[str, Any] = {
        "format": "corsewind.ml_data_collector.status.v1",
        "generated_at_utc": utc_now(),
        "mode": args.mode,
        "dry_run": bool(args.dry_run),
        "ml_root": str(args.ml_root),
        "registry": str(args.registry),
        "selected_sources": selected,
        "validation_errors": validation_errors,
        "sources": {},
    }
    if validation_errors:
        status["result"] = "blocked"
        status["elapsed_s"] = round(time.time() - started, 3)
        write_json(args.status_file, status)
        print(json.dumps(status, indent=2, ensure_ascii=False))
        raise SystemExit(2)

    for key in selected:
        status["sources"][key] = run_source(sources_by_key[key], args, state, now)
        write_json(args.status_file, status)
        write_json(args.state_file, state)

    failed = [key for key, item in status["sources"].items() if item.get("status") == "failed"]
    unavailable = [
        key for key, item in status["sources"].items()
        if item.get("status") == "temporarily_unavailable"
    ]
    status["result"] = "failed" if failed else "ok"
    status["failed_sources"] = failed
    status["unavailable_sources"] = unavailable
    status["elapsed_s"] = round(time.time() - started, 3)
    state.setdefault("history", []).append({
        "at_utc": utc_now(),
        "mode": args.mode,
        "result": status["result"],
        "selected_sources": selected,
        "failed_sources": failed,
        "unavailable_sources": unavailable,
    })
    state["history"] = state["history"][-50:]
    write_json(args.status_file, status)
    write_json(args.state_file, state)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
