#!/usr/bin/env python3
"""Archive and sample Wind2D model layers from a remote forecast-engine container."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import archive_model_layer_snapshot
import sample_model_layers_at_spots


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_ROOT = Path(os.getenv("ML_DATASET_ROOT", str(ROOT / "data/processed/ml_dataset")))
DEFAULT_REGISTRY = ROOT / "configs/ml_spots.json"
DEFAULT_MODEL_RUNS_ROOT = DEFAULT_ML_ROOT / "model_runs"
DEFAULT_MODEL_SAMPLES_ROOT = DEFAULT_ML_ROOT / "model_samples"

SOURCE_LAYER_PATHS = {
    "arome": "/app/visualizations/wind2d/arome-corsica-latest.json",
    "aromepi": "/app/visualizations/wind2d/aromepi-corsica-latest.json",
    "moloch": "/app/visualizations/wind2d/moloch-corsica-latest.json",
    "icon2i": "/app/visualizations/wind2d/icon2i-corsica-latest.json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_sources(values: list[str]) -> list[str]:
    raw = []
    for value in values:
        raw.extend(item.strip() for item in value.split(","))
    sources = [item for item in raw if item]
    invalid = sorted(set(sources) - set(SOURCE_LAYER_PATHS))
    if invalid:
        raise SystemExit(f"Unknown source(s): {', '.join(invalid)}")
    return sources or list(SOURCE_LAYER_PATHS)


def parse_remote_container_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise SystemExit("--remote-container-ssh must look like user@host:docker://container")
    host, remote = spec.split(":", 1)
    if not host or not remote.startswith("docker://"):
        raise SystemExit("--remote-container-ssh must look like user@host:docker://container")
    container = remote.removeprefix("docker://").strip("/")
    if not container or "/" in container:
        raise SystemExit("--remote-container-ssh must point to a container, not a file path")
    return host, container


def ssh_base_command(identity_file: str | None) -> list[str]:
    command = [
        "ssh",
        "-F",
        "/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "UserKnownHostsFile=~/.ssh/known_hosts",
    ]
    if identity_file:
        command.extend(["-i", identity_file])
    return command


def read_remote_layer(
    remote_container_ssh: str,
    container_path: str,
    *,
    timeout_sec: int,
    identity_file: str | None,
) -> dict[str, Any]:
    host, container = parse_remote_container_spec(remote_container_ssh)
    remote_command = f"docker exec {shlex.quote(container)} cat {shlex.quote(container_path)}"
    command = [*ssh_base_command(identity_file), host, remote_command]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"Failed to read remote layer {container_path}: {completed.stderr[-1200:]}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Remote layer is not valid JSON: {container_path}: {exc}") from exc
    if not payload.get("run_time_utc"):
        raise SystemExit(f"Remote layer has no run_time_utc: {container_path}")
    if not payload.get("forecast_steps"):
        raise SystemExit(f"Remote layer has no forecast_steps: {container_path}")
    return payload


def archive_exists(model_runs_root: Path, source: str, run_time_utc: str) -> bool:
    run_slug = archive_model_layer_snapshot.safe_time_slug(run_time_utc)
    return (model_runs_root / source / f"run_{run_slug}" / "summary.json").exists()


def write_temp_layer(source: str, payload: dict[str, Any], tmp_root: Path) -> Path:
    path = tmp_root / f"{source}.json"
    path.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    return path


def collect_source(
    source: str,
    args: argparse.Namespace,
    spots: list[dict[str, Any]],
    tmp_root: Path,
) -> dict[str, Any]:
    payload = read_remote_layer(
        args.remote_container_ssh,
        SOURCE_LAYER_PATHS[source],
        timeout_sec=args.ssh_timeout_sec,
        identity_file=args.ssh_identity_file,
    )
    run_time_utc = str(payload["run_time_utc"])
    existed = archive_exists(args.model_runs_root, source, run_time_utc)
    result: dict[str, Any] = {
        "source": source,
        "remote_path": SOURCE_LAYER_PATHS[source],
        "run_time_utc": run_time_utc,
        "generated_at_utc": payload.get("generated_at_utc"),
        "forecast_step_count": len(payload.get("forecast_steps") or []),
        "already_archived": existed,
    }
    if existed and args.skip_existing_runs and not args.force:
        result["status"] = "skipped_existing"
        return result

    temp_layer = write_temp_layer(source, payload, tmp_root)
    archive_summary = archive_model_layer_snapshot.archive_layer(source, temp_layer, args.model_runs_root)
    rows = sample_model_layers_at_spots.sample_layer(source, payload, spots, args.sample_method)
    written = sample_model_layers_at_spots.write_jsonl_by_valid_day(args.model_samples_root, source, rows)
    result.update({
        "status": "collected",
        "archive": archive_summary,
        "spot_count": len(spots),
        "row_count": len(rows),
        "written": written,
    })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-container-ssh", required=True, help="Remote forecast engine, e.g. home@192.168.1.101:docker://corsewind-forecast-engine.")
    parser.add_argument("--source", action="append", default=[], help="Source to collect: arome, aromepi, moloch, icon2i. Repeatable or comma-separated.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--model-runs-root", type=Path, default=DEFAULT_MODEL_RUNS_ROOT)
    parser.add_argument("--model-samples-root", type=Path, default=DEFAULT_MODEL_SAMPLES_ROOT)
    parser.add_argument("--sample-method", choices=["bilinear", "nearest"], default="bilinear")
    parser.add_argument("--include-context-spots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing-runs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ssh-identity-file", default=os.getenv("ML_REMOTE_MODEL_SSH_IDENTITY_FILE"))
    parser.add_argument("--ssh-timeout-sec", type=int, default=int(os.getenv("ML_REMOTE_MODEL_SSH_TIMEOUT_SEC", "120")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.registry = resolve_path(args.registry)
    args.model_runs_root = resolve_path(args.model_runs_root)
    args.model_samples_root = resolve_path(args.model_samples_root)
    sources = parse_sources(args.source)
    spots = sample_model_layers_at_spots.load_spots(args.registry, args.include_context_spots, set())

    with tempfile.TemporaryDirectory(prefix="corsewind_remote_layers_") as tmp:
        tmp_root = Path(tmp)
        results = [collect_source(source, args, spots, tmp_root) for source in sources]

    print(json.dumps({
        "generated_at_utc": utc_now(),
        "remote_container_ssh": args.remote_container_ssh,
        "sources": sources,
        "registry": str(args.registry),
        "model_runs_root": str(args.model_runs_root),
        "model_samples_root": str(args.model_samples_root),
        "sample_method": args.sample_method,
        "include_context_spots": bool(args.include_context_spots),
        "spot_count": len(spots),
        "collected_count": sum(1 for item in results if item.get("status") == "collected"),
        "skipped_existing_count": sum(1 for item in results if item.get("status") == "skipped_existing"),
        "row_count": sum(int(item.get("row_count") or 0) for item in results),
        "results": results,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
