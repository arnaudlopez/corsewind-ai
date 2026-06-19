#!/usr/bin/env python3
"""Validate prod-like FastEddy packages without requiring FastEddy binaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS = ROOT / "data/processed/benchmarks/fasteddy/prod_like_status.json"
DEFAULT_ICBC_CONTRACT = ROOT / "benchmarks/fasteddy/icbc_contract.json"
DEFAULT_WIND2D_CONTRACT = ROOT / "benchmarks/fasteddy/wind2d_output_contract.json"
DEFAULT_REPORT_JSON = ROOT / "data/processed/benchmarks/fasteddy/prod_like_validation.json"
DEFAULT_REPORT_MD = ROOT / "reports/fasteddy_prod_like_validation.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str, blockers: list[str]) -> None:
    if not condition:
        blockers.append(message)


def finite_range(values: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(values)
    if not finite.any():
        return {"finite": False, "min": None, "max": None}
    return {
        "finite": bool(finite.all()),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
    }


def validate_roughness_table(path: Path) -> tuple[dict[int, float], list[str]]:
    blockers: list[str] = []
    lookup: dict[int, float] = {}
    require(path.exists(), f"missing roughness table {display_path(path)}", blockers)
    if not path.exists():
        return lookup, blockers
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"class", "name", "z0m", "source", "calibration_status"}
        require(set(reader.fieldnames or []) >= required_columns, f"roughness table missing columns: {required_columns}", blockers)
        for row in reader:
            try:
                class_id = int(row["class"])
                z0m = float(row["z0m"])
            except (KeyError, ValueError) as exc:
                blockers.append(f"invalid roughness row {row}: {exc}")
                continue
            require(z0m > 0.0, f"roughness z0m must be positive for class {class_id}", blockers)
            lookup[class_id] = z0m
    return lookup, blockers


def validate_gis(path: Path, expected_classes: list[int], roughness_lookup: dict[int, float]) -> tuple[dict[str, Any], list[str]]:
    from netCDF4 import Dataset

    blockers: list[str] = []
    summary: dict[str, Any] = {"path": display_path(path)}
    require(path.exists(), f"missing GeoSpec GIS NetCDF {display_path(path)}", blockers)
    if not path.exists():
        return summary, blockers
    with Dataset(path) as ds:
        variables = set(ds.variables)
        for name in ["x", "y", "cellsize", "elevation", "lat", "lon", "LandCover"]:
            require(name in variables, f"GIS file missing variable {name}", blockers)
        if blockers:
            return summary, blockers
        elevation = np.array(ds.variables["elevation"][:], dtype=np.float32)
        lat = np.array(ds.variables["lat"][:], dtype=np.float64)
        lon = np.array(ds.variables["lon"][:], dtype=np.float64)
        landcover = np.array(ds.variables["LandCover"][:], dtype=np.int32)
        classes = sorted(int(value) for value in np.unique(landcover))
        summary.update(
            {
                "shape_yx": list(elevation.shape),
                "elevation": finite_range(elevation),
                "lat": finite_range(lat),
                "lon": finite_range(lon),
                "landcover_classes": classes,
            }
        )
        require(elevation.shape == landcover.shape == lat.shape == lon.shape, "GIS variable shapes are inconsistent", blockers)
        require(finite_range(elevation)["finite"], "GIS elevation contains non-finite values", blockers)
        require(finite_range(lat)["finite"] and finite_range(lon)["finite"], "GIS lat/lon contains non-finite values", blockers)
        require(set(classes).issubset(set(roughness_lookup)), f"GIS contains landcover classes without z0m lookup: {classes}", blockers)
        require(set(expected_classes).issubset(set(classes)) or set(classes).issubset(set(expected_classes)), "GIS classes differ from case manifest classes", blockers)
    return summary, blockers


def validate_bridge(path: Path, icbc_contract: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    import xarray as xr

    blockers: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {"path": display_path(path)}
    require(path.exists(), f"missing AROME bridge NetCDF {display_path(path)}", blockers)
    if not path.exists():
        return summary, blockers, warnings
    ds = xr.open_dataset(path)
    required_variables = set(icbc_contract["required_bridge_variables"])
    missing = sorted(required_variables - set(ds.data_vars))
    require(not missing, f"AROME bridge missing variables: {missing}", blockers)
    required_dims = set(icbc_contract["required_bridge_dimensions"])
    missing_dims = sorted(required_dims - set(ds.dims))
    require(not missing_dims, f"AROME bridge missing dimensions: {missing_dims}", blockers)
    fields: dict[str, Any] = {}
    for name in sorted(set(ds.data_vars) & required_variables):
        values = ds[name].values
        fields[name] = {"dims": list(ds[name].dims), "shape": list(ds[name].shape), **finite_range(values)}
        require(fields[name]["finite"], f"AROME bridge variable {name} contains non-finite values", blockers)
    if "QVAPOR" in ds:
        qv = ds["QVAPOR"]
        require(bool(((qv >= 0.0) & (qv < 0.05)).all()), "QVAPOR outside expected 0..0.05 kg/kg range", blockers)
    if "RH" in ds:
        rh = ds["RH"]
        require(bool(((rh >= 0.0) & (rh <= 100.0)).all()), "RH outside 0..100 percent", blockers)
    if "W" in ds:
        units = str(ds["W"].attrs.get("units", ""))
        if "m" not in units:
            warnings.append("W variable units do not explicitly indicate geometric m/s")
    summary["fields"] = fields
    return summary, blockers, warnings


def validate_case(case: dict[str, Any], icbc_contract: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    case_dir = ROOT / case["case_dir"]
    require(case_dir.exists(), f"missing case dir {display_path(case_dir)}", blockers)
    roughness_table = ROOT / case["surface"]["landcover_table"]
    roughness_lookup, roughness_blockers = validate_roughness_table(roughness_table)
    blockers.extend(roughness_blockers)
    gis_summary, gis_blockers = validate_gis(ROOT / case["surface"]["geospec_input"], case["surface"]["worldcover_classes"], roughness_lookup)
    blockers.extend(gis_blockers)
    bridge_summary, bridge_blockers, bridge_warnings = validate_bridge(ROOT / case["forcing"]["path"], icbc_contract)
    blockers.extend(bridge_blockers)
    warnings.extend(bridge_warnings)
    for label, rel_path in case["configs"].items():
        path = ROOT / rel_path
        require(path.exists(), f"missing config {label}: {display_path(path)}", blockers)
        if path.suffix == ".json" and path.exists():
            try:
                load_json(path)
            except json.JSONDecodeError as exc:
                blockers.append(f"invalid JSON config {display_path(path)}: {exc}")
    params = ROOT / case["configs"]["fasteddy_params"]
    require(params.exists(), f"missing FastEddy params {display_path(params)}", blockers)
    if params.exists():
        text = params.read_text(encoding="utf-8")
        for token in ["Nx =", "Ny =", "Nz =", "d_xi =", "d_eta =", "d_zeta =", "hydroBCs ="]:
            require(token in text, f"FastEddy params missing token {token}", blockers)
    grid = case["grid"]
    require(grid["surface_shape_yx"][0] > 0 and grid["surface_shape_yx"][1] > 0, "surface grid dimensions must be positive", blockers)
    require(grid["solver_shape_xyz"][0] > 0 and grid["solver_shape_xyz"][1] > 0 and grid["solver_shape_xyz"][2] > 0, "solver grid dimensions must be positive", blockers)
    return {
        "zone_id": case["zone"]["id"],
        "case_dir": case["case_dir"],
        "blockers": blockers,
        "warnings": warnings,
        "gis": gis_summary,
        "bridge": bridge_summary,
    }


def validate_contract(path: Path, required_top_keys: set[str]) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    require(path.exists(), f"missing contract {display_path(path)}", blockers)
    if not path.exists():
        return {}, blockers
    payload = load_json(path)
    missing = sorted(required_top_keys - set(payload))
    require(not missing, f"contract {display_path(path)} missing top-level keys: {missing}", blockers)
    return payload, blockers


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# FastEddy Prod-Like Package Validation",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Status: `{report['status']}`",
        f"- Blockers: `{len(report['blockers'])}`",
        f"- Warnings: `{len(report['warnings'])}`",
        "",
    ]
    if report["blockers"]:
        lines.extend(["## Blockers", ""])
        lines.extend(f"- {item}" for item in report["blockers"])
        lines.append("")
    if report["warnings"]:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {item}" for item in report["warnings"])
        lines.append("")
    lines.extend(["## Cases", ""])
    for case in report["cases"]:
        lines.extend(
            [
                f"### {case['zone_id']}",
                "",
                f"- Case dir: `{case['case_dir']}`",
                f"- Blockers: `{case['blockers']}`",
                f"- Warnings: `{case['warnings']}`",
                f"- GIS shape: `{case['gis'].get('shape_yx')}`",
                f"- Bridge fields: `{sorted(case['bridge'].get('fields', {}))}`",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--icbc-contract", type=Path, default=DEFAULT_ICBC_CONTRACT)
    parser.add_argument("--wind2d-contract", type=Path, default=DEFAULT_WIND2D_CONTRACT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_REPORT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status_path = args.status if args.status.is_absolute() else ROOT / args.status
    icbc_path = args.icbc_contract if args.icbc_contract.is_absolute() else ROOT / args.icbc_contract
    wind2d_path = args.wind2d_contract if args.wind2d_contract.is_absolute() else ROOT / args.wind2d_contract
    json_output = args.json_output if args.json_output.is_absolute() else ROOT / args.json_output
    markdown_output = args.markdown_output if args.markdown_output.is_absolute() else ROOT / args.markdown_output
    status = load_json(status_path)
    blockers: list[str] = []
    warnings: list[str] = []
    icbc_contract, icbc_blockers = validate_contract(icbc_path, {"format", "input", "required_bridge_variables", "required_outputs"})
    wind2d_contract, wind2d_blockers = validate_contract(wind2d_path, {"format", "required_product", "required_fields_at_display_height", "quality_gates"})
    blockers.extend(icbc_blockers)
    blockers.extend(wind2d_blockers)
    require(status.get("readiness", {}).get("prod_like_package_ready") is True, "prod_like_package_ready is not true", blockers)
    case_reports = [validate_case(case, icbc_contract) for case in status.get("cases", [])]
    for case_report in case_reports:
        blockers.extend(f"{case_report['zone_id']}: {item}" for item in case_report["blockers"])
        warnings.extend(f"{case_report['zone_id']}: {item}" for item in case_report["warnings"])
    if status.get("readiness", {}).get("stock_genicbcs_compatible_now") is not False:
        warnings.append("stock_genicbcs_compatible_now should remain false until the AROME adapter is implemented")
    report = {
        "format": "corsewind.fasteddy.prod_like_validation.v1",
        "generated_at_utc": utc_now(),
        "status": "pass" if not blockers else "fail",
        "source_status": display_path(status_path),
        "icbc_contract": display_path(icbc_path),
        "wind2d_contract": display_path(wind2d_path),
        "blockers": blockers,
        "warnings": warnings,
        "cases": case_reports,
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, markdown_output)
    print(f"wrote {display_path(json_output)}")
    print(f"wrote {display_path(markdown_output)}")
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
