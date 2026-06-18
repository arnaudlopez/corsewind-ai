#!/usr/bin/env python3
"""Validate the AROME parent dataset against FastEddy coupling needs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NETCDF = ROOT / "data/processed/benchmarks/fasteddy/arome_poc/fasteddy_parent_poc.nc"
DEFAULT_MANIFEST = ROOT / "data/processed/benchmarks/fasteddy/arome_poc/fasteddy_parent_poc_manifest.json"
DEFAULT_OUTPUT = ROOT / "reports/fasteddy_parent_input_validation.md"


SOLVER_REQUIREMENTS = [
    {
        "field": "u",
        "expected": "3D eastward wind component",
        "status_if_present": "observed_arome_grib",
        "required_for": "FastEddy IC/BC wind state",
    },
    {
        "field": "v",
        "expected": "3D northward wind component",
        "status_if_present": "observed_arome_grib",
        "required_for": "FastEddy IC/BC wind state",
    },
    {
        "field": "temperature",
        "expected": "3D temperature",
        "status_if_present": "observed_arome_grib",
        "required_for": "Potential-temperature/base-state construction",
    },
    {
        "field": "relative_humidity",
        "expected": "3D humidity",
        "status_if_present": "observed_arome_grib_requires_conversion",
        "required_for": "Moist thermodynamic state",
    },
    {
        "field": "geopotential_or_height",
        "expected": "3D geopotential/height information",
        "status_if_present": "observed_arome_grib",
        "required_for": "Vertical coordinate mapping",
    },
    {
        "field": "height_m",
        "expected": "3D height in meters",
        "status_if_present": "physically_derived_from_geopotential",
        "required_for": "Vertical coordinate mapping",
    },
    {
        "field": "potential_temperature",
        "expected": "3D potential temperature",
        "status_if_present": "physically_derived_from_temperature_pressure",
        "required_for": "FastEddy thermodynamic state",
    },
    {
        "field": "surface_temperature",
        "expected": "2D ground/surface temperature",
        "status_if_present": "observed_arome_grib",
        "required_for": "Surface thermal forcing",
    },
    {
        "field": "topography_m",
        "expected": "2D terrain elevation",
        "status_if_present": "external_copernicus_dem",
        "required_for": "GeoSpec/SimGrid terrain",
    },
    {
        "field": "landmask",
        "expected": "2D land/sea mask",
        "status_if_present": "derived_from_dem_threshold",
        "required_for": "GeoSpec/SimGrid surface classification",
    },
    {
        "field": "z0m",
        "expected": "2D momentum roughness length",
        "status_if_present": "poc_heuristic_replace_before_production",
        "required_for": "GeoSpec/SimGrid roughness",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def finite_range(data_array: Any) -> dict[str, Any]:
    import numpy as np

    values = data_array.values
    finite = np.isfinite(values)
    if not finite.any():
        return {"finite": False, "min": None, "max": None}
    return {"finite": bool(finite.all()), "min": float(np.nanmin(values)), "max": float(np.nanmax(values))}


def validate_dataset(netcdf_path: Path, manifest_path: Path) -> dict[str, Any]:
    import numpy as np
    import xarray as xr

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    dataset = xr.open_dataset(netcdf_path)
    fields = []
    blockers = []
    warnings = []

    for requirement in SOLVER_REQUIREMENTS:
        name = requirement["field"]
        present = name in dataset
        field_status = "missing"
        field_range: dict[str, Any] | None = None
        attrs: dict[str, Any] = {}
        if present:
            field_status = requirement["status_if_present"]
            field_range = finite_range(dataset[name])
            attrs = {key: str(value) for key, value in dataset[name].attrs.items()}
            if field_range and not field_range["finite"]:
                blockers.append(f"{name}: contains non-finite values")
            if field_status.startswith("poc_heuristic"):
                warnings.append(f"{name}: heuristic only, not a validated source field")
        else:
            blockers.append(f"{name}: missing")
        fields.append(
            {
                **requirement,
                "present": present,
                "validation_status": field_status,
                "range": field_range,
                "attrs": attrs,
            }
        )

    if "relative_humidity" in dataset:
        rh = dataset["relative_humidity"]
        if not bool(((rh >= 0) & (rh <= 100)).all()):
            blockers.append("relative_humidity: outside 0-100%")
    if "temperature" in dataset:
        temp = dataset["temperature"]
        if not bool(((temp > 180) & (temp < 330)).all()):
            blockers.append("temperature: outside broad physical sanity range 180-330 K")
    if "height_m" in dataset and "pressure_hpa" in dataset.coords:
        mean_heights = dataset["height_m"].mean(dim=("latitude", "longitude")).values
        if not bool(np.all(np.diff(mean_heights) > 0)):
            blockers.append("height_m: mean heights do not increase from 850 to 300 hPa")
    if "landmask" in dataset:
        unique = sorted(float(value) for value in np.unique(dataset["landmask"].values))
        if unique != [0.0, 1.0]:
            warnings.append(f"landmask: expected binary 0/1, got {unique}")

    missing_inputs = list(manifest.get("missing_inputs") or [])
    if "surface_pressure" in missing_inputs:
        warnings.append("surface_pressure: not present; desired for hydrostatic/base-state consistency")

    return {
        "format": "corsewind.fasteddy_parent_input_validation.v1",
        "generated_at_utc": utc_now(),
        "netcdf": display_path(netcdf_path),
        "manifest": display_path(manifest_path),
        "source_run_time_utc": manifest.get("plan_run_time_utc") or manifest.get("source_run_time_utc"),
        "decoded_grib_inputs": manifest.get("decoded_grib_inputs"),
        "expected_grib_inputs": manifest.get("expected_grib_inputs"),
        "verdict": {
            "solver_input_directly_runnable": False,
            "parent_dataset_ready_for_icbc_converter_poc": not blockers,
            "no_invented_meteorology": True,
            "production_truth_ready": False,
            "blockers": blockers,
            "warnings": warnings,
        },
        "fields": fields,
        "strategy_to_remove_heuristics": [
            "Use official AROME isobaric GRIB fields for 3D U, V, T, relative humidity, geopotential and pressure levels.",
            "Convert relative humidity to specific humidity or mixing ratio before final GenICBCs/FastEddy coupling.",
            "Convert VV__ISOBARIC pressure vertical velocity from Pa/s only if the selected FastEddy converter path requires geometric w.",
            "Use Copernicus GLO-30, or higher-resolution local DEM where available, for elevation.",
            "Replace POC z0m constants with coastline plus land-cover based roughness, ideally Copernicus/Corine classes mapped to FastEddy land-cover metadata.",
            "Fetch or derive surface pressure if needed by the final base-state/ICBC conversion.",
            "Generate real FastEddy GeoSpec, SimGrid and GenICBCs outputs; do not pass this parent NetCDF directly to FastEddy as a final input.",
        ],
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# FastEddy Parent Input Validation",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- NetCDF: `{report['netcdf']}`",
        f"- Manifest: `{report['manifest']}`",
        f"- Parent dataset ready for IC/BC converter POC: `{report['verdict']['parent_dataset_ready_for_icbc_converter_poc']}`",
        f"- Solver input directly runnable: `{report['verdict']['solver_input_directly_runnable']}`",
        f"- Production truth-ready: `{report['verdict']['production_truth_ready']}`",
        "",
        "## Verdict",
        "",
    ]
    for blocker in report["verdict"]["blockers"]:
        lines.append(f"- BLOCKER: {blocker}")
    for warning in report["verdict"]["warnings"]:
        lines.append(f"- WARNING: {warning}")
    if not report["verdict"]["blockers"]:
        lines.append("- No blocking issue in the parent NetCDF POC fields.")
    lines.extend(["", "## Field Audit", "", "| Field | Present | Status | Range | Expected |", "|---|---:|---|---|---|"])
    for field in report["fields"]:
        value_range = field["range"]
        if value_range:
            range_text = f"{value_range['min']:.6g}..{value_range['max']:.6g}"
        else:
            range_text = "n/a"
        lines.append(
            f"| `{field['field']}` | `{field['present']}` | `{field['validation_status']}` | `{range_text}` | {field['expected']} |"
        )
    lines.extend(["", "## Strategy To Remove Heuristics", ""])
    for item in report["strategy_to_remove_heuristics"]:
        lines.append(f"- {item}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--netcdf", type=Path, default=DEFAULT_NETCDF)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    netcdf = args.netcdf if args.netcdf.is_absolute() else ROOT / args.netcdf
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = validate_dataset(netcdf, manifest)
    write_markdown(report, output)
    if args.json_output:
        json_output = args.json_output if args.json_output.is_absolute() else ROOT / args.json_output
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {display_path(output)}")
    print(json.dumps(report["verdict"], indent=2))


if __name__ == "__main__":
    main()
