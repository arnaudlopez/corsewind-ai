# FastEddy IC/BC Adapter Contract

## Scope

This document defines the contract for the missing adapter between the validated AROME parent package and FastEddy initial/boundary-condition files.

The current repository can generate:

```text
icbc/arome_fasteddy_bridge.nc
geospec.json
simgrid.json
fasteddy_real.in
```

It does not yet generate final FastEddy IC/BC files. That is the remaining adapter/direct-writer work.

Machine-readable contract:

```text
benchmarks/fasteddy/icbc_contract.json
```

## Input

Required bridge file:

```text
icbc/arome_fasteddy_bridge.nc
```

Required bridge variables:

```text
U       m/s
V       m/s
W       m/s
T       K
THETA   K
RH      %
QVAPOR  kg/kg
P       Pa
HEIGHT  m
RHO     kg/m3
ALT     m3/kg
TSK     K
PSFC    Pa
HGT     m
XLAT    degrees_north
XLONG   degrees_east
```

Required static context:

```text
simgrid output
fasteddy_real.in
GeoSpec GIS product
WorldCover z0m table version
```

## Output

The adapter must produce FastEddy-compatible files equivalent to:

```text
FE_interp_INITIAL.0
FE_Bndys.*
adapter_manifest.json
```

The exact filenames may change if the direct-writer path differs from stock GenICBCs, but the manifest must record every generated file and the target FastEddy parameter file.

## Rules

- No synthetic meteorology may be introduced without an explicit field-level flag.
- Every unit conversion must be recorded.
- Every vertical interpolation must record source pressure/height levels and target z levels.
- Every temporal interpolation must record source valid times and weights.
- Initial state and boundary state must come from the same AROME run.
- Missing required fields must fail closed.
- Non-finite values must fail closed unless a documented mask is used.

## Validation

Before running FastEddy:

```bash
.venv/bin/python scripts/validate_fasteddy_prod_like_package.py
```

After implementing the adapter, extend that validator to inspect:

```text
FE_interp_INITIAL.0
FE_Bndys.*
adapter_manifest.json
```

## Current Status

```text
AROME bridge: ready
GeoSpec/SimGrid configs: ready
Stock GenICBCs direct compatibility: false
Adapter/direct writer: not implemented
```
