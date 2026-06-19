# FastEddy Linux/GPU Runbook

## Purpose

This runbook is for the first machine that actually has FastEddy and CUDA available.

The Mac/local repo can prepare and validate the prod-like package. The Linux/GPU machine validates the external FastEddy stages:

```text
GeoSpec
SimGrid
AROME IC/BC adapter
FastEddy
output conversion
```

## Prerequisites

Install or provide:

```text
NVIDIA GPU + working driver
CUDA toolkit compatible with FastEddy
MPI
FastEddy-model checkout/build
Python environment with this repo requirements
CorseWind AROME-to-FastEddy adapter/direct writer
```

Environment variables:

```bash
export FASTEDDY_COUPLER_DIR=/path/to/FastEddy-model/scripts/python_utilities/coupler
export FASTEDDY_BIN=/path/to/FastEddy-model/SRC/FEMAIN/FastEddy
export CORSEWIND_FASTEDDY_ADAPTER=/path/to/corsewind-arome-fasteddy-adapter
```

## Prepare Package

From the repo root:

```bash
.venv/bin/python scripts/prepare_fasteddy_prod_like_case.py \
  --allow-parent-warnings
```

Validate without running FastEddy:

```bash
.venv/bin/python scripts/validate_fasteddy_prod_like_package.py
```

Expected:

```text
status = pass
```

## Run Stages

Run static surface preprocessing:

```bash
.venv/bin/python scripts/run_fasteddy_prod_like_pipeline.py \
  --stages geospec simgrid
```

Run IC/BC generation:

```bash
.venv/bin/python scripts/run_fasteddy_prod_like_pipeline.py \
  --stages icbc
```

Run FastEddy:

```bash
.venv/bin/python scripts/run_fasteddy_prod_like_pipeline.py \
  --stages fasteddy
```

Or run all:

```bash
.venv/bin/python scripts/run_fasteddy_prod_like_pipeline.py
```

Run status:

```text
data/processed/benchmarks/fasteddy/prod_like_run_status.json
```

## First Acceptance Criteria

The first Linux/GPU test is useful only if these pass:

- GeoSpec accepts `gis/input_gis.nc`.
- SimGrid accepts the GeoSpec output and `fasteddy_real.in`.
- The IC/BC adapter creates complete initial and boundary files.
- FastEddy starts without missing-input errors.
- FastEddy runs a short test without numerical blow-up.
- Output contains `u/v/w` or equivalent fields required by the Wind2D contract.

## Do Not Claim Production Until

- `z0m` lookup is locally calibrated.
- IC/BC adapter output has its own validator.
- FastEddy output is compared against WindNinja and observations.
- Runtime budget is measured for realistic session windows.
- Wind2D conversion and publication gates are implemented.
