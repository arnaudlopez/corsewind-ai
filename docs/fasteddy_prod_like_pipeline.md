# FastEddy Prod-Like Pipeline

## Goal

This pipeline turns the validated AROME parent dataset into a prod-like FastEddy real-case package.

It is stricter than the old smoke benchmark:

- uses the AROME 3D parent NetCDF, not a single mean wind vector;
- uses Copernicus GLO-30 topography;
- uses ESA WorldCover 10 m land cover;
- uses a versioned WorldCover-to-`z0m` table;
- writes a GeoSpec-compatible GIS NetCDF;
- writes GeoSpec, SimGrid and IC/BC adapter configs;
- records exactly what is ready and what remains external.

FastEddy's official real-case workflow has three preprocessing steps: GeoSpec, SimGrid and GenICBCs. GeoSpec consumes GIS terrain/land-cover data, SimGrid defines the FastEddy grid, and GenICBCs creates the initial and boundary files from a mesoscale parent. See the FastEddy real-case tutorial:

```text
https://fasteddy-model.readthedocs.io/en/latest/Tutorials/cases_real/WRF_coupling_case0.html
```

## Static Versus Dynamic

Prepare once or when surface data changes:

```text
DEM
land cover
landmask
roughness lookup
domain definitions
GeoSpec/SimGrid static GIS input
```

Regenerate for each AROME run:

```text
AROME parent GRIB downloads
fasteddy_parent_poc.nc
arome_fasteddy_bridge.nc
IC/BC files
FastEddy outputs
Wind2D rasters/manifests
```

## Current Prod-Like Status

Current generated package:

```text
data/processed/benchmarks/fasteddy/prod_like_status.json
reports/fasteddy_prod_like_plan.md
```

Current readiness:

```text
prod_like_package_ready = true
stock_geospec_simgrid_ready = true
stock_genicbcs_compatible_now = false
```

`stock_genicbcs_compatible_now` is false because stock FastEddy GenICBCs expects WRF/FastEddy parent files. We now generate a normalized AROME bridge NetCDF, but the remaining production step is the CorseWind AROME-to-FastEddy IC/BC adapter or direct writer.

## Prepare The Package

After refreshing AROME and building `fasteddy_parent_poc.nc`:

```bash
.venv/bin/python scripts/prepare_fasteddy_prod_like_case.py \
  --allow-parent-warnings
```

Outputs for each enabled zone:

```text
gis/input_gis.nc
gis/worldcover_z0m_corse_v1.csv
geospec.json
simgrid.json
genicbcs_arome_adapter.json
icbc/arome_fasteddy_bridge.nc
fasteddy_real.in
```

The warning currently allowed is the expected `z0m` local-calibration warning. It does not mean `z0m` is invented; it means the WorldCover lookup must be tuned against Corsican local observations before production claims.

## Run Or Dry-Run The Pipeline

Dry-run on a machine without FastEddy installed:

```bash
.venv/bin/python scripts/run_fasteddy_prod_like_pipeline.py \
  --dry-run \
  --allow-missing-tools
```

On a Linux/GPU machine with FastEddy installed:

```bash
export FASTEDDY_COUPLER_DIR=/path/to/FastEddy-model/scripts/python_utilities/coupler
export CORSEWIND_FASTEDDY_ADAPTER=/path/to/corsewind-arome-fasteddy-adapter
export FASTEDDY_BIN=/path/to/FastEddy-model/SRC/FEMAIN/FastEddy

.venv/bin/python scripts/run_fasteddy_prod_like_pipeline.py
```

Stage by stage:

```bash
.venv/bin/python scripts/run_fasteddy_prod_like_pipeline.py --stages geospec simgrid
.venv/bin/python scripts/run_fasteddy_prod_like_pipeline.py --stages icbc
.venv/bin/python scripts/run_fasteddy_prod_like_pipeline.py --stages fasteddy
```

Run status:

```text
data/processed/benchmarks/fasteddy/prod_like_run_status.json
```

## What Is Still Needed For Production

- Implement the AROME-to-FastEddy IC/BC adapter/direct writer.
- Run GeoSpec and SimGrid using the FastEddy coupler utilities on a Linux/GPU environment.
- Calibrate `benchmarks/fasteddy/worldcover_z0m_corse_v1.csv` with local station/rider/coastline evidence.
- Convert FastEddy outputs into the Wind2D raster/data-tile contract.
- Add this pipeline to the autonomous forecast engine after solver runtime is validated.

## Additional Contracts

```text
docs/fasteddy_icbc_contract.md
docs/fasteddy_wind2d_output_contract.md
docs/fasteddy_linux_gpu_runbook.md
benchmarks/fasteddy/icbc_contract.json
benchmarks/fasteddy/wind2d_output_contract.json
```

Validate the package without FastEddy installed:

```bash
.venv/bin/python scripts/validate_fasteddy_prod_like_package.py
```

## Why This Is Prod-Like But Not Yet Prod

The data package is prod-like because it uses real forecast and real static surface sources and produces the right classes of inputs for the official FastEddy real-case workflow.

It is not yet production because the IC/BC adapter is not implemented and validated, and because `z0m` values are professional initial values rather than locally calibrated values.
