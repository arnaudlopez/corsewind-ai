# AROME 3D -> FastEddy POC

## Purpose

FastEddy is only useful for CorseWind if it is driven by a credible 3D atmospheric parent, not by a single AROME 10 m wind layer.

This POC makes the Ajaccio test case ready for a GPU solver benchmark:

1. inventory the current Meteo-France AROME WCS coverages;
2. select the useful AROME 0.025 deg isobaric variables;
3. download a small Ajaccio parent-data package;
4. decode the GRIB slices into a compact NetCDF parent state;
5. add Copernicus GLO-30 derived topography, land mask and roughness fields;
6. write an explicit readiness manifest for the future FastEddy IC/BC converter.

It does not claim to produce final production FastEddy IC/BC files yet.

## Why AROME 10 m Is Not Enough

The operational Wind2D/WindNinja path can work from:

```text
U / V / WIND_SPEED at height(10)
```

That is useful for WindNinja and for visualization, but not enough for a professional FastEddy run. FastEddy needs a 3D parent state with wind profiles, temperature or potential temperature, humidity, pressure or height, lateral-boundary evolution and surface parameters.

## Environment

Install the benchmark dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-benchmark.txt
```

Provide the Meteo-France API key through `.env`:

```text
METEOFRANCE_API_KEY=...
```

Do not commit `.env` or generated data.

## Step 1: Inventory AROME

Use AROME `0025`, not `001`. The 0.025 deg product exposes the isobaric 3D fields needed for the parent POC.

```bash
.venv/bin/python scripts/inventory_arome_fasteddy_inputs.py \
  --product arome \
  --resolution 0025 \
  --capabilities-output data/raw/arome_fasteddy_capabilities.xml
```

Outputs:

```text
data/processed/benchmarks/fasteddy/arome_fasteddy_inventory.json
reports/fasteddy_arome_inventory.md
```

The current useful fields are:

```text
U__ISOBARIC
V__ISOBARIC
VV__ISOBARIC
T__ISOBARIC
HU__ISOBARIC
Z__ISOBARIC
T__GROUND
```

## Step 2: Build The Download Plan

The mapping lives in:

```text
benchmarks/fasteddy/arome_to_fasteddy_requirements.json
```

Build the Ajaccio mini-case plan:

```bash
.venv/bin/python scripts/prepare_arome_fasteddy_poc.py \
  --product arome \
  --resolution 0025 \
  --bbox 8.62 41.82 8.90 42.00 \
  --lead-hours 0
```

Outputs:

```text
data/processed/benchmarks/fasteddy/arome_poc/arome_fasteddy_poc_download_plan.json
data/processed/benchmarks/fasteddy/arome_poc/fasteddy_parent_schema.json
reports/fasteddy_arome_poc_readiness.md
```

Meteo-France WCS requires a single `pressure(...)` slice per request. The default planned levels are:

```text
850, 700, 600, 500, 300 hPa
```

For a small smoke test:

```bash
.venv/bin/python scripts/prepare_arome_fasteddy_poc.py \
  --product arome \
  --resolution 0025 \
  --bbox 8.62 41.82 8.90 42.00 \
  --lead-hours 0 \
  --pressure-levels-hpa 850 700 \
  --download
```

For the full current POC:

```bash
.venv/bin/python scripts/prepare_arome_fasteddy_poc.py \
  --product arome \
  --resolution 0025 \
  --bbox 8.62 41.82 8.90 42.00 \
  --lead-hours 0 \
  --download
```

This downloads 31 GRIB inputs for H+0: 6 isobaric requirements x 5 pressure levels, plus surface temperature.

## Step 3: Add Copernicus GLO-30 DEM

Download the DEM tile needed by the Ajaccio POC bbox:

```bash
.venv/bin/python scripts/download_copernicus_dem_tiles.py \
  --bbox 8.62 41.82 8.90 42.00
```

Output:

```text
data/raw/dem/copernicus_glo30/Copernicus_DSM_COG_10_N41_00_E008_00_DEM.tif
```

The parent builder samples this DEM onto the AROME parent grid and derives:

```text
topography_m
landmask
z0m
```

The roughness field is intentionally simple for the POC: sea `0.0002 m`, low land `0.03 m`, rough terrain above 200 m `0.08 m`.

## Step 4: Build The Parent NetCDF

```bash
.venv/bin/python scripts/build_fasteddy_parent_poc.py
```

Outputs:

```text
data/processed/benchmarks/fasteddy/arome_poc/fasteddy_parent_poc.nc
data/processed/benchmarks/fasteddy/arome_poc/fasteddy_parent_poc_manifest.json
```

The NetCDF contains:

```text
u
v
w
temperature
relative_humidity
geopotential_or_height
height_m
potential_temperature
pressure_pa
surface_temperature
topography_m
landmask
z0m
```

Current validated POC dimensions:

```text
pressure_hpa: 5
latitude: 8
longitude: 12
pressure levels: 850, 700, 600, 500, 300 hPa
expected_grib_inputs: 31
decoded_grib_inputs: 31
```

Current readiness gate:

```text
fasteddy_parent_test_ready = true
production_fasteddy_ready = false
```

`production_fasteddy_ready` remains false until we generate the final FastEddy IC/BC files and run the solver.

## Validation Command

```bash
.venv/bin/python - <<'PY'
import json
import xarray as xr

manifest = json.load(open("data/processed/benchmarks/fasteddy/arome_poc/fasteddy_parent_poc_manifest.json"))
print(manifest["fasteddy_parent_test_ready"])
print(manifest["expected_grib_inputs"], manifest["decoded_grib_inputs"])
print(manifest["surface_fields"])

ds = xr.open_dataset("data/processed/benchmarks/fasteddy/arome_poc/fasteddy_parent_poc.nc")
print(dict(ds.sizes))
print(list(ds.data_vars))
print([float(x) for x in ds.pressure_hpa.values])
PY
```

Expected result:

```text
True
31 31
surface_fields.added = True
```

## Known Limits Before A Real FastEddy Run

- `VV__ISOBARIC` is pressure vertical velocity in `Pa/s`; the final converter must convert or remap it if FastEddy requires geometric vertical velocity.
- `Z__ISOBARIC` is geopotential; the POC derives `height_m = Z / 9.80665`.
- `surface_pressure` is still not provided as a source field in the current mapping.
- The current file is a parent-state NetCDF, not a complete FastEddy `FE_Bndys` / `FE_interp` package.
- The next step is the IC/BC writer plus a real Ajaccio/Bonifacio GPU benchmark against the WindNinja baseline.
