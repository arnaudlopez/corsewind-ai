# AROME 3D -> FastEddy POC

## Purpose

FastEddy is only useful for CorseWind if it is driven by a credible 3D atmospheric parent, not by a single AROME 10 m wind layer.

This POC makes the Ajaccio test case ready for a GPU solver benchmark:

1. inventory the current Meteo-France AROME WCS coverages;
2. select the useful AROME 0.025 deg isobaric variables;
3. download a small Ajaccio parent-data package;
4. decode the GRIB slices into a compact NetCDF parent state;
5. add Copernicus GLO-30 derived topography plus ESA WorldCover land cover, land mask and roughness fields;
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

## Step 3: Add Static Surface Truth Sources

Download the DEM tile needed by the Ajaccio POC bbox:

```bash
.venv/bin/python scripts/download_copernicus_dem_tiles.py \
  --bbox 8.62 41.82 8.90 42.00
```

Download the ESA WorldCover 10 m tile needed by the same bbox:

```bash
.venv/bin/python scripts/download_esa_worldcover_tiles.py \
  --bbox 8.62 41.82 8.90 42.00
```

Outputs:

```text
data/raw/dem/copernicus_glo30/Copernicus_DSM_COG_10_N41_00_E008_00_DEM.tif
data/raw/landcover/esa_worldcover_v200_2021/ESA_WorldCover_10m_2021_v200_N42E006_Map.tif
```

The parent builder samples this DEM onto the AROME parent grid and derives:

```text
topography_m
landcover_class
landmask
z0m
```

Provider strategy:

- DEM: Copernicus GLO-30, because it is global, stable and already used by the WindNinja pipeline.
- Land cover: ESA WorldCover 10 m 2021 v200, because it is global, coastal-friendly, cloud optimized and much finer than CORINE 100 m.
- Fallback/context: CORINE Land Cover 2018 100 m can be useful for Europe-wide QA, but it is too coarse to be the primary source for windsurf beach/coast roughness.

`z0m` is now derived from ESA WorldCover classes through an explicit lookup table. It is no longer an invented constant field. The lookup still needs local calibration against observations and coastline QA before production.

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
surface_pressure, when the AROME P__GROUND / pressure-at-ground coverage has been downloaded
topography_m
landcover_class
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

For the stricter source/solver audit:

```bash
.venv/bin/python scripts/validate_fasteddy_parent_inputs.py
```

Output:

```text
reports/fasteddy_parent_input_validation.md
```

The current expected verdict is:

```text
parent_dataset_ready_for_icbc_converter_poc = true
solver_input_directly_runnable = false
production_truth_ready = false
```

The parent dataset has no invented meteorology: wind, temperature, humidity, geopotential and surface temperature come from AROME GRIB files. Surface pressure is expected from AROME `P__GROUND` or `PRESSURE__GROUND_OR_WATER_SURFACE` on the next authenticated refresh. Surface properties come from Copernicus GLO-30 plus ESA WorldCover 10 m; `z0m` is land-cover-derived but still requires local calibration before production.

## Known Limits Before A Real FastEddy Run

- `VV__ISOBARIC` is pressure vertical velocity in `Pa/s`; the final converter must convert or remap it if FastEddy requires geometric vertical velocity.
- `Z__ISOBARIC` is geopotential; the POC derives `height_m = Z / 9.80665`.
- `surface_pressure` is mapped to AROME `P__GROUND` / `PRESSURE__GROUND_OR_WATER_SURFACE`, but the current local generated NetCDF must be refreshed with an authenticated Meteo-France key to include it.
- `z0m` now comes from ESA WorldCover classes, but the class-to-roughness lookup must be locally calibrated.
- The current file is a parent-state NetCDF, not a complete FastEddy `FE_Bndys` / `FE_interp` package.
- The next step is the IC/BC writer plus a real Ajaccio/Bonifacio GPU benchmark against the WindNinja baseline.
