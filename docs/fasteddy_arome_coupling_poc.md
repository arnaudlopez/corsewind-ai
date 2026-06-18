# AROME 3D -> FastEddy POC

## Purpose

FastEddy is only useful for CorseWind if it is driven by a credible 3D atmospheric parent, not by a single AROME 10 m wind layer.

This POC prepares the first four steps needed to become test-ready:

1. inventory all AROME WCS coverages currently exposed by Meteo-France;
2. map those coverages to FastEddy parent-model requirements;
3. produce/download a mini Ajaccio AROME parent-data package;
4. build an explicit parent-schema/manifest for the future FastEddy IC/BC converter.

It does not claim to produce final production IC/BC files yet.

## Why the Current AROME Layer Is Not Enough

The current operational layer downloads only:

```text
WIND_SPEED / U / V at height(10)
```

That is enough for WindNinja and useful for QES smoke tests, but not enough for a professional FastEddy run. FastEddy needs a 3D parent state with wind profiles, thermodynamics, humidity, pressure/height, lateral boundary evolution and surface parameters.

## Step 1: Inventory AROME

With a Meteo-France API key in `.env`:

```bash
python3 scripts/inventory_arome_fasteddy_inputs.py \
  --product arome \
  --resolution 0025 \
  --capabilities-output data/raw/arome_fasteddy_capabilities.xml
```

Outputs:

```text
data/processed/benchmarks/fasteddy/arome_fasteddy_inventory.json
reports/fasteddy_arome_inventory.md
```

The report tells us whether each FastEddy requirement is available as a real 3D field, only as a 10 m fallback, or missing.

## Step 2: Requirements Mapping

The mapping lives in:

```text
benchmarks/fasteddy/arome_to_fasteddy_requirements.json
```

Core required fields:

- 3D U wind;
- 3D V wind;
- 3D temperature or potential temperature;
- 3D humidity;
- pressure/geopotential information to map vertical levels to height.

Desired fields:

- vertical velocity;
- surface pressure;
- surface or skin temperature.

External required surface fields if AROME does not expose them cleanly:

- land/sea mask;
- roughness length;
- land cover / coastline;
- DEM.

## Step 3: Prepare Ajaccio Mini Case

After the inventory exists:

```bash
python3 scripts/prepare_arome_fasteddy_poc.py \
  --product arome \
  --resolution 0025 \
  --bbox 8.62 41.82 8.90 42.00 \
  --lead-hours 0
```

This writes:

```text
data/processed/benchmarks/fasteddy/arome_poc/arome_fasteddy_poc_download_plan.json
data/processed/benchmarks/fasteddy/arome_poc/fasteddy_parent_schema.json
reports/fasteddy_arome_poc_readiness.md
```

To attempt the downloads:

```bash
python3 scripts/prepare_arome_fasteddy_poc.py \
  --product arome \
  --resolution 0025 \
  --bbox 8.62 41.82 8.90 42.00 \
  --lead-hours 0 \
  --download
```

The script intentionally marks 10 m variables as fallback, not production-ready fields.

For the current public API, AROME `0025` exposes useful isobaric fields such as:

```text
U__ISOBARIC
V__ISOBARIC
VV__ISOBARIC
T__ISOBARIC
HU__ISOBARIC
Z__ISOBARIC
```

Meteo-France WCS requires a single pressure slice per request. The POC therefore downloads one GRIB per variable and pressure level. Default planned levels are:

```text
850, 700, 600, 500, 300 hPa
```

Use `--pressure-levels-hpa` to limit a test run, for example:

```bash
python3 scripts/prepare_arome_fasteddy_poc.py \
  --product arome \
  --resolution 0025 \
  --bbox 8.62 41.82 8.90 42.00 \
  --lead-hours 0 \
  --pressure-levels-hpa 850 700 \
  --download
```

## Step 4: Build Parent POC Manifest

```bash
python3 scripts/build_fasteddy_parent_poc.py
```

Outputs:

```text
data/processed/benchmarks/fasteddy/arome_poc/fasteddy_parent_poc_manifest.json
data/processed/benchmarks/fasteddy/arome_poc/fasteddy_parent_poc.nc
```

The NetCDF is only written when readable fallback GeoTIFFs exist. GRIB 3D fields are currently reported as `unsupported_grib_inputs`; decoding them into a true 4D parent package is the next implementation step.

## Readiness Decision

Use the POC report as the gate:

```text
production_fasteddy_ready = true
```

only when required fields are available as direct 3D fields, not fallback 10 m fields.

If required fields are missing or fallback-only, FastEddy should not be treated as product-ready. The next move is either:

- add the missing AROME product/resolution/source;
- decode GRIB 3D fields and build true IC/BC files;
- or park FastEddy and continue with QES/WindNinja.
