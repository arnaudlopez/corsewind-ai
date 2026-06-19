# MOLOCH 1.2 km Integration

This repository supports an optional raw MOLOCH wind layer for the Wind2D viewer.
It is a display/parent-model layer, not a replacement for the WindNinja 50 m
downscaling pipeline.

## Source

CNR-ISAC documents MOLOCH as a high-resolution Italy forecast with 10 m wind
products, about 1.2 to 1.25 km horizontal spacing, 60 atmospheric levels, and
about 48 hours / 2 days of forecast range. MeteoHub lists CNR-ISAC
BOLAM-MOLOCH data as GRIB products and states that data bundles can be
downloaded without login, while extraction workflows require registration.

Useful source pages:

- CNR-ISAC forecast overview: https://www.isac.cnr.it/dinamica/projects/forecasts/
- CNR-ISAC MOLOCH product page: https://www.isac.cnr.it/dinamica/projects/forecast_dpc/moloch_en.htm
- MeteoHub user guide: https://meteohub.agenziaitaliameteo.it/ui/user-guide
- ItaliaMeteo CKAN dataset: https://dati.agenziaitaliameteo.it/dataset/previsioni-meteorologiche-modelli-bolam-moloch-globo-di-isac-cnr

## Builder

Install the optional decoder stack:

```bash
pip install -r requirements-moloch.txt
```

Auto-discover and build from the latest public MeteoHub bundle:

```bash
python3 scripts/build_moloch_corsica_wind_layer.py
```

Build from a local GRIB/NetCDF file:

```bash
python3 scripts/build_moloch_corsica_wind_layer.py \
  --input data/raw/moloch/moloch_bundle.grib2
```

By default, the builder selects the newest public `MOLOCH` bundle containing
10 m U/V wind variables and publishes every forecast lead hour available in the
source bundle. Pass `--lead-hours` only when you intentionally want a smaller
subset.

Build from a direct MeteoHub bundle URL:

```bash
MOLOCH_SOURCE_URL="https://..." \
python3 scripts/build_moloch_corsica_wind_layer.py
```

The output is:

```text
visualizations/wind2d/moloch-corsica-latest.json
```

The JSON follows the same lightweight contract as
`arome-corsica-latest.json`: `bbox_wgs84`, `run_time_utc`,
`forecast_steps[]`, and per-step `speed_ms`, `u_ms`, `v_ms`.

## Forecast Engine

MOLOCH is disabled by default. Enable it for a cycle with:

```bash
python3 scripts/run_forecast_update_engine.py \
  --once \
  --enable-moloch \
  --moloch-input data/raw/moloch/moloch_bundle.grib2
```

or configure `.env`:

```bash
MOLOCH_SOURCE_URL=https://...
```

then run:

```bash
python3 scripts/run_forecast_update_engine.py --once --enable-moloch
```

If no MOLOCH source is configured, the engine now auto-discovers the latest
public `MOLOCH` bundle through MeteoHub. `MOLOCH_SOURCE_URL` remains useful as
a debug or replay override.

## Viewer

The Wind2D viewer loads `moloch-corsica-latest.json` opportunistically.
When present, the `M` layer button becomes active and displays MOLOCH as a raw
10 m wind field. When absent, the button stays disabled with a tooltip.

AROME and MOLOCH are exclusive raw layers in the UI. WindNinja remains the
preferred downscaled layer when a matching 50 m step is available.

The forecast strip is rebuilt from the active raw model, so a MOLOCH payload
with hourly lead times displays every published hourly forecast step.
