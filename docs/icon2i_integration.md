# ICON-2I 2.2 km Integration

This repository supports an optional raw ICON-2I wind layer for model
comparison. The layer is generated for the Wind2D-compatible JSON contract, but
the forecast engine integration is the first step; frontend display can be
added once the multi-model UI is generalized beyond AROME/MOLOCH.

## Source

ItaliaMeteo documents ICON-2I as the operational short-term model managed with
ARPAE and run on CINECA systems. The deterministic ICON-2I product is produced
at 00 and 12 UTC, covers Italy on a 2.2 km grid, and forecasts up to 72 hours.

MeteoHub exposes the public deterministic bundle as:

```text
ICON_2I_SURFACE_PRESSURE_LEVELS
```

The current public bundle endpoint is:

```text
https://meteohub.agenziaitaliameteo.it/api/datasets/ICON_2I_SURFACE_PRESSURE_LEVELS/opendata
```

The builder selects the newest bundle with U/V wind variables and downloads it
through:

```text
https://meteohub.agenziaitaliameteo.it/api/opendata/{filename}
```

`ICON_2I_RUC` is listed by MeteoHub, but its bundle endpoint currently returns
an empty list. Keep RUC as a future integration once single-variable browsing
or public bundles are stable enough for the watcher.

## Builder

Install the optional decoder stack:

```bash
pip install -r requirements-moloch.txt
```

Auto-discover and build from the latest public ICON-2I deterministic bundle:

```bash
python3 scripts/build_icon2i_corsica_wind_layer.py
```

Build from a local GRIB/NetCDF file:

```bash
python3 scripts/build_icon2i_corsica_wind_layer.py \
  --input data/raw/icon2i/icon2i_bundle.grib \
  --lead-hours 0 1 3 6 9 12 24 36 48 72
```

The output is:

```text
visualizations/wind2d/icon2i-corsica-latest.json
```

## Forecast Engine

ICON-2I is disabled by default. Enable it for a cycle with:

```bash
python3 scripts/run_forecast_update_engine.py \
  --once \
  --enable-icon2i
```

The default dataset is `ICON_2I_SURFACE_PRESSURE_LEVELS`. Override the source
for tests or local cache reuse with:

```bash
ICON2I_SOURCE_URL=https://... \
python3 scripts/run_forecast_update_engine.py --once --enable-icon2i
```

The engine records `icon2i_enabled`, `icon2i_dataset`, and
`icon2i_lead_hours` in the status file. ICON-2I is independent from the AROME
run used to decide whether WindNinja 50 m should be rebuilt.
